import os
import math
import time
import copy 
import numpy as np
from logging import getLogger
from collections import OrderedDict

import apex
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.optim import get_optimizer
from src.utils import to_cuda, dict_to_cuda
from src.dataset.wiki_dataset import wiki_rr_trainset, wiki_qlm_trainset, DataCollatorForRelevanceRanking, DataCollatorForMaskedQueryPrediction

logger = getLogger()


class Trainer(object):

    def __init__(self, model, tokenizer, params):
        """
        Initialize trainer.
        """

        self.model = model
        self.params = params

        # epoch / iteration size
        self.epoch_size = params.epoch_size

        # tokenizer
        self.tokenizer = tokenizer

        # data iterators
        self.iterators = {}

        # data collators
        self.rr_collator = DataCollatorForRelevanceRanking(self.tokenizer, "long" in params.model_type)
        self.qlm_collator = DataCollatorForMaskedQueryPrediction(self.tokenizer, params.mlm_probability, "long" in params.model_type, params.qlm_mask_mode)
        
        # set parameters
        self.set_parameters()

        # float16 / distributed (no AMP)
        assert params.amp >= 1 or not params.fp16
        assert params.amp >= 0 or params.accumulate_gradients == 1
        if params.multi_gpu and params.amp == -1:
            logger.info("Using nn.parallel.DistributedDataParallel ...")
            self.model = nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[params.local_rank], 
                output_device=params.local_rank, 
                broadcast_buffers=True,
                find_unused_parameters=True
                )

        # set optimizers
        self.set_optimizers()

        # float16 / distributed (AMP)
        if params.amp >= 0:
            self.init_amp()
            if params.multi_gpu:
                logger.info("Using apex.parallel.DistributedDataParallel ...")
                self.model = apex.parallel.DistributedDataParallel(self.model, delay_allreduce=True)

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_pairs = 0
        stat_keys = [('processed_p', 0)]
        if params.qlm_steps is not None:
            stat_keys += [('QLM-%s' % lang_pair, []) for lang_pair in params.qlm_steps]
        if params.rr_steps is not None:
            stat_keys += [('RR-%s' % lang_pair, []) for lang_pair in params.rr_steps]
        self.stats = OrderedDict(stat_keys)
        stat_keys.pop(0)
        self.epoch_scores = OrderedDict(copy.deepcopy(stat_keys))
        self.last_time = time.time()

    def set_parameters(self):
        """
        Set parameters.
        """
        params = self.params
        self.parameters = {}
        named_params = [(k, p) for k, p in self.model.named_parameters() if p.requires_grad]

        # model (excluding memory values)
        self.parameters['model'] = [p for k, p in named_params]

        # log
        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1

    def set_optimizers(self):
        """
        Set optimizers.
        """
        params = self.params
        self.optimizers = {}

        # model optimizer (excluding memory values)
        self.optimizers['model'] = get_optimizer(self.parameters['model'], params.optimizer)

        # log
        logger.info("Optimizers: %s" % ", ".join(self.optimizers.keys()))

    def init_amp(self):
        """
        Initialize AMP optimizer.
        """
        params = self.params
        assert params.amp == 0 and params.fp16 is False or params.amp in [1, 2, 3] and params.fp16 is True
        opt_names = self.optimizers.keys()
        self.model, optimizers = apex.amp.initialize(
            self.model,
            [self.optimizers[k] for k in opt_names],
            opt_level=('O%i' % params.amp)
        )
        self.optimizers = {
            opt_name: optimizer
            for opt_name, optimizer in zip(opt_names, optimizers)
        }

    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")
            # exit()

        params = self.params

        # optimizers
        names = self.optimizers.keys()
        optimizers = [self.optimizers[k] for k in names]

        # regular optimization
        if params.amp == -1:
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            if params.clip_grad_norm > 0:
                for name in names:
                    clip_grad_norm_(self.parameters[name], params.clip_grad_norm)
            for optimizer in optimizers:
                optimizer.step()

        # AMP optimization
        else:
            if self.n_iter % params.accumulate_gradients == 0:
                with apex.amp.scale_loss(loss, optimizers) as scaled_loss:
                    scaled_loss.backward()
                if params.clip_grad_norm > 0:
                    for name in names:
                        clip_grad_norm_(apex.amp.master_params(self.optimizers[name]), params.clip_grad_norm)
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                with apex.amp.scale_loss(loss, optimizers, delay_unscale=True) as scaled_loss:
                    scaled_loss.backward()

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % 5 != 0:
            return

        s_iter = "%7i - " % self.n_total_iter
        s_stat = ' || '.join([
            '{}: {:7.3f}'.format(k, np.mean(v)) for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # learning rates
        s_lr = " - "
        for k, v in self.optimizers.items():
            s_lr = s_lr + (" - %s LR: " % k) + " / ".join("{:.3e}".format(group['lr']) for group in v.param_groups)
        
        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        p_speed = "{:7.2f} qd pair/s - ".format(
            self.stats['processed_p'] * 1.0 / diff
            )
        self.stats['processed_p'] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + p_speed + s_stat + s_lr)


    def save_checkpoint(self):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return

        # huggingface saves (more useful in our case for finetuning)

        logger.info(f"Saving epoch {self.epoch} ...")
        path = os.path.join(self.params.dump_path, f"huggingface-{self.epoch}")
        if not os.path.exists(path): os.makedirs(path)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def end_epoch(self):
        """
        End the epoch.
        """

        # print epoch loss 

        self.epoch_stat = ' || '.join([
            '{}: {:7.3f}'.format(k, np.mean(v)) for k, v in self.epoch_scores.items()
            if type(v) is list and len(v) > 0
        ])

        for k in self.epoch_scores.keys():
            if type(self.epoch_scores[k]) is list:
                del self.epoch_scores[k][:]

        logger.info("EPOCH LOSS: " + self.epoch_stat)
        self.save_checkpoint()
        self.epoch += 1
        self.n_iter = 0

    def get_iterator(self, obj_name, lang_pair):

        params = self.params
        
        if obj_name == "rr":
            dataset = wiki_rr_trainset(
                    lang_pair = lang_pair, 
                    num_neg = params.num_neg, 
                    neg_val = params.neg_val,
                    params=params
                    )
        elif obj_name == "qlm":
            dataset = wiki_qlm_trainset(
                lang_pair = lang_pair, 
                neg_val = params.neg_val,
                params=params
                )
        
        sampler = DistributedSampler(dataset, shuffle=True)

        dataloader = DataLoader(
            dataset,
            batch_size = params.batch_size,
            shuffle = False,
            num_workers = 0,
            collate_fn = self.rr_collator if obj_name == "rr" else self.qlm_collator,
            sampler = sampler
        )

        iterator = iter(dataloader)
        self.iterators[(obj_name, lang_pair)] = iterator
        logger.info("Created new training data iterator (%s) ..." % ','.join([str(x) for x in [obj_name, lang_pair]]))
        
        return iterator

    def get_batch(self, obj_name, lang_pair):
        
        iterator = self.iterators.get(
            (obj_name, lang_pair), 
            None
            )

        if iterator is None:
            iterator = self.get_iterator(obj_name, lang_pair) # if there is no such iterator, create one
        
        try:
            x = next(iterator)
        except StopIteration:
            iterator = self.get_iterator(obj_name, lang_pair)
            x = next(iterator)

        return x

    def qlm_step(self, lang_pair, lambda_coeff):

        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        
        params = self.params
        self.model.train()

        inputs = self.get_batch("qlm", lang_pair)

        # if 'long' in params.model_type:
        #     if self.check_for_long_queries(inputs['attention_mask']):
        #         ## fail the test: long queries detected
        #         logger.info("QLM step skipping long queries")
        #         return

        if 'long' in params.model_type:
            inputs['attention_mask'] = self.global_attention_safety_check(inputs['attention_mask'])

        inputs = dict_to_cuda(inputs)
        inputs["mode"] = "mlm"
        outputs = self.model(inputs)
        loss = outputs[0]
        self.stats[('QLM-%s' % lang_pair)].append(loss.item())
        self.epoch_scores[('QLM-%s' % lang_pair)].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.stats['processed_p'] += inputs["attention_mask"].size(0)
        self.n_pairs += inputs["attention_mask"].size(0)

    def rr_step(self, lang_pair, lambda_coeff):
        
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        
        params = self.params
        self.model.train()

        inputs = self.get_batch("rr", lang_pair)

        # if 'long' in params.model_type:
        #     if self.check_for_long_queries(inputs['attention_mask']):
        #         ## fail the test: long queries detected
        #         logger.info("RR step skipping long queries")
        #         return
        
        if 'long' in params.model_type:
            inputs['attention_mask'] = self.global_attention_safety_check(inputs['attention_mask'])

        inputs = dict_to_cuda(inputs)
        inputs["mode"] = "seqcls"
        outputs = self.model(inputs)
        loss = outputs[0]
        self.stats[('RR-%s' % lang_pair)].append(loss.item())
        self.epoch_scores[('RR-%s' % lang_pair)].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        qd_pairs = inputs["attention_mask"].size(0)
        pos_qd_pairs = int(qd_pairs / (1 + params.num_neg))
        self.stats['processed_p'] += inputs["attention_mask"].size(0)
        self.n_pairs += pos_qd_pairs

    def check_for_long_queries(self, tensor, length=128):
        
        ## for models that use longformer attention mechanism
        ## when query is obnormally long, it may cause unusual high GPU memory usage
        ## and therefore program failure
        ## thus, we check for those long queries and skip them!

        ## 07/24/2020: deprecating this method because skipping batches can cause waiting with DDP

        if 'long' not in self.params.model_type:
            assert False, "only check for long queries with mBERT-long!"
        
        return any((tensor==2).sum(dim=1) >= length)

    def global_attention_safety_check(self, tensor):
        
        if 'long' not in self.params.model_type:
            return tensor
        else:
            idxs = ((tensor==2).sum(dim=1) >= 256).nonzero().squeeze()
            if len(idxs.shape) != 0:
                if idxs.shape[0] == 0:
                    return tensor
            else:
                # just one row to replace
                idxs = idxs.unsqueeze(dim=0)

            replacement_attention_mask = torch.LongTensor([1]*512 + [0]*512)
            for idx in idxs:
                tensor[idx] = replacement_attention_mask
            return tensor
import time
import glob
import copy
import json
import torch
import pickle
import random
import os, sys
import logging
import argparse
import pytrec_eval
import numpy as np
from collections import OrderedDict

from torch import nn
from apex import amp
import torch.distributed as dist
from torch.utils.data import DataLoader
from apex.parallel import DistributedDataParallel
from torch.distributed import get_rank, get_world_size
from torch.utils.data.distributed import DistributedSampler

from src.utils import CustomFormatter, set_seed
from src.dataset.clef_dataset import Retrieval_Trainset, Retrieval_Testset, train_collate, test_collate


from transformers import XLMTokenizer, XLMForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from BertForXLRetrieval import BertForXLRetrieval
from BertLongForXLRetrieval import BertLongForXLRetrieval

torch.set_printoptions(profile="full")

parser = argparse.ArgumentParser()

parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

parser.add_argument('--model_type', choices=["mbert", "mbert-long", "xlm100", "xlmr"])
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--source_lang', type=str)
parser.add_argument('--target_lang', type=str)

parser.add_argument("--eval_step", type=int, default=1)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_neg', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=20)

parser.add_argument('--cased', action='store_true', default=False)

parser.add_argument('--encoder_lr', type=float, default=2e-5)
parser.add_argument('--projector_lr', type=float, default=1e-4)
parser.add_argument('--num_ft_encoders', type=int, default=3)

parser.add_argument('--dataset', type=str, choices=["clef", "wiki-clir","mix"])

parser.add_argument('--apex_level', type=str, default="O2")
parser.add_argument('--seed', type=int, default=611)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--full_doc_length', action='store_true', default=True)
args = parser.parse_args()

dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)

# set up logging-related stuff
# slurm_id = os.environ.get('SLURM_JOB_ID')
model_type_in_path = args.model_path
model_type_in_path = model_type_in_path.replace("/", "-")
if model_type_in_path.startswith("-"):
    model_type_in_path = model_type_in_path[1:]

cased_dir = "cased" if args.cased else "uncased"
log_dir = f'finetune-logs/{args.dataset}/{cased_dir}/{args.model_type}-{args.source_lang}{args.target_lang}'
log_dir += f"/{model_type_in_path}"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(f'{log_dir}/process-{get_rank()}.log')
formatter = CustomFormatter('%(adjustedTime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info(args)

if args.model_type == "mbert":
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = BertForXLRetrieval.from_pretrained(args.model_path)
    max_len, out_dim = model.bert.embeddings.position_embeddings.weight.shape
    num_encoder = len(model.bert.encoder.layer)
elif args.model_type == "mbert-long":
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = BertLongForXLRetrieval.from_pretrained(args.model_path)
    max_len, out_dim = model.bert.embeddings.position_embeddings.weight.shape
    num_encoder = len(model.bert.encoder.layer)
    logger.info(model)
elif args.model_type == "xlm100":
    if not args.cased:
        assert False
    tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
    model = XLMForSequenceClassification.from_pretrained('xlm-mlm-100-1280')
    max_len, out_dim = model.transformer.position_embeddings.weight.shape
    num_encoder = len(model.transformer.attentions)
elif args.model_type == "xlmr":
    if not args.cased:
        assert False

    args.apex_level = "O1"
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base')
    max_len, out_dim = model.roberta.embeddings.position_embeddings.weight.shape
    num_encoder = len(model.roberta.encoder.layer)
else:
    assert False

class CLIR:
    
    def __init__(self, model):
        
        self._model = model
        self._rank = get_rank()
        self.best_dev_map, self.best_test_map, self.best_epoch = 0.0, 0.0, 0
        self.eval_interval = args.eval_step
        self.cased_dir = "cased" if args.cased else "uncased"

        logger.info(f"Evaluating every {self.eval_interval} epochs ...")

        if os.path.exists("/home/puxuan"):
            home_dir = "/home/puxuan" # GPU server
        else:
            home_dir = "/mnt/home/puxuan" # cluster
        
        # read data
        if args.dataset != "mix":

            if args.dataset == 'clef':

                # CLEF data
                data_dir = f"{home_dir}/CLIR-project/Evaluation_data/process-clef/{self.cased_dir}"

                logger.info(f"reading data from {data_dir}")

                rel = pickle.load(open(f"{data_dir}/relevance/{args.target_lang}_rel.pkl", 'rb'))
                split = pickle.load(open(f"{data_dir}/relevance/{args.target_lang}_split.pkl", 'rb'))
                queries = pickle.load(open(f"{data_dir}/queries/{args.source_lang}_query.pkl", 'rb'))
                documents = pickle.load(open(f"{data_dir}/full_documents/{args.target_lang}_document.pkl", 'rb')) if args.full_doc_length \
                            else pickle.load(open(f"{data_dir}/documents/{args.target_lang}_document.pkl", 'rb'))

            elif args.dataset == "wiki-clir":

                # wiki-CLIR data
                data_dir = f"{home_dir}/wiki-clir/{self.cased_dir}"
                
                logger.info(f"reading data from {data_dir}")
                
                rel = pickle.load(open(f"{data_dir}/relevance/{args.target_lang}_rel.pkl", 'rb'))
                split = pickle.load(open(f"{data_dir}/relevance/{args.target_lang}_split.pkl", 'rb'))
                queries = pickle.load(open(f"{data_dir}/queries/{args.source_lang}_query.pkl", 'rb'))
                documents = pickle.load(open(f"{data_dir}/documents/{args.target_lang}_document.pkl", 'rb'))

            else:
                assert False

            # MAP evaluator
            self.qrel = self.get_qrel(rel)
            self.evaluator = pytrec_eval.RelevanceEvaluator(self.qrel, {'map'})
            
            # dataloaders
            train_dataset = Retrieval_Trainset(id2query=queries, id2doc=documents, rel=rel, split=split, num_neg=args.num_neg, neg_value=0)
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=train_collate, sampler=train_sampler)
            
            dev_dataset = Retrieval_Testset(id2query=queries, id2doc=documents,rel=rel, split=split, mode="dev", neg_value=0)
            dev_sampler = DistributedSampler(dev_dataset, shuffle=False)
            self.dev_loader = DataLoader(dev_dataset, batch_size=2*args.batch_size*(1+args.num_neg), shuffle=False, num_workers=0, collate_fn=test_collate, sampler=dev_sampler)

            test_dataset = Retrieval_Testset(id2query=queries, id2doc=documents,rel=rel, split=split, mode="test", neg_value=0)
            test_sampler = DistributedSampler(test_dataset, shuffle=False)
            self.test_loader = DataLoader(test_dataset, batch_size=2*args.batch_size*(1+args.num_neg), shuffle=False, num_workers=0, collate_fn=test_collate, sampler=test_sampler)
        
        else:
            # mixed evaluation
            # train with wiki-clir data, and test on clef data (2fold)

            wiki_data_dir = f"{home_dir}/wiki-clir/{self.cased_dir}"
            rel = pickle.load(open(f"{wiki_data_dir}/relevance/{args.target_lang}_rel.pkl", 'rb'))
            split = pickle.load(open(f"{wiki_data_dir}/relevance/{args.target_lang}_split.pkl", 'rb'))
            queries = pickle.load(open(f"{wiki_data_dir}/queries/{args.source_lang}_query.pkl", 'rb'))
            documents = pickle.load(open(f"{wiki_data_dir}/documents/{args.target_lang}_document.pkl", 'rb'))

            train_dataset = Retrieval_Trainset(id2query=queries, id2doc=documents, rel=rel, split=split, num_neg=args.num_neg, neg_value=0)
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=train_collate, sampler=train_sampler)

            clef_data_dir = f"{home_dir}/CLIR-project/Evaluation_data/process-clef/{self.cased_dir}"
            rel = pickle.load(open(f"{clef_data_dir}/relevance/{args.target_lang}_rel.pkl", 'rb'))
            split = pickle.load(open(f"{clef_data_dir}/relevance/{args.target_lang}_split_2f.pkl", 'rb')) # different split file here
            queries = pickle.load(open(f"{clef_data_dir}/queries/{args.source_lang}_query.pkl", 'rb'))
            documents = pickle.load(open(f"{clef_data_dir}/full_documents/{args.target_lang}_document.pkl", 'rb')) if args.full_doc_length \
                        else pickle.load(open(f"{clef_data_dir}/documents/{args.target_lang}_document.pkl", 'rb'))

            logger.info(f"reading data from {wiki_data_dir} and {clef_data_dir}")

            dev_dataset = Retrieval_Testset(id2query=queries, id2doc=documents,rel=rel, split=split, mode="f1", neg_value=0)
            dev_sampler = DistributedSampler(dev_dataset, shuffle=False)
            self.dev_loader = DataLoader(dev_dataset, batch_size=2*args.batch_size*(1+args.num_neg), shuffle=False, num_workers=0, collate_fn=test_collate, sampler=dev_sampler)

            test_dataset = Retrieval_Testset(id2query=queries, id2doc=documents,rel=rel, split=split, mode="f2", neg_value=0)
            test_sampler = DistributedSampler(test_dataset, shuffle=False)
            self.test_loader = DataLoader(test_dataset, batch_size=2*args.batch_size*(1+args.num_neg), shuffle=False, num_workers=0, collate_fn=test_collate, sampler=test_sampler)

            # MAP evaluator
            self.qrel = self.get_qrel(rel)
            self.evaluator = pytrec_eval.RelevanceEvaluator(self.qrel, {'map'})

            logger.info(f"f1 has {len(self.dev_loader.dataset.query_ids)} queries ...")
            logger.info(f"f2 has {len(self.test_loader.dataset.query_ids)} queries ...")

        logger.info("Data reading done ...")
        
    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
    
    def run(self):

        set_seed(args.seed)

        self.model = copy.deepcopy(self._model)
        self.model.cuda()

        encoder_params = []
        projecter_params = []

        if 'mbert' in args.model_type:

            for param in self.model.bert.embeddings.parameters():
                param.requires_grad = False
            for l in range(0, num_encoder - args.num_ft_encoders):
                for param in self.model.bert.encoder.layer[l].parameters():
                    param.requires_grad = False

            # Apex DDP
            self.model = DistributedDataParallel(self.model)

            for l in range(num_encoder - args.num_ft_encoders, num_encoder):
                logger.info("adding {}-th encoder to optimizer...".format(l))
                encoder_params += self.model.module.bert.encoder.layer[l].parameters()
            encoder_params += self.model.module.bert.pooler.parameters()

            projecter_params += self.model.module.seqcls_classifier.parameters()
        
        elif args.model_type == 'xlm100':

            self.model.transformer.position_embeddings.requires_grad = False
            self.model.transformer.embeddings.requires_grad = False
            self.model.transformer.layer_norm_emb.requires_grad = False

            for l in range(0, num_encoder - args.num_ft_encoders):
                for param in self.model.transformer.attentions[l].parameters():
                    param.requires_grad = False
                for param in self.model.transformer.layer_norm1[l].parameters():
                    param.requires_grad = False
                for param in self.model.transformer.ffns[l].parameters():
                    param.requires_grad = False
                for param in self.model.transformer.layer_norm2[l].parameters():
                    param.requires_grad = False

            # Apex DDP
            self.model = DistributedDataParallel(self.model)

            for l in range(num_encoder - args.num_ft_encoders, num_encoder):
                logger.info("adding {}-th encoder to optimizer...".format(l))
                encoder_params += self.model.module.transformer.attentions[l].parameters()
                encoder_params += self.model.module.transformer.layer_norm1[l].parameters()
                encoder_params += self.model.module.transformer.ffns[l].parameters()
                encoder_params += self.model.module.transformer.layer_norm2[l].parameters()
            
            projecter_params += self.model.module.sequence_summary.parameters()

        elif args.model_type == "xlmr":
            
            for param in self.model.roberta.embeddings.parameters():
                param.requires_grad = False
            
            for l in range(0, num_encoder - args.num_ft_encoders):
                for param in self.model.roberta.encoder.layer[l].parameters():
                    param.requires_grad = False
            
            # Apex DDP
            self.model = DistributedDataParallel(self.model)

            for l in range(num_encoder - args.num_ft_encoders, num_encoder):
                logger.info("adding {}-th encoder to optimizer...".format(l))
                encoder_params += self.model.module.roberta.encoder.layer[l].parameters()
            encoder_params += self.model.module.roberta.pooler.parameters()

            projecter_params += self.model.module.classifier.parameters()

        else:
            assert False

        # reset top layers
        for param in encoder_params:
            self.init_weights(param)
        
        self.optimizer = torch.optim.Adam([
            {'params': encoder_params},
            {'params': projecter_params, 'lr': args.projector_lr}
        ], lr=args.encoder_lr)

        # apex
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=args.apex_level)

        if args.dataset != "mix":

            for epoch in range(args.num_epochs+1):

                self.epoch = epoch
                
                logger.info("process[{}]: training epoch {} ...".format(self._rank, self.epoch))
                self.train()

                if (self.epoch + 1) % self.eval_interval == 0:
                    # skip half of evaluation for speed
                    # dev eval first: if it is the best ever, we do test
                    logger.info("process[{}]: evaluating epoch {} on dev ...".format(self._rank, self.epoch))
                    with torch.no_grad():
                        dev_map = self.eval("dev")
                        if dev_map > self.best_dev_map:
                            self.best_dev_map = dev_map
                            self.best_epoch = self.epoch
                            logger.info("process[{}]: evaluating epoch {} on test ...".format(self._rank, self.epoch))
                            self.best_test_map = self.eval("test")
                        else:
                            pass
            
            if self._rank == 0:
                logger.info("best test MAP: {:.3f} @ epoch {}".format(self.best_test_map, self.best_epoch))

        else:
            self.f1_maps, self.f2_maps = [], []
            for epoch in range(args.num_epochs):

                self.epoch = epoch
                
                logger.info("process[{}]: training epoch {} ...".format(self._rank, self.epoch))
                # self.train()

                if self.epoch % self.eval_interval == 0 and args.num_epochs - self.epoch <= 3:
                    
                    logger.info("process[{}]: evaluating epoch {} on f1 ...".format(self._rank, self.epoch))
                    with torch.no_grad():
                        dev_map = self.eval("f1")
                        self.f1_maps.append(dev_map)
                    
                    logger.info("process[{}]: evaluating epoch {} on f2 ...".format(self._rank, self.epoch))
                    with torch.no_grad():
                        test_map = self.eval("f2")
                        self.f2_maps.append(test_map)

            dist.barrier()

            if self._rank == 0:
                dev_len, test_len = len(self.dev_loader.dataset.query_ids), len(self.test_loader.dataset.query_ids)
                best_f1_map = self.f1_maps[np.argmax(self.f2_maps)]
                best_f2_map = self.f2_maps[np.argmax(self.f1_maps)]
                best_map = (best_f1_map * dev_len + best_f2_map * test_len) / (dev_len + test_len)
                logger.info(f"best MAP: {best_map:.3f}")
                    
    def train(self):
        
        self.model.train()
        losses = []
        nw = 0
        n_pos_qd_pair = 0
        t = time.time()
        
        if isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(self.epoch)
        
        for qids, dids, queries, documents, y in self.train_loader:

            n_pos_qd_pair += len(queries) / (1+args.num_neg)
            
            encoded = tokenizer.batch_encode_plus(batch_text_or_text_pairs=list(zip(queries, documents)),
                                                    truncation="longest_first", add_special_tokens=True, 
                                                    max_length = max_len, padding="max_length", 
                                                    is_pretokenized=False, return_tensors="pt", 
                                                    return_attention_mask=True, return_token_type_ids=True)

            input_ids = encoded["input_ids"].cuda()
            attention_mask = encoded["attention_mask"].cuda()
            token_type_ids = encoded["token_type_ids"].cuda()
            y = torch.tensor(y).unsqueeze(1).cuda()

            # get lengths
            lengths = (max_len - (input_ids == tokenizer.pad_token_id).sum(dim=1))

            # longformer's global attention
            if args.model_type == "mbert-long":
                attention_mask = 2*attention_mask - token_type_ids
                attention_mask[attention_mask<0] = 0

            if args.debug:
                # check data
                print(tokenizer.decode(input_ids[0].detach().cpu().tolist()))
                print(tokenizer.decode(input_ids[1].detach().cpu().tolist()))
                print(attention_mask[0].detach().cpu().tolist())
                print(attention_mask[1].detach().cpu().tolist())
                print(token_type_ids[0].detach().cpu().tolist())
                print(token_type_ids[1].detach().cpu().tolist())
                assert False

            self.optimizer.zero_grad()

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": y,
                "mode": "seqcls"
            }

            if 'mbert' in args.model_type:
                outputs = self.model(inputs)
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=y
                )
            
            loss = outputs[0]

            if torch.isnan(loss):

                logger.info(tokenizer.decode(input_ids[0].detach().cpu().tolist()))
                logger.info(tokenizer.decode(input_ids[1].detach().cpu().tolist()))
                logger.info(tokenizer.decode(input_ids[2].detach().cpu().tolist()))
                logger.info(tokenizer.decode(input_ids[3].detach().cpu().tolist()))
                logger.info(input_ids.shape)
                logger.info(attention_mask.shape)
                logger.info(token_type_ids.shape)

                logger.info(outputs)

                assert False

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            
            self.optimizer.step()
            
            nw += lengths.sum().item()
            losses.append(loss.item())
            
        # log
        logger.info(f"process[{self._rank}] - epoch {self.epoch} - train iter {n_pos_qd_pair} - {nw / (time.time() - t):.1f} words/s - loss: {sum(losses) / len(losses):.4f}")
        nw, t = 0, time.time()
        losses = []
        
    def eval(self, splt):

        self.model.eval()
        
        if splt in ["dev","f1"]: 
            loader = self.dev_loader
        elif splt in ["test","f2"]: 
            loader = self.test_loader
        else: 
            assert False

        os.makedirs("tmp", exist_ok=True)
        record_path = f"tmp/{args.model_type}_{args.source_lang}{args.target_lang}_{splt}_{self._rank}_{self.epoch}.txt"
        fout = open(record_path, 'w')

        for qids, dids, queries, documents, y in loader:

            encoded = tokenizer.batch_encode_plus(batch_text_or_text_pairs=list(zip(queries, documents)),
                                                truncation="longest_first", add_special_tokens=True, 
                                                max_length = max_len, padding="max_length", 
                                                is_pretokenized=False, return_tensors="pt", 
                                                return_attention_mask=True, return_token_type_ids=True)

            input_ids = encoded["input_ids"].cuda()
            attention_mask = encoded["attention_mask"].cuda()
            token_type_ids = encoded["token_type_ids"].cuda()

            # longformer's global attention
            if args.model_type == "mbert-long":
                attention_mask = 2*attention_mask - token_type_ids
                attention_mask[attention_mask<0] = 0

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "mode": "seqcls"
            }

            # outputs = self.model(inputs)

            if 'mbert' in args.model_type:
                outputs = self.model(inputs)
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

            logits = outputs[0][:,1].detach().cpu().numpy().squeeze()

            if len(qids) > 1:
                for q, d, s in zip(qids, dids, logits):
                    fout.write("{}\t{}\t{}\n".format(q,d,s))
            else:
                fout.write("{}\t{}\t{}\n".format(qids[0],dids[0],logits))

        fout.flush()
        fout.close()
        
        dist.barrier()
            
        num_record = 0
        run = {}
        for rank in range(get_world_size()):
            log_path = f"tmp/{args.model_type}_{args.source_lang}{args.target_lang}_{splt}_{rank}_{self.epoch}.txt"
            with open(log_path, 'r') as fin:
                for line in fin.readlines():
                    q, d, s = line.strip("\n").split("\t")
                    if q not in run:
                        run[q] = {}
                    run[q][d] = float(s)
                    num_record += 1

        if num_record > len(loader.dataset):
            logger.info("{} set during evaluation: {}/{}".format(splt, num_record, len(loader.dataset)))
        results = self.evaluator.evaluate(run)
        mean_ap = np.mean([v["map"] for k, v in results.items()])
        logger.info(f"process[{self._rank}] - {splt} - epoch {self.epoch} - mAP: {mean_ap:.3f} w/ {len(results)} queries")

        dist.barrier()
        
        if self._rank == 0:
            # delete data from all ranks
            for file in glob.glob(f"tmp/{args.model_type}_{args.source_lang}{args.target_lang}_{splt}*{self.epoch}.txt"):
                logger.info("removing file {}".format(file))
                os.remove(file)

        return mean_ap
            
    def get_qrel(self, rel):

        qrel = {}
        
        for query_id, tmp in rel.items():
            qrel[query_id] = {}
            for pos_doc_id in tmp["p"]:
                qrel[query_id][pos_doc_id] = 1

        return qrel

clir = CLIR(model)
clir.run()
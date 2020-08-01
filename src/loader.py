## this is deprecated

from logging import getLogger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .dataset.wiki_dataset import wiki_rr_trainset, wiki_qlm_trainset, DataCollatorForRelevanceRanking, DataCollatorForMaskedQueryPrediction

# logger = getLogger()

def load_data(params, tokenizer):

    iterators = {}
    rr_collator = DataCollatorForRelevanceRanking(tokenizer, "long" in params.model_type)
    qlm_collator = DataCollatorForMaskedQueryPrediction(tokenizer, params.mlm_probability, "long" in params.model_type, params.qlm_mask_mode)

    if params.rr_steps is not None:
        # relevance ranking datasets
        for lang_pair in params.rr_steps:
            
            dataset = wiki_rr_trainset(
                lang_pair = lang_pair, 
                num_neg = params.num_neg, 
                neg_val = params.neg_val
                )
            sampler = DistributedSampler(dataset, shuffle=True)
            iterator = iter(DataLoader(
                dataset,
                batch_size = params.batch_size,
                shuffle = False,
                num_workers = 0,
                collate_fn = rr_collator,
                sampler = sampler
            ))

            iterators[('rr', lang_pair)] = iterator

    if params.qlm_steps is not None:
        # query language model datasets
        for lang_pair in params.qlm_steps:
            dataset = wiki_qlm_trainset(
                lang_pair = lang_pair, 
                neg_val = params.neg_val
                )
            sampler = DistributedSampler(dataset, shuffle=True)
            iterator = iter(DataLoader(
                dataset,
                batch_size = params.batch_size,
                shuffle = False,
                num_workers = 0,
                collate_fn = qlm_collator,
                sampler = sampler
            ))

            iterators[('qlm', lang_pair)] = iterator

    return iterators



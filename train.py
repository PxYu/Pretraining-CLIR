import json
import torch
import random
import argparse

from trainer import Trainer
from src.utils import bool_flag, shuf_order, initialize_exp
from src.slurm import init_signal_handler, init_distributed_mode

from transformers import BertTokenizer
from BertForXLRetrieval import BertForXLRetrieval
from BertLongForXLRetrieval import BertLongForXLRetrieval

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Experiment name")
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")

    # float16 / AMP API
    parser.add_argument("--fp16", type=bool_flag, default=False,
                        help="Run model with float16")
    parser.add_argument("--amp", type=int, default=-1,
                        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.")

    # model type
    parser.add_argument("--model_type", type=str, default="mbert-long", choices=["mbert-long", "mbert"],
                        help="Use normal mbert or mbert-long (with Longformer attention)")
    parser.add_argument("--model_path", type=str, default="bert-base-multilingual-uncased",
                        help="Pretrained model name or local path to pretrained models")

    # QLM task parameters
    parser.add_argument("--qlm_mask_mode", type=str, default="query",
                        help="masking mode for QLM (query, document, or mixed)")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="to be filled") # FIXME: ...

    # batch parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of sentences per batch")
    
    # optimization
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=100000,
                        help="Epoch size / evaluation frequency (-1 for parallel data size)")
    parser.add_argument("--max_epoch", type=int, default=100000,
                        help="Maximum epoch size")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Accumulate model gradients over N iterations (N times larger batch sizes)")

    # training coefficients
    parser.add_argument("--lambda_qlm", type=float, default="1",
                        help="QLM coefficient")
    parser.add_argument("--lambda_rr", type=float, default="1",
                        help="RR coefficient")

    
    # dataset setup
    # parser.add_argument("--lang_pairs", type=lambda s: s.split(','), default="",
    #                     help="language pairs for loading data (usually the same as qlm/rr steps)")
    parser.add_argument("--max_pairs", type=int, default=50000,
                        help="")
    parser.add_argument("--num_neg", type=int, default=1,
                        help="number of negative qd pair per positive pair")
    parser.add_argument("--neg_val", type=int, default=0,
                        help="label value for negative samples")
    
    # training steps
    parser.add_argument("--qlm_steps", type=lambda s: s.split(','), default=None,
                        help="Masked prediction steps (MLM / TLM)")
    parser.add_argument("--rr_steps", type=lambda s: s.split(','), default=None,
                        help="Parallel classification steps")

    # multi-gpu / multi-node
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")
    parser.add_argument("--slurm_debug", type=bool_flag, default=False)

    return parser


def main(params):

    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)

    # initialize the experiment
    logger = initialize_exp(params)

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()

    # build model
    if params.model_type == "mbert":
        model = BertForXLRetrieval.from_pretrained(params.model_path).cuda()
    elif params.model_type == "mbert-long":
        model = BertLongForXLRetrieval.from_pretrained(params.model_path).cuda()
    else:
        assert False

    # tokenizer (initialized earlier for loading data)    
    tokenizer = BertTokenizer.from_pretrained(params.model_path)
    
    # build trainer, reload potential checkpoints / build evaluator
    trainer = Trainer(model, tokenizer, params)

    # language model training
    for epoch in range(params.max_epoch):

        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.epoch = epoch

        while trainer.n_iter < trainer.epoch_size:

            # Query language modeling
            if params.qlm_steps is not None:
                for lang_pair in shuf_order(params.qlm_steps):
                    trainer.qlm_step(lang_pair, params.lambda_qlm)

            # Relevance classification
            if params.rr_steps is not None:
                for lang_pair in shuf_order(params.rr_steps):
                    trainer.rr_step(lang_pair, params.lambda_rr)

            trainer.iter()

        # end of epoch
        logger.info("============ End of epoch %i ============" % trainer.epoch)
        trainer.end_epoch()
        # torch.distributed.barrier()

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
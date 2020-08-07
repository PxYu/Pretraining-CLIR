import os
import torch
import logging
import argparse
import numpy as np
from apex import amp
from tqdm.auto import tqdm
# import torch.distributed as dist
from src.utils import CustomFormatter, set_seed
from src.dataset.mlqa_dataset import mlqa_dataset
# from apex.parallel import DistributedDataParallel
# from torch.distributed import get_rank, get_world_size
from transformers.data.processors.squad import SquadResult
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import XLMRobertaTokenizer, XLMRobertaForQuestionAnswering
from transformers.modeling_longformer import LongformerSelfAttention
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader

from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)

parser = argparse.ArgumentParser()

# parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--model_type', choices=["mbert", "mbert-long",'xlmr'])
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--apex_level', type=str, default="O2")
parser.add_argument('--seed', type=int, default=611)

args = parser.parse_args()

batch_size = args.batch_size
model_name = args.model_name
model_type = args.model_type

# slurm_id = os.environ.get('SLURM_JOB_ID')
model_name_in_path = args.model_name.replace("/", "-")
if model_name_in_path.startswith("-"):
    model_name_in_path = model_name_in_path[1:]
log_dir = f'logs/QA/{args.model_type}/{model_name_in_path}'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(f'{log_dir}/qa.log')
formatter = CustomFormatter('%(adjustedTime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info(args)

lang_pairs = []
for q_lang in ['en', 'de', 'es']:
    for a_lang in ['en', 'de', 'es']:
        if q_lang != "en" or a_lang != "en":
            lang_pairs.append((q_lang, a_lang))

logger.info(lang_pairs)

############################################################################## longformer for QA class ##############################################################################

class BertLongForQuestionAnswering(BertForQuestionAnswering):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.bert.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = LongformerSelfAttention(config, layer_id=i)

############################################################################## model loading ##############################################################################
            
if 'mbert' in model_type:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    if 'long' in model_type:
        model = BertLongForQuestionAnswering.from_pretrained(model_name)
    else:
        model = BertForQuestionAnswering.from_pretrained(model_name)
else:
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    model = XLMRobertaForQuestionAnswering.from_pretrained(model_name)

## model

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

model = model.cuda()

encoder_params = []
projector_params = []

if 'mbert' in model_type:

    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for l in range(0, 9):
        for param in model.bert.encoder.layer[l].parameters():
            param.requires_grad = False

    for l in range(9, 12):
        encoder_params += model.bert.encoder.layer[l].parameters()
    encoder_params += model.bert.pooler.parameters()
    projector_params += model.qa_outputs.parameters()

else:

    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False
    for l in range(0, 9):
        for param in model.roberta.encoder.layer[l].parameters():
            param.requires_grad = False
    
    for l in range(9, 12):
        encoder_params += model.roberta.encoder.layer[l].parameters()
    encoder_params += model.roberta.pooler.parameters()
    projector_params += model.qa_outputs.parameters()

for param in encoder_params:
    init_weights(param)

optimizer = torch.optim.Adam([
            {'params': encoder_params},
            {'params': projector_params, 'lr': 0.00001}
        ], lr=0.00001)

model, optimizer = amp.initialize(model, optimizer, opt_level=args.apex_level)

############################################################################## training data ##############################################################################

_, train_set = mlqa_dataset(
    is_training = True,
    is_squad = True,
    split = None,
    a_lang = None,
    q_lang = None,
    tokenizer = tokenizer,
    model_type = model_type
).squad_convert_examples_to_features(threads=4, global_attention='long' in model_type)
sampler = RandomSampler(train_set, True)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, sampler=sampler)

############################################################################## evaluation data ##############################################################################

def get_eval_datasets(mode, a_lang, q_lang):
    
    mlqa = mlqa_dataset(
        is_training = False,
        is_squad = False,
        split = mode,
        a_lang = a_lang,
        q_lang = q_lang,
        tokenizer = tokenizer,
        model_type = model_type
    )
    
    features, dset = mlqa.squad_convert_examples_to_features(threads=4, global_attention='long' in model_type)
    
    return [mlqa, features, dset]

dev_datasets = {}
test_datasets = {}

for (q_lang, a_lang) in lang_pairs:
    logger.info(f"building datasets for {q_lang}-{a_lang}")
    dev_datasets[f'{q_lang}{a_lang}'] = get_eval_datasets("dev", a_lang, q_lang)
    test_datasets[f'{q_lang}{a_lang}'] = get_eval_datasets("test", a_lang, q_lang)

############################################################################## training function ##############################################################################

def train():
    
    model.train()

    for batch in tqdm(train_loader):
        losses = []
        optimizer.zero_grad()

        batch = tuple(t.cuda() for t in batch)
        inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "start_positions": batch[3],
                    "end_positions": batch[4],
                }

        outputs = model(**inputs)
        loss = outputs[0]
        # loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return np.mean(losses)

############################################################################## evaluation function ##############################################################################

n_best_size = 3
max_answer_length = 256
do_lower_case = True
null_score_diff_threshold = 0

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def evaluate(mode, q_lang, a_lang):

    model.eval()

    if mode == "dev":
        
        [mlqa, features, dataset] = dev_datasets[f'{q_lang}{a_lang}']
        examples = mlqa.examples
    
    elif mode == "test":

        [mlqa, features, dataset] = test_datasets[f'{q_lang}{a_lang}']
        examples = mlqa.examples

    else:

        assert False

    with torch.no_grad():

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)

        all_results = []

        for batch in tqdm(eval_dataloader, desc=f"Evaluating {q_lang}-{a_lang} on {mode}"):
            model.eval()
            batch = tuple(t.cuda() for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                feature_indices = batch[3]
                outputs = model(**inputs)

            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)
                output = [to_list(output[i]) for output in outputs]
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)

        # Compute predictions
        output_prediction_file = os.path.join("qa-pred", f"predictions-{model_name_in_path}.json")
        output_nbest_file = os.path.join("qa-pred", f"nbest_predictions-{model_name_in_path}.json")

        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            n_best_size, # not sure
            max_answer_length,
            do_lower_case,
            output_prediction_file,
            output_nbest_file,
            None,
            True,
            False,
            null_score_diff_threshold,
            tokenizer,
        )

        # Compute the F1 and exact scores.
        results = squad_evaluate(examples, predictions)
        return results['f1'], results['total']

############################################################################## result recording ##############################################################################

best_dev_f1 = {f"{q_lang}{a_lang}": 0.0 for (q_lang, a_lang) in lang_pairs}
best_test_f1 = {f"{q_lang}{a_lang}": 0.0 for (q_lang, a_lang) in lang_pairs}

for epoch in range(args.num_epochs):

    logger.info(f"training loss @ epoch {epoch}: {train():.3f}")
    torch.cuda.empty_cache()

    for (q_lang, a_lang) in lang_pairs:

        key = f"{q_lang}{a_lang}"
        dev_f1, dev_n_question = evaluate("dev", q_lang, a_lang)
        if dev_f1 > best_dev_f1[key]:
            logger.info(f"epoch {epoch}: dev of {key} increased from {best_dev_f1[key]:.3f} to {dev_f1:.3f}")
            logger.info(f"evaluating test ...")
            best_dev_f1[key] = dev_f1
            best_test_f1[key], _ = evaluate("test", q_lang, a_lang)

    logger.info(best_test_f1)


for (q_lang, a_lang) in lang_pairs:
    key = f"{q_lang}{a_lang}"
    logger.info(f"{key}: {best_test_f1[key]:.3f}")
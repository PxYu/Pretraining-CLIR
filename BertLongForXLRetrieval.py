import os
import math
import logging
from transformers import BertTokenizer
from transformers.modeling_longformer import LongformerSelfAttention


from BertForXLRetrieval import BertForXLRetrieval

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

max_pos=1024
attention_window=64

class BertLongForXLRetrieval(BertForXLRetrieval):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.bert.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = LongformerSelfAttention(config, layer_id=i)

def create_long_model(save_model_to, attention_window, max_pos, copy_pos):
    '''
    max_pos: max sequence length of the new model (after BPE/wp encoding)
    '''
    model = BertForXLRetrieval.from_pretrained('bert-base-multilingual-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    config = model.config

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.bert.embeddings.position_embeddings.weight.shape
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.bert.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    
    if copy_pos:
        # copy position embeddings over and over to initialize the new position embeddings
        k = 0
        step = current_max_pos
        while k < max_pos - 1:
            logger.info(f"copying weights {model.bert.embeddings.position_embeddings.weight.shape} into new_pos_embed[{k}:{k+step}]")
            new_pos_embed[k:(k + step)] = model.bert.embeddings.position_embeddings.weight
            k += step
    
    model.bert.embeddings.position_embeddings.weight.data = new_pos_embed

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.bert.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = layer.attention.self.query
        longformer_self_attn.key_global = layer.attention.self.key
        longformer_self_attn.value_global = layer.attention.self.value

        layer.attention.self = longformer_self_attn

    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    
    return model, tokenizer

if __name__ == "__main__":

    mbert_base = BertForXLRetrieval.from_pretrained('bert-base-multilingual-uncased')
    mbert_base_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

    # create mbert-long models
    # hugging face interpret the last number after '-' as the global step, so I add a 0 to the end
    model_path = f'../mBertLongForXLRetrieval/mBert-base-p{max_pos}-w{attention_window}-0'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logger.info(f'Converting mbert-base into mBert-base-p{max_pos}-w{attention_window}-0')
    model, tokenizer = create_long_model(
        save_model_to=model_path, 
        attention_window=attention_window, 
        max_pos=max_pos,
        copy_pos=True
        )
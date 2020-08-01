import os
import math
import logging
import warnings

import torch
from torch import nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.modeling_bert import BertModel, BertOnlyMLMHead, BertPreTrainedModel

class BertForXLRetrieval(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        assert (
            not config.is_decoder
        ), "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for bi-directional self-attention."

        # shared
        self.bert = BertModel(config)

        # for mlm
        self.mlm_cls = BertOnlyMLMHead(config)

        # for seqcls
        self.num_labels = config.num_labels
        self.seqcls_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.seqcls_classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    # def get_output_embeddings(self):
    #     return self.cls.predictions.decoder

    def forward(self, *input, **kwargs):
        
        # FIXME: not sure why parameters are passed like this
        input = input[0]
        mode = input['mode']        
        del input['mode']
        kwargs = input

        if mode == "mlm":
            return self.forward_mlm(**kwargs)
        elif mode == "seqcls":
            return self.forward_seqcls(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def forward_mlm(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):

        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                DeprecationWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        prediction_scores = self.mlm_cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


    def forward_seqcls(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]

        pooled_output = self.seqcls_dropout(pooled_output)
        logits = self.seqcls_classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs
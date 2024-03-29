import os
import json
import torch
import logging
import numpy as np
from tqdm.auto import tqdm
from functools import partial
from torch.utils.data import TensorDataset
from multiprocessing import Pool, cpu_count
from transformers.data.processors.utils import DataProcessor
from transformers.tokenization_bert import whitespace_tokenize
from transformers.data.processors.squad import SquadResult, SquadFeatures, SquadExample

logger = logging.getLogger()

class SquadProcessor(DataProcessor):

    def get_train_examples(self, data_dir, filename=None):
        
        with open(
            os.path.join(data_dir, filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "train")


    def get_dev_examples(self, data_dir, filename=None):
        
        with open(
            os.path.join(data_dir, filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "dev")


    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            title = entry["title"].lower()
            # title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"].lower()
                # context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"].lower()
                    # question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False

                    if not is_impossible:
                        if is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"].lower()
                            # answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = [{'text': x['text'].lower(), 'answer_start': x['answer_start']} for x in qa["answers"]]
                            # answers = [{'text': x['text'], 'answer_start': x['answer_start']} for x in qa["answers"]]

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )

                    examples.append(example)

        return examples

class mlqa_dataset():
    def __init__(
        self,
        is_training,
        is_squad,
        split,
        q_lang,
        a_lang,
        tokenizer,
        model_type,
        squad_version
        ):

        self.is_training = is_training
        self.is_squad = is_squad
        self.q_lang = q_lang
        self.a_lang = a_lang
        self.processor = SquadProcessor()
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.squad_version = squad_version

        if os.path.exists("/home/puxuan"): 
            home_dir = "/home/puxuan"
        else: 
            home_dir = "/mnt/home/puxuan"

        if self.is_squad:
            self.data_path = f"{home_dir}/MLQA/MLQA_V1"
            filename = f"train-v{self.squad_version}.json"
        else:
            self.data_path = f"{home_dir}/MLQA/MLQA_V1/{split}"
            filename = f"{split}-context-{q_lang}-question-{a_lang}.json"

        
        if self.is_training:
            self.examples = self.processor.get_train_examples(self.data_path, filename)
        else:
            self.examples = self.processor.get_dev_examples(self.data_path, filename)
      
        logger.info(f"#examples in [{split}] set: {len(self.examples)}")

    def squad_convert_examples_to_features(
        self,
        threads=4,
        global_attention=False
    ):
        cache_path = f"{self.data_path}/cache"
        os.makedirs(cache_path, exist_ok=True)
        if self.is_training:
            cache_file = f"{cache_path}/modeltype-{self.model_type}_squad-{self.squad_version}_isglobal-{global_attention}_vocabsize-{len(self.tokenizer)}.pkl"
        else:
            cache_file = f"{cache_path}/modeltype-{self.model_type}_mlqa_isglobal-{global_attention}_langs-{self.a_lang}{self.q_lang}_vocabsize-{len(self.tokenizer)}.pkl"

        if not os.path.exists(cache_file):

            # Defining helper methods
            features = []
            threads = min(threads, cpu_count())
            with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(self.tokenizer,)) as p:
                annotate_ = partial(
                    squad_convert_example_to_features,
                    max_seq_length=self.tokenizer.model_max_length,
                    doc_stride=self.tokenizer.model_max_length,
                    max_query_length=128,
                    is_training=self.is_training,
                )
                features = list(
                    tqdm(
                        p.imap(annotate_, self.examples, chunksize=32),
                        total=len(self.examples),
                        desc="convert squad examples to features",
                        disable=False,
                    )
                )
            new_features = []
            unique_id = 1000000000
            example_index = 0
            for example_features in tqdm(
                features, total=len(features), desc="add example index and unique id", disable=False
            ):
                if not example_features:
                    continue
                for example_feature in example_features:
                    example_feature.example_index = example_index
                    example_feature.unique_id = unique_id
                    new_features.append(example_feature)
                    unique_id += 1
                example_index += 1
            features = new_features
            del new_features
            
            if global_attention:
                for feature in features:
                    feature.attention_mask = 2*np.asarray(feature.attention_mask) - np.asarray(feature.token_type_ids)
                    feature.attention_mask[feature.attention_mask<0] = 0
                    feature.attention_mask = feature.attention_mask.tolist()

            # Convert to Tensors and build dataset
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
            all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
            all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

            if not self.is_training:
                all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
                dataset = TensorDataset(
                    all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index, all_cls_index, all_p_mask
                )
            else:
                all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
                all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
                dataset = TensorDataset(
                    all_input_ids,
                    all_attention_masks,
                    all_token_type_ids,
                    all_start_positions,
                    all_end_positions,
                    all_cls_index,
                    all_p_mask,
                    all_is_impossible,
                )

            torch.save({"features": features, "dataset": dataset}, cache_file)
        
        else:

            tmp = torch.load(cache_file)
            features = tmp["features"]
            dataset = tmp["dataset"]

        return features, dataset

### below are copied from transformers library

# Store the tokenizers which insert 2 separators tokens
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart"}

def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def squad_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length, is_training):
    features = []
    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    spans = []

    truncated_query = tokenizer.encode(
        example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
    )

    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.max_len - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.max_len - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
            truncated_query if tokenizer.padding_side == "right" else span_doc_tokens,
            span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
            truncation="only_second" if tokenizer.padding_side == "right" else "only_first",
            padding="max_length",
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            return_token_type_ids=True,
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                    len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or (
            "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0)
        p_mask = np.ones_like(span["token_type_ids"])
        if tokenizer.padding_side == "right":
            p_mask[len(truncated_query) + sequence_added_tokens :] = 0
        else:
            p_mask[-len(span["tokens"]) : -(len(truncated_query) + sequence_added_tokens)] = 0

        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        features.append(
            SquadFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
                qas_id=example.qas_id,
            )
        )
    return features

def squad_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert
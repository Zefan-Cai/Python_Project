import torch
from torch.utils import data
from torch.utils.data import Dataset
from datasets.arrow_dataset import Dataset as HFDataset
from datasets.load import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import numpy as np
import logging

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


class GlueDataset():
    def __init__(self, tokenizer, model_args, data_args, training_args) -> None:
        super().__init__()
        self.raw_datasets = load_dataset("/home/sankuai/cephfs_caizefan/P-tuning-v2/tasks/glue/glue_dataset.py", data_args.dataset_name)
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        # labels
        self.is_regression = data_args.dataset_name == "stsb"
        if not self.is_regression:
            self.label_list = self.raw_datasets["train"].features["label"].names
            self.num_labels = len(self.label_list)
        else:
            self.num_labels = 1

        # Preprocessing the raw_datasets
        self.sentence1_key, self.sentence2_key = task_to_keys[data_args.dataset_name]

        # Padding strategy
        if data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = "longest"

        # Some models have set the order of the labels to use, so let's make sure we do use it.
        if not self.is_regression:
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}

        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

#         raw_datasets = raw_datasets.map(
#             self.preprocess_function,
#             batched=True,
#             load_from_cache_file=not data_args.overwrite_cache,
#             desc="Running tokenizer on dataset",
#         )

#         if training_args.do_train:
#             self.train_dataset = raw_datasets["train"]
#             if data_args.max_train_samples is not None:
#                 self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))

#         if training_args.do_eval:
#             self.eval_dataset = raw_datasets["validation_matched" if data_args.dataset_name == "mnli" else "validation"]
#             if data_args.max_eval_samples is not None:
#                 self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))

#         if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
#             self.predict_dataset = raw_datasets["test_matched" if data_args.dataset_name == "mnli" else "test"]
#             if data_args.max_predict_samples is not None:
#                 self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))

        self.metric = load_metric("/home/sankuai/cephfs_caizefan/P-tuning-v2/tasks/glue/glue_metric.py", data_args.dataset_name)

        if data_args.pad_to_max_length:
            self.data_collator = default_data_collator
        elif training_args.fp16:
            self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)


    def preprocess_function(self, examples):
        # Tokenize the texts
        args = (
            (examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key], examples[self.sentence2_key])
        )
        result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)

        return result

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
        if self.data_args.dataset_name is not None:
            result = self.metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif self.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

class GlueDatasetForLM(GlueDataset):
    def __init__(self, tokenizer: AutoTokenizer, model_args, data_args, training_args) -> None:
        super().__init__(tokenizer, model_args, data_args, training_args)
        self.verbalizer_dict = {
            "2": {
                "verbalizer_0": {"0": "no", "1": "yes", "-1": "yes"},
                "verbalizer_1": {"0": "No", "1": "Yes", "-1": "yes"},
                "verbalizer_2": {"0": "false", "1": "true", "-1": "yes"},
                "verbalizer_3": {"0": "False", "1": "True", "-1": "yes"}
            },
            "3": {
                "verbalizer_0": {"0": "no", "1": "maybe", "2": "yes", "-1": "yes"},
                "verbalizer_1": {"0": "No", "1": "maybe", "2": "yes", "-1": "yes"},
                "verbalizer_2": {"0": "false", "1": "maybe", "2": "yes", "-1": "yes"},
                "verbalizer_3": {"0": "False", "1": "maybe", "2": "yes", "-1": "yes"}
            },
        }
        self.label2token = self.verbalizer_dict[str(self.num_labels)][self.model_args.verbalizer_id]
        self.label_token_list = [v for _, v in self.verbalizer_dict[str(self.num_labels)][self.model_args.verbalizer_id].items()]
        self.token2label = {v: k for k, v in self.verbalizer_dict[str(self.num_labels)][self.model_args.verbalizer_id].items()}

    def get_input_ids(self, input):
        pass

    def data_collator_glue_lm(self, features):
        # if not isinstance(features[0], (dict, BatchEncoding)):
        #     features = [vars(f) for f in features]
        first = features[0]
        batch = {}

        
        
        for f in features:
            f["input_ids"], f["sentences_ids"] = self.get_input_ids(f)
            f["label_token_id"] = f["label_token_id"][1:-1][0] # [103, 222, 104] -> 222
            label_ids = [-100 for _ in range(len(f["input_ids"]))]
            mask_index = f["input_ids"].index(self.tokenizer.mask_token_id)
            f["mask_index"] = mask_index
            label_ids[mask_index] = f["label_token_id"]
            f["label_ids"] = label_ids
            f["label_token_id_list"] = [self.tokenizer.encode(l)[1:-1][0] for l in self.label_token_list]
            # for k, v in f.items():
            #     if k not in ["input_ids", "sentences_ids", "label_ids", "label", "idx", "label_token_id", "label_token_id_list", "mask_index"]:
            #         def f[k]

        # sentence_ids，label_ids和input_ids需要pad
        # label_token_id，label, idx, label_token_id_list, mask_index都是固定长度
        # pad_to_max_length一般默认为True，所以padidng一般为max_length，这里一般pad到512.
        # 这个返回值是有attention_mask的。
        input_ids_result = self.tokenizer.pad(
            {"input_ids": [f["input_ids"] for f in features]},
            padding=self.padding,
            max_length=self.max_seq_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch["input_ids"] = input_ids_result["input_ids"]
        batch["attention_mask"] = input_ids_result["attention_mask"]
        
        label_ids_result = self.tokenizer.pad(
            {"input_ids": [f["label_ids"] for f in features]},
            padding=self.padding,
            max_length=self.max_seq_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        # batch["labels"] = label_ids_result["input_ids"]
        batch["label_ids"] = label_ids_result["input_ids"]
        
        label_ids_result = self.tokenizer.pad(
            {"input_ids": [f["sentences_ids"] for f in features]},
            padding="longest",
            max_length=self.max_seq_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch["sentences_ids"] = label_ids_result["input_ids"]
        
        for k, v in first.items():
        #     print(k)
        #     if type(v) == list: print([len(f[k]) for f in features])
        #     else: print(features[0][k])
            # if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if v is not None and not isinstance(v, str) and k not in ["idx", "input_ids", "attention_mask", "labels", "label_ids", "sentences_ids", "sentence1", "sentence2", "input_id_sentence1", "input_id_sentence2"]:
                batch[k] = torch.tensor([f[k] for f in features])
                # print(batch[k].shape)
        batch["labels"] = batch["label"]
        return batch



class GlueDatasetForLMForNLI(GlueDatasetForLM):
    def __init__(self, tokenizer: AutoTokenizer, model_args, data_args, training_args) -> None:
        super().__init__(tokenizer, model_args, data_args, training_args)

        self.template_id = self.model_args.template_id
        self.template_dict = {
            "template_0": self.get_input_ids_0,
            "template_1": self.get_input_ids_1,
            "template_2": self.get_input_ids_2,
            "template_3": self.get_input_ids_3,
            "template_6": self.get_input_ids_6,
            "template_66": self.get_input_ids_66,
            "template_12": self.get_input_ids_12,
        }
        self.pre_seq_len_dict = {
            "template_0": 0,
            "template_1": 1,
            "template_2": 2,
            "template_3": 3,
            "template_6": 6,
            "template_66": 12,
            "template_12": 12
        }
        self.pre_seq_len = self.pre_seq_len_dict[self.template_id]
        self.get_input_ids = self.template_dict[self.template_id]
        
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["歌"]})

        self.raw_datasets = self.raw_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    
        if training_args.do_train:
            self.train_dataset = self.raw_datasets["train"]
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))

        if training_args.do_eval:
            self.eval_dataset = self.raw_datasets["validation_matched" if data_args.dataset_name == "mnli" else "validation"]
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))

        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            self.predict_dataset = self.raw_datasets["test_matched" if data_args.dataset_name == "mnli" else "test"]
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))
                
        self.data_collator = self.data_collator_glue_lm
        
    def preprocess_function(self, examples):
        # Tokenize the textspre_seq_len
        result = {
            "input_id_sentence1": self.tokenizer(examples[self.sentence1_key], padding='do_not_pad', max_length=512, truncation=True)["input_ids"],
            "input_id_sentence2": self.tokenizer(examples[self.sentence2_key], padding='do_not_pad', max_length=512, truncation=True)["input_ids"],
            "label_token_id": self.tokenizer([self.label2token[str(label)]for label in examples["label"]], padding='do_not_pad', max_length=512, truncation=True)["input_ids"] # 1 to 'yes' or 'true'
        }
        return result

    def get_input_ids_0(self, f):
        input_ids = (f[ "input_id_sentence1"][:-1] + [self.tokenizer.mask_token_id] + f[ "input_id_sentence2"][1:])[:512]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:512]
        return input_ids, sentences_ids
    
    def get_input_ids_1(self, f):
        input_ids = (f[ "input_id_sentence1"][:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 1 + [self.tokenizer.mask_token_id] + f[ "input_id_sentence2"][1:])[:512]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:512]
        return input_ids, sentences_ids
    
    def get_input_ids_2(self, f):
        input_ids = (f[ "input_id_sentence1"][:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 2 + [self.tokenizer.mask_token_id] + f[ "input_id_sentence2"][1:])[:512]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:512]
        return input_ids, sentences_ids
    
    def get_input_ids_3(self, f):
        input_ids = (f[ "input_id_sentence1"][:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 3 + [self.tokenizer.mask_token_id] + f[ "input_id_sentence2"][1:])[:512]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:512]
        return input_ids, sentences_ids

    def get_input_ids_6(self, f):
        input_ids = (f[ "input_id_sentence1"][:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + [self.tokenizer.mask_token_id] + f[ "input_id_sentence2"][1:])[:512]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:512]
        return input_ids, sentences_ids
    
    def get_input_ids_66(self, f):
        input_ids = (f[ "input_id_sentence1"][:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + [self.tokenizer.mask_token_id] + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + f[ "input_id_sentence2"][1:])[:512]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:512]
        return input_ids, sentences_ids
    
    def get_input_ids_12(self, f):
        input_ids = (f[ "input_id_sentence1"][:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 12 + [self.tokenizer.mask_token_id] + f[ "input_id_sentence2"][1:])[:512]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:512]
        return input_ids, sentences_ids
    
class GlueDatasetForLMForClassification(GlueDatasetForLM):
    def __init__(self, tokenizer: AutoTokenizer, model_args, data_args, training_args) -> None:
        super().__init__(tokenizer, model_args, data_args, training_args)

        self.template_id = self.model_args.template_id
        self.template_dict = {
            "template_0": self.get_input_ids_0,
            "template_1": self.get_input_ids_1,
            "template_2": self.get_input_ids_2,
            "template_3": self.get_input_ids_3,
            "template_6": self.get_input_ids_6,
            "template_66": self.get_input_ids_66,
            "template_12": self.get_input_ids_12,
        }
        self.pre_seq_len_dict = {
            "template_0": 0,
            "template_1": 1,
            "template_2": 2,
            "template_3": 3,
            "template_6": 6,
            "template_66": 12,
            "template_12": 12
        }
        self.pre_seq_len = self.pre_seq_len_dict[self.template_id]
        self.get_input_ids = self.template_dict[self.template_id]
        
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["歌"]})

        self.raw_datasets = self.raw_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    
        if training_args.do_train:
            self.train_dataset = self.raw_datasets["train"]
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))

        if training_args.do_eval:
            self.eval_dataset = self.raw_datasets["validation_matched" if data_args.dataset_name == "mnli" else "validation"]
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))

        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            self.predict_dataset = self.raw_datasets["test_matched" if data_args.dataset_name == "mnli" else "test"]
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))
                
        self.data_collator = self.data_collator_glue_lm
        
    def preprocess_function(self, examples):
        # Tokenize the textspre_seq_len
        result = {
            "input_id_sentence": self.tokenizer(examples["sentence"], padding='do_not_pad', max_length=512, truncation=True)["input_ids"],
            "label_token_id": self.tokenizer([self.label2token[str(label)]for label in examples["label"]], padding='do_not_pad', max_length=512, truncation=True)["input_ids"] # 1 to 'yes' or 'true'
        }
        return result

    def get_input_ids_0(self, f):
        input_ids = f[ "input_id_sentence1"]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:512] + [self.tokenizer.sep_token_id]
        return input_ids, sentences_ids
    
    def get_input_ids_1(self, f):
        input_ids = f[ "input_id_sentence1"][:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 1 + [self.tokenizer.mask_token_id] + [self.tokenizer.sep_token_id]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:512]
        return input_ids, sentences_ids
    
    def get_input_ids_2(self, f):
        input_ids = f[ "input_id_sentence1"][:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 2 + [self.tokenizer.mask_token_id] + [self.tokenizer.sep_token_id]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:512]
        return input_ids, sentences_ids
    
    def get_input_ids_3(self, f):
        input_ids = f[ "input_id_sentence1"][:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 3 + [self.tokenizer.mask_token_id] + [self.tokenizer.sep_token_id]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:512]
        return input_ids, sentences_ids

    def get_input_ids_6(self, f):
        input_ids = f[ "input_id_sentence1"][:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + [self.tokenizer.mask_token_id] + [self.tokenizer.sep_token_id]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:512]
        return input_ids, sentences_ids
    
    def get_input_ids_66(self, f):
        input_ids = f[ "input_id_sentence1"][:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + [self.tokenizer.mask_token_id] + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + [self.tokenizer.sep_token_id]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:512]
        return input_ids, sentences_ids
    
    def get_input_ids_12(self, f):
        input_ids = f[ "input_id_sentence1"][:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 12 + [self.tokenizer.mask_token_id] + [self.tokenizer.sep_token_id]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:512]
        return input_ids, sentences_ids
    

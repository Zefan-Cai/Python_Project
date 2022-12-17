from datasets import load_dataset, load_metric, concatenate_datasets
import random
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import numpy as np
import logging
from collections import defaultdict

task_to_keys = {
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    "rte": ("premise", "hypothesis"),
    "wic": ("processed_sentence1", None),
    "wsc": ("span2_word_text", "span1_text"),
    "copa": (None, None),
    "record": (None, None),
    "multirc": ("paragraph", "question_answer")
}


logger = logging.getLogger(__name__)


class SuperGlueDataset():
    def __init__(self, tokenizer, model_args, data_args, training_args) -> None:
        super().__init__()
        raw_datasets = load_dataset("../../../tasks/superglue/superglue_dataset.py", data_args.dataset_name, ignore_verifications=True),
        self.raw_datasets = raw_datasets[0]
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args        

        self.multiple_choice = data_args.dataset_name in ["copa"]

        if self.data_args.dataset_name == "record":
            self.num_labels = 2
            self.label_list = ["0", "1"]
        elif not self.multiple_choice:
            self.label_list = self.raw_datasets["train"].features["label"].names
            self.num_labels = len(self.label_list)
        else:
            # self.num_labels = 1
            self.num_labels = 2

        # Preprocessing the raw_datasets
        self.sentence1_key, self.sentence2_key = task_to_keys[data_args.dataset_name]

        # Padding strategy
        if self.data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = "longest"

        if not self.multiple_choice:
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}
            print(f"{self.label2id}")
            print(f"{self.id2label}")

        if self.data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

#         if data_args.dataset_name == "record":
#             self.raw_datasets = self.raw_datasets.map(
#                 self.record_preprocess_function,
#                 batched=True,
#                 load_from_cache_file=not data_args.overwrite_cache,
#                 remove_columns=raw_datasets["train"].column_names,
#                 desc="Running tokenizer on dataset",
#             )
#         else:
#             self.raw_datasets = self.raw_datasets.map(
#                 self.preprocess_function,
#                 batched=True,
#                 load_from_cache_file=not data_args.overwrite_cache,
#                 desc="Running tokenizer on dataset",
#             )

#         if training_args.do_train:
#             self.train_dataset = self.raw_datasets["train"]
#             if data_args.max_train_samples is not None:
#                 self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))

#         if training_args.do_eval:
#             self.eval_dataset = self.raw_datasets["validation"]
#             if data_args.max_eval_samples is not None:
#                 self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))

#         if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
#             self.predict_dataset = self.raw_datasets["test"]
#             if data_args.max_predict_samples is not None:
#                 self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))

        self.metric = load_metric("../../../tasks/superglue/superglue_metric.py", data_args.dataset_name)
    
        if data_args.pad_to_max_length:
            self.data_collator = default_data_collator
        elif training_args.fp16:
            self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

        self.test_key = "accuracy" if data_args.dataset_name not in ["record", "multirc"] else "f1"


    def preprocess_function(self, examples):
        # WSC
        if self.data_args.dataset_name == "wsc":
            examples["span2_word_text"] = []
            for text, span2_index, span2_word in zip(examples["text"], examples["span2_index"], examples["span2_text"]):
                if self.data_args.template_id == 0:
                    examples["span2_word_text"].append(span2_word + ": " + text)
                elif self.data_args.template_id == 1:
                    words_a = text.split()
                    words_a[span2_index] = "*" + words_a[span2_index] + "*"
                    examples["span2_word_text"].append(' '.join(words_a))

        # WiC
        if self.data_args.dataset_name == "wic":
            examples["processed_sentence1"] = []
            if self.data_args.template_id == 1:
                self.sentence2_key = "processed_sentence2"
                examples["processed_sentence2"] = []
            for sentence1, sentence2, word, start1, end1, start2, end2 in zip(examples["sentence1"], examples["sentence2"], examples["word"], examples["start1"], examples["end1"], examples["start2"], examples["end2"]):
                if self.data_args.template_id == 0: #ROBERTA
                    examples["processed_sentence1"].append(f"{sentence1} {sentence2} Does {word} have the same meaning in both sentences?")
                elif self.data_args.template_id == 1: #BERT
                    examples["processed_sentence1"].append(word + ": " + sentence1)
                    examples["processed_sentence2"].append(word + ": " + sentence2)

        # MultiRC
        if self.data_args.dataset_name == "multirc":
            examples["question_answer"] = []
            for question, asnwer in zip(examples["question"], examples["answer"]):
                examples["question_answer"].append(f"{question} {asnwer}")

        # COPA
        if self.data_args.dataset_name == "copa":
            examples["text_a"] = []
            for premise, question in zip(examples["premise"], examples["question"]):
                joiner = "because" if question == "cause" else "so"
                text_a = f"{premise} {joiner}"                    
                examples["text_a"].append(text_a)

            result1 = self.tokenizer(examples["text_a"], examples["choice1"], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
            result2 = self.tokenizer(examples["text_a"], examples["choice2"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
            result = {}  
            for key in ["input_ids", "attention_mask", "token_type_ids"]:
                if key in result1 and key in result2:
                    result[key] = []
                    for value1, value2 in zip(result1[key], result2[key]):
                        result[key].append([value1, value2])
            return result
        
        # Boolq
        if self.data_args.dataset_name == "boolq":
            args = (
                (examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key], examples[self.sentence2_key])
            )
            result = self.tokenizer(*args, padding="do_not_pad", max_length=self.max_seq_length, truncation=True)
            result["sentences_ids"] = self.tokenizer(examples[self.sentence1_key], padding="do_not_pad", max_length=self.max_seq_length, truncation=True)["input_ids"]
            return result
        
        # cb
        if self.data_args.dataset_name in ["cb", "rte"]:
            args = (
                (examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key], examples[self.sentence2_key])
            )
            result = self.tokenizer(*args, padding="do_not_pad", max_length=self.max_seq_length, truncation=True)
            result["sentences_ids"] = self.tokenizer(*args, padding="do_not_pad", max_length=self.max_seq_length, truncation=True)["input_ids"]
            return result

        args = (
            (examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key], examples[self.sentence2_key])
        )
        result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)

        return result

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        print(type(preds))

        if self.data_args.dataset_name == "record":
            return self.reocrd_compute_metrics(p)

        if self.data_args.dataset_name == "multirc":
            from sklearn.metrics import f1_score
            return {"f1": f1_score(preds, p.label_ids)}

        if self.data_args.dataset_name is not None:
            result = self.metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif self.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    def reocrd_compute_metrics(self, p: EvalPrediction):
        from tasks.superglue.utils import f1_score, exact_match_score, metric_max_over_ground_truths
        probs = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        examples = self.eval_dataset
        qid2pred = defaultdict(list)
        qid2ans = {}
        for prob, example in zip(probs, examples):
            qid = example['question_id']
            qid2pred[qid].append((prob[1], example['entity']))
            if qid not in qid2ans:
                qid2ans[qid] = example['answers']
        n_correct, n_total = 0, 0
        f1, em = 0, 0
        for qid in qid2pred:
            preds = sorted(qid2pred[qid], reverse=True)
            entity = preds[0][1]
            n_total += 1
            n_correct += (entity in qid2ans[qid])
            f1 += metric_max_over_ground_truths(f1_score, entity, qid2ans[qid])
            em += metric_max_over_ground_truths(exact_match_score, entity, qid2ans[qid])
        acc = n_correct / n_total
        f1 = f1 / n_total
        em = em / n_total
        return {'f1': f1, 'exact_match': em}

    def record_preprocess_function(self, examples, split="train"):
        results = {
            "index": list(),
            "question_id": list(),
            "input_ids": list(),
            "attention_mask": list(),
            "token_type_ids": list(),
            "label": list(),
            "entity": list(),
            "answers": list()
        }
        for idx, passage in enumerate(examples["passage"]):
            query, entities, answers =  examples["query"][idx], examples["entities"][idx], examples["answers"][idx]
            index = examples["idx"][idx]
            passage = passage.replace("@highlight\n", "- ")
            
            for ent_idx, ent in enumerate(entities):
                question = query.replace("@placeholder", ent)
                result = self.tokenizer(passage, question, padding=self.padding, max_length=self.max_seq_length, truncation=True)
                label = 1 if ent in answers else 0

                results["input_ids"].append(result["input_ids"])
                results["attention_mask"].append(result["attention_mask"])
                if "token_type_ids" in result: results["token_type_ids"].append(result["token_type_ids"])
                results["label"].append(label)
                results["index"].append(index)
                results["question_id"].append(index["query"])
                results["entity"].append(ent)
                results["answers"].append(answers)

        return results
    
class SuperGlueDatasetForSequenceClassification(SuperGlueDataset):
    def __init__(self, tokenizer, model_args, data_args, training_args) -> None:
        super().__init__(tokenizer, model_args, data_args, training_args)
        
        if data_args.dataset_name == "record":
            self.raw_datasets = self.raw_datasets.map(
                self.record_preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on dataset",
            )
        else:
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
            self.eval_dataset = self.raw_datasets["validation"]
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))

        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            self.predict_dataset = self.raw_datasets["test"]
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))
        
        self.data_collator = self.data_collator_glue
        
    def data_collator_glue(self, features):
        first = features[0]
        batch = {}

        # sentence_ids，label_ids和input_ids需要pad
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
            {"input_ids": [f["sentences_ids"] for f in features]},
            padding=self.padding,
            max_length=self.max_seq_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch["sentences_ids"] = label_ids_result["input_ids"]
        
        reduced_column = []
        reduced_column.extend(["idx", "input_ids", "attention_mask", "labels", "label_ids", "sentences_ids", "token_type_ids"]) # general
        reduced_column.extend(["input_id", "sentence_id"]) # boolq
        reduced_column.extend(["sentence1", "sentence2", "input_id_sentence1", "input_id_sentence2"]) # nli(cb, rte)
        reduced_column.extend(["span1_text", "span1_index", "span2_text", "span2_index"]) # wsc
        reduced_column.extend(["text", "word", "start1", "start2", "end1", "end2"]) # wic
        reduced_column.extend(["paragraph", "question", "answer"]) # multirc
        
        for k, v in first.items():
            if v is not None and not isinstance(v, str) and k not in reduced_column:
                batch[k] = torch.tensor([f[k] for f in features])
        batch["labels"] = batch["label"]
        del(batch["label"])
        return batch
    
    def preprocess_function(self, examples):
        # Boolq
        if self.data_args.dataset_name == "boolq":
            args = (
                (examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key], examples[self.sentence2_key])
            )
            result = self.tokenizer(*args, padding="do_not_pad", max_length=self.max_seq_length, truncation=True)
            result["sentences_ids"] = self.tokenizer(examples[self.sentence1_key], padding="do_not_pad", max_length=self.max_seq_length, truncation=True)["input_ids"]
            return result
        
        # cb
        if self.data_args.dataset_name in ["cb", "rte"]:
            args = (
                (examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key], examples[self.sentence2_key])
            )
            result = self.tokenizer(*args, padding="do_not_pad", max_length=self.max_seq_length, truncation=True)
            result["sentences_ids"] = self.tokenizer(*args, padding="do_not_pad", max_length=self.max_seq_length, truncation=True)["input_ids"]
            return result

        args = (
            (examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key], examples[self.sentence2_key])
        )
        result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)

        return result

class SuperGlueDatasetForLM(SuperGlueDataset):
    def __init__(self, tokenizer, model_args, data_args, training_args) -> None:
        super().__init__(tokenizer, model_args, data_args, training_args)
        self.verbalizer_dict = {
            "2": {
                "verbalizer_0": {"0": "no", "1": "yes", "-1": "a"},
                "verbalizer_1": {"0": "No", "1": "Yes", "-1": "a"},
                "verbalizer_2": {"0": "false", "1": "true", "-1": "a"},
                "verbalizer_3": {"0": "False", "1": "True", "-1": "a"},
            },
            "3": {
                "verbalizer_0": {"0": "no", "1": "maybe", "2": "yes", "-1": "a"},
                "verbalizer_1": {"0": "No", "1": "maybe", "2": "yes", "-1": "a"},
                "verbalizer_2": {"0": "false", "1": "maybe", "2": "yes", "-1": "a"},
                "verbalizer_3": {"0": "False", "1": "maybe", "2": "yes", "-1": "a"},
                "verbalizer_4": {"0": "False No", "1": "maybe maybe", "2": "True Yes", "-1": "a me"}
            },
        }
        self.label2token = self.verbalizer_dict[str(self.num_labels)][self.model_args.verbalizer_id]
        self.label_token_list = [v for _, v in self.verbalizer_dict[str(self.num_labels)][self.model_args.verbalizer_id].items()]
        self.token2label = {v: k for k, v in self.verbalizer_dict[str(self.num_labels)][self.model_args.verbalizer_id].items()}

    def get_input_ids(self, input):
        pass

    def data_collator_glue_lm(self, features):
        """
        返回：input_ids, sentence_dis, labels, attention_mask, label_token_id_list
        """

        first = features[0]
        batch = {}

        for f in features:
            f["input_ids"], f["sentences_ids"] = self.get_input_ids(f)
            label_ids = [-100 for _ in range(len(f["input_ids"]))]
            mask_start = f["input_ids"].index(self.tokenizer.mask_token_id)
            f["label_token_id"] = f["label_token_id"][1: -1]
            label_token_id = f["label_token_id"]
            mask_end = mask_start + len(label_token_id)
            label_ids[mask_start: mask_end] = label_token_id
            f["labels"] = label_ids
            f["label_token_id_list"] = [self.tokenizer.encode(l)[1:-1] for l in self.label_token_list]

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
            {"input_ids": [f["labels"] for f in features]},
            padding=self.padding,
            max_length=self.max_seq_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch["labels"] = label_ids_result["input_ids"]
        
        sentences_ids_result = self.tokenizer.pad(
            {"input_ids": [f["sentences_ids"] for f in features]},
            padding="longest",
            max_length=self.max_seq_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch["sentences_ids"] = sentences_ids_result["input_ids"]
        
        reduced_column = []
        reduced_column.extend(["idx", "input_ids", "attention_mask", "labels", "label_ids", "sentences_ids"]) # general
        reduced_column.extend(["input_id", "sentence_id"]) # boolq
        reduced_column.extend(["sentence1", "sentence2", "input_id_sentence1", "input_id_sentence2"]) # nli(cb, rte)
        reduced_column.extend(["span1_text", "span1_index", "span2_text", "span2_index"]) # wsc
        reduced_column.extend(["text", "word", "start1", "start2", "end1", "end2"]) # wic
        reduced_column.extend(["paragraph", "question", "answer"]) # multirc
        
        for k, v in first.items():
        #     print(k)
        #     if type(v) == list: print([len(f[k]) for f in features])
        #     else: print(features[0][k])
            # if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if v is not None and not isinstance(v, str) and k not in reduced_column:
                batch[k] = torch.tensor([f[k] for f in features])
                # print(batch[k].shape)
        return batch

class SuperGlueDatasetForLMForBoolQ(SuperGlueDatasetForLM):
    def __init__(self, tokenizer, model_args, data_args, training_args) -> None:
        super().__init__(tokenizer, model_args, data_args, training_args)

        self.template_id = self.model_args.template_id
        self.template_dict = {
            "template_6": self.get_input_ids_6,
        }
        self.pre_seq_len_dict = {
            "template_6": 6,
        }
        self.pre_seq_len = self.pre_seq_len_dict[self.template_id]
        self.get_input_ids = self.template_dict[self.template_id]
        
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["歌"]})

        self.raw_datasets = self.raw_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
#             desc="Running tokenizer on dataset",
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
            "input_id": self.tokenizer(examples[self.sentence2_key], examples[self.sentence1_key], padding='do_not_pad', max_length=505, truncation=True)["input_ids"],
            "sentence_id": self.tokenizer(examples[self.sentence1_key], padding='do_not_pad', max_length=512, truncation=True)["input_ids"],
            "label_token_id": self.tokenizer([self.label2token[str(label)]for label in examples["label"]], padding='do_not_pad', max_length=512, truncation=True)["input_ids"] # 1 to 'yes' or 'true'
        }
        return result
    def get_input_ids_6(self, f):
        input_ids = ([self.tokenizer.cls_token_id] + f["input_id"][1:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + [self.tokenizer.mask_token_id] + [self.tokenizer.cls_token_id])[:512]
        sentences_ids = f["sentence_id"][1:-1][:512]
        return input_ids, sentences_ids

class SuperGlueDatasetForLMForNLI(SuperGlueDatasetForLM):
    def __init__(self, tokenizer, model_args, data_args, training_args) -> None:
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
#             desc="Running tokenizer on dataset",
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
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:511]
        return input_ids, sentences_ids
    
    def get_input_ids_2(self, f):
        input_ids = (f[ "input_id_sentence1"][:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 2 + [self.tokenizer.mask_token_id] + f[ "input_id_sentence2"][1:])[:512]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:510]
        return input_ids, sentences_ids
    
    def get_input_ids_3(self, f):
        input_ids = (f[ "input_id_sentence1"][:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 3 + [self.tokenizer.mask_token_id] + f[ "input_id_sentence2"][1:])[:512]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:509]
        return input_ids, sentences_ids

    def get_input_ids_6(self, f):
        input_ids = (f[ "input_id_sentence1"][:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + [self.tokenizer.mask_token_id] + f[ "input_id_sentence2"][1:])[:512]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:506]
        return input_ids, sentences_ids
    
    def get_input_ids_66(self, f):
        input_ids = (f[ "input_id_sentence1"][:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + [self.tokenizer.mask_token_id] + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + f[ "input_id_sentence2"][1:])[:512]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:500]
        return input_ids, sentences_ids
    
    def get_input_ids_12(self, f):
        input_ids = (f[ "input_id_sentence1"][:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 12 + [self.tokenizer.mask_token_id] + f[ "input_id_sentence2"][1:])[:512]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:512]
        return input_ids, sentences_ids


# TODO: Finish the data script of WiC, WSC, COPA, MultiRC and ReCoRD in May 3
class SuperGlueDatasetForLMForWiC(SuperGlueDatasetForLM):
    def __init__(self, tokenizer, model_args, data_args, training_args) -> None:
        super().__init__(tokenizer, model_args, data_args, training_args)

        self.template_id = self.model_args.template_id
        self.template_dict = {
            "template_6": self.get_input_ids_6,
            "template_66": self.get_input_ids_66,
            "template_33": self.get_input_ids_33,
            "template_333": self.get_input_ids_333,
            "template_666": self.get_input_ids_666
        }
        self.pre_seq_len_dict = {
            "template_6": 6,
            "template_33": 6,
            "template_66": 12,
            "template_333": 9,
            "template_666": 18
        }
        self.pre_seq_len = self.pre_seq_len_dict[self.template_id]
        self.get_input_ids = self.template_dict[self.template_id]
        
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["歌"]})

        self.raw_datasets = self.raw_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
#             desc="Running tokenizer on dataset",
        )

        self.text_a = self.tokenizer("?", padding='do_not_pad', max_length=512, truncation=True)["input_ids"][1:-1]
    
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
        result = {
            "input_id_sentence1": self.tokenizer(examples["sentence1"], padding='do_not_pad', max_length=512, truncation=True)["input_ids"],
            "input_id_sentence2": self.tokenizer(examples["sentence2"], padding='do_not_pad', max_length=512, truncation=True)["input_ids"],
            "word": self.tokenizer(examples["word"], padding='do_not_pad', max_length=512, truncation=True)["input_ids"],
            "label_token_id": self.tokenizer([self.label2token[str(label)]for label in examples["label"]], padding='do_not_pad', max_length=512, truncation=True)["input_ids"] # 1 to 'yes' or 'true'
        }
        return result

    def get_input_ids_6(self, f):
        input_ids = ([self.tokenizer.cls_token_id] + f["input_id_sentence1"][1:-1] + [self.tokenizer.sep_token_id] + f[ "input_id_sentence2"][1:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + f["word"][1:-1] + self.text_a + [self.tokenizer.mask_token_id] + [self.tokenizer.sep_token_id])[:512]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:512]
        return input_ids, sentences_ids

    def get_input_ids_33(self, f):
        input_ids = ([self.tokenizer.cls_token_id] + f["input_id_sentence1"][1:-1] + [self.tokenizer.sep_token_id] + f[ "input_id_sentence2"][1:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 3 + f["word"][1:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 3 + self.text_a + [self.tokenizer.mask_token_id] + [self.tokenizer.sep_token_id])[:512]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:512]
        return input_ids, sentences_ids

    def get_input_ids_66(self, f):
        input_ids = ([self.tokenizer.cls_token_id] + f["input_id_sentence1"][1:-1] + [self.tokenizer.sep_token_id] + f[ "input_id_sentence2"][1:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + f["word"][1:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + self.text_a + [self.tokenizer.mask_token_id] + [self.tokenizer.sep_token_id])[:512]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:512]
        return input_ids, sentences_ids
    
    def get_input_ids_333(self, f):
        input_ids = ([self.tokenizer.cls_token_id] + f["input_id_sentence1"][1:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 3 + f["word"][1:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 3 + [self.tokenizer.mask_token_id] + [self.tokenizer.additional_special_tokens_ids[0]] * 3 + f[ "input_id_sentence2"][1:-1] + [self.tokenizer.sep_token_id])[:512]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:512]
        return input_ids, sentences_ids
    
    def get_input_ids_666(self, f):
        input_ids = ([self.tokenizer.cls_token_id] + f["input_id_sentence1"][1:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + f["word"][1:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + [self.tokenizer.mask_token_id] + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + f[ "input_id_sentence2"][1:-1] + [self.tokenizer.sep_token_id])[:512]
        sentences_ids = (f[ "input_id_sentence1"][1:-1] +  f[ "input_id_sentence2"][1:-1])[:512]
        return input_ids, sentences_ids
    
    
class SuperGlueDatasetForLMForWSC(SuperGlueDatasetForLM):
    def __init__(self, tokenizer, model_args, data_args, training_args) -> None:
        super().__init__(tokenizer, model_args, data_args, training_args)

        self.rng = random.Random(training_args.seed)

        self.template_id = self.model_args.template_id
        self.template_dict = {
            "template_6": self.get_input_ids_6,
            "template_333": self.get_input_ids_333,
            "template_666": self.get_input_ids_666
        }
        self.pre_seq_len_dict = {
            "template_6": 6,
            "template_333": 9,
            "template_666": 18
        }
        self.pre_seq_len = self.pre_seq_len_dict[self.template_id]
        self.get_input_ids = self.template_dict[self.template_id]
        
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["歌"]})

        self.raw_datasets["train"] = self.raw_datasets["train"].map(lambda example: {"set_type": "train"}).filter(lambda example: example['label'] == 1)
        self.raw_datasets["validation"] = self.raw_datasets["validation"].map(lambda example: {"set_type": "validation"})
        self.raw_datasets = self.raw_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache
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
                
        self.data_collator = self.data_collator_wsc

    def data_collator_wsc(self, features):
        """
        返回：input_ids, sentence_dis, labels, attention_mask, label_token_id
        """

        first = features[0]
        batch = {}

        text_a = self.tokenizer("the pronoun '*", padding='do_not_pad', max_length=512, truncation=True)["input_ids"][1:-1]
        text_b = self.tokenizer("*' refers to", padding='do_not_pad', max_length=512, truncation=True)["input_ids"][1:-1]
        text_c = self.tokenizer(".", padding='do_not_pad', max_length=512, truncation=True)["input_ids"][1:-1]

        for f in features:
            num_pad = self.rng.randint(0, 3) if f["set_type"] in ["train"] else 1
            num_masks = len(f["span1_text"]) + num_pad
            f["input_ids"], f["sentences_ids"] = self.get_input_ids(f, num_masks, text_a, text_b, text_c)
            f["label_token_id"] = f["span1_text"][1: -1]
            label_ids = [-100 for _ in range(len(f["input_ids"]))]
            mask_index = f["input_ids"].index(self.tokenizer.mask_token_id)
            label_ids[mask_index: mask_index + len(f["label_token_id"])] = f["label_token_id"]
            label_ids[mask_index + len(f["label_token_id"]): mask_index + len(f["label_token_id"]) + num_pad] = [self.tokenizer.pad_token_id] * num_pad
            f["labels"] = label_ids

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
            {"input_ids": [f["labels"] for f in features]},
            padding=self.padding,
            max_length=self.max_seq_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch["labels"] = label_ids_result["input_ids"]
        
        sentences_ids_result = self.tokenizer.pad(
            {"input_ids": [f["sentences_ids"] for f in features]},
            padding="longest",
            max_length=self.max_seq_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch["sentences_ids"] = sentences_ids_result["input_ids"]
        
        reduced_column = []
        reduced_column.extend(["idx", "input_ids", "attention_mask", "labels", "label_ids", "sentences_ids"]) # general
        reduced_column.extend(["input_id", "sentence_id"]) # boolq
        reduced_column.extend(["sentence1", "sentence2", "input_id_sentence1", "input_id_sentence2"]) # nli(cb, rte)
        reduced_column.extend(["span1_text", "span1_index", "span2_text", "span2_index", "label_token_id"]) # wsc
        reduced_column.extend(["text", "word", "start1", "start2", "end1", "end2"]) # wic
        reduced_column.extend(["paragraph", "question", "answer"]) # multirc
        
        for k, v in first.items():
            if v is not None and not isinstance(v, str) and k not in reduced_column:
                batch[k] = torch.tensor([f[k] for f in features])
        batch["label_token_id"] = [f["label_token_id"] for f in features]
        return batch

    def preprocess_function(self, examples):
        result = {
            "text": self.tokenizer(examples["text"], padding='do_not_pad', max_length=512, truncation=True)["input_ids"],
            "span1_text": self.tokenizer(examples["span1_text"], padding='do_not_pad', max_length=512, truncation=True)["input_ids"],
            "span2_text": self.tokenizer(examples["span2_text"], padding='do_not_pad', max_length=512, truncation=True)["input_ids"],
        }
        return result
    
    def get_input_ids_6(self, f, num_masks, text_a, text_b, text_c):
        input_ids = ([self.tokenizer.cls_token_id] + f["text"][1:480] + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + text_a + f["span2_text"][1: -1] + text_b + [self.tokenizer.mask_token_id] * num_masks + text_c + [self.tokenizer.sep_token_id])[:512]
        sentences_ids = f["text"][1: -1]
        return input_ids, sentences_ids
    
    def get_input_ids_333(self, f):
        input_ids = ([self.tokenizer.cls_token_id] + f["text"][1:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 3 + f["span2_text"][1:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 3 + f[ "span1_text"][1:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 3 + [self.tokenizer.mask_token_id] + [self.tokenizer.sep_token_id])[:512]
        sentences_ids = f["text"][1: -1]
        return input_ids, sentences_ids
    
    def get_input_ids_666(self, f):
        input_ids = ([self.tokenizer.cls_token_id] + f["text"][1:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + f["span2_text"][1:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + f[ "span1_text"][1:-1] + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + [self.tokenizer.mask_token_id] + [self.tokenizer.sep_token_id])[:512]
        sentences_ids = f["text"][1: -1]
        return input_ids, sentences_ids
    
class SuperGlueDatasetForLMForMultiRC(SuperGlueDatasetForLM):
    def __init__(self, tokenizer, model_args, data_args, training_args) -> None:
        super().__init__(tokenizer, model_args, data_args, training_args)

        self.template_id = self.model_args.template_id
        self.template_dict = {
            "template_6": self.get_input_ids_6,
        }
        self.pre_seq_len_dict = {
            "template_6": 6,
        }
        self.pre_seq_len = self.pre_seq_len_dict[self.template_id]
        self.get_input_ids = self.template_dict[self.template_id]

        self.text_a = self.tokenizer(". Question: ", padding='do_not_pad', max_length=512, truncation=True)["input_ids"][1:-1]
        self.text_b = self.tokenizer("? Is it ", padding='do_not_pad', max_length=512, truncation=True)["input_ids"][1:-1]
        self.text_c = self.tokenizer("?", padding='do_not_pad', max_length=512, truncation=True)["input_ids"][1:-1]
        self.text_d = self.tokenizer(".", padding='do_not_pad', max_length=512, truncation=True)["input_ids"][1:-1]
        
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["歌"]})

        self.raw_datasets = self.raw_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
#             desc="Running tokenizer on dataset",
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
        result = {
            "paragraph": self.tokenizer(examples["paragraph"], padding='do_not_pad', max_length=512, truncation=True)["input_ids"],
            "question": self.tokenizer(examples["question"], padding='do_not_pad', max_length=512, truncation=True)["input_ids"],
            "answer": self.tokenizer(examples["answer"], padding='do_not_pad', max_length=512, truncation=True)["input_ids"],
            "label_token_id": self.tokenizer([self.label2token[str(label)]for label in examples["label"]], padding='do_not_pad', max_length=512, truncation=True)["input_ids"] # 1 to 'yes' or 'true'
        }
        return result
    
    def get_input_ids_6(self, f):
        input_ids = ([self.tokenizer.cls_token_id] + f["paragraph"][:360] + self.text_a + f["question"][1: -1] + self.text_b + f["answer"][1: -1] + self.text_c + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + [self.tokenizer.mask_token_id] + self.text_d + [self.tokenizer.sep_token_id])[:512]
        sentences_ids = f["question"][1: -1]
        return input_ids, sentences_ids
    
class SuperGlueDatasetForLMForCOPA(SuperGlueDatasetForLM):
    def __init__(self, tokenizer, model_args, data_args, training_args) -> None:
        super().__init__(tokenizer, model_args, data_args, training_args)

        self.template_id = self.model_args.template_id
        self.template_dict = {
            "template_6": self.get_input_ids_6
        }
        self.pre_seq_len_dict = {
            "template_6": 6,
        }
        self.pre_seq_len = self.pre_seq_len_dict[self.template_id]
        self.get_input_ids = self.template_dict[self.template_id]
        
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["歌"]})

        train_len = len(self.raw_datasets["train"])
        updated_dataset =  self.raw_datasets["train"].map(lambda example: {'idx': example['idx'] + train_len, 'choice1': example['choice2'], 'choice2': example["choice1"], 'label': 1 - example['label']})
        self.raw_datasets["train"] = concatenate_datasets([self.raw_datasets["train"], updated_dataset])
        self.raw_datasets = self.raw_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
#             desc="Running tokenizer on dataset",
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
                
        self.data_collator = self.data_collator_copa
        
    def data_collator_copa(self, features):
        first = features[0]
        batch = {}
        choice_list = ["choice1", "choice2"]

        text_a = self.tokenizer("or", padding='do_not_pad', max_length=512, truncation=True)["input_ids"][1:-1]
        text_b = self.tokenizer("?", padding='do_not_pad', max_length=512, truncation=True)["input_ids"][1:-1]
        text_c = self.tokenizer(".", padding='do_not_pad', max_length=512, truncation=True)["input_ids"][1:-1]
        joiner_because = self.tokenizer("because", padding='do_not_pad', max_length=512, truncation=True)["input_ids"][1:-1]
        joiner_so = self.tokenizer("so", padding='do_not_pad', max_length=512, truncation=True)["input_ids"][1:-1]
        
        for f in features:
            num_masks = max(len(f[c]) for c in choice_list)

            if f["question"] == "cause":
                joiner = joiner_because
            else:
                joiner = joiner_so

            f["input_ids"], f["sentences_ids"] = self.get_input_ids(f, num_masks, text_a, text_b, text_c, joiner)

            mask_start = f["input_ids"].index(self.tokenizer.mask_token_id)

            for choice in choice_list:
                choice_token_ids = f[choice][1:-1]
                mask_end = mask_start + len(choice_token_ids)
                num_pad = num_masks - mask_end
                f[f'{choice}_token_ids'] = [-100] * len(f["input_ids"])
                f[f'{choice}_token_ids'][mask_start: mask_end] = choice_token_ids
                # f[f'{choice}_token_ids'][mask_end: mask_end + num_pad] = [self.tokenizer.pad_token_id] * num_pad

            if f["label"] == 0:
                f["labels"] = f["choice1_token_ids"]
            else:
                f["labels"] = f["choice2_token_ids"]
                
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
        
        for choice in choice_list:
            label_ids_result = self.tokenizer.pad(
                {"input_ids": [f[f'{choice}_token_ids'] for f in features]},
                padding=self.padding,
                max_length=self.max_seq_length,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
            batch[f'{choice}_token_ids'] = label_ids_result["input_ids"]

        label_ids_result = self.tokenizer.pad(
            {"input_ids": [f["labels"] for f in features]},
            padding=self.padding,
            max_length=self.max_seq_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch["labels"] = label_ids_result["input_ids"]
        
        sentences_ids_result = self.tokenizer.pad(
            {"input_ids": [f["sentences_ids"] for f in features]},
            padding="longest",
            max_length=self.max_seq_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch["sentences_ids"] = sentences_ids_result["input_ids"]
        
        reduced_column = []
        reduced_column.extend(["idx", "input_ids", "attention_mask", "labels", "label_ids", "sentences_ids"]) # general
        reduced_column.extend(["input_id", "sentence_id"]) # boolq
        reduced_column.extend(["sentence1", "sentence2", "input_id_sentence1", "input_id_sentence2"]) # nli(cb, rte)
        reduced_column.extend(["span1_text", "span1_index", "span2_text", "span2_index", "label_token_id"]) # wsc
        reduced_column.extend(["text", "word", "start1", "start2", "end1", "end2"]) # wic
        reduced_column.extend(["paragraph", "question", "answer"]) # multirc
        reduced_column.extend(["premise", "question", "choice1", "choice2", "choice1_token_ids", "choice2_token_ids"]) # copa
        
        for k, v in first.items():
            if v is not None and not isinstance(v, str) and k not in reduced_column:
                batch[k] = torch.tensor([f[k] for f in features])
        batch["choice1"] = [f["choice1"] for f in features]
        batch["choice2"] = [f["choice2"] for f in features]
        return batch
        
    def preprocess_function(self, examples):
        result = {
            "premise": self.tokenizer(examples["premise"], padding='do_not_pad', max_length=512, truncation=True)["input_ids"],
            "choice1": self.tokenizer(examples["choice1"], padding='do_not_pad', max_length=512, truncation=True)["input_ids"],
            "choice2": self.tokenizer(examples["choice2"], padding='do_not_pad', max_length=512, truncation=True)["input_ids"],
        }
        return result
    
    def get_input_ids_6(self, f, num_masks, text_a, text_b, text_c, joiner):
        input_ids = ([self.tokenizer.cls_token_id] + f["choice1"][1:-1] + text_a + f["choice2"][1:-1] + text_b + f["premise"][1:-1] + joiner + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + [self.tokenizer.mask_token_id] * num_masks + text_c + [self.tokenizer.sep_token_id])[:512]
        sentences_ids = (f["premise"] +  joiner)[:512]
        return input_ids, sentences_ids



class SuperGlueDatasetForLMForRECORD(SuperGlueDatasetForLM):
    def __init__(self, tokenizer, model_args, data_args, training_args) -> None:
        super().__init__(tokenizer, model_args, data_args, training_args)

        self.rng = random.Random(training_args.seed)

        self.template_id = self.model_args.template_id
        self.template_dict = {
            "template_6": self.get_input_ids_6
        }
        self.pre_seq_len_dict = {
            "template_6": 6
        }
        self.pre_seq_len = self.pre_seq_len_dict[self.template_id]
        self.get_input_ids = self.template_dict[self.template_id]
        
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["歌"]})

        # print(self.raw_datasets["train"].features)
        # print(self.raw_datasets["train"][0])
        # print(self.raw_datasets["train"][1])
        # print(self.raw_datasets["train"][2])

        self.raw_datasets = self.raw_datasets.map(
            self.preprocess_function,
            load_from_cache_file=not data_args.overwrite_cache
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
                
        self.data_collator = self.data_collator_record

    def data_collator_record(self, features):
        """
        返回：input_ids, sentence_dis, labels, attention_mask, label_token_id
        """

        batch_size = len(features)
        first = features[0]
        batch = {}

        text_a = self.tokenizer(".", padding='do_not_pad', max_length=512, truncation=True)["input_ids"][1:-1]



        for f in features:
            f["input_ids"], f["sentences_ids"] = self.get_input_ids(f, text_a)
            
            mask_start = f["input_ids"].index(self.tokenizer.mask_token_id)

            f['candidate_label_ids'] = []
            f['candidate_labels'] = []

            for idx, choice_text in enumerate(f["entities_token_ids"]):
                choice_label = 1 if choice_text in f["answers_token_ids"] else 0

                mask_end = mask_start + len(choice_text)
                candidate_label_ids = [-100] * len(f["input_ids"])
                candidate_label_ids[mask_start:mask_end] = choice_text

                if choice_label == 1:
                    f["labels"] = candidate_label_ids

                f['candidate_label_ids'].append(candidate_label_ids)
                f['candidate_labels'].append(choice_label)

        max_num_candidates = max(len(f['candidate_label_ids']) for f in features)
        for feature in features:
            while len(feature['candidate_label_ids']) < max_num_candidates:
                feature['candidate_label_ids'].append([-100] * len(feature["input_ids"]))
                feature['candidate_labels'].append(-100)

        input_ids_result = self.tokenizer.pad(
            {"input_ids": [f["input_ids"] for f in features]},
            padding=self.padding,
            max_length=self.max_seq_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch["input_ids"] = input_ids_result["input_ids"]
        batch["attention_mask"] = input_ids_result["attention_mask"]
        
        candidate_input_ids = []
        for f in features:
            for candidate_label_ids in f["candidate_label_ids"]:
                candidate_input_ids.append(candidate_label_ids)
        candidate_input_ids = self.tokenizer.pad(
            {"input_ids": candidate_input_ids},
            padding=self.padding,
            max_length=self.max_seq_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch["candidate_label_ids"] = candidate_input_ids["input_ids"].view(batch_size, max_num_candidates, -1)
        batch["labels"] = candidate_input_ids["input_ids"].view(batch_size, max_num_candidates, -1)[:, 0, :]
        
        sentences_ids_result = self.tokenizer.pad(
            {"input_ids": [f["sentences_ids"] for f in features]},
            padding="longest",
            max_length=self.max_seq_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch["sentences_ids"] = sentences_ids_result["input_ids"]
        
        reduced_column = []
        reduced_column.extend(["idx", "input_ids", "attention_mask", "labels", "label_ids", "sentences_ids"]) # general
        reduced_column.extend(["input_id", "sentence_id"]) # boolq
        reduced_column.extend(["sentence1", "sentence2", "input_id_sentence1", "input_id_sentence2"]) # nli(cb, rte)
        reduced_column.extend(["span1_text", "span1_index", "span2_text", "span2_index", "label_token_id"]) # wsc
        reduced_column.extend(["text", "word", "start1", "start2", "end1", "end2"]) # wic
        reduced_column.extend(["paragraph", "question", "answer"]) # multirc
        reduced_column.extend(["passage", "question", "entities", "answers", "entities_token_ids", "answers_token_ids", "candidate_label_ids"]) # record
        
        for k, v in first.items():
            if v is not None and not isinstance(v, str) and k not in reduced_column:
                print(k)
                batch[k] = torch.tensor([f[k] for f in features])
        batch["label_token_id"] = [f["label_token_id"] for f in features]
        return batch

    def preprocess_function(self, examples):
        result = {}

        result["idx"] = examples["idx"]

        text = examples["passage"].replace("@highlight\n", "- ")
        result["passage"] = self.tokenizer(text, padding='do_not_pad', max_length=512, truncation=True)["input_ids"][1: -1]

        result["entities_token_ids"] = []
        for entity in examples["entities"]:
            result["entities_token_ids"].append(self.tokenizer(entity, padding='do_not_pad', max_length=512, truncation=True)["input_ids"][1: -1])

        result["answers_token_ids"] = []
        for answer in examples["answers"]:
            result["answers_token_ids"].append(self.tokenizer(answer, padding='do_not_pad', max_length=512, truncation=True)["input_ids"][1: -1])

        num_masks = max(len(entity) for entity in result["entities_token_ids"])
        question = examples["query"].replace('@placeholder', self.tokenizer.mask_token * num_masks)
        result["question"] = self.tokenizer(question, padding='do_not_pad', max_length=512, truncation=True)["input_ids"][1: -1]
        
        return result
    
    def get_input_ids_6(self, f, text_a):
        input_ids = ([self.tokenizer.cls_token_id] + f["passage"][0:480] + [self.tokenizer.additional_special_tokens_ids[0]] * 6 + f["question"] + text_a + [self.tokenizer.sep_token_id])[:512]
        sentences_ids = f["question"]
        return input_ids, sentences_ids
    
import logging
import os
import random
import sys

from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from transformers.optimization import Adafactor, AdafactorSchedule

from model.utils import get_model, TaskType
from tasks.glue.dataset import GlueDataset, GlueDatasetForLMForNLI, GlueDatasetForLMForClassification
from training.trainer_base import BaseTrainer

logger = logging.getLogger(__name__)


datasets = {
    "cola": GlueDatasetForLMForClassification,
    'qnli': GlueDatasetForLMForNLI,
    'qqp': GlueDatasetForLMForNLI,
    'rte': GlueDatasetForLMForNLI,
    'sst2': GlueDatasetForLMForClassification,
    "stsb": None,
    'wnli': GlueDatasetForLMForNLI
}

def get_trainer(args):
    model_args, data_args, training_args, _ = args

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    # dataset = GlueDataset(tokenizer, data_args, training_args)
    dataset = datasets[data_args.dataset_name](tokenizer, model_args, data_args, training_args)

    if not dataset.is_regression:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )
    
    config.pre_seq_len = dataset.pre_seq_len
    config.pseudo_token_id = tokenizer.additional_special_tokens_ids[0]
    config.pad_token_id = tokenizer.pad_token_id
    config.unk_token_id = tokenizer.unk_token_id
    config.mask_token_id = tokenizer.mask_token_id
    config.tokenizer = tokenizer
    tasktype = {
        "language_modeling": TaskType.LANGUAGE_MODELING,
        "sequence_classification": TaskType.SEQUENCE_CLASSIFICATION
    }
    model = get_model(model_args, tasktype[model_args.task_type], config)
    
    # replace AdamW with Adafactor
    optimizer = Adafactor(
        model.parameters(),
        lr=training_args.learning_rate,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=0.98,
        beta1=None,
        weight_decay=training_args.weight_decay,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    
    lr_scheduler = AdafactorSchedule(optimizer)


    # Initialize our Trainer
    trainer = BaseTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        optimizers=(optimizer, lr_scheduler)
    )

    return trainer, None
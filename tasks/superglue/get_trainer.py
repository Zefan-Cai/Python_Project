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
from tasks.superglue.dataset import (
    SuperGlueDatasetForSequenceClassification,
    SuperGlueDatasetForLMForBoolQ,
    SuperGlueDatasetForLMForNLI,
    SuperGlueDatasetForLMForWSC, 
    SuperGlueDatasetForLMForWiC, 
    SuperGlueDatasetForLMForMultiRC,
    SuperGlueDatasetForLMForCOPA,
    SuperGlueDatasetForLMForRECORD
)
from training.trainer_base import BaseTrainer
from training.trainer_lm import LMTrainer
from training.trainer_exp import ExponentialTrainer

logger = logging.getLogger(__name__)

datasets = {
    "boolq": SuperGlueDatasetForLMForBoolQ,
    "cb": SuperGlueDatasetForLMForNLI,
    "rte": SuperGlueDatasetForLMForNLI,
    "wic": SuperGlueDatasetForLMForWiC,
    "wsc": SuperGlueDatasetForLMForWSC,
    "multirc": SuperGlueDatasetForLMForMultiRC,
    "copa": SuperGlueDatasetForLMForCOPA,
    "record": SuperGlueDatasetForLMForRECORD
}

def get_trainer(args):
    model_args, data_args, training_args, _ = args

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    if model_args.task_type == "sequence_classification":
        dataset = SuperGlueDatasetForSequenceClassification(tokenizer, model_args, data_args, training_args)
    elif model_args.task_type == "language_modeling":
        dataset = datasets[data_args.dataset_name](tokenizer, model_args, data_args, training_args)

    if training_args.do_train:
        for index in random.sample(range(len(dataset.train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {dataset.train_dataset[index]}.")

    if not dataset.multiple_choice:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
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
        
    if model_args.task_type == "language_modeling":
        model_args.pre_seq_len = dataset.pre_seq_len
    model_args.multiple_choice = dataset.multiple_choice
    model_args.tokenizer = tokenizer


    # if not dataset.multiple_choice:
    model = get_model(model_args, config)
    # else:
    #     model = get_model(model_args, TaskType.MULTIPLE_CHOICE, config, fix_bert=True)
        
    # replace AdamW with Adafactor
#     optimizer = Adafactor(
#         model.parameters(),
#         lr=training_args.learning_rate,
#         eps=(1e-30, 1e-3),
#         clip_threshold=1.0,
#         decay_rate=0.98,
#         beta1=None,
#         weight_decay=training_args.weight_decay,
#         relative_step=False,
#         scale_parameter=False,
#         warmup_init=False,
#     )
    
#     lr_scheduler = AdafactorSchedule(optimizer)

    # Initialize our Trainer
    trainer = LMTrainer(
        model=model,
        model_args = model_args,
        data_args=data_args,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        test_key=dataset.test_key,
        # optimizers=(optimizer, lr_scheduler)
    )


    return trainer, None

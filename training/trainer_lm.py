import math
import torch
import torch.nn as nn
import logging
import os
from typing import Dict, OrderedDict, Union, Any, Optional, List, Tuple

from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach
from transformers.file_utils import is_sagemaker_mp_enabled


if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

from tasks.superglue.task_helpers import TASK_HELPERS


from utils.embedding_encoder import PromptEncoder, EmbeddingEncoder

from utils.data_augmentation import compute_kl_loss

logger = logging.getLogger(__name__)

_default_log_level = logging.INFO
logger.setLevel(_default_log_level)

class LMTrainer(Trainer):
    def __init__(self, *args, model_args, data_args, predict_dataset = None, test_key = "accuracy", **kwargs):
        super().__init__(*args, **kwargs)

        #TODO 这个trainer类是可以拥有data_args, model_args, training_args和config的。
        # training_args就是self.args
        self.model_args = model_args
        self.data_args = data_args
        self.config = self.model.config

        self.task_helper = TASK_HELPERS[self.data_args.dataset_name](model_args=self.model_args, model=self.model, tokenizer=self.tokenizer)

        self.embedding_encoder = EmbeddingEncoder(self.config, self.model_args, self.model)
        if self.place_model_on_device:
            self._move_model_to_device(self.embedding_encoder, self.args.device)

        self.predict_dataset = predict_dataset
        self.test_key = test_key
        self.best_metrics = OrderedDict({
            "best_epoch": 0,
            f"best_eval_{self.test_key}": 0,
        })

        self.label_names = ["label"]

    def log_best_metrics(self):
        self.log_metrics("best", self.best_metrics)
        self.save_metrics("best", self.best_metrics, combined=False)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs)

        eval_metrics = None
        if self.control.should_evaluate:
            if self.model_args.eval_type == "train":
                logger.info(f"***** Running Evaluation for train dataset *****")
                train_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval, eval_dataset=self.train_dataset)
                self._report_to_hp_search(trial, epoch, train_metrics)
            logger.info(f"***** Running Evaluation for eval dataset *****")    
            eval_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, eval_metrics)

            if eval_metrics["eval_"+self.test_key] > self.best_metrics["best_eval_"+self.test_key]:
                self.best_metrics["best_epoch"] = epoch
                self.best_metrics["best_eval_"+self.test_key] = eval_metrics["eval_"+self.test_key]

                if self.predict_dataset is not None:
                    if isinstance(self.predict_dataset, dict):
                        for dataset_name, dataset in self.predict_dataset.items():
                            _, _, test_metrics = self.predict(dataset, metric_key_prefix="test")
                            self.best_metrics[f"best_test_{dataset_name}_{self.test_key}"] = test_metrics["test_"+self.test_key]
                    else:
                        _, _, test_metrics = self.predict(self.predict_dataset, metric_key_prefix="test")
                        self.best_metrics["best_test_"+self.test_key] = test_metrics["test_"+self.test_key]

            logger.info(f"***** Epoch {epoch}: Best results *****")
            for key, value in self.best_metrics.items():
                logger.info(f"{key} = {value}")
            self.log(self.best_metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=eval_metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.

        We will conly keep inputs_embeds, labels and attention_mask in our input to model.
        """

        inputs = self._prepare_input(inputs)
        inputs["inputs_embeds"] = self.embedding_encoder.id2embedding(inputs["input_ids"], inputs["sentences_ids"])
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        input_models = {}
        # 这三样就是训练的时候必须要有的column，predict和evaluate的时候另外说。
        for key in ["inputs_embeds", "attention_mask", "labels"]:
            input_models[key] = inputs[key]

        return inputs, input_models


# TODO: 你需要在_prepare_inputs中把把input中的input_embeding, attention_mask和labels输入进去
# TODO: 然后把input中不需要输入的内容用来在计算logit的时候用
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        _, input_models = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, input_models, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.autocast_smart_context_manager():
            if self.data_augmentation != "rdrop":
                loss, outputs = self.compute_loss(model, input_models, return_outputs=True)
                if self.data_args.dataset_name in ["copa", "record"]:
                    loss = self.task_helper.logits2loss(inputs, outputs)
            else:
                loss1, outputs1 = self.compute_loss(model, input_models, return_outputs=True)
                    if self.data_args.dataset_name in ["copa", "record"]:
                        loss1 = self.task_helper.logits2loss(inputs, outputs1)
                loss2, outputs2 = self.compute_loss(model, input_models, return_outputs=True)
                    if self.data_args.dataset_name in ["copa", "record"]:
                        loss2 = self.task_helper.logits2loss(inputs, outputs2)
                bce_loss = 0.5 * loss1 + 0.5 * loss2
                kl_loss = compute_kl_loss(logits1, logits2)
                loss = bce_loss + kl_loss


        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        #TODO 这里把用来计算acc的东西存下来
        inputs, input_models = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, input_models)
                if has_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels:
                    with self.autocast_smart_context_manager():
                        loss, outputs = self.compute_loss(model, input_models, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]

                else:
                    loss = None
                    with self.autocast_smart_context_manager():
                        outputs = model(**input_models)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]


        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        logits = self.task_helper.logits2pred(inputs, logits)

        return (loss, logits, labels)


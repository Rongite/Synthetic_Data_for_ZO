# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import inspect
import math
import os
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import copy
from metrics import f1
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
import transformers
from transformers import Trainer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from metrics import calculate_metric

# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    get_reporting_integration_callbacks,
    hp_params,
    is_optuna_available,
    is_ray_tune_available,
    is_sigopt_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
    run_hp_search_wandb,
)

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    WEIGHTS_NAME,
    find_labels,
    is_datasets_available,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)
from transformers.utils.generic import ContextManagers
import wandb
from utils import encode_prompt, Prediction

_is_native_cpu_amp_available = True

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_datasets_available():
    import datasets


logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class OurTrainer(Trainer):

    from transformers.trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state

    def get_modules(self, layer, model_type):
        # NOTE: This is llama-specific
        # For other models, replace this with proper names for all linear layers
        if model_type in ['mistral', 'llama']:
            return [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
                layer.mlp.gate_proj,
                layer.mlp.up_proj,
                layer.mlp.down_proj,
            ]
        elif model_type in ['opt']:
            return [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.out_proj,
                layer.fc1,
                layer.fc2,
            ]
        else:
            raise NotImplementedError()

    def get_grad_and_load_to_model(self, model: nn.Module, grad_save_path,
                                   microbatch=1, minibatch=1000):
        count = 0
        self._train_batch_size = microbatch

        _model = model.model
        if 'mistral' in self.args.model_name.lower():
            _layers = _model.layers
            model_type = 'mistral'

        elif 'llama' in self.args.model_name.lower():
            _layers = _model.layers
            model_type = 'llama'

        elif 'opt' in self.args.model_name.lower():
            _layers = _model.decoder.layers
            model_type = 'opt'

        else:
            raise NotImplementedError()

        for n, p in model.named_parameters():
            p.requires_grad_(False)

        for layer in _layers:
            for module in self.get_modules(layer, model_type):
                module.weight.requires_grad_(True)

        minibatch = min(minibatch, len(self.get_train_dataloader()))
        counter = tqdm(range(minibatch))
        continue_count = 0
        with torch.enable_grad():
            for sampled_batch in self.get_train_dataloader():
                sampled_batch = {k: v.cuda() for k, v in sampled_batch.items()}
                if sampled_batch['input_ids'].shape[0] * sampled_batch['input_ids'].shape[1] > 2700:
                    continue_count += 1
                    print(continue_count)
                    continue
                outputs = model(**sampled_batch)
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                else:
                    loss = outputs[0]
                loss.backward()
                torch.cuda.empty_cache()
                count += 1
                counter.update(1)

                if count >= minibatch:
                    break

        _grad_module_dict = {(i, j): torch.zeros_like(module.weight, device='cpu')
                             for i, layer in enumerate(_layers)
                             for j, module in enumerate(self.get_modules(layer, model_type))
                             }
        for i, layer in enumerate(_layers):
            for j, module in enumerate(self.get_modules(layer, model_type)):
                _grad_module_dict[(i, j)] = module.weight.grad.data / count
        model.zero_grad(set_to_none=True)

        torch.save(_grad_module_dict, grad_save_path)

    @torch.no_grad()
    def get_outlier_masks(self, model, percentile=0.005):
        mask_dict = dict()
        for n, p in model.named_parameters():
            if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n:
                continue
            top_cutoff = int(p.numel() * percentile)
            mask = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)
            mask[-p.view(-1).abs().argsort()[:top_cutoff]] = True
            mask_dict[n] = torch.arange(p.numel(), device=p.device)[mask]
        return mask_dict

    @torch.no_grad()
    def get_smaller_weight_masks(self, model, percentile=0.20):
        mask_dict = dict()
        for n, p in model.named_parameters():
            if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n:
                continue
            top_cutoff = int(p.numel() * percentile)
            mask = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)
            mask[(p.view(-1).abs()).argsort()[:top_cutoff]] = True
            mask_dict[n] = torch.arange(p.numel(), device=p.device)[mask].clone()
            del mask
            torch.cuda.empty_cache()
        return mask_dict

    @torch.no_grad()
    def get_random_masks(self, model, percentile=0.005):
        mask_dict = dict()
        for n, p in model.named_parameters():
            if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n:
                continue
            random_indices = torch.randperm(p.numel(), device=p.device)[:int(p.numel() * percentile)]
            mask_dict[n] = random_indices.clone()
            torch.cuda.empty_cache()
        return mask_dict

    @torch.no_grad()
    def get_GraSP_mask(self, model: nn.Module,
                       percentile=5e-3, microbatch=1, minibatch=16):
        assert minibatch % microbatch == 0
        count = 0
        old_batch_size = self._train_batch_size
        self._train_batch_size = microbatch

        weights = []
        for n, p in model.named_parameters():
            if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n:
                p.requires_grad_(False)
            else:
                weights.append(p)

        hessian_grad_prod = {n: torch.zeros_like(p, device='cpu')
                             for n, p in model.named_parameters() if p.requires_grad}

        weight_params = {n: p for n, p in model.named_parameters()}

        with torch.enable_grad():
            for sampled_batch in self.get_train_dataloader():
                sampled_batch = {k: v.cuda() for k, v in sampled_batch.items()}
                count += 1
                outputs = model(**sampled_batch)
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                else:
                    loss = outputs[0]
                loss /= (minibatch / microbatch)
                grads = torch.autograd.grad(loss, weights, create_graph=True, retain_graph=True)

                inner_prod = torch.tensor(0.0, requires_grad=True, device=grads[0].device, dtype=torch.bfloat16)
                for g in grads:
                    inner_prod = inner_prod + torch.inner(g.view(-1), g.data.view(-1))
                del grads

                inner_prod.backward()
                for n, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    hessian_grad_prod[n] = (hessian_grad_prod[n].cuda() + p.grad).cpu()
                    p.grad = None

                torch.cuda.empty_cache()

                if count >= minibatch:
                    break

        mask_dict = dict()

        with torch.no_grad():
            for n, g in hessian_grad_prod.items():
                per_layer_scores = -g.cuda() * weight_params[n]
                top_cutoff = int(g.numel() * percentile)
                mask = torch.zeros(g.numel(), dtype=torch.bool)
                mask[per_layer_scores.view(-1).argsort()[:top_cutoff]] = True
                mask_dict[n] = torch.arange(g.numel())[mask].to(device='cuda')

        self._train_batch_size = old_batch_size
        return mask_dict

    @torch.no_grad()
    def get_gradient_masks(self, model: nn.Module,
                           percentile=5e-3, microbatch=1, minibatch=16):
        assert minibatch % microbatch == 0
        count = 0
        old_batch_size = self._train_batch_size
        self._train_batch_size = microbatch
        with torch.enable_grad():
            for sampled_batch in self.get_train_dataloader():
                sampled_batch = {k: v.cuda() for k, v in sampled_batch.items()}
                count += 1
                outputs = model(**sampled_batch)
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                else:
                    loss = outputs[0]
                loss /= (minibatch / microbatch)
                loss.backward()
                if count >= minibatch:
                    break

        mask_dict = dict()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n:
                    continue
                per_layer_scores = p.grad ** 2
                top_cutoff = int(p.numel() * percentile)
                mask = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)
                mask[(-per_layer_scores.view(-1)).argsort()[:top_cutoff]] = True
                mask_dict[n] = torch.arange(p.numel(), device=p.device)[mask]
                p.grad = None

        self._train_batch_size = old_batch_size
        return mask_dict

    @torch.no_grad()
    def get_C4_gradient_masks(self, model, C4_grad_addr, percentile):
        mask_dict = dict()
        grad = transformers.AutoModelForCausalLM.from_pretrained(
            C4_grad_addr,
        )
        grad_dict = {n: p for n, p in grad.named_parameters()}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n:
                    continue
                per_layer_scores = grad_dict[n].cuda().abs()
                top_cutoff = int(p.numel() * percentile)
                mask = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)
                mask[(-per_layer_scores.view(-1)).argsort()[:top_cutoff]] = True
                mask_dict[n] = torch.arange(p.numel(), device=p.device)[mask]

        return mask_dict

    @torch.no_grad()
    def add_to_weight(self, model, vec, scale):
        counter = 0
        for p in model.parameters():
            if p.requires_grad:
                p.data.add_(vec[counter: counter + p.numel()].view(p.shape), alpha=scale)
                counter += p.numel()

    def get_grad(self, model, inputs, random_seed):
        self.zo_perturb_parameters(scaling_factor=1, random_seed=random_seed)
        loss1 = self.zo_forward(model, inputs)
        self.zo_perturb_parameters(scaling_factor=-2, random_seed=random_seed)
        loss2 = self.zo_forward(model, inputs)
        self.zo_perturb_parameters(scaling_factor=1, random_seed=random_seed)
        global_projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        torch.manual_seed(random_seed)
        ret_grad_dict = dict()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            ret_grad_dict[name] = global_projected_grad * z
        return ret_grad_dict, loss1

    def get_grad_rademacher(self, model, inputs, random_seed):
        self.zo_perturb_parameters_rademacher(scaling_factor=1, random_seed=random_seed)
        loss1 = self.zo_forward(model, inputs)
        self.zo_perturb_parameters_rademacher(scaling_factor=-2, random_seed=random_seed)
        loss2 = self.zo_forward(model, inputs)
        self.zo_perturb_parameters_rademacher(scaling_factor=1, random_seed=random_seed)
        global_projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        torch.manual_seed(random_seed)
        ret_grad_dict = dict()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            z = torch.randint(low=0, high=2, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            z[z == 0] = -1
            ret_grad_dict[name] = global_projected_grad * z
        return ret_grad_dict, loss1

    @torch.inference_mode()
    def set_grad_with_mask(self, model, inputs, mask_dict, random_seed):
        self.zo_perturb_parameters_with_mask(scaling_factor=1, random_seed=random_seed, mask_dict=mask_dict)
        loss1 = self.zo_forward(model, inputs)
        self.zo_perturb_parameters_with_mask(scaling_factor=-2, random_seed=random_seed, mask_dict=mask_dict)
        loss2 = self.zo_forward(model, inputs)
        self.zo_perturb_parameters_with_mask(scaling_factor=1, random_seed=random_seed, mask_dict=mask_dict)
        global_projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        torch.manual_seed(random_seed)
        for name, selected_param in self.named_parameters_to_optim.items():
            z = torch.normal(mean=0, std=1, size=selected_param.size(), device=selected_param.device, dtype=selected_param.dtype)
            selected_param.grad = global_projected_grad * z
        return loss1

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs using standard first-order backpropagation.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        return loss.detach()

    @torch.inference_mode()
    def zo_forward(self, model, inputs):
        model.eval()
        if self.args.non_diff:
            return self.zo_forward_nondiff(model, inputs)
        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()
        return loss.detach()

    def zo_forward_nondiff(self, model, inputs):
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."
        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = self.model.generate(
                inputs["input_ids"],
                do_sample=args.sampling,
                temperature=args.temperature,
                num_beams=args.num_beams,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)),
                num_return_sequences=1,
                eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[0], self.tokenizer.eos_token_id],
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]
        return -torch.tensor(np.mean(f1s), dtype=torch.float32)

    @torch.inference_mode()
    def set_squeezellm_sparse_grad(self, model, inputs):
        self.zo_random_seed = np.random.randint(1000000000)
        self.squeezellm_zo_perturb_parameters(self.zo_random_seed, scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)
        self.squeezellm_zo_perturb_parameters(self.zo_random_seed, scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        self.squeezellm_zo_perturb_parameters(self.zo_random_seed, scaling_factor=1)
        torch.manual_seed(self.zo_random_seed)
        for name, buffer in self.dequantized_weight_list:
            sensitive_vals = self.sqllm_sparse_weight_dict[name]
            z = torch.normal(mean=0, std=1, size=sensitive_vals.size(), device=sensitive_vals.device, dtype=sensitive_vals.dtype)
            sensitive_vals.grad = self.projected_grad * z
        return (loss1 + loss2) / 2

    @torch.inference_mode()
    def perturb_and_update_squeezellm_sparse_grad_memory_limited(self, model, inputs):
        self.zo_random_seed = np.random.randint(1000000000)
        self.squeezellm_zo_perturb_parameters_memory_limited(self.zo_random_seed, scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)
        self.squeezellm_zo_perturb_parameters_memory_limited(self.zo_random_seed, scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        self.squeezellm_zo_perturb_parameters_memory_limited(self.zo_random_seed, scaling_factor=1)
        torch.manual_seed(self.zo_random_seed)
        for name, buffer in self.sqllm_sparse_weight_dict.items():
            z = torch.randn_like(buffer, device=buffer.device, dtype=buffer.dtype)
            buffer.add_(z, alpha=-(self.args.learning_rate * self.projected_grad))
        return (loss1 + loss2) / 2

    @torch.inference_mode()
    def zo_step(self, model, inputs):
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
        self.zo_random_seed = np.random.randint(1000000000)
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)
        self.zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        self.zo_perturb_parameters(scaling_factor=1)
        return (loss1 + loss2) / 2

    def set_zo_grad_as_grad(self):
        torch.manual_seed(self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            param.grad = self.projected_grad * z

    @torch.no_grad()
    def full_zo_update(self):
        torch.manual_seed(self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            param.data.add_(z, alpha=-(self.args.learning_rate * self.projected_grad))

    ############## Misc overload functions ##############

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir
        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        elif self.deepspeed:
            if self.args.should_save:
                self._save(output_dir)
            if is_deepspeed_zero3_enabled():
                file = os.path.join(output_dir, WEIGHTS_NAME)
                if os.path.isfile(file):
                    os.remove(file)
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)
        elif self.args.should_save:
            self._save(output_dir)
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")

    def evaluate_test_set(self):
        predictions = []
        train_samples = self.train_samples
        for eval_id, eval_sample in enumerate(tqdm(self.test_samples)):
            predictions.append(
                self.one_step_pred(train_samples, eval_sample)
            )
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        print(metrics, flush=True)
        sys.stdout.flush()
        return metrics

    def one_step_pred(self, train_samples, eval_sample):
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer,
            max_length=self.args.max_length,
            generation=self.task.generation,
            max_new_tokens=self.args.max_new_tokens
        )
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(
                self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer,
                max_length=self.args.max_length,
                sfc=self.args.sfc,
                icl_sfc=self.args.icl_sfc,
                generation=self.task.generation,
                max_new_tokens=self.args.max_new_tokens
            )
        outputs = []
        if self.task.generation:
            output_text = self.test_forward(encoded_candidates[0], generation=True)
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.test_forward(encoded_candidate, option_len=option_lens[candidate_id])
                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.test_forward(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id])
                outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})
            if self.args.sfc or self.args.icl_sfc:
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                scores = [x['log_probs'].mean().item() for x in outputs]
            if isinstance(eval_sample.correct_candidate, list):
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)
            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        重载内部训练循环，支持两种模式：
        - 当 self.args.trainer == "zo" 时，使用零阶优化（MeZO）的更新方式；
        - 否则（例如当 trainer=="first"）直接使用标准第一阶反向传播进行更新。
        其它数据预处理、评估、日志输出和checkpoint保存均保持一致，均只保存dev_acc最高时的模型checkpoint。
        """
        self._train_batch_size = batch_size
        train_dataloader = self.get_train_dataloader()

        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                num_train_samples = args.max_steps * (args.train_batch_size * args.gradient_accumulation_steps * args.world_size)
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * (args.train_batch_size * args.gradient_accumulation_steps * args.world_size)
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if self.args.linear_probing:
            def _get_token_prediction_layer(model):
                if model.config.model_type == "opt":
                    return model.lm_head
                else:
                    raise NotImplementedError(model.config.model_type)

            def _extract_features(model, *args, **kwargs):
                features = {}
                def __hook(model_, input_, output_):
                    features["features"] = input_[0].detach()
                _get_token_prediction_layer(model).register_forward_hook(__hook)
                model.forward(*args, **kwargs)
                return features["features"]

            logger.info("Linear probing")
            logger.info("Starting to get features for training dataset")
            targets = []
            features = []
            with torch.inference_mode():
                for step, inputs in enumerate(tqdm(train_dataloader)):
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.model.device)
                    feature = _extract_features(self.model, **inputs)
                    target = inputs["labels"]
                    assert not self.args.train_as_classification and self.args.only_train_option
                    feature, target = feature[:, :-1], target[:, 1:]
                    for _i, _len in enumerate(inputs["option_len"]):
                        features.append(feature[_i, -_len:])
                        targets.append(target[_i, -_len:])
            logger.info("Finished getting features for training dataset")
            features = torch.cat(features, dim=0).cpu().numpy()
            targets = torch.cat(targets, dim=0).cpu().numpy()
            if self.model.config.model_type in ["opt", "gpt2"]:
                use_bias = False
            else:
                raise NotImplementedError
            tol = 0.01 if self.args.lp_early_stopping else 1e-4
            max_iter = 1000 if self.args.lp_early_stopping else 5000
            logger.info("Fitting logistic regression...")
            reg = LogisticRegressionCV(max_iter=max_iter, fit_intercept=use_bias, multi_class="multinomial",
                                       random_state=0, tol=tol, n_jobs=-1).fit(features, targets)
            logger.info("Done")
            logger.info("Assigning weights to model")
            decoder = _get_token_prediction_layer(self.model)
            coef_torch = torch.tensor(reg.coef_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if use_bias:
                bias_torch = torch.tensor(reg.intercept_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if coef_torch.shape[0] == 1:
                assert len(reg.classes_) == 2
                coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)
                if use_bias:
                    bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)
            for _i, token_id in enumerate(reg.classes_):
                decoder.weight.data[token_id] = coef_torch[_i]
                if use_bias:
                    decoder.bias.data[token_id] = bias_torch[_i]
            return None

        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if self.args.save_grad:
            self.get_grad_and_load_to_model(self.model, self.args.grad_save_path + os.path.sep + f'start.pt')
            raise RuntimeError('finished')

        delay_optimizer_creation = False
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        has_mask = False
        if self.args.smaller_weight_mask:
            print(f'smaller weight mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_smaller_weight_masks(model, self.args.outlier_percentage)
            has_mask = True
        elif self.args.outlier:
            print(f'outlier mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_outlier_masks(model, self.args.outlier_percentage)
            has_mask = True
        elif self.args.random_subset_weights:
            print(f'random mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_random_masks(model, self.args.outlier_percentage)
            has_mask = True
        elif self.args.grad_mask:
            print(f'grad mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_gradient_masks(model, self.args.outlier_percentage)
            has_mask = True
        elif self.args.C4_grad_mask:
            print(f'C4 grad mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_C4_gradient_masks(model, self.args.C4_grad_addr, self.args.outlier_percentage)
            has_mask = True
        elif self.args.GraSP_mask:
            print(f'GraSP mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_GraSP_mask(model, self.args.outlier_percentage)
            has_mask = True

        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        self.named_parameters_to_optim = [(name, param) for name, param in model.named_parameters() if param.requires_grad]

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} batches in the first epoch."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        tr_loss = torch.tensor(0.0).to(args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad(set_to_none=True)

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        if args.report_to != []:
            wandb.run.name = self.args.tag

        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                _ = list(train_dataloader.sampler)

        from eventProfiler import EventProfiler
        self.profiler = EventProfiler(torch.device('cuda'), active=False)
        self.record_loss = []

        if self.args.use_squeezellm:
            self.named_buffers = {name: buffer for name, buffer in model.named_buffers()}
            self.sqllm_sparse_weight_dict = dict()
            if self.args.memory_limit_scenario:
                self.sqllm_sparse_weight_dict = {name: buffer.requires_grad_(True) for name, buffer in model.named_buffers() if 'sensitive_vals' in name}
            else:
                self.dequantized_weight_list = [(name, buffer) for name, buffer in model.named_buffers() if 'dequantized_weight' in name]
                for name, buffer in self.dequantized_weight_list:
                    indices = self.named_buffers[name.replace('dequantized_weight', 'sensitive_indices')]
                    self.sqllm_sparse_weight_dict[name] = buffer[indices[0], indices[1]].clone().detach().requires_grad_(True)

        if has_mask:
            self.named_parameters_to_optim = {
                n: p.view(-1)[self.outlier_masks[n]].detach().clone().requires_grad_(True)
                for n, p in model.named_parameters() if n in self.outlier_masks
            }

        if args.use_adam:
            if self.args.use_squeezellm:
                self.optimizer = torch.optim.Adam(list(self.sqllm_sparse_weight_dict.values()), lr=self.args.learning_rate)
            elif has_mask:
                self.optimizer = torch.optim.Adam(list(self.named_parameters_to_optim.values()), lr=self.args.learning_rate)
            else:
                self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)
        elif args.use_momentum:
            if self.args.use_squeezellm:
                self.optimizer = torch.optim.SGD(list(self.sqllm_sparse_weight_dict.values()), lr=self.args.learning_rate, momentum=0.9)
            elif has_mask:
                self.optimizer = torch.optim.SGD(list(self.named_parameters_to_optim.values()), lr=self.args.learning_rate, momentum=0.9)
            else:
                self.optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate, momentum=0.9)
        else:
            if self.args.use_squeezellm:
                self.optimizer = torch.optim.SGD(list(self.sqllm_sparse_weight_dict.values()), lr=self.args.learning_rate)
            elif has_mask:
                self.optimizer = torch.optim.SGD(list(self.named_parameters_to_optim.values()), lr=self.args.learning_rate)
            else:
                self.optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate)

        print(self.optimizer)
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = len(epoch_iterator) if len_dataloader is not None else args.max_steps * args.gradient_accumulation_steps
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            self.deepspeed = False

            for step, inputs in enumerate(epoch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if args.trainer == 'zo' and args.use_squeezellm:
                    if self.args.memory_limit_scenario:
                        tr_loss_step = self.perturb_and_update_squeezellm_sparse_grad_memory_limited(model, inputs)
                    else:
                        tr_loss_step = self.set_squeezellm_sparse_grad(model, inputs)
                elif args.trainer == 'zo' and has_mask:
                    zo_random_seed = np.random.randint(1000000000)
                    tr_loss_step = self.set_grad_with_mask(model, inputs, self.outlier_masks, zo_random_seed)
                    if self.args.record_time:
                        self.record_loss.append(tr_loss_step)
                elif args.trainer == 'zo':
                    tr_loss_step = self.zo_step(model, inputs)
                    if self.args.record_time:
                        self.record_loss.append(tr_loss_step)
                    if self.args.use_full_zo_update:
                        self.full_zo_update()
                    else:
                        self.set_zo_grad_as_grad()
                else:
                    # First-order fine-tuning: directly use standard backpropagation via training_step.
                    tr_loss_step = self.training_step(model, inputs)
                    with torch.no_grad():
                        if has_mask:
                            grad_dict = {n: p.grad for n, p in self.model.named_parameters() if p.requires_grad}
                            for name, selected_param in self.named_parameters_to_optim.items():
                                selected_param.grad = grad_dict[name].view(-1)[self.outlier_masks[name]].clone()
                            for n, p in self.model.named_parameters():
                                p.grad = None

                if (args.logging_nan_inf_filter and not is_torch_tpu_available() and
                    (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))):
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                ):
                    if args.trainer == 'zo' and (not args.memory_limit_scenario):
                        if not args.use_full_zo_update:
                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                        if self.args.use_squeezellm:
                            for name, buffer in self.dequantized_weight_list:
                                sensitive_vals = self.sqllm_sparse_weight_dict[name]
                                indices = self.named_buffers[name.replace('dequantized_weight', 'sensitive_indices')]
                                buffer[indices[0], indices[1]] = sensitive_vals
                    elif args.trainer != 'zo':
                        with self.profiler('optimizer'):
                            self.optimizer.step()
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    if has_mask:
                        with torch.no_grad():
                            for name, param in model.named_parameters():
                                if name not in self.named_parameters_to_optim:
                                    continue
                                param.view(-1)[self.outlier_masks[name]] = self.named_parameters_to_optim[name]
                        if self.args.dynamic_mask and (self.state.global_step + 1) % self.args.dynamic_mask_step == 0:
                            assert not self.args.use_momentum and not self.args.use_adam
                            if self.args.smaller_weight_mask:
                                del self.outlier_masks
                                self.outlier_masks = self.get_smaller_weight_masks(model, self.args.outlier_percentage)
                            elif self.args.outlier:
                                del self.outlier_masks
                                self.outlier_masks = self.get_outlier_masks(model, self.args.outlier_percentage)
                            elif self.args.random_subset_weights:
                                del self.outlier_masks
                                self.outlier_masks = self.get_random_masks(model, self.args.outlier_percentage)
                            elif self.args.grad_mask:
                                del self.outlier_masks
                                self.outlier_masks = self.get_gradient_masks(model, self.args.outlier_percentage)
                            elif self.args.GraSP_mask:
                                del self.outlier_masks
                                self.outlier_masks = self.get_GraSP_mask(model, self.args.outlier_percentage)
                            self.named_parameters_to_optim = {
                                n: p.view(-1)[self.outlier_masks[n]].detach().clone().requires_grad_(True)
                                for n, p in model.named_parameters() if n in self.outlier_masks
                            }
                            self.optimizer = torch.optim.SGD(list(self.named_parameters_to_optim.values()), lr=self.args.learning_rate)
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    if self.args.save_strategy == 'steps' and (self.state.global_step > 0 and self.state.global_step % self.args.save_steps == 0):
                        self.control.should_save = True
                        self.control.should_evaluate = True
                    torch.cuda.empty_cache()
                    self._maybe_log_save_evaluate(tr_loss, None, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}!"
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)

            if self.args.save_model_addr and self.control.should_save:
                if not os.path.exists(self.args.save_model_addr):
                    os.makedirs(self.args.save_model_addr)
                model_params = {n: p for n, p in model.named_parameters()}
                torch.save(model_params, self.args.save_model_addr + os.sep + f'epoch-{epoch}.pt')
                del model_params
                torch.cuda.empty_cache()

            self._maybe_log_save_evaluate(tr_loss, None, model, trial, epoch, ignore_keys_for_eval)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        self._load_best_model()
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        return TrainOutput(self.state.global_step, train_loss, metrics)

    ############## MeZO ##############
    @torch.no_grad()
    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            param.add_(z, alpha=(scaling_factor * self.args.zo_eps))

    @torch.no_grad()
    def squeezellm_zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        for name, buffer in self.dequantized_weight_list:
            indices = self.named_buffers[name.replace('dequantized_weight', 'sensitive_indices')]
            z = torch.normal(mean=0, std=1, size=(indices.shape[1],), device=buffer.device, dtype=buffer.dtype)
            buffer[indices[0], indices[1]] += (scaling_factor * self.args.zo_eps) * z

    @torch.no_grad()
    def squeezellm_zo_perturb_parameters_memory_limited(self, random_seed=None, scaling_factor=1):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        for name, buffer in self.sqllm_sparse_weight_dict.items():
            z = torch.randn_like(buffer, device=buffer.device, dtype=buffer.dtype)
            buffer.add_(z, alpha=(scaling_factor * self.args.zo_eps))

    def zo_perturb_parameters_rademacher(self, random_seed=None, scaling_factor=1):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.randint(low=0, high=2, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            z[z == 0] = -1
            param.data.add_(z, alpha=scaling_factor * self.args.zo_eps)

    @torch.inference_mode()
    def zo_perturb_parameters_with_mask(self, random_seed=None, mask_dict=None, scaling_factor=1):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        for name, param in self.model.named_parameters():
            if not name in self.named_parameters_to_optim:
                continue
            selected_param = self.named_parameters_to_optim[name]
            z = torch.normal(mean=0, std=1, size=selected_param.size(), device=selected_param.device, dtype=selected_param.dtype)
            param.view(-1)[mask_dict[name]] += (scaling_factor * self.args.zo_eps) * z

    def norm_zo_perturb_parameters(self, random_seed=None, scaling_factor=1, normalize_dict=None):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.args.zo_eps * normalize_dict[name]

    def zo_forward(self, model, inputs):
        model.eval()
        if self.args.non_diff:
            return self.zo_forward_nondiff(model, inputs)
        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()
        return loss.detach()

    def zo_forward_nondiff(self, model, inputs):
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."
        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = self.model.generate(
                inputs["input_ids"],
                do_sample=args.sampling,
                temperature=args.temperature,
                num_beams=args.num_beams,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)),
                num_return_sequences=1,
                eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[0], self.tokenizer.eos_token_id],
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]
        return -torch.tensor(np.mean(f1s), dtype=torch.float32)

    @torch.inference_mode()
    def set_squeezellm_sparse_grad(self, model, inputs):
        self.zo_random_seed = np.random.randint(1000000000)
        self.squeezellm_zo_perturb_parameters(self.zo_random_seed, scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)
        self.squeezellm_zo_perturb_parameters(self.zo_random_seed, scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        self.squeezellm_zo_perturb_parameters(self.zo_random_seed, scaling_factor=1)
        torch.manual_seed(self.zo_random_seed)
        for name, buffer in self.dequantized_weight_list:
            sensitive_vals = self.sqllm_sparse_weight_dict[name]
            z = torch.normal(mean=0, std=1, size=sensitive_vals.size(), device=sensitive_vals.device, dtype=sensitive_vals.dtype)
            sensitive_vals.grad = self.projected_grad * z
        return (loss1 + loss2) / 2

    @torch.inference_mode()
    def perturb_and_update_squeezellm_sparse_grad_memory_limited(self, model, inputs):
        self.zo_random_seed = np.random.randint(1000000000)
        self.squeezellm_zo_perturb_parameters_memory_limited(self.zo_random_seed, scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)
        self.squeezellm_zo_perturb_parameters_memory_limited(self.zo_random_seed, scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        self.squeezellm_zo_perturb_parameters_memory_limited(self.zo_random_seed, scaling_factor=1)
        torch.manual_seed(self.zo_random_seed)
        for name, buffer in self.sqllm_sparse_weight_dict.items():
            z = torch.randn_like(buffer, device=buffer.device, dtype=buffer.dtype)
            buffer.add_(z, alpha=-(self.args.learning_rate * self.projected_grad))
        return (loss1 + loss2) / 2

    @torch.inference_mode()
    def zo_step(self, model, inputs):
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
        self.zo_random_seed = np.random.randint(1000000000)
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)
        self.zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        self.zo_perturb_parameters(scaling_factor=1)
        return (loss1 + loss2) / 2

    def set_zo_grad_as_grad(self):
        torch.manual_seed(self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            param.grad = self.projected_grad * z

    @torch.no_grad()
    def full_zo_update(self):
        torch.manual_seed(self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            param.data.add_(z, alpha=-(self.args.learning_rate * self.projected_grad))

    ############## Misc overload functions ##############

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir
        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        elif self.deepspeed:
            if self.args.should_save:
                self._save(output_dir)
            if is_deepspeed_zero3_enabled():
                file = os.path.join(output_dir, WEIGHTS_NAME)
                if os.path.isfile(file):
                    os.remove(file)
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)
        elif self.args.should_save:
            self._save(output_dir)
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")

    def evaluate_test_set(self):
        predictions = []
        train_samples = self.train_samples
        for eval_id, eval_sample in enumerate(tqdm(self.test_samples)):
            predictions.append(
                self.one_step_pred(train_samples, eval_sample)
            )
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        print(metrics, flush=True)
        sys.stdout.flush()
        return metrics

    def one_step_pred(self, train_samples, eval_sample):
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer,
            max_length=self.args.max_length,
            generation=self.task.generation,
            max_new_tokens=self.args.max_new_tokens
        )
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(
                self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer,
                max_length=self.args.max_length,
                sfc=self.args.sfc,
                icl_sfc=self.args.icl_sfc,
                generation=self.task.generation,
                max_new_tokens=self.args.max_new_tokens
            )
        outputs = []
        if self.task.generation:
            output_text = self.test_forward(encoded_candidates[0], generation=True)
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.test_forward(encoded_candidate, option_len=option_lens[candidate_id])
                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.test_forward(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id])
                outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})
            if self.args.sfc or self.args.icl_sfc:
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                scores = [x['log_probs'].mean().item() for x in outputs]
            if isinstance(eval_sample.correct_candidate, list):
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)
            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        重载内部训练循环，支持两种模式：
          - 当 self.args.trainer == "zo" 时，使用零阶优化(MeZO)的更新方式；
          - 否则（例如当 trainer=="first"）直接使用标准一阶反向传播进行更新。
        其它数据预处理、评估、日志输出和checkpoint保存均保持一致。
        """
        self._train_batch_size = batch_size
        train_dataloader = self.get_train_dataloader()

        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                num_train_samples = args.max_steps * (args.train_batch_size * args.gradient_accumulation_steps * args.world_size)
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * (args.train_batch_size * args.gradient_accumulation_steps * args.world_size)
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if self.args.linear_probing:
            def _get_token_prediction_layer(model):
                if model.config.model_type == "opt":
                    return model.lm_head
                else:
                    raise NotImplementedError(model.config.model_type)

            def _extract_features(model, *args, **kwargs):
                features = {}
                def __hook(model_, input_, output_):
                    features["features"] = input_[0].detach()
                _get_token_prediction_layer(model).register_forward_hook(__hook)
                model.forward(*args, **kwargs)
                return features["features"]

            logger.info("Linear probing")
            logger.info("Starting to get features for training dataset")
            targets = []
            features = []
            with torch.inference_mode():
                for step, inputs in enumerate(tqdm(train_dataloader)):
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.model.device)
                    feature = _extract_features(self.model, **inputs)
                    target = inputs["labels"]
                    assert not self.args.train_as_classification and self.args.only_train_option
                    feature, target = feature[:, :-1], target[:, 1:]
                    for _i, _len in enumerate(inputs["option_len"]):
                        features.append(feature[_i, -_len:])
                        targets.append(target[_i, -_len:])
            logger.info("Finished getting features for training dataset")
            features = torch.cat(features, dim=0).cpu().numpy()
            targets = torch.cat(targets, dim=0).cpu().numpy()
            if self.model.config.model_type in ["opt", "gpt2"]:
                use_bias = False
            else:
                raise NotImplementedError
            tol = 0.01 if self.args.lp_early_stopping else 1e-4
            max_iter = 1000 if self.args.lp_early_stopping else 5000
            logger.info("Fitting logistic regression...")
            reg = LogisticRegressionCV(max_iter=max_iter, fit_intercept=use_bias, multi_class="multinomial",
                                       random_state=0, tol=tol, n_jobs=-1).fit(features, targets)
            logger.info("Done")
            logger.info("Assigning weights to model")
            decoder = _get_token_prediction_layer(self.model)
            coef_torch = torch.tensor(reg.coef_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if use_bias:
                bias_torch = torch.tensor(reg.intercept_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if coef_torch.shape[0] == 1:
                assert len(reg.classes_) == 2
                coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)
                if use_bias:
                    bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)
            for _i, token_id in enumerate(reg.classes_):
                decoder.weight.data[token_id] = coef_torch[_i]
                if use_bias:
                    decoder.bias.data[token_id] = bias_torch[_i]
            return None

        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if self.args.save_grad:
            self.get_grad_and_load_to_model(self.model, self.args.grad_save_path + os.path.sep + f'start.pt')
            raise RuntimeError('finished')

        delay_optimizer_creation = False
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        has_mask = False
        if self.args.smaller_weight_mask:
            print(f'smaller weight mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_smaller_weight_masks(model, self.args.outlier_percentage)
            has_mask = True
        elif self.args.outlier:
            print(f'outlier mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_outlier_masks(model, self.args.outlier_percentage)
            has_mask = True
        elif self.args.random_subset_weights:
            print(f'random mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_random_masks(model, self.args.outlier_percentage)
            has_mask = True
        elif self.args.grad_mask:
            print(f'grad mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_gradient_masks(model, self.args.outlier_percentage)
            has_mask = True
        elif self.args.C4_grad_mask:
            print(f'C4 grad mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_C4_gradient_masks(model, self.args.C4_grad_addr, self.args.outlier_percentage)
            has_mask = True
        elif self.args.GraSP_mask:
            print(f'GraSP mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_GraSP_mask(model, self.args.outlier_percentage)
            has_mask = True

        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        self.named_parameters_to_optim = [(name, param) for name, param in model.named_parameters() if param.requires_grad]

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} batches in the first epoch."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        tr_loss = torch.tensor(0.0).to(args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad(set_to_none=True)

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        if args.report_to != []:
            wandb.run.name = self.args.tag

        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                _ = list(train_dataloader.sampler)

        from eventProfiler import EventProfiler
        self.profiler = EventProfiler(torch.device('cuda'), active=False)
        self.record_loss = []

        if self.args.use_squeezellm:
            self.named_buffers = {name: buffer for name, buffer in model.named_buffers()}
            self.sqllm_sparse_weight_dict = dict()
            if self.args.memory_limit_scenario:
                self.sqllm_sparse_weight_dict = {name: buffer.requires_grad_(True) for name, buffer in model.named_buffers() if 'sensitive_vals' in name}
            else:
                self.dequantized_weight_list = [(name, buffer) for name, buffer in model.named_buffers() if 'dequantized_weight' in name]
                for name, buffer in self.dequantized_weight_list:
                    indices = self.named_buffers[name.replace('dequantized_weight', 'sensitive_indices')]
                    self.sqllm_sparse_weight_dict[name] = buffer[indices[0], indices[1]].clone().detach().requires_grad_(True)

        if has_mask:
            self.named_parameters_to_optim = {
                n: p.view(-1)[self.outlier_masks[n]].detach().clone().requires_grad_(True)
                for n, p in model.named_parameters() if n in self.outlier_masks
            }

        if args.use_adam:
            if self.args.use_squeezellm:
                self.optimizer = torch.optim.Adam(list(self.sqllm_sparse_weight_dict.values()), lr=self.args.learning_rate)
            elif has_mask:
                self.optimizer = torch.optim.Adam(list(self.named_parameters_to_optim.values()), lr=self.args.learning_rate)
            else:
                self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)
        elif args.use_momentum:
            if self.args.use_squeezellm:
                self.optimizer = torch.optim.SGD(list(self.sqllm_sparse_weight_dict.values()), lr=self.args.learning_rate, momentum=0.9)
            elif has_mask:
                self.optimizer = torch.optim.SGD(list(self.named_parameters_to_optim.values()), lr=self.args.learning_rate, momentum=0.9)
            else:
                self.optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate, momentum=0.9)
        else:
            if self.args.use_squeezellm:
                self.optimizer = torch.optim.SGD(list(self.sqllm_sparse_weight_dict.values()), lr=self.args.learning_rate)
            elif has_mask:
                self.optimizer = torch.optim.SGD(list(self.named_parameters_to_optim.values()), lr=self.args.learning_rate)
            else:
                self.optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate)

        print(self.optimizer)
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = len(epoch_iterator) if len_dataloader is not None else args.max_steps * args.gradient_accumulation_steps
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            self.deepspeed = False

            for step, inputs in enumerate(epoch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if args.trainer == 'zo' and args.use_squeezellm:
                    if self.args.memory_limit_scenario:
                        tr_loss_step = self.perturb_and_update_squeezellm_sparse_grad_memory_limited(model, inputs)
                    else:
                        tr_loss_step = self.set_squeezellm_sparse_grad(model, inputs)
                elif args.trainer == 'zo' and has_mask:
                    zo_random_seed = np.random.randint(1000000000)
                    tr_loss_step = self.set_grad_with_mask(model, inputs, self.outlier_masks, zo_random_seed)
                    if self.args.record_time:
                        self.record_loss.append(tr_loss_step)
                elif args.trainer == 'zo':
                    tr_loss_step = self.zo_step(model, inputs)
                    if self.args.record_time:
                        self.record_loss.append(tr_loss_step)
                    if self.args.use_full_zo_update:
                        self.full_zo_update()
                    else:
                        self.set_zo_grad_as_grad()
                else:
                    # First-order fine-tuning分支：直接使用标准反向传播
                    tr_loss_step = self.training_step(model, inputs)
                    with torch.no_grad():
                        if has_mask:
                            grad_dict = {n: p.grad for n, p in self.model.named_parameters() if p.requires_grad}
                            for name, selected_param in self.named_parameters_to_optim.items():
                                selected_param.grad = grad_dict[name].view(-1)[self.outlier_masks[name]].clone()
                            for n, p in self.model.named_parameters():
                                p.grad = None

                if (args.logging_nan_inf_filter and not is_torch_tpu_available() and
                    (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))):
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                ):
                    if args.trainer == 'zo' and (not args.memory_limit_scenario):
                        if not args.use_full_zo_update:
                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                        if self.args.use_squeezellm:
                            for name, buffer in self.dequantized_weight_list:
                                sensitive_vals = self.sqllm_sparse_weight_dict[name]
                                indices = self.named_buffers[name.replace('dequantized_weight', 'sensitive_indices')]
                                buffer[indices[0], indices[1]] = sensitive_vals
                    elif args.trainer != 'zo':
                        with self.profiler('optimizer'):
                            self.optimizer.step()
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    if has_mask:
                        with torch.no_grad():
                            for name, param in model.named_parameters():
                                if name not in self.named_parameters_to_optim:
                                    continue
                                param.view(-1)[self.outlier_masks[name]] = self.named_parameters_to_optim[name]
                        if self.args.dynamic_mask and (self.state.global_step + 1) % self.args.dynamic_mask_step == 0:
                            assert not self.args.use_momentum and not self.args.use_adam
                            if self.args.smaller_weight_mask:
                                del self.outlier_masks
                                self.outlier_masks = self.get_smaller_weight_masks(model, self.args.outlier_percentage)
                            elif self.args.outlier:
                                del self.outlier_masks
                                self.outlier_masks = self.get_outlier_masks(model, self.args.outlier_percentage)
                            elif self.args.random_subset_weights:
                                del self.outlier_masks
                                self.outlier_masks = self.get_random_masks(model, self.args.outlier_percentage)
                            elif self.args.grad_mask:
                                del self.outlier_masks
                                self.outlier_masks = self.get_gradient_masks(model, self.args.outlier_percentage)
                            elif self.args.GraSP_mask:
                                del self.outlier_masks
                                self.outlier_masks = self.get_GraSP_mask(model, self.args.outlier_percentage)
                            self.named_parameters_to_optim = {
                                n: p.view(-1)[self.outlier_masks[n]].detach().clone().requires_grad_(True)
                                for n, p in model.named_parameters() if n in self.outlier_masks
                            }
                            self.optimizer = torch.optim.SGD(list(self.named_parameters_to_optim.values()), lr=self.args.learning_rate)
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    if self.args.save_strategy == 'steps' and (self.state.global_step > 0 and self.state.global_step % self.args.save_steps == 0):
                        self.control.should_save = True
                        self.control.should_evaluate = True
                    torch.cuda.empty_cache()
                    self._maybe_log_save_evaluate(tr_loss, None, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}!"
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)

            if self.args.save_model_addr and self.control.should_save:
                if not os.path.exists(self.args.save_model_addr):
                    os.makedirs(self.args.save_model_addr)
                model_params = {n: p for n, p in model.named_parameters()}
                torch.save(model_params, self.args.save_model_addr + os.sep + f'epoch-{epoch}.pt')
                del model_params
                torch.cuda.empty_cache()

            self._maybe_log_save_evaluate(tr_loss, None, model, trial, epoch, ignore_keys_for_eval)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        self._load_best_model()
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        return TrainOutput(self.state.global_step, train_loss, metrics)

    ############## MeZO ##############
    @torch.no_grad()
    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            param.add_(z, alpha=(scaling_factor * self.args.zo_eps))

    @torch.no_grad()
    def squeezellm_zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        for name, buffer in self.dequantized_weight_list:
            indices = self.named_buffers[name.replace('dequantized_weight', 'sensitive_indices')]
            z = torch.normal(mean=0, std=1, size=(indices.shape[1],), device=buffer.device, dtype=buffer.dtype)
            buffer[indices[0], indices[1]] += (scaling_factor * self.args.zo_eps) * z

    @torch.no_grad()
    def squeezellm_zo_perturb_parameters_memory_limited(self, random_seed=None, scaling_factor=1):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        for name, buffer in self.sqllm_sparse_weight_dict.items():
            z = torch.randn_like(buffer, device=buffer.device, dtype=buffer.dtype)
            buffer.add_(z, alpha=(scaling_factor * self.args.zo_eps))

    def zo_perturb_parameters_rademacher(self, random_seed=None, scaling_factor=1):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.randint(low=0, high=2, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            z[z == 0] = -1
            param.data.add_(z, alpha=scaling_factor * self.args.zo_eps)

    @torch.inference_mode()
    def zo_perturb_parameters_with_mask(self, random_seed=None, mask_dict=None, scaling_factor=1):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        for name, param in self.model.named_parameters():
            if not name in self.named_parameters_to_optim:
                continue
            selected_param = self.named_parameters_to_optim[name]
            z = torch.normal(mean=0, std=1, size=selected_param.size(), device=selected_param.device, dtype=selected_param.dtype)
            param.view(-1)[mask_dict[name]] += (scaling_factor * self.args.zo_eps) * z

    def norm_zo_perturb_parameters(self, random_seed=None, scaling_factor=1, normalize_dict=None):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.args.zo_eps * normalize_dict[name]

    def zo_forward(self, model, inputs):
        model.eval()
        if self.args.non_diff:
            return self.zo_forward_nondiff(model, inputs)
        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()
        return loss.detach()

    def zo_forward_nondiff(self, model, inputs):
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."
        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = self.model.generate(
                inputs["input_ids"],
                do_sample=args.sampling,
                temperature=args.temperature,
                num_beams=args.num_beams,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)),
                num_return_sequences=1,
                eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[0], self.tokenizer.eos_token_id],
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]
        return -torch.tensor(np.mean(f1s), dtype=torch.float32)

    @torch.inference_mode()
    def set_squeezellm_sparse_grad(self, model, inputs):
        self.zo_random_seed = np.random.randint(1000000000)
        self.squeezellm_zo_perturb_parameters(self.zo_random_seed, scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)
        self.squeezellm_zo_perturb_parameters(self.zo_random_seed, scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        self.squeezellm_zo_perturb_parameters(self.zo_random_seed, scaling_factor=1)
        torch.manual_seed(self.zo_random_seed)
        for name, buffer in self.dequantized_weight_list:
            sensitive_vals = self.sqllm_sparse_weight_dict[name]
            z = torch.normal(mean=0, std=1, size=sensitive_vals.size(), device=sensitive_vals.device, dtype=sensitive_vals.dtype)
            sensitive_vals.grad = self.projected_grad * z
        return (loss1 + loss2) / 2

    @torch.inference_mode()
    def perturb_and_update_squeezellm_sparse_grad_memory_limited(self, model, inputs):
        self.zo_random_seed = np.random.randint(1000000000)
        self.squeezellm_zo_perturb_parameters_memory_limited(self.zo_random_seed, scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)
        self.squeezellm_zo_perturb_parameters_memory_limited(self.zo_random_seed, scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        self.squeezellm_zo_perturb_parameters_memory_limited(self.zo_random_seed, scaling_factor=1)
        torch.manual_seed(self.zo_random_seed)
        for name, buffer in self.sqllm_sparse_weight_dict.items():
            z = torch.randn_like(buffer, device=buffer.device, dtype=buffer.dtype)
            buffer.add_(z, alpha=-(self.args.learning_rate * self.projected_grad))
        return (loss1 + loss2) / 2

    @torch.inference_mode()
    def zo_step(self, model, inputs):
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
        self.zo_random_seed = np.random.randint(1000000000)
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)
        self.zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        self.zo_perturb_parameters(scaling_factor=1)
        return (loss1 + loss2) / 2

    def set_zo_grad_as_grad(self):
        torch.manual_seed(self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            param.grad = self.projected_grad * z

    @torch.no_grad()
    def full_zo_update(self):
        torch.manual_seed(self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            param.data.add_(z, alpha=-(self.args.learning_rate * self.projected_grad))

    ############## Misc overload functions ##############

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir
        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        elif self.deepspeed:
            if self.args.should_save:
                self._save(output_dir)
            if is_deepspeed_zero3_enabled():
                file = os.path.join(output_dir, WEIGHTS_NAME)
                if os.path.isfile(file):
                    os.remove(file)
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)
        elif self.args.should_save:
            self._save(output_dir)
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")

    def evaluate_test_set(self):
        predictions = []
        train_samples = self.train_samples
        for eval_id, eval_sample in enumerate(tqdm(self.test_samples)):
            predictions.append(
                self.one_step_pred(train_samples, eval_sample)
            )
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        print(metrics, flush=True)
        sys.stdout.flush()
        return metrics

    def one_step_pred(self, train_samples, eval_sample):
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer,
            max_length=self.args.max_length,
            generation=self.task.generation,
            max_new_tokens=self.args.max_new_tokens
        )
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(
                self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer,
                max_length=self.args.max_length,
                sfc=self.args.sfc,
                icl_sfc=self.args.icl_sfc,
                generation=self.task.generation,
                max_new_tokens=self.args.max_new_tokens
            )
        outputs = []
        if self.task.generation:
            output_text = self.test_forward(encoded_candidates[0], generation=True)
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.test_forward(encoded_candidate, option_len=option_lens[candidate_id])
                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.test_forward(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id])
                outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})
            if self.args.sfc or self.args.icl_sfc:
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                scores = [x['log_probs'].mean().item() for x in outputs]
            if isinstance(eval_sample.correct_candidate, list):
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)
            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        重载内部训练循环，支持两种模式：
          - 当 self.args.trainer == "zo" 时，使用零阶优化(MeZO)的更新方式；
          - 否则（例如当 trainer=="first"）直接使用标准一阶反向传播进行更新。
        其它数据预处理、评估、日志输出和checkpoint保存均保持一致。
        """
        self._train_batch_size = batch_size
        train_dataloader = self.get_train_dataloader()

        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                num_train_samples = args.max_steps * (args.train_batch_size * args.gradient_accumulation_steps * args.world_size)
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * (args.train_batch_size * args.gradient_accumulation_steps * args.world_size)
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if self.args.linear_probing:
            def _get_token_prediction_layer(model):
                if model.config.model_type == "opt":
                    return model.lm_head
                else:
                    raise NotImplementedError(model.config.model_type)

            def _extract_features(model, *args, **kwargs):
                features = {}
                def __hook(model_, input_, output_):
                    features["features"] = input_[0].detach()
                _get_token_prediction_layer(model).register_forward_hook(__hook)
                model.forward(*args, **kwargs)
                return features["features"]

            logger.info("Linear probing")
            logger.info("Starting to get features for training dataset")
            targets = []
            features = []
            with torch.inference_mode():
                for step, inputs in enumerate(tqdm(train_dataloader)):
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.model.device)
                    feature = _extract_features(self.model, **inputs)
                    target = inputs["labels"]
                    assert not self.args.train_as_classification and self.args.only_train_option
                    feature, target = feature[:, :-1], target[:, 1:]
                    for _i, _len in enumerate(inputs["option_len"]):
                        features.append(feature[_i, -_len:])
                        targets.append(target[_i, -_len:])
            logger.info("Finished getting features for training dataset")
            features = torch.cat(features, dim=0).cpu().numpy()
            targets = torch.cat(targets, dim=0).cpu().numpy()
            if self.model.config.model_type in ["opt", "gpt2"]:
                use_bias = False
            else:
                raise NotImplementedError
            tol = 0.01 if self.args.lp_early_stopping else 1e-4
            max_iter = 1000 if self.args.lp_early_stopping else 5000
            logger.info("Fitting logistic regression...")
            reg = LogisticRegressionCV(max_iter=max_iter, fit_intercept=use_bias, multi_class="multinomial",
                                       random_state=0, tol=tol, n_jobs=-1).fit(features, targets)
            logger.info("Done")
            logger.info("Assigning weights to model")
            decoder = _get_token_prediction_layer(self.model)
            coef_torch = torch.tensor(reg.coef_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if use_bias:
                bias_torch = torch.tensor(reg.intercept_, device=decoder.weight.device, dtype=decoder.weight.dtype)
            if coef_torch.shape[0] == 1:
                assert len(reg.classes_) == 2
                coef_torch = torch.cat([-coef_torch / 2, coef_torch / 2], dim=0)
                if use_bias:
                    bias_torch = torch.cat([-bias_torch / 2, bias_torch / 2], dim=0)
            for _i, token_id in enumerate(reg.classes_):
                decoder.weight.data[token_id] = coef_torch[_i]
                if use_bias:
                    decoder.bias.data[token_id] = bias_torch[_i]
            return None

        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if self.args.save_grad:
            self.get_grad_and_load_to_model(self.model, self.args.grad_save_path + os.path.sep + f'start.pt')
            raise RuntimeError('finished')

        delay_optimizer_creation = False
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        has_mask = False
        if self.args.smaller_weight_mask:
            print(f'smaller weight mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_smaller_weight_masks(model, self.args.outlier_percentage)
            has_mask = True
        elif self.args.outlier:
            print(f'outlier mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_outlier_masks(model, self.args.outlier_percentage)
            has_mask = True
        elif self.args.random_subset_weights:
            print(f'random mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_random_masks(model, self.args.outlier_percentage)
            has_mask = True
        elif self.args.grad_mask:
            print(f'grad mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_gradient_masks(model, self.args.outlier_percentage)
            has_mask = True
        elif self.args.C4_grad_mask:
            print(f'C4 grad mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_C4_gradient_masks(model, self.args.C4_grad_addr, self.args.outlier_percentage)
            has_mask = True
        elif self.args.GraSP_mask:
            print(f'GraSP mask {self.args.outlier_percentage}', flush=True)
            sys.stdout.flush()
            self.outlier_masks = self.get_GraSP_mask(model, self.args.outlier_percentage)
            has_mask = True

        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        self.named_parameters_to_optim = [(name, param) for name, param in model.named_parameters() if param.requires_grad]

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} batches in the first epoch."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        tr_loss = torch.tensor(0.0).to(args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad(set_to_none=True)

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        if args.report_to != []:
            wandb.run.name = self.args.tag

        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                _ = list(train_dataloader.sampler)

        from eventProfiler import EventProfiler
        self.profiler = EventProfiler(torch.device('cuda'), active=False)
        self.record_loss = []

        if self.args.use_squeezellm:
            self.named_buffers = {name: buffer for name, buffer in model.named_buffers()}
            self.sqllm_sparse_weight_dict = dict()
            if self.args.memory_limit_scenario:
                self.sqllm_sparse_weight_dict = {name: buffer.requires_grad_(True) for name, buffer in model.named_buffers() if 'sensitive_vals' in name}
            else:
                self.dequantized_weight_list = [(name, buffer) for name, buffer in model.named_buffers() if 'dequantized_weight' in name]
                for name, buffer in self.dequantized_weight_list:
                    indices = self.named_buffers[name.replace('dequantized_weight', 'sensitive_indices')]
                    self.sqllm_sparse_weight_dict[name] = buffer[indices[0], indices[1]].clone().detach().requires_grad_(True)

        if has_mask:
            self.named_parameters_to_optim = {
                n: p.view(-1)[self.outlier_masks[n]].detach().clone().requires_grad_(True)
                for n, p in model.named_parameters() if n in self.outlier_masks
            }

        if args.use_adam:
            if self.args.use_squeezellm:
                self.optimizer = torch.optim.Adam(list(self.sqllm_sparse_weight_dict.values()), lr=self.args.learning_rate)
            elif has_mask:
                self.optimizer = torch.optim.Adam(list(self.named_parameters_to_optim.values()), lr=self.args.learning_rate)
            else:
                self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)
        elif args.use_momentum:
            if self.args.use_squeezellm:
                self.optimizer = torch.optim.SGD(list(self.sqllm_sparse_weight_dict.values()), lr=self.args.learning_rate, momentum=0.9)
            elif has_mask:
                self.optimizer = torch.optim.SGD(list(self.named_parameters_to_optim.values()), lr=self.args.learning_rate, momentum=0.9)
            else:
                self.optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate, momentum=0.9)
        else:
            if self.args.use_squeezellm:
                self.optimizer = torch.optim.SGD(list(self.sqllm_sparse_weight_dict.values()), lr=self.args.learning_rate)
            elif has_mask:
                self.optimizer = torch.optim.SGD(list(self.named_parameters_to_optim.values()), lr=self.args.learning_rate)
            else:
                self.optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate)

        print(self.optimizer)
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = len(epoch_iterator) if len_dataloader is not None else args.max_steps * args.gradient_accumulation_steps
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            self.deepspeed = False

            for step, inputs in enumerate(epoch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if args.trainer == 'zo' and args.use_squeezellm:
                    if self.args.memory_limit_scenario:
                        tr_loss_step = self.perturb_and_update_squeezellm_sparse_grad_memory_limited(model, inputs)
                    else:
                        tr_loss_step = self.set_squeezellm_sparse_grad(model, inputs)
                elif args.trainer == 'zo' and has_mask:
                    zo_random_seed = np.random.randint(1000000000)
                    tr_loss_step = self.set_grad_with_mask(model, inputs, self.outlier_masks, zo_random_seed)
                    if self.args.record_time:
                        self.record_loss.append(tr_loss_step)
                elif args.trainer == 'zo':
                    tr_loss_step = self.zo_step(model, inputs)
                    if self.args.record_time:
                        self.record_loss.append(tr_loss_step)
                    if self.args.use_full_zo_update:
                        self.full_zo_update()
                    else:
                        self.set_zo_grad_as_grad()
                else:
                    # First-order fine-tuning：使用标准训练步
                    tr_loss_step = self.training_step(model, inputs)
                    with torch.no_grad():
                        if has_mask:
                            grad_dict = {n: p.grad for n, p in self.model.named_parameters() if p.requires_grad}
                            for name, selected_param in self.named_parameters_to_optim.items():
                                selected_param.grad = grad_dict[name].view(-1)[self.outlier_masks[name]].clone()
                            for n, p in self.model.named_parameters():
                                p.grad = None

                if (args.logging_nan_inf_filter and not is_torch_tpu_available() and
                    (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))):
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                ):
                    if args.trainer == 'zo' and (not args.memory_limit_scenario):
                        if not args.use_full_zo_update:
                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                        if self.args.use_squeezellm:
                            for name, buffer in self.dequantized_weight_list:
                                sensitive_vals = self.sqllm_sparse_weight_dict[name]
                                indices = self.named_buffers[name.replace('dequantized_weight', 'sensitive_indices')]
                                buffer[indices[0], indices[1]] = sensitive_vals
                    elif args.trainer != 'zo':
                        with self.profiler('optimizer'):
                            self.optimizer.step()
                            self.lr_scheduler.step()
                            self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    if has_mask:
                        with torch.no_grad():
                            for name, param in model.named_parameters():
                                if name not in self.named_parameters_to_optim:
                                    continue
                                param.view(-1)[self.outlier_masks[name]] = self.named_parameters_to_optim[name]
                        if self.args.dynamic_mask and (self.state.global_step + 1) % self.args.dynamic_mask_step == 0:
                            assert not self.args.use_momentum and not self.args.use_adam
                            if self.args.smaller_weight_mask:
                                del self.outlier_masks
                                self.outlier_masks = self.get_smaller_weight_masks(model, self.args.outlier_percentage)
                            elif self.args.outlier:
                                del self.outlier_masks
                                self.outlier_masks = self.get_outlier_masks(model, self.args.outlier_percentage)
                            elif self.args.random_subset_weights:
                                del self.outlier_masks
                                self.outlier_masks = self.get_random_masks(model, self.args.outlier_percentage)
                            elif self.args.grad_mask:
                                del self.outlier_masks
                                self.outlier_masks = self.get_gradient_masks(model, self.args.outlier_percentage)
                            elif self.args.GraSP_mask:
                                del self.outlier_masks
                                self.outlier_masks = self.get_GraSP_mask(model, self.args.outlier_percentage)
                            self.named_parameters_to_optim = {
                                n: p.view(-1)[self.outlier_masks[n]].detach().clone().requires_grad_(True)
                                for n, p in model.named_parameters() if n in self.outlier_masks
                            }
                            self.optimizer = torch.optim.SGD(list(self.named_parameters_to_optim.values()), lr=self.args.learning_rate)
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    if self.args.save_strategy == 'steps' and (self.state.global_step > 0 and self.state.global_step % self.args.save_steps == 0):
                        self.control.should_save = True
                        self.control.should_evaluate = True
                    torch.cuda.empty_cache()
                    self._maybe_log_save_evaluate(tr_loss, None, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}!"
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)

            if self.args.save_model_addr and self.control.should_save:
                if not os.path.exists(self.args.save_model_addr):
                    os.makedirs(self.args.save_model_addr)
                model_params = {n: p for n, p in model.named_parameters()}
                torch.save(model_params, self.args.save_model_addr + os.sep + f'epoch-{epoch}.pt')
                del model_params
                torch.cuda.empty_cache()

            self._maybe_log_save_evaluate(tr_loss, None, model, trial, epoch, ignore_keys_for_eval)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        self._load_best_model()

        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        return TrainOutput(self.state.global_step, train_loss, metrics)

    ############## End of _inner_training_loop ##############
# End of OurTrainer class
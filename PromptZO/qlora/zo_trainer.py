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
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
import transformers
from transformers import Trainer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV


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

from transformers import Seq2SeqTrainer
from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
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


class ZOTrainer(Seq2SeqTrainer):

    from transformers.trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state

    def get_modules(self, layer, model_type):
        # NOTE: This is llama-specific
        # For other models, replace this with proper names for all linear layers
        if model_type in ['mistral', 'llama']:
            return[
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
                layer.mlp.gate_proj,
                layer.mlp.up_proj,
                layer.mlp.down_proj,
            ]
        elif model_type in ['opt']:
            return[
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

        _grad_module_dict = { (i, j): torch.zeros_like(module.weight, device='cpu')
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
            if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n: continue
            top_cutoff = int(p.numel() * percentile)
            mask = torch.zeros(p.numel(), dtype=torch.bool, device=p.device)
            mask[-p.view(-1).abs().argsort()[:top_cutoff]] = True
            mask_dict[n] = torch.arange(p.numel(), device=p.device)[mask]
        return mask_dict

    @torch.no_grad()
    def get_smaller_weight_masks(self, model, percentile=0.20):
        mask_dict = dict()
        for n, p in model.named_parameters():
            if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n: continue
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
            if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n: continue
            random_indices = torch.randperm(p.numel(), device=p.device)[:int(p.numel() * percentile)]
            mask_dict[n] = random_indices.clone()
            torch.cuda.empty_cache()
        return mask_dict

    @torch.no_grad()
    def full_zo_update(self):
        """
        Update the parameters with the estimated gradients.
        """
        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_random_seed)     
        for name, param in self.named_parameters_to_optim:
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            param.data.add_(z, alpha=-(self.args.learning_rate * self.projected_grad))


    # https://arxiv.org/pdf/1810.02340.pdf, SNIP approach
    # def get_gradient_weight_product_masks(self, model: nn.Module,
    #                                 percentile=5e-3, microbatch=1, minibatch=32):
    #     assert minibatch % microbatch == 0
    #     count = 0
    #     old_batch_size = self._train_batch_size
    #     self._train_batch_size = microbatch
    #     with torch.enable_grad():
    #         for sampled_batch in self.get_train_dataloader():
    #             sampled_batch = {k: v.cuda() for k, v in sampled_batch.items()}    
    #             count += len(sampled_batch['input_ids'])
    #             outputs = model(**sampled_batch)
    #             if hasattr(outputs, 'loss'):
    #                 loss = outputs.loss
    #             else:
    #                 loss = outputs[0]
    #             loss /= (minibatch/microbatch)
    #             loss.backward()
    #             if count >= minibatch: 
    #                 break
        
    #     mask_dict = dict()
    #     with torch.no_grad():
    #         for n, p in model.named_parameters():
    #             if not p.requires_grad: continue
    #             if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n: continue
    #             per_layer_scores = p.grad.float() ** 2
    #             top_cutoff = int(p.numel() * percentile)
    #             mask = torch.zeros(*p.shape, dtype=torch.bool, device=p.device)
    #             mask.view(-1)[(-per_layer_scores.view(-1)).argsort()[:top_cutoff]] = True
    #             mask_dict[n] = mask
    #             p.grad = None

    #     return mask_dict

    @torch.no_grad()
    # https://arxiv.org/pdf/2002.07376.pdf, GraSP paper
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

        hessian_grad_prod = {n : torch.zeros_like(p, device='cpu') 
                             for n, p in model.named_parameters() if p.requires_grad}
        
        weight_params = {n : p for n, p in model.named_parameters()}

        with torch.enable_grad():
            for sampled_batch in self.get_train_dataloader():
                sampled_batch = {k: v.cuda() for k, v in sampled_batch.items()}    
                count += 1
                outputs = model(**sampled_batch)
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                else:
                    loss = outputs[0]
                loss /= (minibatch/microbatch)
                grads = torch.autograd.grad(loss, weights, 
                                            create_graph=True, retain_graph=True)

                inner_prod = torch.tensor(0.0, requires_grad=True, device=grads[0].device, dtype=torch.bfloat16)
                for g in grads:
                    inner_prod = inner_prod + torch.inner(g.view(-1), g.data.view(-1))
                del grads
                
                inner_prod.backward()
                for n, p in model.named_parameters():
                    if not p.requires_grad: continue
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
                loss /= (minibatch/microbatch)
                loss.backward()
                if count >= minibatch: 
                    break
        
        mask_dict = dict()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if not p.requires_grad: continue
                if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n: continue
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
        grad_dict = {n : p for n, p in grad.named_parameters()}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if not p.requires_grad: continue
                if 'weight' not in n or 'layer_norm' in n or 'embed' in n or 'lm_head' in n or 'norm' in n: continue
                per_layer_scores = grad_dict[n].cuda() ** 2
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
                p.data.add_(vec[counter : counter + p.numel()].view(p.shape), alpha=scale)
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
            if not param.requires_grad: continue
            # Resample z
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
            if not param.requires_grad: continue
            # Resample z
            z = torch.randint(low=0, high=2, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            z[z == 0] = -1
            ret_grad_dict[name] = global_projected_grad * z
        return ret_grad_dict, loss1

    @torch.inference_mode()
    def set_grad_with_mask(self, model, inputs, mask_dict, random_seed):
        # with self.profiler('perturb'):
        self.zo_perturb_parameters_with_mask(scaling_factor=1, random_seed=random_seed, mask_dict=mask_dict)
        # with self.profiler('forward'):
        loss1 = self.zo_forward(model, inputs)
        # with self.profiler('perturb'):
        self.zo_perturb_parameters_with_mask(scaling_factor=-2, random_seed=random_seed, mask_dict=mask_dict)
        # with self.profiler('forward'):
        loss2 = self.zo_forward(model, inputs)
        # with self.profiler('perturb'):
        self.zo_perturb_parameters_with_mask(scaling_factor=1, random_seed=random_seed, mask_dict=mask_dict)

        # with self.profiler('optimizer'):
        global_projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        torch.manual_seed(random_seed)
        for name, selected_param in self.named_parameters_to_optim.items():
            z = torch.normal(mean=0, std=1, size=selected_param.size(), device=selected_param.device, dtype=selected_param.dtype)
            selected_param.grad = global_projected_grad * z
        return loss1


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
        inputs = self._prepare_inputs(inputs)

        # with self.profiler('forward-backward'):
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            # with self.profiler('forward-backward'):
            loss.backward()

        return loss.detach()


    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        We overload the original training loop to add linear probing and MeZO. Search key word "MeZO added"
        for those updates.
        """
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
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
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        self.state = TrainerState()

        model = self._wrap_model(self.model_wrapped)
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        has_mask = False
        # if self.args.smaller_weight_mask:
        #     print(f'smaller weight mask {self.args.outlier_percentage}', flush=True)
        #     sys.stdout.flush()
        #     self.outlier_masks = self.get_smaller_weight_masks(model, self.args.outlier_percentage)
        #     has_mask = True
            
        # elif self.args.outlier:
        #     print(f'outlier mask {self.args.outlier_percentage}', flush=True)
        #     sys.stdout.flush()
        #     self.outlier_masks = self.get_outlier_masks(model, self.args.outlier_percentage)
        #     has_mask = True

        # elif self.args.random_subset_weights:
        #     print(f'random mask {self.args.outlier_percentage}', flush=True)
        #     sys.stdout.flush()
        #     self.outlier_masks = self.get_random_masks(model, self.args.outlier_percentage)
        #     has_mask = True

        # elif self.args.grad_mask:
        #     print(f'grad mask {self.args.outlier_percentage}', flush=True)
        #     sys.stdout.flush()
        #     self.outlier_masks = self.get_gradient_masks(model, self.args.outlier_percentage)
        #     has_mask = True

        # elif self.args.C4_grad_mask:
        #     print(f'C4 grad mask {self.args.outlier_percentage}', flush=True)
        #     sys.stdout.flush()
        #     self.outlier_masks = self.get_C4_gradient_masks(model, self.args.C4_grad_addr, self.args.outlier_percentage)
        #     # self.outlier_masks = torch.load('C4_grad_mask_1e_3.pt')
        #     has_mask = True

        # elif self.args.GraSP_mask:
        #     print(f'GraSP mask {self.args.outlier_percentage}', flush=True)
        #     sys.stdout.flush()
        #     self.outlier_masks = self.get_GraSP_mask(model, self.args.outlier_percentage)
        #     has_mask = True

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        self.named_parameters_to_optim = [(name, param) for name, param in model.named_parameters() if param.requires_grad]

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad(set_to_none=True)

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                _ = list(train_dataloader.sampler)

        self.record_loss = []

        if self.args.use_squeezellm and (not self.args.lora and not self.args.prefix_tuning):
            self.named_buffers = { name : buffer for name, buffer in model.named_buffers() }
            self.sqllm_sparse_weight_dict = dict()
            self.dequantized_weight_list = [(name, buffer) for name, buffer in model.named_buffers() if 'dequantized_weight' in name]
            for name, buffer in self.dequantized_weight_list:
                indices = self.named_buffers[name.replace('dequantized_weight', 'sensitive_indices')]
                self.sqllm_sparse_weight_dict[name] = buffer[indices[0], indices[1]].clone().detach().requires_grad_(True)

        if has_mask:
            self.named_parameters_to_optim = {
                n : p.view(-1)[self.outlier_masks[n]].detach().clone().requires_grad_(True) for n, p in model.named_parameters() if n in self.outlier_masks
            }

        if self.args.use_squeezellm and (not self.args.lora and not self.args.prefix_tuning):
            # with self.profiler('optimizer'):
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

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
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

                if args.use_squeezellm and (not self.args.lora and not self.args.prefix_tuning):
                    tr_loss_step = self.set_squeezellm_sparse_grad(model, inputs)
                    
                elif has_mask:
                    zo_random_seed = np.random.randint(1000000000)
                    tr_loss_step = self.set_grad_with_mask(model, inputs, self.outlier_masks, zo_random_seed)
                    if self.args.record_time:
                        self.record_loss.append(tr_loss_step)
                    
                else:
                    tr_loss_step = self.zo_step(model, inputs)
                    if self.args.use_full_zo_update:
                        self.full_zo_update()
                    else:
                        self.set_zo_grad_as_grad()
                    # print(tr_loss_step)

                # with torch.no_grad():
                #     if has_mask:
                #         grad_dict = {n : p.grad for n, p in self.model.named_parameters() if p.requires_grad}
                #         for name, selected_param in self.named_parameters_to_optim.items():
                #             selected_param.grad = grad_dict[name].view(-1)[self.outlier_masks[name]].clone()
                #         for n, p in self.model.named_parameters():
                #             p.grad = None

                tr_loss += tr_loss_step

                if not args.use_full_zo_update:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
        
                if self.args.use_squeezellm and (not args.lora and not args.prefix_tuning):
                    # with self.profiler('copy'):              
                    for name, buffer in self.dequantized_weight_list:
                        sensitive_vals = self.sqllm_sparse_weight_dict[name]
                        indices = self.named_buffers[name.replace('dequantized_weight', 'sensitive_indices')]
                        buffer[indices[0], indices[1]] = sensitive_vals

                torch.cuda.empty_cache()
                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    if has_mask:
                        # with self.profiler('copy'):
                        with torch.no_grad():
                            # with self.profiler('optimizer'):
                            for name, param in model.named_parameters():
                                if not name in self.named_parameters_to_optim: continue
                                param.view(-1)[self.outlier_masks[name]] = self.named_parameters_to_optim[name]

                        if self.args.dynamic_mask and \
                                (self.state.global_step + 1) % self.args.dynamic_mask_step == 0:
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
                                n : p.view(-1)[self.outlier_masks[n]].detach().clone().requires_grad_(True) for n, p in model.named_parameters() if n in self.outlier_masks
                            }
                            self.optimizer = torch.optim.SGD(list(self.named_parameters_to_optim.values()), lr=self.args.learning_rate)

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    # if self.state.global_step > 0 and self.state.global_step % 1000 == 0:
                    #     self.evaluate_test_set()
                    if self.args.save_strategy == 'steps' and \
                        (self.state.global_step > 0 and self.state.global_step % self.args.save_steps == 0):
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
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)

            self._maybe_log_save_evaluate(tr_loss, None, model, trial, epoch, ignore_keys_for_eval)

            if self.control.should_training_stop:
                break

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")      
        self._load_best_model()

        # add remaining tr_loss
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

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        # if self.args.record_time:
        #     torch.save(self.profiler.summary(), f'time{os.sep}{self.args.model_name.replace(os.sep, "-")}-{self.args.task_name}-{self.args.tag.replace(os.sep, "-")}-time.pt')
        #     torch.save(self.record_loss, f'time{os.sep}{self.args.model_name.replace(os.sep, "-")}-{self.args.task_name}-{self.args.tag.replace(os.sep, "-")}-loss.pt')

        return TrainOutput(self.state.global_step, train_loss, metrics)


    ############## MeZO ##############

    @torch.no_grad()
    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
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
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            z = torch.randint(low=0, high=2, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            z[z == 0] = -1
            param.data.add_(z, alpha=scaling_factor * self.args.zo_eps)

    @torch.inference_mode()
    def zo_perturb_parameters_with_mask(self, random_seed=None, mask_dict=None, scaling_factor=1):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.model.named_parameters():
            if not name in self.named_parameters_to_optim: continue
            selected_param = self.named_parameters_to_optim[name]
            z = torch.normal(mean=0, std=1, size=selected_param.size(), device=selected_param.device, dtype=selected_param.dtype)
            param.view(-1)[mask_dict[name]] += (scaling_factor * self.args.zo_eps) * z

    def norm_zo_perturb_parameters(self, random_seed=None, scaling_factor=1, normalize_dict=None):
        """
        Perturb the parameters with random vector z.
        Input: 
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.args.zo_eps * normalize_dict[name]


    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()
        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()


    def zo_forward_nondiff(self, model, inputs):
        """
        Get (no gradient) non-diffiable loss from the model.
        """
        model.eval()
        assert self.args.task_name == "SQuAD", "Non differentiable objective only supports SQuAD for now."

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            args = self.args
            outputs = self.model.generate(
                inputs["input_ids"], do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - inputs["input_ids"].size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[0], self.tokenizer.eos_token_id],
            )
            output_text = []
            for i in range(len(outputs)):
                output_text.append(self.tokenizer.decode(outputs[i][inputs["input_ids"].size(1):], skip_special_tokens=True).strip())
            f1s = [f1(output_text[i], inputs['gold'][i]) for i in range(len(output_text))]
        
        return -torch.tensor(np.mean(f1s), dtype=torch.float32)
    

    @torch.inference_mode()
    def set_squeezellm_sparse_grad(self, model, inputs):
        self.zo_random_seed = np.random.randint(1000000000)
        # with self.profiler('perturb'):
        self.squeezellm_zo_perturb_parameters(self.zo_random_seed, scaling_factor=1)
        # with self.profiler('forward'):
        loss1 = self.zo_forward(model, inputs)

        # with self.profiler('perturb'):
        self.squeezellm_zo_perturb_parameters(self.zo_random_seed, scaling_factor=-2)
        # with self.profiler('forward'):
        loss2 = self.zo_forward(model, inputs)

        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

        # with self.profiler('perturb'):
        self.squeezellm_zo_perturb_parameters(self.zo_random_seed, scaling_factor=1)
        
        torch.manual_seed(self.zo_random_seed)

        # with self.profiler('set_grad'):
        for name, buffer in self.dequantized_weight_list:
            sensitive_vals =  self.sqllm_sparse_weight_dict[name]
            z = torch.normal(mean=0, std=1, size=sensitive_vals.size(), device=sensitive_vals.device, dtype=sensitive_vals.dtype)
            sensitive_vals.grad = self.projected_grad * z

        return (loss1 + loss2) / 2

    @torch.inference_mode()
    def perturb_and_update_squeezellm_sparse_grad_memory_limited(self, model, inputs):
        self.zo_random_seed = np.random.randint(1000000000)
        # First function evaluation
        # with self.profiler('perturb'):
        self.squeezellm_zo_perturb_parameters_memory_limited(self.zo_random_seed, scaling_factor=1)
        # with self.profiler('forward'):
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        # with self.profiler('perturb'):
        self.squeezellm_zo_perturb_parameters_memory_limited(self.zo_random_seed, scaling_factor=-2)
        # with self.profiler('forward'):
        loss2 = self.zo_forward(model, inputs)

        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

        # with self.profiler('perturb'):
        self.squeezellm_zo_perturb_parameters_memory_limited(self.zo_random_seed, scaling_factor=1)
        torch.manual_seed(self.zo_random_seed)

        for name, buffer in self.sqllm_sparse_weight_dict.items():
            z = torch.randn_like(buffer, device=buffer.device, dtype=buffer.dtype)
            buffer.add_(z, alpha=-(self.args.learning_rate * self.projected_grad))

        return (loss1 + loss2) / 2


    @torch.inference_mode()
    def zo_step(self, model, inputs):
        """
        Estimate gradient by MeZO. 
        """
        # What parameters to optimize 
        # 
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # with self.profiler('perturb'):
        self.zo_perturb_parameters(scaling_factor=1)
        # with self.profiler('forward'):
        loss1 = self.zo_forward(model, inputs)

        # with self.profiler('perturb'):
        self.zo_perturb_parameters(scaling_factor=-2)
        # with self.profiler('forward'):
        loss2 = self.zo_forward(model, inputs)

        self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
        # print(self.projected_grad, flush=True)
        # with self.profiler('perturb'):
        self.zo_perturb_parameters(scaling_factor=1)
        
        return (loss1 + loss2) / 2

    def set_zo_grad_as_grad(self):
        """
        Update the parameters with the estimated gradients.
        """
        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_random_seed)     
        for name, param in self.named_parameters_to_optim:
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            param.grad = self.projected_grad * z

    @torch.no_grad()
    def full_zo_update(self):
        """
        Update the parameters with the estimated gradients.
        """
        # Reset the random seed for sampling zs
        torch.manual_seed(self.zo_random_seed)     
        for name, param in self.named_parameters_to_optim:
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            param.data.add_(z, alpha=-(self.args.learning_rate * self.projected_grad))

    ############## Misc overload functions ##############

    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        We overload this function to fix an FSDP saving bug (before fix, it will likely cause OOM) 
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            
        elif self.deepspeed:
            # this takes care of everything as long as we aren't under zero3
            if self.args.should_save:
                self._save(output_dir)

            if is_deepspeed_zero3_enabled():
                # It's too complicated to try to override different places where the weights dump gets
                # saved, so since under zero3 the file is bogus, simply delete it. The user should
                # either user deepspeed checkpoint to resume or to recover full weights use
                # zero_to_fp32.py stored in the checkpoint.
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                        os.remove(file)

                # now save the real model if stage3_gather_16bit_weights_on_model_save=True
                # if false it will not be saved.
                # This must be called on all ranks
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")
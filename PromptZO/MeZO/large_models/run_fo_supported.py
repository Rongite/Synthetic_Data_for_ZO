import logging
import math
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import time
import tasks
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, HfArgumentParser, TrainingArguments, DataCollatorWithPadding, DataCollatorForTokenClassification
from typing import Union, Optional
import torch
from torch.nn.parameter import Parameter
import numpy as np
from dataclasses import dataclass, is_dataclass, asdict
from tqdm import tqdm
from tasks import get_task
import json
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from metrics import calculate_metric
from utils import *
from trainer import OurTrainer  # 使用自定义 Trainer
import random
from model_opt_soft_prompt_learning import OPTPromptTuningLM
import os
import wandb
from eventProfiler import EventProfiler
import shutil

os.environ['TMPDIR'] = '~/.tmp'


@dataclass
class OurArguments(TrainingArguments):
    # 任务和数据集相关
    task_name: str = "SST2"
    num_train: int = 1000
    num_dev: Optional[int] = None
    num_eval: int = 1000
    num_train_sets: int = None
    train_set_seed: int = None
    result_file: str = None

    # 模型加载相关
    model_name: str = "facebook/opt-125m"
    load_float16: bool = False
    load_bfloat16: bool = False
    load_int8: bool = False
    max_length: int = 2048
    no_auto_device: bool = False

    # 校准相关
    sfc: bool = False
    icl_sfc: bool = False

    # Trainer 模式：支持 "none"（仅评估）、"first"（传统一阶微调）和 "zo"（零阶优化）
    trainer: str = "first"  # 默认采用传统一阶微调

    # 其它训练参数
    only_train_option: bool = True
    train_as_classification: bool = False
    zo_eps: float = 1e-3

    # 前缀调优参数
    prefix_tuning: bool = False
    num_prefix: int = 5
    no_reparam: bool = True
    prefix_init_by_real_act: bool = True

    # LoRA 参数
    lora: bool = False
    lora_alpha: int = 16
    lora_rank: int = 16

    # 生成相关
    sampling: bool = True
    temperature: float = 1.0
    num_beams: int = 1
    top_k: int = None
    top_p: float = 0.95
    max_new_tokens: int = 50
    eos_token: str = "\n"

    # 保存与评估
    save_model: bool = False
    no_eval: bool = False
    tag: str = ""

    # 线性探测相关
    linear_probing: bool = False
    lp_early_stopping: bool = False
    head_tuning: bool = False

    # 其它显示与优化参数
    verbose: bool = False
    non_diff: bool = False
    save_on_interrupt: bool = False
    n_tokens: int = 2
    use_sgd: bool = False
    optim: str = 'sgd'
    samples: int = 1
    use_adam: bool = False
    use_momentum: bool = False
    prefix_tuning_one_layer: bool = False
    prefix_layer_id: int = None
    per_device_eval_batch_size: int = 8
    logging_steps: int = 10

    # 与梯度掩码和零阶优化相关
    outlier: bool = False
    outlier_percentage: float = 0.005
    random_subset_weights: bool = False
    grad_mask: bool = False
    use_squeezellm: bool = False
    squeezellm_ckpt: str = None
    squeezellm_wbits: int = 4

    # 更多零阶优化与 mask 参数
    eval_steps: int = 200
    mask_path: str = None
    other_task_mask: bool = False
    lr_scheduler_type = 'constant'
    save_grad: bool = False
    grad_save_path: str = None
    model_weight_path: str = None
    use_squeezellm_peft: bool = False
    no_flash_attn_2: bool = False
    memory_limit_scenario: bool = False
    use_full_zo_update: bool = False
    dynamic_mask: bool = False
    GraSP_mask: bool = False
    dynamic_mask_step: int = 100
    save_model_addr: str = None
    C4_grad_mask: bool = False
    C4_grad_addr: str = None


def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


def parse_args():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Framework:
    """
    加载模型、分词器以及基于任务模板对数据进行转换，支持训练与评估
    """
    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
        with count_time("Loading model"):
            tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.add_eos_token = True
            tokenizer.padding_side = 'right'
            if "opt" in self.args.model_name:
                tokenizer.bos_token_id = 0
            if self.args.use_squeezellm:
                from squeezellm_quant import load_quant
                if self.args.memory_limit_scenario:
                    torch.cuda.empty_cache()
                    total_cuda_mem_in_GB = torch.cuda.mem_get_info(0)[1] /(1024**3)
                    if total_cuda_mem_in_GB > 10.75:
                        torch.cuda.set_per_process_memory_fraction(10.75/total_cuda_mem_in_GB)
                    model = load_quant(self.args.model_name, 
                                       self.args.squeezellm_ckpt, 
                                       self.args.squeezellm_wbits, True, topX=0,
                                       sparse_dtype=torch.float32,  
                                       fake_quant=False,
                                       use_flash_attn_2=False,
                                       use_cuda=False)
                    model = model.cuda()
                    torch.cuda.empty_cache()
                else:
                    model = load_quant(self.args.model_name, 
                                       self.args.squeezellm_ckpt, 
                                       self.args.squeezellm_wbits, True, topX=0, 
                                       sparse_dtype=torch.float16, 
                                       fake_quant=True,
                                       use_flash_attn_2=(not self.args.no_flash_attn_2))
                model.eval()
                return model, tokenizer
            elif self.args.use_squeezellm_peft:
                from squeezellm_quant import load_quant, transfer_quant_linear_to_nn_linear
                model = load_quant(self.args.model_name, 
                                   self.args.squeezellm_ckpt, 
                                   self.args.squeezellm_wbits, 
                                   True, topX=0, fake_quant=True, 
                                   sparse_dtype=torch.float16,
                                   use_flash_attn_2=((not self.args.prefix_tuning) and (not self.args.no_flash_attn_2)))
                model.eval()
                transfer_quant_linear_to_nn_linear(model)
            else:
                config = AutoConfig.from_pretrained(self.args.model_name)
                config.use_cache = False
                torch_dtype = torch.float16
                attn_impl = 'flash_attention_2' if not self.args.no_flash_attn_2 else 'eager'
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name,
                    config=config,
                    device_map='auto',
                    torch_dtype=torch_dtype,
                    attn_implementation=attn_impl
                )
        if self.args.prefix_tuning and not self.args.prefix_tuning_one_layer:
            from prefix import PrefixTuning
            PrefixTuning(model, num_prefix=self.args.num_prefix, reparam=not self.args.no_reparam, float16=self.args.load_float16, init_by_real_act=self.args.prefix_init_by_real_act)
        if self.args.lora:
            from lora import LoRA
            LoRA(model, r=self.args.lora_rank, alpha=self.args.lora_alpha)
        return model, tokenizer

    def forward(self, input_ids, option_len=None, generation=False):
        input_ids = torch.tensor([input_ids]).to(self.model.device)
        if generation:
            args = self.args
            outputs = self.model.generate(
                input_ids, do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[0], self.tokenizer.eos_token_id],
            )
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_ids).logits
                labels = input_ids[0, 1:]
                logits = logits[0, :-1]
                log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            return selected_log_probs[-option_len:]

    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        verbose = verbose or self.args.verbose
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length, 
            generation=self.task.generation, max_new_tokens=self.args.max_new_tokens
        )
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(self.task, self.task.get_template(), 
                train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length,
                sfc=self.args.sfc, icl_sfc=self.args.icl_sfc, generation=self.task.generation, 
                max_new_tokens=self.args.max_new_tokens
            )
        outputs = []
        if self.task.generation:
            output_text = self.forward(encoded_candidates[0], generation=True)
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id])
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

    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        predictions = []
        for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
            predictions.append(
                self.one_step_pred(train_samples[eval_id] if one_train_set_per_eval_sample else train_samples, eval_sample, verbose=(eval_id < 3))
            )
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics

    def train(self, train_samples, eval_samples, test_samples=None):
        self.tokenizer.padding_side = "left"

        class HFDataset(Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]

        def _convert(samples):
            data = []
            for sample in samples:
                encoded_candidates, option_lens = encode_prompt(
                    self.task, self.task.get_template(), [], sample, self.tokenizer, 
                    max_length=self.args.max_length, generation=self.task.generation, generation_with_gold=True, 
                    max_new_tokens=self.args.max_new_tokens
                )
                if self.task.generation:
                    correct_candidate_id = 0
                elif isinstance(sample.correct_candidate, list):
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
                else:
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate)
                if self.args.non_diff:
                    encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][:-option_lens[correct_candidate_id]]
                if self.args.train_as_classification:
                    data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id, "option_len": option_lens[_i], "num_options": len(sample.candidates)} for _i in range(len(encoded_candidates))])
                elif self.args.only_train_option:
                    if self.args.non_diff:
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate})
                    else:
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id]})
                else:
                    data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id]})
            return data

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(_convert(train_samples))
            eval_dataset = HFDataset(_convert(eval_samples))
            if test_samples:
                test_dataset = HFDataset(_convert(test_samples))
                eval_data_for_trainer = {"dev": eval_dataset, "test": test_dataset}
                self.args.metric_for_best_model = 'eval_dev_loss'
            else:
                eval_data_for_trainer = eval_dataset

        if self.args.only_train_option and not self.args.non_diff:
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))

        if self.args.non_diff:
            collator = NondiffCollator
        else:
            collator = DataCollatorForTokenClassification

        self.args.report_to = []
        self.trainer = OurTrainer(
            model=self.model, 
            args=self.args,
            train_dataset=train_dataset, 
            eval_dataset=eval_data_for_trainer,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
        )
        self.trainer.train_samples = train_samples
        self.trainer.test_samples = test_samples
        self.trainer.task = self.task
        self.trainer.test_forward = self.forward
        if self.args.save_on_interrupt:
            self.trainer.add_callback(SIGUSR1Callback())
        last_checkpoint = None
        self.trainer.train(resume_from_checkpoint=last_checkpoint)
        if self.args.save_model:
            logger.warning("Saving the final best checkpoint (best dev)...")
            self.trainer.save_model()
        self.model = self.trainer.model
        if self.args.only_train_option and not self.args.non_diff:
            if type(self.model) == FSDP:
                print("This is an FSDP model now. Be careful when assigning back the original forward function")
                self.model._fsdp_wrapped_module.forward = self.model._fsdp_wrapped_module.original_forward
            else:
                self.model.forward = self.model.original_forward


def result_file_tag(args):
    save_model_name = args.model_name.split("/")[-1]
    sfc_tag = "-sfc" if args.sfc else ""
    icl_sfc_tag = "-icl_sfc" if args.icl_sfc else ""
    sample_eval_tag = "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    sample_dev_tag = "-ndev%d" % args.num_dev if args.num_dev is not None else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    return f"{args.task_name}-{save_model_name}" + sfc_tag + icl_sfc_tag + sample_eval_tag + sample_train_tag + sample_dev_tag + customized_tag.replace(os.sep, '-')


def main():
    args = parse_args()
    print(args, flush=True)
    set_seed(args.seed)
    task = get_task(args.task_name, args)
    train_sets = task.ordered_train_sets()
    framework = Framework(args, task)
    if args.train_set_seed is not None or args.num_train_sets is not None:
        for train_set_id, train_samples in enumerate(train_sets):
            train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed
            if args.num_eval is not None:
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
            else:
                eval_samples = task.valid_samples
            if args.trainer != "none":
                if args.num_dev is not None:
                    dev_samples = train_samples[-args.num_dev:] 
                    train_samples = train_samples[:-args.num_dev]
                else:
                    dev_samples = None
                framework.train(train_samples, dev_samples if dev_samples is not None else eval_samples, eval_samples)
                if not args.no_eval:
                    metrics = framework.evaluate([], eval_samples)
                    if dev_samples is not None:
                        dev_metrics = framework.evaluate([], dev_samples)
                        for m in dev_metrics:
                            metrics["dev_" + m] = dev_metrics[m]
            else:
                assert args.num_dev is None
                metrics = framework.evaluate(train_samples, eval_samples, one_train_set_per_eval_sample=True)
            if not args.no_eval:
                print("===== Train set %d =====" % train_set_seed)
                print(metrics)
                the_output_file = os.path.dirname(args.output_dir)
                if args.local_rank <= 0:
                    write_metrics_to_file(metrics, "result" + result_file_tag(args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file)
    else:
        assert args.trainer == "none"
        if args.num_eval is not None:
            eval_samples = task.sample_subset(data_split="valid", seed=0, num=args.num_eval)
        else:
            eval_samples = task.valid_samples
        metrics = framework.evaluate(train_sets, eval_samples, one_train_set_per_eval_sample=True)
        print(metrics)
        if args.local_rank <= 0:
            write_metrics_to_file(metrics, "result" + result_file_tag(args) + "-onetrainpereval.json" if args.result_file is None else args.result_file)
    shutil.rmtree(framework.trainer.state.best_model_checkpoint)


if __name__ == "__main__": 
    main()
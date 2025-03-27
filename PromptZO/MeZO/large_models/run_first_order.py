import logging
import math
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import time
import torch
import numpy as np
import random
import os
import shutil

from dataclasses import dataclass
from typing import Optional, Union
from tqdm import tqdm

# Hugging Face
from transformers import (
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForTokenClassification,
    set_seed,
)

# Local modules
from tasks import get_task
from lora import LoRA
from templates import *
from metrics import calculate_metric
from utils import (
    write_metrics_to_file,
    encode_prompt,
    DataCollatorWithPaddingAndNesting,
)

# -------------------------------------------------------------------------
# 1) Our custom argument class with a new 'precision' argument
#    and a default per_device_eval_batch_size=8
# -------------------------------------------------------------------------
@dataclass
class OurArguments(TrainingArguments):
    """
    Extends Hugging Face TrainingArguments to allow:
    - trainer: 'none' or 'regular' for first-order finetuning
    - precision: choose "fp16", "bf16", "fp32", or "fp64"
    - lora toggles
    - learning_rate from TrainingArguments
    - per_device_eval_batch_size = 8 by default
    """
    task_name: str = "SST2"
    num_train: int = 1000
    num_dev: Optional[int] = None
    num_eval: int = 1000
    train_set_seed: Optional[int] = None
    result_file: Optional[str] = None

    model_name: str = "facebook/opt-125m"
    max_length: int = 2048

    trainer: str = "none"  # "none" or "regular"

    # New argument for controlling model dtype
    precision: str = "fp32"  # options: "fp16", "bf16", "fp32", "fp64"

    no_eval: bool = False
    tag: str = ""

    finetune_method: str = "full"
    lora: bool = False
    lora_alpha: int = 1
    lora_rank: int = 16
    need_all_linear: bool = False

    max_new_tokens: int = 50
    eos_token: str = "\n"

    # Set default evaluation batch size to 8
    per_device_eval_batch_size: int = 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    return args

def set_seed_everywhere(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class HFDataset(torch.utils.data.Dataset):
    """Minimal dataset for train/eval."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Framework:
    """
    Loads model in the dtype chosen by `args.precision`, optionally injects LoRA,
    trains/eval using HF Trainer with AdamW.
    """
    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        logger.info(f"Loading model from {self.args.model_name} ...")

        config = AutoConfig.from_pretrained(self.args.model_name)
        config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Map user-friendly precision strings -> torch dtypes
        precision_map = {
            "fp16":  torch.float16,
            "bf16":  torch.bfloat16,
            "fp32":  torch.float32,
            "fp64":  torch.float64,
        }
        if self.args.precision not in precision_map:
            raise ValueError(f"Unsupported precision {self.args.precision}, choose one of: {list(precision_map.keys())}")

        chosen_dtype = precision_map[self.args.precision]

        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            config=config,
            torch_dtype=chosen_dtype
        )

        if torch.cuda.is_available():
            model.cuda()

        if self.args.lora or self.args.finetune_method == "lora":
            logger.info(f"Injecting LoRA (alpha={self.args.lora_alpha}, rank={self.args.lora_rank})...")
            LoRA(model, r=self.args.lora_rank, alpha=self.args.lora_alpha, need_all_linear=self.args.need_all_linear)
            self.args.lora = True
        else:
            logger.info("Performing full finetuning (no LoRA).")

        return model, tokenizer

    def _convert_dataset(self, samples):
        data = []
        for sample in samples:
            encoded_candidates, option_lens = encode_prompt(
                self.task,
                self.task.get_template(),
                [],
                sample,
                self.tokenizer,
                max_length=self.args.max_length,
                generation=False,
                generation_with_gold=True,
                max_new_tokens=self.args.max_new_tokens
            )
            if hasattr(sample, 'correct_candidate'):
                if isinstance(sample.correct_candidate, list):
                    cid = sample.candidates.index(sample.correct_candidate[0])
                else:
                    cid = sample.candidates.index(sample.correct_candidate)
                data.append({
                    "input_ids": encoded_candidates[cid],
                    "labels": encoded_candidates[cid],
                })
            else:
                data.append({
                    "input_ids": encoded_candidates[0],
                    "labels": encoded_candidates[0],
                })
        return HFDataset(data)

    def train(self, train_samples, eval_samples):
        logger.info("Building train dataset ...")
        train_dataset = self._convert_dataset(train_samples)
        eval_dataset = self._convert_dataset(eval_samples) if eval_samples else None

        collator = DataCollatorForTokenClassification(self.tokenizer, pad_to_multiple_of=8)

        # (CHANGED) We configure TrainerArguments to evaluate frequently but only keep the best dev checkpoint
        # We do so by setting things like load_best_model_at_end, metric_for_best_model, etc.

        # Overwrite or set relevant training args:
        self.args.evaluation_strategy = "steps"  # or "epoch"
        self.args.eval_steps = 100  # Evaluate every 100 steps (modify as needed)
        self.args.save_strategy = "steps"  # Must match for best checkpoint
        self.args.save_steps = 100  # same freq as eval
        self.args.save_total_limit = 1  # keep only 1 checkpoint
        self.args.load_best_model_at_end = True
        # If dev accuracy is your measure, do:
        self.args.metric_for_best_model = "eval_accuracy"  # or "eval_loss", etc.
        self.args.greater_is_better = True  # If you're tracking accuracy
        # If you use dev loss, then do greater_is_better=False

        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=collator,
        )

        logger.info(f"Starting training in precision={self.args.precision} with evaluation batch size={self.args.per_device_eval_batch_size} ...")
        trainer.train()

        # The best checkpoint is automatically reloaded if load_best_model_at_end=True
        # So now trainer.model is the best model on dev set

        # If user wants to save that best model to self.args.output_dir, do:
        # (CHANGED) There's no "save every 500 steps" anymore, we rely on the best dev checkpoint only
        logger.warning("Saving the final best checkpoint (best dev) ...")
        trainer.save_model(self.args.output_dir)  # This overwrites output_dir with best checkpoint

    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        logger.info(f"Evaluating model in precision={self.args.precision} ...")
        if not eval_samples:
            logger.info("No eval samples, skipping evaluation.")
            return {}

        eval_dataset = self._convert_dataset(eval_samples)
        collator = DataCollatorForTokenClassification(self.tokenizer, pad_to_multiple_of=8)

        trainer = Trainer(
            model=self.model,
            args=self.args,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=collator,
        )
        results = trainer.evaluate()
        logger.info(f"Eval results: {results}")
        return results

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
    logger.info(args)

    set_seed_everywhere(args.seed)

    task = get_task(args.task_name, args)
    train_sets = task.ordered_train_sets()
    if not train_sets:
        logger.error("No training sets found, exit.")
        return

    train_samples = train_sets[0]
    eval_samples = task.valid_samples

    framework = Framework(args, task)

    if args.trainer == "regular":
        logger.info("** Using HF Trainer, AdamW, plus user-chosen precision. **")

        dev_samples = None
        if args.num_dev and len(train_samples) > args.num_dev:
            dev_samples = train_samples[-args.num_dev:]
            train_samples = train_samples[:-args.num_dev]

        framework.train(train_samples, dev_samples if dev_samples else eval_samples)

        if not args.no_eval and eval_samples:
            metrics = framework.evaluate([], eval_samples)
            if dev_samples:
                dev_metrics = framework.evaluate([], dev_samples)
                for k, v in dev_metrics.items():
                    metrics[f"dev_{k}"] = v
            logger.info("Final evaluation metrics:")
            logger.info(metrics)
            if args.local_rank <= 0 and args.result_file:
                write_metrics_to_file(metrics, args.result_file)
    else:
        if not args.no_eval and eval_samples:
            logger.info("No training => zero-shot or existing model evaluation ...")
            metrics = framework.evaluate(train_sets, eval_samples, one_train_set_per_eval_sample=True)
            logger.info(metrics)
            if args.result_file:
                write_metrics_to_file(metrics, args.result_file)

if __name__ == "__main__":
    main()

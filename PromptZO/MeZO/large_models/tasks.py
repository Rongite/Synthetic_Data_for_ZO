from templates import *
from utils import temp_seed
import json
import os
from datasets import load_dataset
from dataclasses import dataclass
from typing import List, Union
import string
import random
import datasets
import sys
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_task(task_name, args):
    # aa = task_name.split("__")
    # if len(aa) == 2:
    #     task_group, subtask = aa
    # else:
    #     task_group = aa[0]
    #     subtask = None
    # class_ = getattr(sys.modules[__name__], f"{task_group}Dataset")
    # instance = class_(subtask, args=args)
    # return instance
    # 初始化 task_group 和 subtask
    """
    根据任务名称动态加载任务类，同时支持从 Hugging Face 和本地 JSONL 文件加载数据。
    """
    # 初始化 task_group 和 subtask
    task_group = None
    subtask = None

    # 创建一个字典来存储 args 对象的属性
    task_args = {k: v for k, v in vars(args).items()}

    # 检查 task_name 是否为本地路径
    if os.path.exists(task_name) and os.path.isdir(task_name):
        print(f"检测到本地路径：{task_name}")
        task_args['path'] = task_name  # 设置本地路径
        task_group = "sst2"  # 假设本地数据集为 SST2
    elif task_name.lower() == "sst2":
        task_group = "sst2"
        task_args['path'] = None  # 不设置本地路径
    else:
        # 如果 task_name 包含子任务，则解析子任务
        aa = task_name.split("__")
        if len(aa) == 2:
            task_group, subtask = aa
        else:
            task_group = aa[0]

    # 如果 task_group 未正确设置，抛出异常
    if not task_group:
        raise ValueError("未能正确设置 task_group")

    # 生成类名
    class_name = f"{task_group.upper()}Dataset" if task_group.lower() == 'sst2' else f"{task_group.capitalize()}Dataset"

    # 动态加载数据集类
    try:
        class_ = getattr(sys.modules[__name__], class_name)
    except AttributeError as e:
        print(f"错误：找不到类 '{class_name}'。请检查任务名称。")
        raise e

    # 实例化数据集类，并确保传递 task_args
    instance = class_(subtask=subtask, path=task_args.get('path'), args=task_args)
    return instance

@dataclass
class Sample:
    id: int = None
    data: dict = None
    correct_candidate: Union[str, List[str]] = None
    candidates: List[str] = None


class Dataset:
    mixed_set = False
    train_sep = "\n\n"
    generation = False # whether this is a generation task

    def __init__(self, subtask=None, **kwargs) -> None:
        self.subtask = subtask
    
    def get_task_name(self):
        return self.subtask
        
    def load_dataset():
        raise NotImplementedError
    
    def get_template(self, template_version=0):
       templates = {0: Template}
       return templates[template_version]
   
    def build_sample(self, example):
        return 
     
    def sample_train_sets(self, num_train=32, num_dev=None, num_eval=None, num_train_sets=None, seed=None):
        if seed is not None:
            # one train/demo set using the designated seed
            seeds = [seed]
        elif num_train_sets is not None:
            # num_train_sets train/demo sets
            seeds = list(range(num_train_sets))
        else: 
            # one train/demo set per evaluation sample
            assert num_dev is None # not supported
            len_valid_samples = len(self.samples["valid"]) if num_eval is None else num_eval
            with temp_seed(0):
                seeds = np.random.randint(0, 10000, len_valid_samples)

        train_samples = [] 
        for i, set_seed in enumerate(seeds):
            if self.mixed_set:
                raise NotImplementedError
                train_samples.append(self.sample_subset(data_split="valid", seed=set_seed, num=num_train, exclude=i))
            else:
                if num_dev is not None:
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed, num=num_train+num_dev)) # dev set is included at the end of train set
                    if num_train + num_dev > len(self.samples["train"]):
                        logger.warn("num_train + num_dev > available training examples")
                else:
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed, num=num_train))
                if num_dev is not None:
                    print(f"Sample train set {len(train_samples[-1])}/{len(self.samples['train'])}")
                    print(f"... including dev set {num_dev} samples")
        return train_samples

    def sample_subset(self, data_split="train", seed=0, num=100, exclude=None):
        with temp_seed(seed):
            samples = self.samples[data_split] 
            lens = len(samples)
            index = np.random.permutation(lens).tolist()[:num if exclude is None else num+1]
            if exclude is not None and exclude in index:
                index.remove(exclude)
            else:
                index = index[:num]
            return [samples[i] for i in index]
    
    @property
    def valid_samples(self):
        return self.samples["valid"]


class SST2Dataset(Dataset):
    train_sep = "\n\n"
    # def __init__(self, subtask=None, **kwargs) -> None:
        # self.load_dataset(subtask, **kwargs)
        # if 'llama' in kwargs['args'].model_name.lower():
        #     self.template = SST2Template_LLama2
        # elif 'mistral' in kwargs['args'].model_name.lower():
        #     self.template = SST2Template
        # else:
        #     self.template = SST2Template
    def __init__(self, subtask=None, path=None, **kwargs) -> None:
        """
        初始化数据集，根据参数决定是从本地 JSONL 文件还是 Hugging Face Hub 加载数据
        """
        self.args = kwargs.get('args', {})
        self.load_dataset(path, **kwargs)
        # 根据模型名称选择模板
        model_name = self.args.get('model_name', '').lower()
        if 'llama' in model_name:
            self.template = SST2Template_LLama2
        elif 'mistral' in model_name:
            self.template = SST2Template
        else:
            self.template = SST2Template

    # def load_dataset(self, path, **kwargs):
        # d = load_dataset('sst2')
        # train_d = d["train"]
        # validation_d = d["validation"]
        
        # train_samples = [self.build_sample(example) for example in train_d]
        # valid_samples = [self.build_sample(example) for example in validation_d]
        
        # self.samples = {"train": train_samples, "valid": valid_samples}
    def load_dataset(self, path=None, **kwargs):
        print(f"加载数据集的路径: {path}")
        if path and os.path.exists(path):
            print("从本地 JSONL 文件加载 SST-2 数据集...")
            train_samples = self.load_jsonl(os.path.join(path, "sst2_train.jsonl"))
            valid_samples = self.load_jsonl(os.path.join(path, "sst2_validation.jsonl"))
        else:
            print("从 Hugging Face 加载 SST-2 数据集...")
            dataset = load_dataset('glue', 'sst2')
            train_samples = dataset["train"]
            valid_samples = dataset["validation"]
        
        self.samples = {
            "train": [self.build_sample(example) for example in train_samples],
            "valid": [self.build_sample(example) for example in valid_samples]
        }

    # 新加的方法
    def load_jsonl(self, file_path):
        """加载 JSONL 文件并返回字典列表"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    
    # for generative tasks, candidates are []
    def build_sample(self, example):
        label = int(example["label"])
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1])
        
    def get_template(self, template_version=0):
        return {0: self.template}[template_version]()
        
    
class CopaDataset(Dataset):
    train_sep = "\n\n"
    mixed_set = False

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):
        train_examples = load_dataset('super_glue', "copa")["train"]
        valid_examples = load_dataset('super_glue', "copa")["validation"]
    
        train_samples = [self.build_sample(example) for example in train_examples]
        valid_samples = [self.build_sample(example) for example in valid_examples]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example):
        sample = \
            Sample(
                id=example["idx"],
                data=example,
                candidates=[example["choice1"], example["choice2"]],
                correct_candidate=example[f"choice{example['label'] + 1}"],
            )
        
        return sample
        
    def get_template(self, template_version=0):
        return {0: CopaTemplate}[template_version]()


class BoolQDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
        if 'llama' in kwargs['args'].model_name.lower():
            self.template = BoolQTemplate
        else:
            self.template = BoolQTemplateV3

    def load_dataset(self, path, **kwargs):
        d = load_dataset("boolq")
        train_set = d["train"]
        valid_set = d["validation"]
        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=["Yes", "No"],
                correct_candidate="Yes" if example["answer"] else "No",
            )
        
        return sample
    
    def get_template(self):
        return self.template()


class MultiRCDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "multirc")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: MultiRCTemplate}[template_version]()


class CBDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "cb")
        train_set = d["train"]
        valid_set = d["validation"]
        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1, 2],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: CBTemplate}[template_version]()


class WICDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "wic")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: WICTemplate}[template_version]()


class WSCDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "wsc.fixed")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: WSCTemplate}[template_version]()


class ReCoRDDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "record")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=example['entities'],
                correct_candidate=example['answers']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: ReCoRDTemplateGPT3}[template_version]()


class RTEDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)
        if 'llama' in kwargs['args'].model_name.lower():
            self.template = RTE_Llama2Template
        elif 'mistral' in kwargs['args'].model_name.lower():
            self.template = RTETemplate
        else:
            self.template = RTETemplate
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "rte")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: self.template}[template_version]()
        # return {0: RTE_Llama2Template}[template_version]()

 
class SQuADDataset(Dataset):
    metric_name = "f1"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()
        
    def load_dataset(self):
        dataset = load_dataset("squad")
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        answers = example['answers']['text']
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "title": example['title'],
                "context": example['context'],
                "question": example['question'],
                "answers": answers
            },
            candidates=None,
            correct_candidate=answers
        )
        
    def get_template(self, template_version=0):
        return {0: SQuADv2Template}[template_version]()


class DROPDataset(Dataset):
    metric_name = "f1"
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()
        
    def load_dataset(self):
        dataset = load_dataset("drop")
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        answers = example['answers_spans']['spans']
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "context": example['passage'],
                "question": example['question'],
                "answers": answers
            },
            candidates=None,
            correct_candidate=answers
        )
        
    def get_template(self, template_version=0):
        return {0: DROPTemplate}[template_version]()


class WinoGrandeDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        train_set = load_dataset('winogrande', 'winogrande_m', split='train')
        valid_set = load_dataset('winogrande', 'winogrande_m', split='validation')

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        """
        Prompt adapted from https://arxiv.org/pdf/2110.08207.pdf
        """
        sentence = example["sentence"]
        context, target = sentence.split("_")
        sample = Sample(
            data=example,
            candidates=[example['option1'] + target, example['option2'] + target],
            correct_candidate=example[f'option{example["answer"]}'] + target,
        )
        return sample

    def get_template(self, template_version=0):
        if template_version == 0:
            return WinoGrandeTemplate()
        else:
            raise NotImplementedError(f"Template version {template_version} not implemented for WinoGrande")
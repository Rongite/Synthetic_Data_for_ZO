from templates import *
from utils import temp_seed
import json
import os
from datasets import load_dataset
from datasets import Features, Value, ClassLabel
from datasets import Dataset as HFDataset
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
    # Initialize task_group and subtask
    """
    Dynamically load the task class based on the task name and pass the correct path.
    """
    task_group = None

    # If it's a local path
    if os.path.exists(task_name):
        print(f"Detected local path: {task_name}")
        task_group = os.path.basename(task_name).lower()  # Extract task name
        # import pdb; pdb.set_trace()
        args.path = task_name  # Set local path
    else:
        # If it's a predefined task name
        task_group = task_name.lower()
        args.path = None  # Do not set local path

    # Mapping from task names to class names
    supported_tasks = {
        "sst2": "SST2Dataset",
        "copa": "CopaDataset",
        "boolq": "BoolQDataset",
        "multirc": "MultiRCDataset",
        "cb": "CBDataset",
        "wic": "WICDataset",
        "wsc": "WSCDataset",
        "record": "ReCoRDDataset",
        "rte": "RTEDataset",
        "squad": "SQuADDataset",
        "drop": "DROPDataset",
        "winogrande": "WinoGrandeDataset",
        "arcc_mc": "ArcC_MCDataset",
        "arcc_cloze": "ArcC_ClozeDataset"
    }

    # Check if the task name is supported
    if task_group not in supported_tasks:
        raise ValueError(f"Unknown task name or path: {task_name}")

    # Dynamically load the class
    class_name = supported_tasks[task_group]
    try:
        class_ = getattr(sys.modules[__name__], class_name)
    except AttributeError:
        raise AttributeError(f"Error: Could not find class '{class_name}'. Please check the class definition or module imports.")

    # Instantiate the task class
    return class_(path=args.path, args=args)


@dataclass
class Sample:
    id: int = None
    data: dict = None
    question_string: str = None
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
     
    def ordered_train_sets(self):
        print("The training set is completely sequential")
        train_samples = []
        sample = self.samples["train"]
        train_samples.append(sample)
        return train_samples
    
    def sample_train_sets(self, num_train=32, num_dev=None, num_eval=None, num_train_sets=None, seed=None):
        # print("How large is num_train: ", num_train)
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
                    # print("How large is num_train: ", num_train)
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed, num=num_train+num_dev))  # dev set is included at the end of train set
                    if num_train + num_dev > len(self.samples["train"]):
                        logger.warning("num_train + num_dev > available training examples")
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
            # print("How many samples: ", lens)
            # print("How many num_train+num_dev: ", num)
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
        Initialize the dataset and decide whether to load from local JSONL files or from the Hugging Face Hub based on parameters.
        """
        self.args = kwargs.get('args', {})
        self.subtask = subtask
        self.samples = {"train": [], "valid": []}
        self.load_dataset(path, **kwargs)
        # Choose the template based on the model name
        # model_name = self.args.get('model_name', '').lower()
        model_name = getattr(self.args, "model_name", "").lower()
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
        print(f"Dataset path: {path}")
        try:
            if path and os.path.exists(path):
                print("Loading SST-2 dataset from local JSONL files...")
                train_samples = self.load_jsonl(os.path.join(path, "sst2_train.jsonl"))
                valid_samples = self.load_jsonl(os.path.join(path, "sst2_validation.jsonl"))
            else:
                print("Loading SST-2 dataset from Hugging Face...")
                dataset = load_dataset("glue", "sst2")
                train_samples = dataset["train"]
                valid_samples = dataset["validation"]

            # Build samples
            self.samples = {
                "train": [self.build_sample(example) for example in train_samples],
                "valid": [self.build_sample(example) for example in valid_samples],
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")

    # Newly added method
    def load_jsonl(self, file_path):
        """Load a JSONL file and return a list of dictionaries"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return [json.loads(line.strip()) for line in f]
        except Exception as e:
            raise RuntimeError(f"Unable to load JSONL file {file_path}: {e}")
        
    # For generative tasks, candidates are []
    def build_sample(self, example):
        try:
            label = int(example["label"])
            return Sample(
                id=example["idx"],
                data=example,
                correct_candidate=label,
                candidates=[0, 1],
            )
        except KeyError as e:
            raise ValueError(f"Sample data missing required field: {e}")
        
    def get_template(self, template_version=0):
        return {0: self.template}[template_version]()
        
    
class CopaDataset(Dataset):
    train_sep = "\n\n"
    mixed_set = False

    def __init__(self, subtask=None, path=None, **kwargs):
        """
        Initialize the dataset. Decide to load from Huggingface or local JSONL files.
        """
        self.samples = {"train": [], "valid": []}
        self.args = kwargs.get("args", {})
        self.load_dataset(path)

    def load_jsonl(self, file_path):
        """Load a JSONL file and return a list of dictionaries."""
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error in file {file_path}: {e}")
        print(f"Loaded {len(samples)} samples from {file_path}")
        return samples

    def load_dataset(self, path=None):
        """
        Load the dataset. If `path` is provided and valid, load from local JSONL files.
        Otherwise, load from Huggingface.
        """
        if path and os.path.exists(path):
            print(f"Loading COPA dataset from local JSONL files at: {path}")
            train_file = os.path.join(path, "copa_train.jsonl")
            valid_file = os.path.join(path, "copa_validation.jsonl")
            if not os.path.exists(train_file) or not os.path.exists(valid_file):
                raise FileNotFoundError("Train or validation JSONL file is missing. Check the path.")

            train_samples = self.load_jsonl(train_file)
            valid_samples = self.load_jsonl(valid_file)

        else:
            print("Loading COPA dataset from Huggingface...")
            dataset = load_dataset("super_glue", "copa")
            train_samples = dataset["train"]
            valid_samples = dataset["validation"]

        # Build samples
        self.samples = {
            "train": [self.build_sample(example) for example in train_samples],
            "valid": [self.build_sample(example) for example in valid_samples],
        }
        print(f"Training samples: {len(self.samples['train'])}")
        print(f"Validation samples: {len(self.samples['valid'])}")

    def build_sample(self, example):
        """Construct a sample."""
        try:
            sample = Sample(
                id=example["idx"],
                data=example,
                candidates=[example["choice1"], example["choice2"]],
                correct_candidate=example[f"choice{example['label'] + 1}"],
            )
            return sample
        except KeyError as e:
            print(f"Error building sample, missing key: {e}, example: {example}")
            return None

    def get_template(self, template_version=0):
        return {0: CopaTemplate}[template_version]()


class BoolQDataset(Dataset):
    def __init__(self, subtask=None, path=None, **kwargs) -> None:
        self.args = kwargs.get("args", {})
        self.samples = {"train": [], "valid": []}
        self.load_dataset(path)

        model_name = getattr(self.args, "model_name", "").lower()
        if 'llama' in model_name:
            self.template = BoolQTemplate
        else:
            self.template = BoolQTemplateV3

    def load_dataset(self, path=None):
        def load_jsonl(file_path):
            """Load JSONL file into a list of dicts"""
            samples = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    try:
                        sample = json.loads(line.strip())
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        print(f"[Line {i}] JSON decode error: {e}")
            return samples

        if path and os.path.exists(path):
            print(f"Loading BoolQ dataset from local path: {path}")
            train_data = load_jsonl(os.path.join(path, "boolq_train.jsonl"))
            valid_data = load_jsonl(os.path.join(path, "boolq_validation.jsonl"))
        else:
            print("Loading BoolQ dataset from Hugging Face...")
            dataset = load_dataset("boolq")
            train_data = dataset["train"]
            valid_data = dataset["validation"]

        self.samples = {
            "train": [s for s in (self.build_sample(ex) for ex in train_data) if s is not None],
            "valid": [s for s in (self.build_sample(ex) for ex in valid_data) if s is not None],
        }

    def build_sample(self, example):
        try:
            return Sample(
                data=example,
                candidates=["Yes", "No"],
                correct_candidate="Yes" if example["answer"] else "No",
            )
        except KeyError as e:
            print(f"Missing key in example: {e}")
            return None

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


class SampleCB:
    def __init__(self, data, candidates, correct_candidate):
        self.data = data  # Contains fields like 'premise', 'hypothesis', 'idx', 'label', etc.
        self.candidates = candidates  # List of possible label indices
        self.correct_candidate = correct_candidate  # Correct label index


class CBDataset(Dataset):
    def __init__(self, subtask=None, path=None, **kwargs):
        super().__init__(subtask, **kwargs)
        self.args = kwargs.get("args", {})
        self.samples = {"train": [], "valid": []}
        self.load_dataset(path)

    def load_jsonl(self, file_path):
        """Load a JSONL file and return a list of dictionaries"""
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    print(f"Warning: Line {line_num} in {file_path} is empty, skipped")
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"JSON decoding error at line {line_num} in {file_path}: {e}")
        print(f"Loaded {len(samples)} samples from {file_path}")
        return samples

    def load_dataset(self, path):
        """
        Load data based on the provided path.
        - If a valid path is provided, load from local JSONL files.
        - If the path is empty or invalid, load from Hugging Face Hub.
        """
        label_names = ["entailment", "contradiction", "neutral"]
        label_to_id = {name: idx for idx, name in enumerate(label_names)}

        if path and os.path.exists(path):
            print(f"Loading CB dataset from local JSONL files: {path}")
            train_file = os.path.join(path, "cb_train.jsonl")
            valid_file = os.path.join(path, "cb_validation.jsonl")
            if not os.path.exists(train_file) or not os.path.exists(valid_file):
                raise FileNotFoundError("Training or validation JSONL file not found, please check the path and filenames")

            train_samples = self.load_jsonl(train_file)
            valid_samples = self.load_jsonl(valid_file)

            print(f"Number of training samples loaded: {len(train_samples)}")
            print(f"Number of validation samples loaded: {len(valid_samples)}")

            # Process labels to ensure they are integer indices
            def process_labels(samples):
                label_names = ["entailment", "contradiction", "neutral"]
                label_to_id = {name: idx for idx, name in enumerate(label_names)}
                processed_samples = []
                for sample in samples:
                    label = sample.get('label')
                    if isinstance(label, str):
                        label_lower = label.lower()
                        label_variations = {
                            "entailment": "entailment",
                            "entail": "entailment",
                            "contradiction": "contradiction",
                            "contradict": "contradiction",
                            "neutral": "neutral",
                        }
                        if label_lower in label_variations:
                            sample['label'] = label_to_id[label_variations[label_lower]]
                        else:
                            print(f"Unknown label '{label}', sample skipped: {sample}")
                            continue  # Skip this sample
                    elif isinstance(label, int):
                        if 0 <= label < len(label_names):
                            sample['label'] = label  # Valid label index
                        else:
                            print(f"Invalid label index '{label}', sample skipped: {sample}")
                            continue  # Skip this sample
                    else:
                        print(f"Invalid label type '{type(label)}', sample skipped: {sample}")
                        continue  # Skip this sample
                    processed_samples.append(sample)
                return processed_samples

            train_samples = process_labels(train_samples)
            valid_samples = process_labels(valid_samples)

            print(f"Number of training samples after label processing: {len(train_samples)}")
            print(f"Number of validation samples after label processing: {len(valid_samples)}")

        else:
            print("Loading CB dataset from Hugging Face...")
            dataset = load_dataset("super_glue", "cb")
            train_samples = dataset["train"]
            valid_samples = dataset["validation"]

            print(f"Number of training samples loaded from Hugging Face: {len(train_samples)}")
            print(f"Number of validation samples loaded from Hugging Face: {len(valid_samples)}")

        # Build samples
        train_samples_built = [self.build_sample(example) for example in train_samples]
        valid_samples_built = [self.build_sample(example) for example in valid_samples]

        # Filter out samples that failed to build
        train_samples_built = [sample for sample in train_samples_built if sample is not None]
        valid_samples_built = [sample for sample in valid_samples_built if sample is not None]

        print(f"Number of training samples after building: {len(train_samples_built)}")
        print(f"Number of validation samples after building: {len(valid_samples_built)}")

        # Save samples
        self.samples = {"train": train_samples_built, "valid": valid_samples_built}

        # Output sample counts
        print(f"Training set sample count: {len(self.samples['train'])}")
        print(f"Validation set sample count: {len(self.samples['valid'])}")

        # Prevent empty training set
        if len(self.samples["train"]) == 0:
            raise ValueError("Training dataset is empty, please check the data path and file format")

    def build_sample(self, example):
        """Build a sample"""
        try:
            # Ensure the label is an integer index
            label = example['label']
            if isinstance(label, int):
                pass
            elif isinstance(label, str):
                label_names = ["entailment", "contradiction", "neutral"]
                label = label_names.index(label.lower())
            else:
                print(f"Invalid label type '{type(label)}', sample skipped: {example}")
                return None  # Skip this sample

            sample = SampleCB(
                data=example,
                candidates=[0, 1, 2],
                correct_candidate=label
            )
            return sample
        except Exception as e:
            print(f"Error building sample, sample skipped: {example}\nError message: {e}")
            return None  # Skip problematic sample

    def get_template(self, template_version=0):
        templates = {0: CBTemplate}  # Assume you have a CBTemplate class defined
        return templates[template_version]()


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
    
    def __init__(self, subtask=None, path=None, **kwargs):
        """
        Initialize the dataset. Load from Huggingface or local JSONL files.
        """
        self.args = kwargs.get("args", {})
        self.samples = {"train": [], "valid": []}
        self.load_dataset(path)

    def load_jsonl(self, file_path):
        """Load a JSONL file and return a list of dictionaries."""
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error in file {file_path}: {e}")
        print(f"Loaded {len(samples)} samples from {file_path}")
        return samples

    def load_dataset(self, path=None):
        """
        Load the dataset. If `path` is provided and valid, load from local JSONL files.
        Otherwise, load from Huggingface.
        """
        if path and os.path.exists(path):
            print(f"Loading WSC dataset from local JSONL files at: {path}")
            train_file = os.path.join(path, "wsc_train.jsonl")
            valid_file = os.path.join(path, "wsc_validation.jsonl")
            if not os.path.exists(train_file) or not os.path.exists(valid_file):
                raise FileNotFoundError("Train or validation JSONL file is missing. Check the path.")

            train_samples = self.load_jsonl(train_file)
            valid_samples = self.load_jsonl(valid_file)

        else:
            print("Loading WSC dataset from Huggingface...")
            dataset = load_dataset("super_glue", "wsc.fixed")
            train_samples = dataset["train"]
            valid_samples = dataset["validation"]

        # Build samples
        self.samples = {
            "train": [self.build_sample(example) for example in train_samples],
            "valid": [self.build_sample(example) for example in valid_samples],
        }
        print(f"Training samples: {len(self.samples['train'])}")
        print(f"Validation samples: {len(self.samples['valid'])}")

    def build_sample(self, example):
        """Construct a sample."""
        try:
            sample = Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example["label"]
            )
            return sample
        except KeyError as e:
            print(f"Error building sample, missing key: {e}, example: {example}")
            return None

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


import os
import json
from datasets import load_dataset

class RTEDataset(Dataset):
    def __init__(self, subtask=None, path=None, **kwargs) -> None:
        self.args = kwargs.get("args", {})
        self.samples = {"train": [], "valid": []}
        self.load_dataset(path)

        model_name = getattr(self.args, "model_name", "").lower()
        if 'llama' in model_name:
            self.template = RTE_Llama2Template
        elif 'mistral' in model_name:
            self.template = RTETemplate
        else:
            self.template = RTETemplate

    def load_dataset(self, path=None):
        def load_jsonl(file_path):
            samples = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        print(f"[JSON decode error] Line {line_num}: {e}")
            return samples

        if path and os.path.exists(path):
            print(f"Loading RTE dataset from local JSONL: {path}")
            train_path = os.path.join(path, "rte_train.jsonl")
            valid_path = os.path.join(path, "rte_validation.jsonl")

            train_data = load_jsonl(train_path)
            valid_data = load_jsonl(valid_path)
        else:
            print("Loading RTE dataset from Hugging Face...")
            dataset = load_dataset("super_glue", "rte")
            train_data = dataset["train"]
            valid_data = dataset["validation"]

        self.samples["train"] = [self.build_sample(example) for example in train_data if self._valid_sample(example)]
        self.samples["valid"] = [self.build_sample(example) for example in valid_data if self._valid_sample(example)]

    def _valid_sample(self, example):
        return "label" in example and example["label"] in [0, 1]

    def build_sample(self, example):
        return Sample(
            data=example,
            candidates=[0, 1],
            correct_candidate=example["label"]
        )

    def get_template(self, template_version=0):
        return {0: self.template}[template_version]()

 
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


class ArcC_ClozeDataset(Dataset):  # Output answer text
    def __init__(self, path=None, **kwargs) -> None:
        super().__init__()
        self.samples = {"train": [], "valid": []}
        self.load_dataset(path, **kwargs)

    def load_dataset(self, path, **kwargs):
        if path and os.path.isdir(path):  # Load from local JSONL
            print(f"Loading ARC-Challenge from local path: {path}")
            self.samples["train"] = self.load_jsonl(os.path.join(path, "ARC-Challenge_train.jsonl"))
            self.samples["valid"] = self.load_jsonl(os.path.join(path, "ARC-Challenge_validation.jsonl"))
        else:  # Load from Hugging Face dataset
            print("Loading ARC-Challenge from Hugging Face...")
            train_set = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='train')
            valid_set = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='validation')
            self.samples["train"] = [self.build_sample(example) for example in train_set]
            self.samples["valid"] = [self.build_sample(example) for example in valid_set]

    def load_jsonl(self, file_path):
        samples = []
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    example = json.loads(line.strip())
                    samples.append(self.build_sample(example))
        return samples

    def build_sample(self, example):
        """Ensure returning Sample object instead of dict"""
        return Sample(
            id=example['id'],
            data=example,
            candidates=example['choices']['text'],  # Candidate texts
            correct_candidate=example['choices']['text'][example['choices']['label'].index(example['answerKey'])],
        )

    def get_template(self, template_version=0):
        return Arc_ClozeTemplate() if template_version == 0 else NotImplementedError()


class ArcC_MCDataset(Dataset):  # Output answer label
    def __init__(self, path=None, **kwargs) -> None:
        super().__init__()
        self.samples = {"train": [], "valid": []}
        self.load_dataset(path, **kwargs)

    def load_dataset(self, path, **kwargs):
        if path and os.path.isdir(path):  # Load from local JSONL data
            print(f"Loading ARC-Challenge from local path: {path}")
            self.samples["train"] = self.load_jsonl(os.path.join(path, "ARC-Challenge_train.jsonl"))
            self.samples["valid"] = self.load_jsonl(os.path.join(path, "ARC-Challenge_validation.jsonl"))
        else:  # Load from Hugging Face
            print("Loading ARC-Challenge from Hugging Face...")
            train_set = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='train')
            valid_set = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='validation')
            self.samples["train"] = [self.build_sample(example) for example in train_set]
            self.samples["valid"] = [self.build_sample(example) for example in valid_set]

    def load_jsonl(self, file_path):
        samples = []
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    example = json.loads(line.strip())
                    samples.append(self.build_sample(example))
        return samples

    def build_sample(self, example):
        """Ensure returning Sample object instead of dict"""
        mcf_string = "\n".join([
            f"({l}) {example['choices']['text'][i]}"
            for i, l in enumerate(example['choices']['label'])
        ])
        question_string = f"Question: {example['question']}\n{mcf_string}"

        return Sample(
            id=example['id'],
            data=example,
            question_string=question_string,  # Key: formatted question string for multiple choice
            candidates=example['choices']['label'],  # ["A", "B", "C", "D"]
            correct_candidate=example['answerKey']
        )

    def get_template(self, template_version=0):
        return Arc_MCTemplate() if template_version == 0 else NotImplementedError()

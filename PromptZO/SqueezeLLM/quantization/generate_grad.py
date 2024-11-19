from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import numpy as np
import torch
import transformers

from tqdm import tqdm

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    traindata.with_format('torch')
    testdata.with_format('torch')

    from transformers import AutoTokenizer 
    
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    from transformers import AutoTokenizer 
    
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
    
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', streaming=True, download_mode="force_redownload")
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', streaming=True, download_mode="force_redownload")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        if trainenc.input_ids.shape[1] - seqlen - 1 < 0:
            i = 0
        else:
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        if tmp.input_ids.shape[1] - seqlen - 1 <= 0:
            i = 0
        else:
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 



def get_ptb_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )
    traindata.with_format('torch')
    valdata.with_format('torch')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc
def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model=''
):
    if 'wiki2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    #save_grad_path: str = field(
    #    metadata={"help": "Path to save the gradients"}
    #)


@dataclass
class DataArguments:
    dataset: str = field(default="c4_new")
    num_examples: int = field(default=100, metadata={"help": "Number of calibration examples"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg



def get_modules(layer, model_type):
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
    elif model_type == 'roberta':
        return [
            layer.attention.self.query,
            layer.attention.self.key,
            layer.attention.self.value,
            layer.attention.output.dense,
            layer.intermediate.dense,
            layer.output.dense,
        ]
    else:
        raise NotImplementedError()


def parse_model_type(model_name_or_path):
    model_name_or_path = model_name_or_path.lower()
    if 'opt' in model_name_or_path:
        return 'opt'
    elif 'llama' in model_name_or_path:
        return 'llama'
    elif 'vicuna' in model_name_or_path:
        return 'llama'
    elif 'mistral' in model_name_or_path:
        return 'mistral'
    elif 'roberta' in model_name_or_path:
        return 'roberta'
    else:
        raise NotImplementedError()


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if data_args.dataset in ["c4_new", "c4", 'ptb', 'wiki2']:
        dataloader, testloader = get_loaders(data_args.dataset,  model=model_args.model_name_or_path, seqlen=512,
                                            nsamples=data_args.num_examples)
    else:
        raise NotImplementedError("Please define your own dataset here")

    model_type = parse_model_type(model_args.model_name_or_path)
    if model_type in ['roberta']:
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            device_map='auto',
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True
        )
        data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            device_map='auto',
            torch_dtype=torch.bfloat16,
        )
    
    if model_type in ['mistral', 'llama']:
        _model = model.model
        _layers = _model.layers
    elif model_type in ['opt']:     
        _model = model.model  
        _layers = _model.decoder.layers
    elif model_type == 'roberta':
        _model = model.roberta
        _layers = _model.encoder.layer
    else:
        raise NotImplementedError()

    for data in tqdm(dataloader):
        data = data[0]
        if model_type == 'roberta':
            batch_dict = data_collator(data.tolist())
            batch_dict = {k : v.cuda() for k, v in batch_dict.items()}
            outputs = model(**batch_dict)
        else:
            x = data.cuda()
            outputs = model(input_ids=x, labels=x)
        loss = outputs.loss
        loss /= len(dataloader)
        loss.backward()

    # This is a hacky solution to save the gradients
    # where we overwrite all the weights in the model as the gradients
    # and use HF save_pretrained
    for layer in _layers:
        for module in get_modules(layer, model_type):
            module.weight.data = module.weight.grad

    print(f"saving model gradient at {training_args.output_dir}")
    if model_type == 'roberta':
        # we don't save the whole roberta due to the LM head weight sharing. 
        model.roberta.save_pretrained(training_args.output_dir)
    else:
        model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()

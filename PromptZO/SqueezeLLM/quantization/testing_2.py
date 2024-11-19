import warnings
warnings.filterwarnings('ignore')
import torch
import transformers
from squeezellm.quant import *
from squeezellm.modelutils import find_layers
from squeezellm.datautils import get_loaders
from transformers import AutoModelForCausalLM
import copy
from tqdm.auto import tqdm


@torch.no_grad()
def load_quant(model, checkpoint, wbits, include_sparse, topX, fake_quant=False):
    from transformers import AutoConfig, AutoModelForCausalLM

    target_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=target_dtype,
    )
    layers = find_layers(model)

    state_dict_tmp = torch.load(checkpoint)
    state_dict = state_dict_tmp
    for k, v in state_dict_tmp.items():
        if not isinstance(v, torch.Tensor) or \
                v.dtype == target_dtype or \
                v.dtype not in [torch.float16, torch.float32, torch.float64] or \
                    'vals' in k:
            state_dict[k] = v
        else:
            state_dict[k] = v.to(target_dtype)
    
    # load sparse thresholds from checkpoint
    if include_sparse:
        num_outlier_vals = {}
        outlier_sparse_buffer_dict = {}
        for k, v in state_dict.items():
            if "outlier_sparse_threshold." in k and v is not None:
                key = k.replace("outlier_sparse_threshold.", "")
                num_outlier_vals[key] = v
                outlier_sparse_buffer_dict[key] = \
                    torch.sparse_csr_tensor(
                        state_dict[key + '.outlier_rows'],
                        state_dict[key + '.outlier_cols'],
                        state_dict[key + '.outlier_vals'],
                        size=tuple(state_dict[key + '.dequantized_weight'].shape)
                    )

        for k, v in num_outlier_vals.items():
            del state_dict["outlier_sparse_threshold." + k]
        
        if len(num_outlier_vals) == 0:
            num_outlier_vals = None

        num_sensitive_vals = {}
        sensitive_sparse_buffer_dict = {}
        for k, v in state_dict.items():
            if "sensitive_sparse_threshold." in k:
                key = k.replace("sensitive_sparse_threshold.", "")
                num_sensitive_vals[key] = v
                sensitive_sparse_buffer_dict[key] = \
                    torch.sparse_csr_tensor(
                        state_dict[key + '.sensitive_rows'],
                        state_dict[key + '.sensitive_cols'],
                        state_dict[key + '.sensitive_vals'],
                        size=tuple(state_dict[key + '.dequantized_weight'].shape)
                    )
        
        for k, v in num_sensitive_vals.items():
            del state_dict["sensitive_sparse_threshold." + k]

        if len(num_sensitive_vals) == 0:
            num_sensitive_vals = None

        if fake_quant:
            if len(sensitive_sparse_buffer_dict) > 0:
                for k, v in sensitive_sparse_buffer_dict.items():
                    if k in outlier_sparse_buffer_dict:
                        sparse_buffers = v + outlier_sparse_buffer_dict[k]
                    else:
                        sparse_buffers = v
                    dequantized_weight = state_dict[k + '.dequantized_weight']
                    state_dict[k + '.dequantized_weight'] = dequantized_weight + sparse_buffers
                    state_dict[k + '.sensitive_indices'] = v.to_sparse_coo().indices()            

            elif len(outlier_sparse_buffer_dict) > 0:
                for k, v in outlier_sparse_buffer_dict.items():
                    sparse_buffers = v.clone()
                    dequantized_weight = state_dict[k + '.dequantized_weight']
                    state_dict[k + '.dequantized_weight'] = dequantized_weight + sparse_buffers


    else:
        num_outlier_vals = None
        num_sensitive_vals = None

    # replace layers
    for name in ["lm_head"]:
        if name in layers:
            del layers[name]
    make_quant_lut(
        model, 
        layers, 
        wbits, 
        include_sparse=include_sparse, 
        num_outlier_vals=num_outlier_vals, 
        num_sensitive_vals=num_sensitive_vals, 
        topX=topX, 
        sparse_dtype=target_dtype,
        fake_quant=fake_quant
    )
    del layers
    
    print("Loading SqueezeLLM model ...")
    model.load_state_dict(state_dict, strict=False)
    print("Done.")
    return model


@torch.inference_mode()
def eval(model, testenc, seqlen=2048): # testenc: [1, total_test_length], dev: torch.device('cuda')
    losses = []
    nsamples = testenc.input_ids.numel() // seqlen
    model.config.use_cache = False

    testenc.input_ids = testenc.input_ids[:, :nsamples * seqlen].reshape(nsamples, seqlen)
    for batch in tqdm(torch.utils.data.DataLoader(torch.arange(nsamples), batch_size=4)):
        batch = testenc.input_ids[batch].cuda()
        output = model(input_ids=batch, labels=batch)
        loss = output.loss
        for _ in batch:
            losses.append(loss)

    losses = torch.as_tensor(losses)
    eval_loss = torch.mean(losses)
    perplexity = math.exp(eval_loss)

    return perplexity


# model_name = 'NousResearch/Llama-2-7b-hf'
# model_name ='facebook/opt-6.7b'
model_name ='facebook/opt-1.3b'
unquantized_model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)
# unquantized_model.eval()

torch.manual_seed(0)

dataloader, testloader = get_loaders(
    'wikitext2',
    seed=0,
    model=model_name,
    seqlen=512,
)


ret = []
with torch.no_grad():
    # unquantized_val = eval(unquantized_model, testloader)
    # print(unquantized_val)
    # ret.append(unquantized_val)

    # model = load_quant('NousResearch/Llama-2-7b-hf', f'/share/desa/nfs02/wg247/SqueezeLLM/nuq/llama2-7b/3bit.pt', 3, True, 0, fake_quant=True).cuda()
    # model = load_quant('facebook/opt-6.7b', f'/share/desa/nfs02/wg247/SqueezeLLM/nuq/opt-6.7b/3bit.pt', 3, True, 0, fake_quant=True).cuda()
    # model = load_quant('facebook/opt-1.3b', f'/share/desa/nfs02/wg247/SqueezeLLM/model/opt-1.3b/3bit.pt', 3, True, 0, fake_quant=True).cuda()
    # model.eval()
    # ref_quantized_val = eval(model, testloader)
    # print(ref_quantized_val)
    # ret.append(ref_quantized_val)

    # del model
    for i in ['1e-3', '5e-3']:
    # for i in ['2e-4', '5e-4', '1e-3', '2e-3', '5e-3', '1e-2', '2e-2']:
        # model = load_quant('NousResearch/Llama-2-7b-hf', f'/share/desa/nfs02/wg247/SqueezeLLM/nuq/llama2-7b/3-bit-0-{i}.pt', 3, True, 0, fake_quant=True).cuda()

        model = load_quant('facebook/opt-1.3b', f'/share/desa/nfs02/wg247/SqueezeLLM/nuq/opt-1.3b/3-bit-0-{i}.pt', 3, True, 0, fake_quant=True).cuda()
        # model = load_quant('facebook/opt-6.7b', f'/share/desa/nfs02/wg247/SqueezeLLM/nuq/opt-6.7b/3-bit-0-{i}.pt', 3, True, 0, fake_quant=True).cuda()
        model.eval()

        quantized_val = eval(model, testloader)
        ret.append(quantized_val)
        print(i)
        print(quantized_val)
        print()
        
    print(ret)
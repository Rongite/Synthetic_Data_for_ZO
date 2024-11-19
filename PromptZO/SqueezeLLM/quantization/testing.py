import torch
import transformers
from squeezellm.quant import *
from squeezellm.modelutils import find_layers
import copy


@torch.no_grad()
def load_quant(model, checkpoint, wbits, include_sparse, topX):
    from transformers import AutoConfig, AutoModelForCausalLM

    target_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model,
        device_map='auto',
        torch_dtype=target_dtype,
    )
    model = model.eval()
    layers = find_layers(model)

    state_dict_tmp = torch.load(checkpoint)
    state_dict = state_dict_tmp
    for k, v in state_dict_tmp.items():
        if not isinstance(v, torch.Tensor) or \
                v.dtype == target_dtype or \
                v.dtype not in [torch.float16, torch.float32, torch.float64]:
            state_dict[k] = v
        else:
            state_dict[k] = v.to(target_dtype)
    
    # load sparse thresholds from checkpoint
    if include_sparse:
        num_outlier_vals = {}
        outlier_sparse_buffer_dict = {}
        for k, v in state_dict.items():
            if "outlier_sparse_threshold." in k:
                key = k.replace("outlier_sparse_threshold.", "")
                num_outlier_vals[key] = v
                outlier_sparse_buffer_dict[key + '.outlier_sparse_buffers_trans'] = \
                    torch.sparse_csr_tensor(
                        state_dict[key + '.outlier_rows'],
                        state_dict[key + '.outlier_cols'],
                        state_dict[key + '.outlier_vals'],
                        size=tuple(state_dict[key + '.dequantized_weight'].shape)
                    ).transpose(1, 0).to_sparse_csr()

        for k, v in num_outlier_vals.items():
            del state_dict["outlier_sparse_threshold." + k]
        
        state_dict |= outlier_sparse_buffer_dict

        num_sensitive_vals = {}
        sensitive_sparse_buffer_dict = {}
        for k, v in state_dict.items():
            if "sensitive_sparse_threshold." in k:
                key = k.replace("sensitive_sparse_threshold.", "")
                num_sensitive_vals[key] = v
                sensitive_sparse_buffer_dict[key + '.sensitive_sparse_buffers_trans'] = \
                    torch.sparse_csr_tensor(
                        state_dict[key + '.sensitive_rows'],
                        state_dict[key + '.sensitive_cols'],
                        state_dict[key + '.sensitive_vals'],
                        size=tuple(state_dict[key + '.dequantized_weight'].shape)
                    ).transpose(1, 0).to_sparse_csr()
        
        for k, v in num_sensitive_vals.items():
            del state_dict["sensitive_sparse_threshold." + k]

        state_dict |= sensitive_sparse_buffer_dict

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
        sparse_dtype=target_dtype
    )
    del layers
    
    print("Loading SqueezeLLM model ...")
    model.load_state_dict(state_dict, strict=False)
    print("Done.")
    return model


torch.manual_seed(0)
model = load_quant('facebook/opt-1.3b', '/share/desa/nfs02/wg247/SqueezeLLM/model/opt-1.3b/4bit.pt', 4, True, 0).cuda()
model.eval()

unquantized_model = transformers.AutoModel.from_pretrained('facebook/opt-1.3b', device_map='auto', torch_dtype=torch.bfloat16)
unquantized_model.eval()


with torch.no_grad():
    X = torch.randn(8, 16, 2048).cuda().bfloat16()
    old_res = model.model.decoder.layers[0].self_attn.q_proj.forward(X)
    vanilla_res = unquantized_model.decoder.layers[0].self_attn.q_proj(X)
    print((old_res - vanilla_res).norm())

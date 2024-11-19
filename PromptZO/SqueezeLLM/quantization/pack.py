import json
import os
import pickle
import time

import torch
import torch.nn as nn
import transformers
from squeezellm.modelutils import *
from squeezellm.quant import *


@torch.no_grad()
def opt_sequential(model, folder, include_sparse):
    print("Starting ...")

    layers = model.model.decoder.layers
    quantizers = {}
    for i in range(len(layers)):
        with open(f"{folder}/lut/l{i}.pkl", "rb") as f:
            # dictionary: key ["q", "k", "v", "o", "fc1", "fc2"] -> list of length channel,
            # each of which is a list of #group lists, (here, it's always 1)
            # and each of them are a tuple (centroids, indices)
            lut_layer = pickle.load(f)

        if include_sparse:
            with open(f"{folder}/outliers/l{i}.pkl", "rb") as f:
                # dictionary: key ["q", "k", "v", "o", "gate", "up", "down"] -> list of length channel,
                # each of which is a list of #group lists, (here, it's always 1)
                # and each of them are a tuple (centroids, indices)
                sensitive_outlier_weights, sensitive_grads, outlier_weights = pickle.load(f)

        sequential_lut = ["q", "k", "v", "o", "up", "down"]
        sequential_lut_real_name = {
            "q": "self_attn.q_proj",
            "k": "self_attn.k_proj",
            "v": "self_attn.v_proj",
            "o": "self_attn.out_proj",
            "up": "fc1",
            "down": "fc2",
        }

        for s in sequential_lut:
            lut = lut_layer[s]

            if sensitive_outlier_weights is not None:
                sensitive_weight = sensitive_outlier_weights[s]
                sensitive_grad = sensitive_grads[s]
            else:
                sensitive_weight = None
                sensitive_grad = None

            if outlier_weights is not None:
                outlier_weight = outlier_weights[s]
            else:
                outlier_weight = None

            name = sequential_lut_real_name[s]
            # lut: 2048 rows,
            # lut[0][0] = (2 ** bits values, exact quantized values)
            quantizers["model.decoder.layers.%d.%s" % (i, name)] = [lut, sensitive_weight, sensitive_grad, outlier_weight]

    return quantizers

@torch.no_grad()
def llama_sequential(model, folder, include_sparse):
    print("Starting ...")

    layers = model.model.layers

    quantizers = {}
    for i in range(len(layers)):
        with open(f"{folder}/lut/l{i}.pkl", "rb") as f:
            # dictionary: key ["q", "k", "v", "o", "gate", "up", "down"] -> list of length channel,
            # each of which is a list of #group lists, (here, it's always 1)
            # and each of them are a tuple (centroids, indices)
            lut_layer = pickle.load(f)

        if include_sparse:
            with open(f"{folder}/outliers/l{i}.pkl", "rb") as f:
                # dictionary: key ["q", "k", "v", "o", "gate", "up", "down"] -> list of length channel,
                # each of which is a list of #group lists, (here, it's always 1)
                # and each of them are a tuple (centroids, indices)
                sensitive_outlier_weights, sensitive_grads, outlier_weights = pickle.load(f)

        sequential_lut = ["q", "k", "v", "o", "gate", "up", "down"]
        sequential_lut_real_name = {
            "q": "self_attn.q_proj",
            "k": "self_attn.k_proj",
            "v": "self_attn.v_proj",
            "o": "self_attn.o_proj",
            "gate": "mlp.gate_proj",
            "up": "mlp.up_proj",
            "down": "mlp.down_proj",
        }

        for s in sequential_lut:
            lut = lut_layer[s]

            if sensitive_outlier_weights is not None:
                sensitive_weight = sensitive_outlier_weights[s]
                sensitive_grad = sensitive_grads[s]
            else:
                sensitive_weight = None
                sensitive_grad = None
            
            if outlier_weights is not None:
                outlier_weight = outlier_weights[s]
            else:
                outlier_weight = None

            name = sequential_lut_real_name[s]
            
            quantizers["model.layers.%d.%s" % (i, name)] = [lut, sensitive_weight, sensitive_grad, outlier_weight]

    return quantizers


@torch.no_grad()
def roberta_sequential(model, folder, include_sparse):
    print("Starting ...")

    layers = model.roberta.encoder.layer

    quantizers = {}
    for i in range(len(layers)):
        with open(f"{folder}/lut/l{i}.pkl", "rb") as f:
            lut_layer = pickle.load(f)

        if include_sparse:
            with open(f"{folder}/outliers/l{i}.pkl", "rb") as f:
                sensitive_outlier_weights, sensitive_grads, outlier_weights = pickle.load(f)

        sequential_lut = ["q", "k", "v", "o", "up", "down"]
        sequential_lut_real_name = {
            "q": "attention.self.query",
            "k": "attention.self.key",
            "v": "attention.self.value",
            "o": "attention.output.dense",
            "up": "intermediate.dense",
            "down": "output.dense",
        }

        for s in sequential_lut:
            lut = lut_layer[s]

            if sensitive_outlier_weights is not None:
                sensitive_weight = sensitive_outlier_weights[s]
                sensitive_grad = sensitive_grads[s]
            else:
                sensitive_weight = None
                sensitive_grad = None
            
            if outlier_weights is not None:
                outlier_weight = outlier_weights[s]
            else:
                outlier_weight = None

            name = sequential_lut_real_name[s]
            
            quantizers["roberta.encoder.layer.%d.%s" % (i, name)] = [lut, sensitive_weight, sensitive_grad, outlier_weight]

    return quantizers


def pack(
    model,
    quantizers,
    wbits,
    include_sparse,
    balanced,
    num_nonzero_per_thread,
):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant_lut(
        model,
        quantizers,
        wbits,
        include_sparse=include_sparse,
        balanced=balanced,
    )

    qlayers = find_layers(model, [QuantLinearLUT])
    print("Packing ...")
    sparsedict = {}

    for name in qlayers:
        print(name)
        lookup_table = quantizers[name]
        layers[name].cpu()
        qlayers[name].pack2(
            layers[name],
            lookup_table,
            include_sparse,
            num_nonzero_per_thread=num_nonzero_per_thread,
        )
        if include_sparse:
            sparse_val_pair = ()
            if hasattr(qlayers[name], 'outlier_vals'):
                sparse_val_pair = sparse_val_pair + (qlayers[name].outlier_vals.shape[-1],)
            else:
                sparse_val_pair = sparse_val_pair + (None,)

            if hasattr(qlayers[name], 'sensitive_vals'):
                sparse_val_pair = sparse_val_pair + (qlayers[name].sensitive_vals.shape[-1],)
            else:
                sparse_val_pair = sparse_val_pair + (None,)

            sparsedict[name] = sparse_val_pair

    print("Done.")
    return model, sparsedict


if __name__ == "__main__":
    import argparse

    from squeezellm.datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="llama model to load")
    parser.add_argument(
        "--wbits",
        type=int,
        default=16,
        choices=[3, 4, 16],
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument(
        "--save",
        type=str,
        required=True,
        help="Save quantized checkpoint under this name.",
    )

    # sparse args
    parser.add_argument(
        "--folder",
        type=str,
        default="",
        help="Path to folder containing luts and outliers.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=['opt', 'llama', 'mistral', 'roberta'],
    )
    parser.add_argument(
        "--include_sparse",
        action='store_true',
    )

    #balanced kernel arguments
    parser.add_argument(
        '--balanced', action='store_true',
        help='Whether to use balanced sparse kernel.'
    )
    parser.add_argument(
        '--num_nonzero_per_thread', type=int, default=10,
        help='Num nonzeros assigned to each thread.'
    )


    args = parser.parse_args()
    if args.model_type == 'roberta':
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype="auto"
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype="auto"
        )
    model.eval()

    print("Running sequential")
    tick = time.time()
    if args.model_type == 'opt':
        quantizers = opt_sequential(
            model=model,
            folder=args.folder,
            include_sparse=args.include_sparse,
        )
    elif args.model_type in ['llama', 'mistral']:
        quantizers = llama_sequential(
            model=model,
            folder=args.folder,
            include_sparse=args.include_sparse,
        )
    elif args.model_type == 'roberta':
        quantizers = roberta_sequential(
            model=model,
            folder=args.folder,
            include_sparse=args.include_sparse,
        )
    else:
        raise NotImplementedError()
    
    print("Sequential done:", time.time() - tick)

    print("Running pack")
    tick = time.time()
    model, num_sparse_vals = pack(
        model=model,
        quantizers=quantizers,
        wbits=args.wbits,
        include_sparse=args.include_sparse,
        balanced=args.balanced,
        num_nonzero_per_thread=args.num_nonzero_per_thread,
    )
    print("packing done:", time.time() - tick)

    model_dict = model.state_dict()

    if args.include_sparse:
        # need to merge in sparse dict
        for k, (v1, v2) in num_sparse_vals.items():
            model_dict["outlier_sparse_threshold." + k] = v1
            model_dict["sensitive_sparse_threshold." + k] = v2

    # save model
    torch.save(model_dict, args.save)

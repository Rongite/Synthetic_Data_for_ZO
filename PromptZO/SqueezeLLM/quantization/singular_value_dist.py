import argparse
import os
import pickle

import numpy as np
import torch
from squeezellm.model_parse import get_module_names, parse_model
from squeezellm.outliers import remove_outliers
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, help="model weights to load", required=True)
parser.add_argument(
    "--model_type", type=str, default='opt', help="model type", choices=["llama", "opt"]
)
parser.add_argument(
    "--gradient", type=str, help="model gradients to load", required=True
)
parser.add_argument(
    "--range", type=str, default=None, help="range of layers to quantize"
)
parser.add_argument(
    "--output_folder", type=str, required=None, help="path to dump the output"
)
parser.add_argument(
    "--outlier_percentage",
    type=float,
    default=0,
    help="outlier percentage to remove",
)
parser.add_argument(
    "--sensitivity", type=float, default=0, help="sensitivity for outlier extraction"
)


def effective_rank(S):
    p = S / S.sum()
    return - (p * torch.log(p)).sum()

if __name__ == "__main__":
    args = parser.parse_args()

    # if model type is not explicitly given, infer from the model name
    model_type = args.model_type or parse_model(args.model)

    # Run as a outlier extraction mode if outlier config is given or sensitivity is non-zero
    is_outlier_mode = args.outlier_percentage > 0 or args.sensitivity > 0

    lut_folder = f"{args.output_folder}/lut"

    if args.range:
        ranges = args.range.split(",")
        ranges = [int(r) for r in ranges]
        ran = ranges
    else:
        # Count number of layers based on the chunk item count in the model folder
        # You should not add/delete anything in the folder to make this work
        nlayers = len([f for f in os.listdir(args.model)])
        ran = list(range(nlayers))

    print(ran)
    outlier_folder = f"{args.output_folder}/outliers"

    singular_values_dist = dict()
    singular_values_effective_rank = dict()

    for l in ran:
        lut_file_name = f"{lut_folder}/l{l}.pkl"
        outlier_file_name = f"{outlier_folder}/l{l}.pkl"

        gradient_layer = torch.load(f"{args.gradient}/layer_{l}.pt")
        model_layer = torch.load(f"{args.model}/layer_{l}.pt")

        if is_outlier_mode:
            print(
                f"Removing outliers percentage={args.outlier_percentage}, sensitivity={args.sensitivity}"
            )
            remove_outliers(
                model=model_layer,
                sensitivity=args.sensitivity,
                outlier_percentage=args.outlier_percentage,
                gradients=gradient_layer,
            )


            for name in tqdm(get_module_names(model_type)):
                module_weight = model_layer[name].float()

                _, S, _ = torch.svd(module_weight)
                singular_values_dist[f'{l}-{name}'] = S

                eff_rank = effective_rank(S)
                singular_values_effective_rank[f'{l}-{name}'] = eff_rank
                print(f'Effective rank {l}-{name}: {eff_rank.item():.3f}')
                
                del module_weight

        
        else:

            for name in tqdm(get_module_names(model_type)):
                module_weight = model_layer[name].float()

                _, S, _ = torch.svd(module_weight)
                singular_values_dist[f'{l}-{name}'] = S

                eff_rank = effective_rank(S)
                singular_values_effective_rank[f'{l}-{name}'] = eff_rank 
                print(f'Effective rank {l}-{name}: {eff_rank.item():.3f}')
                
                del module_weight
    

    if is_outlier_mode:
        torch.save(
            (singular_values_dist, singular_values_effective_rank), 
            f'{args.output_folder}{os.sep}{args.sensitivity}-{args.outlier_percentage}-singular-value-dist.pt'
        )
    else:
        torch.save(
            (singular_values_dist, singular_values_effective_rank), 
            f'{args.output_folder}{os.sep}vanilla-singular-value-dist.pt'
        )
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

    # row_sum = dict()
    # column_sum = dict()
    # row_by_128_sum = dict()
    # column_by_128_sum = dict()
    # block_128_sum = dict()
    row_sum_percentage = dict()
    column_sum_percentage = dict()
    row_by_128_sum_percentage = dict()
    column_by_128_sum_percentage = dict()
    block_128_sum_percentage = dict()

    for l in ran:
        lut_file_name = f"{lut_folder}/l{l}.pkl"
        outlier_file_name = f"{outlier_folder}/l{l}.pkl"

        gradient_layer = torch.load(f"{args.gradient}/layer_{l}.pt")
        model_layer = torch.load(f"{args.model}/layer_{l}.pt")


        print(
            f"Removing outliers percentage={args.outlier_percentage}, sensitivity={args.sensitivity}"
        )

        remove_outliers(
            model=model_layer,
            sensitivity=args.sensitivity,
            outlier_percentage=args.outlier_percentage,
            gradients=gradient_layer,
        )

    #     for name in tqdm(get_module_names(model_type)):
    #         module_weight = model_layer[name]
    #         nrows, ncols = module_weight.shape

    #         column_sum[f'{l}-{name}'] = (module_weight == 0).sum(dim=1) / ncols
    #         row_sum[f'{l}-{name}'] = (module_weight == 0).sum(dim=0) / nrows
            
    #         column_by_128_sum[f'{l}-{name}'] = (module_weight.reshape(nrows, ncols // 128, 128) == 0).sum(dim=-1) / 128
    #         row_by_128_sum[f'{l}-{name}'] = (module_weight.reshape(nrows // 128, 128, ncols) == 0).sum(dim=1) / 128
            
    #         block_128_sum[f'{l}-{name}'] = (module_weight.reshape(nrows // 128, ncols // 128, 128, 128) == 0).sum(dim=(-2, -1)) / (128 ** 2)


    # torch.save(
    #     (column_sum, row_sum, column_by_128_sum, row_by_128_sum, block_128_sum),
    #     f'{args.output_folder}{os.sep}{args.sensitivity}-{args.outlier_percentage}-column-sum-dist.pt'
    # )

        for name in tqdm(get_module_names(model_type)):
            module_weight = model_layer[name]
            nrows, ncols = module_weight.shape

            zero_weight = (module_weight == 0)
            nnz = zero_weight.sum()

            column_sum_percentage[f'{l}-{name}'] = zero_weight.sum(dim=1) / nnz
            row_sum_percentage[f'{l}-{name}'] = zero_weight.sum(dim=0) / nnz
            
            column_by_128_sum_percentage[f'{l}-{name}'] = (module_weight.reshape(nrows, ncols // 128, 128) == 0).sum(dim=-1) / nnz
            row_by_128_sum_percentage[f'{l}-{name}'] = (module_weight.reshape(nrows // 128, 128, ncols) == 0).sum(dim=1) / nnz
            
            block_128_sum_percentage[f'{l}-{name}'] = (module_weight.reshape(nrows // 128, ncols // 128, 128, 128) == 0).sum(dim=(-2, -1)) / nnz


    torch.save(
        (column_sum_percentage, row_sum_percentage, column_by_128_sum_percentage, row_by_128_sum_percentage, block_128_sum_percentage),
        f'{args.output_folder}{os.sep}{args.sensitivity}-{args.outlier_percentage}-nnz-percentage-dist.pt'
    )
import argparse
import json
import os
import pickle

import numpy as np
import torch
from sklearn.cluster import KMeans
from squeezellm.model_parse import get_module_names, parse_model
from squeezellm.outliers import remove_outliers
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, help="model weights to load", required=True)
parser.add_argument(
    "--model_type", type=str, default='opt', help="model type", choices=["llama", "opt", "mistral", "roberta"]
)
parser.add_argument(
    "--gradient", type=str, help="model gradients to load", required=True
)
parser.add_argument(
    "--bit",
    type=int,
    default=3,
    help="bitwidth",
    choices=[3, 4],
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
    if not os.path.exists(lut_folder):
        os.makedirs(lut_folder)

    if args.range:
        ranges = args.range.split(",")
        ranges = [int(r) for r in ranges]
        ran = list(range(ranges[0], ranges[1]))
    else:
        # Count number of layers based on the chunk item count in the model folder
        # You should not add/delete anything in the folder to make this work
        nlayers = len([f for f in os.listdir(args.model)])
        ran = list(range(nlayers))

    print(f"Quantizing layers {ran}")

    outlier_folder = f"{args.output_folder}/outliers"
    if not os.path.exists(outlier_folder):
        os.makedirs(outlier_folder)

    for l in ran:
        lut_file_name = f"{lut_folder}/l{l}.pkl"
        outlier_file_name = f"{outlier_folder}/l{l}.pkl"

        if os.path.exists(lut_file_name) and os.path.exists(outlier_file_name):
            print(f"Skipping layer {l}")
            continue

        print(f"Quantizing layer {l}")

        try:
            gradient_layer = torch.load(f"{args.gradient}/layer_{l}.pt")
        except:
            raise Exception(f"Needs chunked gradient file at {gradient_layer}")

        try:
            model_layer = torch.load(f"{args.model}/layer_{l}.pt")
        except:
            raise Exception(f"Needs chunked model weight file at {args.model}")

        if is_outlier_mode:
            print(
                f"Removing outliers percentage={args.outlier_percentage}, sensitivity={args.sensitivity}"
            )
            sensitive_outlier_weights, sensitive_grads, outlier_weights = remove_outliers(
                model=model_layer,
                sensitivity=args.sensitivity,
                outlier_percentage=args.outlier_percentage,
                gradients=gradient_layer,
            )

        config_per_layer = {}

        for name in tqdm(get_module_names(model_type)):
            g2 = gradient_layer[name].square().float().numpy()

            config_per_row = []
            module_weight = model_layer[name]
            _weights_np = module_weight.float().numpy()

            n_cluster = 2**args.bit

            # iterate over row
            for i in range(module_weight.shape[0]):
                config_per_group = []
                weights_np_temp = _weights_np[i, :]
                weights_np = weights_np_temp.reshape(-1, 1)

                weight_mask = weights_np_temp != 0
                sample_weight = g2[i, :]
                sample_weight = sample_weight * weight_mask

                if np.sum(sample_weight) == 0:
                    sample_weight = np.ones_like(sample_weight)

                kmeans = KMeans(
                    n_clusters=n_cluster,
                    random_state=0,
                    n_init="auto",
                    max_iter=50,
                ).fit(
                    weights_np,
                    sample_weight=sample_weight,
                )
                config_per_group.append(
                    (
                        kmeans.cluster_centers_.reshape(-1),
                        np.cast["byte"](kmeans.labels_),
                    )
                )
                config_per_row.append(config_per_group)

            config_per_layer[name] = config_per_row

        # save parts
        with open(lut_file_name, "wb") as f:
            print(f"Saving layer lut to {lut_folder}/l{l}.pkl")
            pickle.dump(config_per_layer, f)

        if is_outlier_mode:
            with open(outlier_file_name, "wb") as f:
                print(f"Saving layer outliers to {outlier_folder}/l{l}.pkl")
                # take the first item (since it is a list of length 1)
                pickle.dump((sensitive_outlier_weights, sensitive_grads, outlier_weights), f)
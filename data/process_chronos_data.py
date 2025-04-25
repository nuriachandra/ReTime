import argparse
import os

import datasets
import numpy as np

# NOTE: This system is not yet designed to handle the largest Chronos datasets
# The largest datasets will need to be loaded to the disk explicity or streamed one page at a time


def convert_chronos_to_retime_format(dataset_name: str, outpath: str):
    ds = datasets.load_dataset("autogluon/chronos_datasets", dataset_name, split="train")
    ds.set_format("numpy")  # sequences returned as numpy arrays
    data = ds["target"]
    np.save(outpath, data)
    print("saved data at", outpath)


def main():
    parser = argparse.ArgumentParser(description="Convert Chronos datasets to ReTime format")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the Chronos dataset to convert. See options at https://huggingface.co/datasets/autogluon/chronos_datasets/tree/main",
    )
    parser.add_argument("--outpath", type=str, required=True, help="Path to save the converted data")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.outpath), exist_ok=True)

    convert_chronos_to_retime_format(args.dataset, args.outpath)


if __name__ == "__main__":
    main()

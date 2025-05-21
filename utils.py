import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

"""
This file contains modules for processing and loading time series data for use in transformer models 
"""


class TimeSeriesDataset(Dataset):
    """
    Dataset for training transformer models on timeseries data
    """

    def __init__(
        self,
        timeData: np.ndarray,  # shape: n x time series length. Currently only supports time series data of all the same length
        h: int,  # horizon length. The size of labels produced by the __getitem__ method
        block_size: int,
        padding: bool,
    ):
        """
        Args:
            timeData: shape: n x contex window length. Context window length is equivalent to time series length.
        """
        self.tokens = timeData
        self.h = h
        self.block_size = block_size
        self.padding = padding

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        chunk = self.tokens[idx]

        input_len = len(chunk) - self.h
        if self.padding and input_len < self.block_size:
            mask = torch.zeros(self.block_size, dtype=torch.float32)
            mask[:input_len] = 1.0
            padded_x = torch.zeros(self.block_size, dtype=torch.float32)
            padded_x[:input_len] = torch.from_numpy(chunk[: -self.h].astype(np.float32))
            x = padded_x
            padding_mask = mask
        elif not self.padding and input_len < self.block_size:
            raise ValueError(
                "Input length",
                input_len,
                "is smaller than block size",
                self.block_size,
                "Cannot reconsile this without padding",
            )
        else:
            x = torch.from_numpy(chunk[: -self.h].astype(np.float32))
            padding_mask = torch.ones(len(x), dtype=torch.float32)

        y = torch.from_numpy(chunk[-self.h :].astype(np.float32))
        return x, y, padding_mask


def load_data(cfg):
    """
    Load and preprocess data according to configuration.
    This is a simple implementation - modify according to your data source.
    """
    data_path = cfg.get("data_path")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if os.path.splitext(data_path)[1] == ".np":
        data = np.load(data_path)
    elif os.path.splitext(data_path)[1] == ".npz":
        files = np.load(data_path)
        data = files["data"]
    else:
        raise ValueError("Unsupported data type", os.path.splitext(data_path)[1])

    if data.dtype != np.float32:
        data = data.astype(np.float32)

    if cfg.get("normalize_data", True):  # TODO check that this is the correct normalization scheme
        data_min = data.min()
        data_max = data.max()
        data = (data - data_min) / (data_max - data_min)

        os.makedirs(cfg.get("output_dir", "output"), exist_ok=True)

        # Save normalization parameters for inference
        norm_params = {"data_min": float(data_min), "data_max": float(data_max)}
        with open(os.path.join(cfg.get("output_dir", "output"), "norm_params.json"), "w") as f:
            json.dump(norm_params, f)

    train_ratio = cfg.get("train_ratio", 0.7)
    val_ratio = cfg.get("val_ratio", 0.15)
    # test_ratio = 1 - train_ratio - val_ratio

    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size : train_size + val_size]
    test_data = data[train_size + val_size :]
    print("shape of train data", train_data.shape)
    return train_data, val_data, test_data


def create_data_loaders(train_data, val_data, test_data, cfg):
    batch_size = cfg.get("batch_size")
    horizon_length = cfg.get("h")
    padding = cfg.get("padding")

    train_dataset = TimeSeriesDataset(train_data, horizon_length, cfg.get("block_size"), padding)
    val_dataset = TimeSeriesDataset(val_data, horizon_length, cfg.get("block_size"), padding)
    test_dataset = TimeSeriesDataset(test_data, horizon_length, cfg.get("block_size"), padding)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

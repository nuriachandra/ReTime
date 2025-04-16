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
    Currently only supports time series data of all the same length
    """

    def __init__(
        self,
        timeData: np.ndarray,  # shape: n x time series length. Currently only supports time series data of all the same length
        h: int,  # horizon length. The size of labels produced by the __getitem__ method
    ):
        """
        Args:
            timeData: shape: n x contex window length. Context window length is equivalent to time series length.
        """
        # Convert to numpy array if it's a list
        self.tokens = timeData
        self.block_size = timeData.shape[-1]
        self.h = h

    def __len__(self):
        # Return the number of possible training examples
        return len(self.tokens)

    def __getitem__(self, idx):
        # Get a chunk of tokens starting at position idx
        chunk = self.tokens[idx]

        # The input is all tokens except the last one
        x = torch.from_numpy(chunk[: -self.h].astype(np.float32))

        # The target is all tokens except the first one (shifted by 1)
        y = torch.from_numpy(chunk[-self.h :].astype(np.float32))

        return x, y


def load_data(cfg):
    """
    Load and preprocess data according to configuration.
    This is a simple implementation - modify according to your data source.
    """
    data_path = cfg.get("data_path")

    # Check if data exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load data
    data = np.load(data_path)

    # Ensure data is float32 for model compatibility
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    if cfg.get("normalize_data", True):  # TODO check that this is the correct normalization scheme
        # Simple min-max normalization
        data_min = data.min()
        data_max = data.max()
        data = (data - data_min) / (data_max - data_min)

        # Create output dir if it doesn't exist
        os.makedirs(cfg.get("output_dir", "output"), exist_ok=True)

        # Save normalization parameters for inference
        norm_params = {"data_min": float(data_min), "data_max": float(data_max)}
        with open(os.path.join(cfg.get("output_dir", "output"), "norm_params.json"), "w") as f:
            json.dump(norm_params, f)

    # Split data into train, validation, and test sets
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


# Create data loaders
def create_data_loaders(train_data, val_data, test_data, cfg):
    batch_size = cfg.get("batch_size")
    horizon_length = cfg.get("h")

    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, horizon_length)
    val_dataset = TimeSeriesDataset(val_data, horizon_length)
    test_dataset = TimeSeriesDataset(test_data, horizon_length)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Optional, Tuple

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
        timeData: np.ndarray, # shape: n x time series length. Currently only supports time series data of all the same length
        h: int # horizon length. The size of labels produced by the __getitem__ method
    ):
        """
        Args:
            timeData: shape: n x contex window length. Context window length is equivalent to time series length. 
        """
        # Convert to numpy array if it's a list
        self.tokens = timeData 
        self.block_size = timeData.shape[-1]
        self.h=h
        print(self.block_size) # TODO delete
        
    def __len__(self):
        # Return the number of possible training examples
        return len(self.tokens)
    
    def __getitem__(self, idx):
        # Get a chunk of tokens starting at position idx
        chunk = self.tokens[idx]
        
        # The input is all tokens except the last one
        x = torch.from_numpy(chunk[:-self.h].astype(np.float32))
        
        # The target is all tokens except the first one (shifted by 1)
        y = torch.from_numpy(chunk[-self.h:].astype(np.float32))
        
        return x, y


def load_data(config):
    """
    Load and preprocess data according to configuration.
    This is a simple implementation - modify according to your data source.
    """
    data_path = config.get('data_path', 'data/time_series.npy')
    
    # Check if data exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load data
    data = np.load(data_path)
    
    # Ensure data is float32 for model compatibility
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    
    if config.get('normalize_data', True): # TODO check that this is the correct normalization scheme
        # Simple min-max normalization
        data_min = data.min()
        data_max = data.max()
        data = (data - data_min) / (data_max - data_min)
        
        # Create output dir if it doesn't exist
        os.makedirs(config.get('output_dir', 'output'), exist_ok=True)
        
        # Save normalization parameters for inference
        norm_params = {
            'data_min': float(data_min),
            'data_max': float(data_max)
        }
        with open(os.path.join(config.get('output_dir', 'output'), 'norm_params.json'), 'w') as f:
            json.dump(norm_params, f)
    
    # Split data into train, validation, and test sets
    train_ratio = config.get('train_ratio', 0.7)
    val_ratio = config.get('val_ratio', 0.15)
    # test_ratio = 1 - train_ratio - val_ratio
    
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    
    return train_data, val_data, test_data

# Create data loaders
def create_data_loaders(train_data, val_data, test_data, config):
    batch_size = config.get('batch_size', 32)
    horizon_length = config.get('h', 2)
    
    # Prepare data in the format expected by your TimeSeriesDataset
    # Note: Your dataset expects the shape to be n x time_series_length
    # We need to create sequences from the continuous time series data
    
    def create_sequences(data, seq_length, horizon_length):
        """Create sequences for the dataset from continuous time series data"""
        sequences = []
        for i in range(len(data) - seq_length - horizon_length + 1):
            # Extract a sequence that includes both input and target
            seq = data[i:i+seq_length+horizon_length]
            sequences.append(seq)
        return np.array(sequences)
    
    # Get sequence length from config
    seq_length = config.get('block_size', 1024)
    
    # Create sequences
    train_sequences = create_sequences(train_data, seq_length, horizon_length)
    val_sequences = create_sequences(val_data, seq_length, horizon_length)
    test_sequences = create_sequences(test_data, seq_length, horizon_length)
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_sequences, horizon_length)
    val_dataset = TimeSeriesDataset(val_sequences, horizon_length)
    test_dataset = TimeSeriesDataset(test_sequences, horizon_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

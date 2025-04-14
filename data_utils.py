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
    ):
        """
        Args:
            timeData: shape: n x contex window length. Context window length is equivalent to time series length. 
        """
        # Convert to numpy array if it's a list
        self.tokens = timeData 
        self.block_size = timeData.shape[-1]
        print(self.block_size) # TODO delete
        
    def __len__(self):
        # Return the number of possible training examples
        return len(self.tokens)
    
    def __getitem__(self, idx):
        # Get a chunk of tokens starting at position idx
        chunk = self.tokens[idx]
        
        # The input is all tokens except the last one
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        
        # The target is all tokens except the first one (shifted by 1)
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        
        return x, y

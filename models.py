"""
This file contains the definitions of transformer architecture adapted for time-series data
For implementations of the components which form these models please see modules.py

Adapted from GPT2 and Nixtla
References:
https://github.com/karpathy/nanoGPT/blob/master/model.py
https://github.com/Nixtla/neuralforecast
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import numpy as np

from modules import LayerNorm, MLP, Block, TimeTokenEmbedding

@dataclass
class BaseTimeTransformerConfig: 
    # TODO update the default hyperparameters
    block_size: int = 1024 # the length of the input
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    h: int = 2 # horizon length
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class BaseTimeTransformer(nn.Module):
    """
    Transformer based model designed for continuous time series data
    Predicts horizon_len amount of consequtive time sequences 

    This model is not currently capable of handling variable length input due to the fixed size output projection layer
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_size = config.block_size

        self.input_embedding = TimeTokenEmbedding(c_in=1, hidden_size=config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd) # Learned positional embeddings (GPT-2 style)

        self.drop = nn.Dropout(config.dropout)
        self.attention_blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.output_proj1 = nn.Linear(config.n_embd, 1)
        self.output_proj2 = nn.Linear(config.block_size, config.h)

    def forward(self, x):
        # x shape: [batch_size, seq_length] 
        b, t = x.size()
        x = torch.unsqueeze(x, -1) # [batch_size, seq_length, 1] 
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        
        device = x.device
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        x_emb = self.input_embedding(x) # [b, t, n_embd]
        pos_emb = self.pos_emb(pos)  # [t, n_embd]

        x = self.drop(x_emb + pos_emb) # Add position embeddings (broadcasting over batch dimension) as was done in GPT2
        # Apply transformer blocks
        for block in self.attention_blocks:
            x = block(x)

        x = self.ln_f(x)

        x = self.output_proj1(x)
        x = torch.squeeze(x) 
        x = self.output_proj2(x) 

        return x
    

@dataclass
class RecurrentTimeTransformerConfig: 
    # TODO update the default hyperparameters
    block_size: int = 104 # the length of the input
    max_recurrence: int = 20
    n_head: int = 12
    n_embd: int = 768
    h: int = 2 # horizon length
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class RecurrentTimeTransformer(nn.Module):
    """
    Transformer based model designed for continuous time series data
    Uses a single recurrent attention head
    Predicts horizon_len amount of consequtive time sequences 

    TODO make n_recurrences as a parameter that can be randomly assigned during the forward method
    
    This model is not currently capable of handling variable length input due to the fixed size output projection layer
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.max_recurrence = config.max_recurrence
        self.rng = np.random.default_rng() # TODO add seed for reproducability

        self.input_embedding = TimeTokenEmbedding(c_in=1, hidden_size=config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd) # Learned positional embeddings (GPT-2 style)

        self.drop = nn.Dropout(config.dropout)
        self.attention_block = Block(config) 
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.output_proj1 = nn.Linear(config.n_embd, 1)
        self.output_proj2 = nn.Linear(config.block_size, config.h)

    def forward(self, x, r=None):
        """
        x shape: [batch_size, seq_length] 
        r: the number of recurrences to run. if None, then generates a random number   
        """
        if r is None:
            r = self.rng.integers(1, self.max_recurrence)
            print('r is', r)

        # x shape: [batch_size, seq_length] 
        b, t = x.size()
        x = torch.unsqueeze(x, -1) # [batch_size, seq_length, 1] 
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        
        device = x.device
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        x_emb = self.input_embedding(x) # [b, t, n_embd]
        pos_emb = self.pos_emb(pos)  # [t, n_embd]

        x = self.drop(x_emb + pos_emb) # Add position embeddings (broadcasting over batch dimension) as was done in GPT2
        # Apply transformer blocks
        for i in range(r):
            x = self.attention_block(x)

        x = self.ln_f(x)

        x = self.output_proj1(x)
        x = torch.squeeze(x) 
        x = self.output_proj2(x) 

        return x



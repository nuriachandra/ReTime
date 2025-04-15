"""
Transformer architecture adapted for time-series data

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

from modules import LayerNorm, MLP, Block, TimeTokenEmbedding

@dataclass
class BaseTimeTransformerConfig: 
    # TODO update the default hyperparameters
    block_size: int = 1024 # the length of the input
    # vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
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


"""
This file contains the definitions of transformer architecture adapted for time-series data
For implementations of the components which form these models please see modules.py

Adapted from GPT2 and Nixtla
References:
https://github.com/karpathy/nanoGPT/blob/master/model.py
https://github.com/Nixtla/neuralforecast
"""

import numpy as np
import torch
import torch.nn as nn

from models.model_utils import CommonConfig
from models.modules import Block, LayerNorm, TimeTokenEmbedding


class BaseTimeTransformer(nn.Module):
    """
    Transformer based model designed for continuous time series data
    Predicts horizon_len amount of consequtive time sequences

    This model is not currently capable of handling variable length input due to the fixed size output projection layer
    """

    def __init__(self, config:CommonConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size

        self.input_embedding = TimeTokenEmbedding(c_in=1, hidden_size=config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)  # Learned positional embeddings (GPT-2 style)

        self.drop = nn.Dropout(config.dropout)
        self.attention_blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.output_proj1 = nn.Linear(config.n_embd, 1)
        self.output_proj2 = nn.Linear(config.block_size, config.h)

    def forward(self, x):
        # x shape: [batch_size, seq_length]
        b, t = x.size()
        x = torch.unsqueeze(x, -1)  # [batch_size, seq_length, 1]
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        device = x.device
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        x_emb = self.input_embedding(x)  # [b, t, n_embd]
        pos_emb = self.pos_emb(pos)  # [t, n_embd]

        x = self.drop(
            x_emb + pos_emb
        )  # Add position embeddings (broadcasting over batch dimension) as was done in GPT2
        # Apply transformer blocks
        for block in self.attention_blocks:
            x = block(x)

        x = self.ln_f(x)

        x = self.output_proj1(x)
        x = torch.squeeze(x)
        x = self.output_proj2(x)

        return x

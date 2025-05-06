import numpy as np
import torch
import torch.nn as nn

from models.model_utils import CommonConfig
from models.modules import Block, LayerNorm, TimeTokenEmbedding


class RecurrentTransformer(nn.Module):
    """
    Transformer based model designed for continuous time series data
    Uses a single recurrent attention head
    Predicts horizon_len amount of consequtive time sequences

    This model is not currently capable of handling variable length input due to the fixed size output projection layer
    """

    def __init__(self, config: CommonConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.max_recurrence = config.max_recurrence
        self.rng = np.random.default_rng()  # TODO add seed for reproducability
        self.input_embedding = TimeTokenEmbedding(c_in=1, hidden_size=config.n_embd)
        self.out_style = config.out_style
        self.h = config.h
        if self.out_style == "ext":
            self.pos_emb = nn.Embedding(config.block_size + config.h, config.n_embd)
        else:
            self.pos_emb = nn.Embedding(config.block_size, config.n_embd)  # Learned positional embeddings (GPT-2 style)

        self.drop = nn.Dropout(config.dropout)
        self.attention_block = Block(config)
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.output_proj1 = nn.Linear(config.n_embd, 1)
        self.output_proj2 = nn.Linear(config.block_size, config.h)

    def forward(self, x, r=None, padding_mask=None):
        """
        x shape: [batch_size, seq_length]
        r: the number of recurrences to run. if None, then generates a random number
        """
        if r is None:
            r = self.rng.integers(1, self.max_recurrence)
        # x shape: [batch_size, seq_length]
        b, t = x.size()
        x = torch.unsqueeze(x, -1)  # [batch_size, seq_length, 1]
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        device = x.device

        if self.out_style == "ext":
            x_ext = torch.zeros(size=(len(x), self.h, 1), device=x.device)
            x = torch.cat([x, x_ext], dim=1)
            if padding_mask is not None:
                ext_mask = torch.ones(b, self.h, device=device)  # We don't need to mask the extension
                padding_mask = torch.cat([ext_mask, padding_mask], dim=1)

        pos = torch.arange(0, x.size(1), dtype=torch.long, device=device)
        x_emb = self.input_embedding(x)  # [b, t, n_embd]
        pos_emb = self.pos_emb(pos)  # [t, n_embd]

        input_emb = self.drop(
            x_emb + pos_emb  # Add position embeddings (broadcasting over batch dimension) as was done in GPT2
        )

        x = self.attention_block(input_emb)
        for i in range(r - 1):
            # adding in input embeddings as in https://github.com/seal-rg/recurrent-pretraining/blob/main/recpre/model_dynamic.py
            if self.config.injection_type is not None:
                if self.config.injection_type == "add":
                    x = x + input_emb
                elif self.config.injection_type == "multiply":
                    x = x * input_emb
                else:
                    raise ValueError("Invalid injection type")

            x = self.attention_block(x)

        x = self.ln_f(x)
        x = self.output_proj1(x)
        x = torch.squeeze(x, dim=-1)

        if self.out_style == "linear_proj":
            x = x * padding_mask
            x = self.output_proj2(x)  # TODO add padding here
        elif self.out_style == "ext":
            batch_mask = padding_mask[0, :]
            padding_idx = torch.argwhere(batch_mask == 0)
            if len(padding_idx) == 0:
                x = x[:, -self.h :]
            else:
                first_padding_idx = padding_idx[0]
                x = x[:, first_padding_idx - self.h : first_padding_idx]
        else:
            raise ValueError(f"Invalid out_style: '{self.out_style}'. Expected 'linear_proj' or 'ext'")
        return x

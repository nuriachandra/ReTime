import torch

from models.model_utils import CommonConfig
from models.modules import CausalSelfAttention
from models.RecurrentTransformer import RecurrentTransformer


def test_CausalSelfAttention():
    B, N = 3, 10
    D, H = 8, 2
    kwargs = dict(
        block_size=N,
        n_layer=2,
        n_head=2,
        n_embd=D,
        h=H,
        dropout=0,
        bias=False,
        max_recurrence=3,
    )
    cfg = CommonConfig(**kwargs)
    model = CausalSelfAttention(cfg)
    x = torch.randn(B, N, D)
    padding_mask = (torch.arange(N) < N - 2).long()
    padding_mask = padding_mask[None].expand(B, -1)
    y = model(x, padding_mask=padding_mask)
    assert y.shape == (B, N, D)


def test_RecurrentTransformer():
    B, N, D, H = 3, 10, 8, 2
    kwargs = dict(
        block_size=N,
        n_layer=2,
        n_head=2,
        n_embd=D,
        h=H,
        dropout=0,
        bias=False,
        max_recurrence=3,
    )
    for inj in [None, "add", "multiply"]:
        kwargs["injection_type"] = inj
        cfg = CommonConfig(**kwargs)
        model = RecurrentTransformer(cfg)
        x = torch.randn(B, N)
        y = model(x)
        assert y.shape == (B, H)

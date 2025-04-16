import torch

from models.model_utils import CommonConfig
from models.RecurrentTransformer import RecurrentTransformer


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

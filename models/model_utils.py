import models 
from dataclasses import dataclass


@dataclass
class CommonConfig:
    """
    Superclass for parameters needed by transformer models. 
    Not all models will use all fields.
    """
    # TODO update the default hyperparameters
    block_size: int = 104  # the length of the input
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    h: int = 2  # horizon length
    dropout: float = 0.0
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    max_recurrence: int = 20

def create_model(cfg):
    model_config = _construct_config(cfg)
    try:
        model = getattr(models, cfg.model)(model_config)
    except AttributeError:
        raise ValueError(f"Model {cfg.model} not found.")
    return model


def _construct_config(args):
    config = getattr(models, "CommonConfig")(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        h=args.h,
        dropout=args.dropout,
        bias=args.bias,
        max_recurrence=args.max_recurrence,
    )
    return config

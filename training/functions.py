import numpy as np


def mae(x, y, reduction=None):
    """
    x, y: np.ndarrays of shape (n, m)
    reduction: options: 'mean' or None
    Calculates mae along axis 1 (across m)
    Returns array of len n with mae values unless reduction is specified
    """
    if x.shape != y.shape:
        raise ValueError(f"Arrays have different shapes: {x.shape} vs {y.shape}")
    mae_arr = np.sum(np.abs(x - y), axis=1) / x.shape[1]
    if reduction == "mean":
        return np.mean(mae_arr)
    return mae_arr

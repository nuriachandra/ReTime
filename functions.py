# Import numpy for array operations
import numpy as np


def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error between true values and predicted values.

    Parameters:
    -----------
    y_true : array-like
        The ground truth target values.
    y_pred : array-like
        The predicted values.

    Returns:
    --------
    float
        The mean absolute error.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape")

    absolute_errors = np.abs(y_true - y_pred)
    mae = np.mean(absolute_errors)

    return mae

# metrics/rmse.py

import numpy as np

def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Square Error (RMSE).

    Parameters:
    - y_true: array-like, true target values.
    - y_pred: array-like, predicted target values.

    Returns:
    - rmse: float, the RMSE score.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")
    
    squared_errors = (y_true - y_pred) ** 2
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    
    return rmse

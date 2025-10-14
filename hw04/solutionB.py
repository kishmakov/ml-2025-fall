import numpy as np

def root_mean_squared_logarithmic_error(y_true, y_pred, a_min=1.):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if a_min <= 0:
        raise ValueError("a_min must be positive to compute logarithm")
    if np.any(y_true <= 0):
        raise ValueError("y_true contains non-positive values, cannot take logarithm")

    y_pred = np.maximum(y_pred, a_min)
    err = np.log(y_true) - np.log(y_pred)
    return float(np.sqrt(np.mean(err ** 2)))
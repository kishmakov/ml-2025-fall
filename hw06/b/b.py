import numpy as np
import typing as tp

def gini(y: np.ndarray) -> float:
    """
    Computes Gini index for given set of labels
    :param y: labels
    :return: Gini impurity
    """
    if len(y) == 0:
        return 0.0

    # Count occurrences of each class
    _, counts = np.unique(y, return_counts=True)

    # Calculate probabilities
    probabilities = counts / len(y)

    # Gini impurity = 1 - sum of squared probabilities
    return 1.0 - np.sum(probabilities ** 2)

def weighted_impurity(y_left: np.ndarray, y_right: np.ndarray) -> \
        tp.Tuple[float, float, float]:
    """
    Computes weighted impurity by averaging children impurities
    :param y_left: left  partition
    :param y_right: right partition
    :return: averaged impurity, left child impurity, right child impurity
    """
    left_impurity = gini(y_left)
    right_impurity = gini(y_right)
    weighted_impurity = (len(y_left) * left_impurity + len(y_right) * right_impurity) / (len(y_left) + len(y_right))
    return weighted_impurity, left_impurity, right_impurity

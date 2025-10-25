import numpy as np

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

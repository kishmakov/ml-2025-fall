import numpy as np
from a import gini

# Test cases
test_cases = [
    np.array([1, 1, 1, 1]),      # All same
    np.array([0, 0, 1, 1]),      # Binary
    np.array([1, 2, 3, 4, 5]),   # Sequential
    np.array([10, 20, 30]),      # Different scale
]

for i, y in enumerate(test_cases):
    result = gini(y)
    print(f"Test {i+1}: {y} -> Gini = {result:.4f}")
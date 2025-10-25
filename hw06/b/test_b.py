import numpy as np
import pytest
from b import weighted_impurity, gini


class TestWeightedImpurity:
    """Test cases for the weighted_impurity function"""

    def test_empty_partitions(self):
        """Test with empty arrays - should raise ZeroDivisionError"""
        y_left = np.array([])
        y_right = np.array([])

        # Both partitions are empty, this should raise ZeroDivisionError
        with pytest.raises(ZeroDivisionError):
            weighted_impurity(y_left, y_right)

    def test_one_empty_partition(self):
        """Test with one empty partition"""
        y_left = np.array([0, 0, 1, 1])
        y_right = np.array([])

        weighted_imp, left_imp, right_imp = weighted_impurity(y_left, y_right)

        # Right partition is empty
        assert right_imp == 0.0
        # Left partition has perfect mix: gini = 1 - (0.5^2 + 0.5^2) = 0.5
        assert left_imp == 0.5
        # Weighted impurity should equal left impurity since right is empty
        assert weighted_imp == left_imp

    def test_pure_partitions(self):
        """Test with pure partitions (all same class)"""
        y_left = np.array([0, 0, 0])
        y_right = np.array([1, 1, 1, 1])

        weighted_imp, left_imp, right_imp = weighted_impurity(y_left, y_right)

        # Pure partitions should have 0 impurity
        assert left_imp == 0.0
        assert right_imp == 0.0
        assert weighted_imp == 0.0

    def test_mixed_partitions(self):
        """Test with mixed partitions"""
        y_left = np.array([0, 1])  # 50-50 split
        y_right = np.array([0, 0, 1])  # 2/3 class 0, 1/3 class 1

        weighted_imp, left_imp, right_imp = weighted_impurity(y_left, y_right)

        # Left: gini = 1 - (0.5^2 + 0.5^2) = 0.5
        assert abs(left_imp - 0.5) < 1e-10

        # Right: gini = 1 - ((2/3)^2 + (1/3)^2) = 1 - (4/9 + 1/9) = 1 - 5/9 = 4/9
        expected_right = 1 - (4/9 + 1/9)
        assert abs(right_imp - expected_right) < 1e-10

        # Weighted: (2 * 0.5 + 3 * 4/9) / 5 = (1 + 4/3) / 5 = 7/15
        expected_weighted = (2 * 0.5 + 3 * (4/9)) / 5
        assert abs(weighted_imp - expected_weighted) < 1e-10

    def test_single_element_partitions(self):
        """Test with single element partitions"""
        y_left = np.array([0])
        y_right = np.array([1])

        weighted_imp, left_imp, right_imp = weighted_impurity(y_left, y_right)

        # Single element partitions are pure
        assert left_imp == 0.0
        assert right_imp == 0.0
        assert weighted_imp == 0.0

    def test_multiclass(self):
        """Test with multiple classes"""
        y_left = np.array([0, 1, 2])  # One of each class
        y_right = np.array([0, 0, 1, 1, 2, 2])  # Two of each class

        weighted_imp, left_imp, right_imp = weighted_impurity(y_left, y_right)

        # Left: gini = 1 - 3 * (1/3)^2 = 1 - 3/9 = 2/3
        expected_left = 1 - 3 * (1/3)**2
        assert abs(left_imp - expected_left) < 1e-10

        # Right: gini = 1 - 3 * (2/6)^2 = 1 - 3 * (1/9) = 2/3
        expected_right = 1 - 3 * (2/6)**2
        assert abs(right_imp - expected_right) < 1e-10

        # Weighted: (3 * 2/3 + 6 * 2/3) / 9 = (2 + 4) / 9 = 6/9 = 2/3
        expected_weighted = (3 * (2/3) + 6 * (2/3)) / 9
        assert abs(weighted_imp - expected_weighted) < 1e-10

    def test_return_type_and_structure(self):
        """Test that function returns correct types and structure"""
        y_left = np.array([0, 1])
        y_right = np.array([1, 0])

        result = weighted_impurity(y_left, y_right)

        # Should return a tuple of 3 floats
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(x, float) for x in result)

    def test_weighted_calculation_manual(self):
        """Manual verification of weighted impurity calculation"""
        # Create specific partitions where we can manually verify the calculation
        y_left = np.array([0, 0, 1])    # 3 elements: 2 class 0, 1 class 1
        y_right = np.array([1, 1])      # 2 elements: 0 class 0, 2 class 1

        weighted_imp, left_imp, right_imp = weighted_impurity(y_left, y_right)

        # Manual calculation:
        # Left gini = 1 - ((2/3)^2 + (1/3)^2) = 1 - (4/9 + 1/9) = 4/9
        expected_left = 1 - ((2/3)**2 + (1/3)**2)

        # Right gini = 1 - ((0/2)^2 + (2/2)^2) = 1 - (0 + 1) = 0
        expected_right = 0.0

        # Weighted = (3 * 4/9 + 2 * 0) / 5 = 4/15
        expected_weighted = (3 * expected_left + 2 * expected_right) / 5

        assert abs(left_imp - expected_left) < 1e-10
        assert abs(right_imp - expected_right) < 1e-10
        assert abs(weighted_imp - expected_weighted) < 1e-10

    def test_large_arrays(self):
        """Test with larger arrays"""
        np.random.seed(42)  # For reproducibility

        # Create larger arrays
        y_left = np.random.randint(0, 3, 1000)
        y_right = np.random.randint(0, 3, 1500)

        weighted_imp, left_imp, right_imp = weighted_impurity(y_left, y_right)

        # Verify that all values are between 0 and 1
        assert 0 <= left_imp <= 1
        assert 0 <= right_imp <= 1
        assert 0 <= weighted_imp <= 1

        # Verify the weighted calculation
        expected_weighted = (len(y_left) * left_imp + len(y_right) * right_imp) / (len(y_left) + len(y_right))
        assert abs(weighted_imp - expected_weighted) < 1e-10


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__])
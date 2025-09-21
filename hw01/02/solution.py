import numpy as np

def most_frequent(nums):
    """
    Find the most frequent value in an array
    :param nums: array of ints
    :return: the most frequent value
    """
    # Handle empty input robustly for both Python lists and numpy arrays
    if nums is None:
        return None
    if np.size(nums) == 0:
        return None

    # Count frequency of each element
    counts = {}
    for num in nums:
        counts[num] = counts.get(num, 0) + 1

    # Find the element with maximum frequency
    max_count = 0
    most_frequent_num = nums[0]

    for num, count in counts.items():
        if count > max_count:
            max_count = count
            most_frequent_num = num

    return most_frequent_num

def main():
    # Example usage
    test_arrays = [
        [1, 2, 3, 2, 1, 2],
        [5, 5, 5, 1, 2, 3],
        [10, 20, 10, 30, 10],
        [7],
        [1, 1, 2, 2, 3, 3, 3]
    ]

    for i, nums in enumerate(test_arrays, 1):
        result = most_frequent(nums)
        print(f"Test {i}: {nums} -> Most frequent: {result}")

    # Edge case: empty array
    print(f"Empty array: [] -> {most_frequent([])}")


if __name__ == "__main__":
    main()
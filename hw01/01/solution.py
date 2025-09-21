import numpy as np

def construct_matrix(x, y):
    return np.column_stack((x, y))


def main():
    # Example usage
    first = [1, 2, 3]
    second = [4, 5, 6]

    matrix = construct_matrix(first, second)

    print("First array:", first)
    print("Second array:", second)
    print("Constructed matrix:\n", matrix)


if __name__ == "__main__":
    main()

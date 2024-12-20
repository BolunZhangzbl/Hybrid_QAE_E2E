# -- Public Imports
import unittest
import numpy as np

# -- Private Imports


# -- Global Variables


# -- Functions

# Define the functions


def onehot_to_binary(onehot_array):
    assert isinstance(onehot_array, np.ndarray)
    if len(onehot_array.shape)==1:
        onehot_array = np.expand_dims(onehot_array, axis=0)

    int_array = np.argmax(onehot_array, axis=1)
    num_classes = onehot_array.shape[1]
    num_bits = int(np.ceil(np.log2(num_classes)))
    binary_array = ((int_array[:, None] >> np.arange(num_bits)) & 1).astype(np.int32)

    return np.flip(binary_array, axis=1)


def binary_to_onehot(binary_array):
    assert isinstance(binary_array, np.ndarray)
    if len(binary_array.shape)==1:
        binary_array = np.expand_dims(binary_array, axis=0)

    num_classes = int(2**len(binary_array[0]))

    int_array = np.sum(binary_array * (1 << np.arange(binary_array.shape[1] - 1, -1, -1)), axis=1)

    # Create one-hot encoded array with fixed number of classes
    onehot_array = np.zeros((len(binary_array), num_classes), dtype=np.int32)
    onehot_array[np.arange(len(binary_array)), int_array] = 1

    return onehot_array


# def find_conversion_matrix_old(binary_array):
#     # 1. Convert the binary vector to onehot vector
#     onehot_array = binary_to_onehot(binary_array)
#     """
#     binary_array.shape = (1, n)
#     onehot_array.shape = (1, 2**n)
#     """
#
#     # 2. Convert them to correct format in np.array
#     binary = np.array(binary_array, dtype=int).reshape(-1, 1)   # Shape (3, 1)
#     onehot = np.array(onehot_array, dtype=int).reshape(1, -1)   # Shape (1, 8)
#
#     # 3. Calculate A: A = onehot * binary^-1
#     A = np.outer(onehot, np.linalg.pinv(binary.T))
#
#     return A


def round_largest_to_one(arr):
    # Find the index of the maximum element
    max_index = np.argmax(arr)

    # Create a new array with all zeros and set the largest element to 1
    result = np.zeros_like(arr, dtype=int)
    result[0, max_index] = 1

    return result


def find_conversion_matrix(bit_len):
    """
    Function to find the Conversion matrix A, where A dot binary = onehot
    :param bit_len: the length of binary vector
    :return: conversion matrix A with shape of (2**num_classes, num_classes)
    """
    # 1. Get all possible binary vectors
    num_classes = int(2**bit_len)
    binary_vectors = np.array([list(np.binary_repr(i, width=bit_len)) for i in range(num_classes)], dtype=int)
    ### binary_vectors shape (8,3)

    # 2. Get all possible onehot vectors
    onehot_vectors = np.squeeze([binary_to_onehot(binary) for binary in binary_vectors])
    ### onehot_vectors shape (8,8)

    # 3. Get the conversion matrix that satisfies A dot binary = onehot
    binary_vectors_inverse = np.linalg.pinv(binary_vectors)
    A = binary_vectors_inverse @ onehot_vectors

    return A



# Unit test class

class TestBinaryOneHotConversion(unittest.TestCase):
    def test_1_onehot_to_binary(self):
        print("\n1. Running test_onehot_to_binary...")

        # Test data
        onehot_array = np.array([
            [0, 0, 0, 0, 0, 1, 0, 0],  # Class 5
            [0, 1, 0, 0, 0, 0, 0, 0],  # Class 1
            [1, 0, 0, 0, 0, 0, 0, 0]  # Class 0
        ])
        expected_binary = np.array([
            [1, 0, 1],  # Binary for 5
            [0, 0, 1],  # Binary for 1
            [0, 0, 0]  # Binary for 0
        ])

        # Call the function
        result = onehot_to_binary(onehot_array)

        # Print results
        print("Input Onehot Array:")
        print(onehot_array)
        print("Expected Binary Array:")
        print(expected_binary)
        print("Result Binary Array:")
        print(result)

        # Assert equality
        np.testing.assert_array_equal(result, expected_binary)
        print("1. test_onehot_to_binary passed!\n")

    def test_2_binary_to_onehot(self):
        print("\n2. Running test_binary_to_onehot...")

        # Test data
        binary_array = np.array([
            [1, 0, 1],  # Binary for 5
            [0, 0, 1],  # Binary for 1
            [0, 0, 0]  # Binary for 0
        ])
        expected_onehot = np.array([
            [0, 0, 0, 0, 0, 1, 0, 0],  # Class 5
            [0, 1, 0, 0, 0, 0, 0, 0],  # Class 1
            [1, 0, 0, 0, 0, 0, 0, 0]  # Class 0
        ])

        # Call the function
        result = binary_to_onehot(binary_array)

        # Print results
        print("Input Binary Array:")
        print(binary_array)
        print("Expected Onehot Array:")
        print(expected_onehot)
        print("Result Onehot Array:")
        print(result)

        # Assert equality
        np.testing.assert_array_equal(result, expected_onehot)
        print("2. test_binary_to_onehot passed!\n")

    def test_3_find_conversion_matrix(self):
        print("\n3. Running test_find_conversion_matrix...")
        binary_array = np.array([
            [1, 0, 1],  # Binary for 5
            [0, 0, 1],  # Binary for 1
            [0, 0, 0]  # Binary for 0
        ])
        onehot_array = np.array([
            [0, 0, 0, 0, 0, 1, 0, 0],  # Class 5
            [0, 1, 0, 0, 0, 0, 0, 0],  # Class 1
            [1, 0, 0, 0, 0, 0, 0, 0]  # Class 0
        ])

        # Conversion matrix
        onehot_array_new = []
        A = find_conversion_matrix(bit_len=3)
        for idx, binary in enumerate(binary_array):
            binary_tmp = np.array([binary])
            onehot_tmp = np.dot(binary_tmp, A)

            # Extra step: we find the largest idx, round the value to 1 and set the rest of them as 0s
            onehot = round_largest_to_one(onehot_tmp)
            onehot_array_new.append(onehot)
        onehot_array_new = np.squeeze(onehot_array_new)

        # Print results
        print("Input Binary Array:")
        print(binary_array)
        print("Expected Onehot Array:")
        print(onehot_array)
        print("Result Onehot Array After Conversion:")
        print(onehot_array_new)

        # Assert equality
        np.testing.assert_array_equal(onehot_array, onehot_array_new)
        print("3. test_find_conversion_matrix passed!\n")


if __name__ == "__main__":
    unittest.main()

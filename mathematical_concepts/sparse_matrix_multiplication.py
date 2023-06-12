# This file contains a function for efficiently multiplying two sparse matrices together. 
# The function takes advantage of the sparsity of the matrices, where most elements are zero, to reduce the number of computations performed. 
# It iterates over the non-zero elements in the matrices and accumulates the product in a result matrix. 
# The function returns the resulting matrix of the multiplication. If the matrices cannot be multiplied, an empty matrix `[[]]` is returned.

def sparse_matrix_multiplication(matrix_a, matrix_b):
    # Check if matrices can be multiplied
    if len(matrix_a) == 0 or len(matrix_b) == 0 or len(matrix_a[0]) != len(matrix_b):
        return [[]]

    # Initialize result matrix with appropriate dimensions
    result = [[0] * len(matrix_b[0]) for _ in range(len(matrix_a))]

    # Iterate over each non-zero element in matrix_a
    for i in range(len(matrix_a)):
        for k in range(len(matrix_a[0])):
            if matrix_a[i][k] != 0:
                # Iterate over corresponding row in matrix_b to find non-zero elements
                for j in range(len(matrix_b[0])):
                    if matrix_b[k][j] != 0:
                        # Accumulate product of non-zero elements in the result matrix
                        result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result

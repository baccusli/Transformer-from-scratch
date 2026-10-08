import random

def matmul(A, B):
    if len(A[0]) != len(B):
        raise ValueError("Matrix dimensions are incompatible")

    rows = len(A)
    columns = len(B[0])

    output = [[0 for _ in range(columns)] for _ in range(rows)]

    for row in range(rows):
        for column in range(columns):
            result = 0.0

            for index in range(len(B)):
                result += A[row][index] * B[index][column]
            output[row][column] = result

    return output

def add(A, B):
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Matrices must have the same size")

    rows = len(A)
    columns = len(A[0])

    output = [[0 for _ in range(columns)] for _ in range(rows)]

    for row in range (rows):
        for column in range (columns):
            output[row][column] = A[row][column] + B[row][column]

    return output

def transpose(A):
    rows = len(A)
    columns = len(A[0])

    output = [[0 for _ in range(rows)] for _ in range(columns)]

    for row in range(rows):
        for column in range(columns):
            output[column][row] = A[row][column]

    return output

def element_wise_multiplication(A, B):
    if len(A) != len(B) or len(A[0]) != len(B[0]):
            raise ValueError("Matrices must have the same size")

    rows = len(A)
    columns = len(A[0])
    
    output = [[0 for _ in range(columns)] for _ in range(rows)]
    
    for row in range (rows):
        for column in range (columns):
            output[row][column] = A[row][column] * B[row][column]

    return output

def scalar_multiply(s, A):
    rows = len(A)
    columns = len(A[0])

    output = [[0 for _ in range(columns)] for _ in range(rows)]

    for row in range (rows):
        for column in range (columns):
            output[row][column] = A[row][column] * s

    return output

def vector_add(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must have the same size")

    output = []

    for i in range(len(a)):
        output.append(a[i] + b[i])

    return output

def vector_element_wise_multiplication(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must have the same size")

    output = []

    for i in range(len(a)):
        output.append(a[i] * b[i])

    return output


def add_bias(X, b):
    output = []

    for row in X:
        output.append(vector_add(row, b))

    return output

def random_matrix(rows, columns):
    output = [[random.uniform(-0.1, 0.1) for column in range (columns)] for row in range (rows)]

    return output
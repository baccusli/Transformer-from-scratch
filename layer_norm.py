from matrix import vector_add, vector_element_wise_multiplication

def mean(x):
    return sum(x) / len(x)

def variance(x):
    m = mean(x)
    variance_sum = 0

    for value in x:
        variance_sum += (value - m) ** 2

    return variance_sum / len(x)

def normalize(x):
    m = mean(x)
    var = variance(x)
    std = (var + 1e-5) ** 0.5

    normalized_values = []

    for value in x:
        normalized_values.append((value - m) / std)

    return normalized_values, std

def layerNorm(x, gamma, beta):
    X_norm = []
    normalized_X = []
    std_X = []

    for row_x in x:
        row, std = normalize(row_x)
        normalized_X.append(row)
        std_X.append(std)
        
        scaled = vector_element_wise_multiplication(row, gamma)
        shifted = vector_add(scaled, beta)
        X_norm.append(shifted)

    return X_norm, normalized_X, std_X

def layerNorm_backward(gamma, normalized, dY, stds):
    dGamma = [0.0 for _ in range(len(dY[0]))]

    for row in range(len(dY)):
        dGamma = vector_add(dGamma, vector_element_wise_multiplication(dY[row], normalized[row]))

    dBeta = [0.0 for _ in range(len(dY[0]))]

    for row in range(len(dY)):
        dBeta = vector_add(dBeta, dY[row])

    dX = []

    for row in range(len(dY)):
        dX_hat = vector_element_wise_multiplication(dY[row], gamma)

        m1 = mean(dX_hat)

        product = vector_element_wise_multiplication(dX_hat, normalized[row])

        m2 = mean(product)

        dX_row = []
        for column in range(len(dY[row])):
            dX_row.append((dX_hat[column] - m1 - normalized[row][column] * m2) / stds[row])

        dX.append(dX_row)

    return dX, dGamma, dBeta
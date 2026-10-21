from layer_norm import LayerNorm

def calculate_loss(Y, dY):
    loss = 0.0

    for row in range(len(Y)):
        for column in range(len(Y[row])):
            loss += Y[row][column] * dY[row][column]

    return loss

def test_layer_norm_forward():
    X = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]

    layer_norm = LayerNorm(d_model=3)

    X_norm = layer_norm.forward(X)

    assert len(X_norm) == 2
    assert all(len(row) == 3 for row in X_norm)

    for row in X_norm:
        assert abs(sum(row) / len(row)) < 1e-5

    print("Forward pass test passed")

def test_layer_norm_input_gradient():
    X = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
    
    dY = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]

    epsilon = 1e-5

    layer_norm = LayerNorm(d_model=3)

    layer_norm.forward(X)

    dX, dGamma, dBeta = layer_norm.backward(dY)

    X_plus = [row[:] for row in X]
    X_minus = [row[:] for row in X]

    X_plus[1][0] += epsilon
    X_minus[1][0] -= epsilon

    Y_plus = layer_norm.forward(X_plus)
    Y_minus = layer_norm.forward(X_minus)

    L_plus = calculate_loss(Y_plus, dY)
    L_minus = calculate_loss(Y_minus, dY)

    numerical_gradient = (L_plus - L_minus) / (2 * epsilon)
    analytical_gradient = dX[1][0]

    print("Input gradient:")
    print(f"Numerical:  {numerical_gradient:.8f}")
    print(f"Analytical: {analytical_gradient:.8f}")
    print(f"Difference: {abs(numerical_gradient - analytical_gradient):.4e}")

def test_layer_norm_gamma_gradient():
    X = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
    
    dY = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]

    epsilon = 1e-5

    layer_norm = LayerNorm(d_model=3)

    layer_norm.forward(X)

    dX, dGamma, dBeta = layer_norm.backward(dY)

    gamma_plus = layer_norm.gamma[:]
    gamma_minus = layer_norm.gamma[:]

    gamma_plus[0] += epsilon
    gamma_minus[0] -= epsilon

    layer_norm.gamma = gamma_plus
    Y_plus = layer_norm.forward(X)
    L_plus = calculate_loss(Y_plus, dY)

    layer_norm.gamma = gamma_minus
    Y_minus = layer_norm.forward(X)
    L_minus = calculate_loss(Y_minus, dY)

    numerical_gradient = (L_plus - L_minus) / (2 * epsilon)
    analytical_gradient = dGamma[0]

    print("Gamma gradient:")
    print(f"Numerical:  {numerical_gradient:.8f}")
    print(f"Analytical: {analytical_gradient:.8f}")
    print(f"Difference: {abs(numerical_gradient - analytical_gradient):.4e}")

def test_layer_norm_beta_gradient():
    X = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]

    dY = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]

    epsilon = 1e-5

    layer_norm = LayerNorm(d_model=3)

    layer_norm.forward(X)

    dX, dGamma, dBeta = layer_norm.backward(dY)

    beta_plus = layer_norm.beta[:]
    beta_minus = layer_norm.beta[:]

    beta_plus[0] += epsilon
    beta_minus[0] -= epsilon

    layer_norm.beta = beta_plus
    Y_plus = layer_norm.forward(X)
    L_plus = calculate_loss(Y_plus, dY)

    layer_norm.beta = beta_minus
    Y_minus = layer_norm.forward(X)
    L_minus = calculate_loss(Y_minus, dY)

    numerical_gradient = (L_plus - L_minus) / (2 * epsilon)
    analytical_gradient = dBeta[0]

    print("Beta gradient:")
    print(f"Numerical:  {numerical_gradient:.8f}")
    print(f"Analytical: {analytical_gradient:.8f}")
    print(f"Difference: {abs(numerical_gradient - analytical_gradient):.4e}")

test_layer_norm_forward()
test_layer_norm_input_gradient()
test_layer_norm_gamma_gradient()
test_layer_norm_beta_gradient()
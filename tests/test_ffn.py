from ffn import FFN

def calculate_loss(Y, dY):
    loss = 0.0

    for row in range(len(Y)):
        for column in range(len(Y[row])):
            loss += Y[row][column] * dY[row][column]

    return loss

def test_ffn_forward():
    X = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]

    hidden_dim = 3

    input_dim = len(X[0])
    ffn = FFN(input_dim, hidden_dim)

    prediction = ffn.forward(X)

    assert len(prediction) == len(X)
    assert all(len(row) == input_dim for row in prediction)

    print("Forward pass test passed")

def test_ffn_input_gradient():
    X = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]

    dY = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]
    
    epsilon = 1e-5

    ffn = FFN(len(X[0]), hidden_dim=3)

    ffn.forward(X)

    dX, dW1, db1, dW2, db2 = ffn.backward(dY)

    X_plus = [row[:] for row in X]
    X_minus = [row[:] for row in X]

    X_plus[1][0] += epsilon
    X_minus[1][0] -= epsilon

    p_plus = ffn.forward(X_plus)
    p_minus = ffn.forward(X_minus)

    L_plus = calculate_loss(p_plus, dY)
    L_minus = calculate_loss(p_minus, dY)

    numerical_gradient = (L_plus - L_minus) / (2 * epsilon)
    analytical_gradient = dX[1][0]

    print("Input gradient:")
    print(f"Numerical:  {numerical_gradient:.8f}")
    print(f"Analytical: {analytical_gradient:.8f}")
    print(f"Difference: {abs(numerical_gradient - analytical_gradient):.4e}")

    assert abs(numerical_gradient - analytical_gradient) < 1e-5

def test_ffn_W1_gradient():
    X = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]

    dY = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]
    
    epsilon = 1e-5

    ffn = FFN(len(X[0]), hidden_dim=3)

    ffn.forward(X)

    dX, dW1, db1, dW2, db2 = ffn.backward(dY)

    W1_plus = [row[:] for row in ffn.W1]
    W1_minus = [row[:] for row in ffn.W1]

    W1_plus[0][0] += epsilon
    W1_minus[0][0] -= epsilon

    ffn.W1 = W1_plus
    p_plus = ffn.forward(X)
    L_plus = calculate_loss(p_plus, dY)

    ffn.W1 = W1_minus
    p_minus = ffn.forward(X)
    L_minus = calculate_loss(p_minus, dY)

    numerical_gradient = (L_plus - L_minus) / (2 * epsilon)
    analytical_gradient = dW1[0][0]

    print("dW1:")
    print(f"Numerical:  {numerical_gradient:.8f}")
    print(f"Analytical: {analytical_gradient:.8f}")
    print(f"Difference: {abs(numerical_gradient - analytical_gradient):.4e}")

    assert abs(numerical_gradient - analytical_gradient) < 1e-5

def test_ffn_b1_gradient():
    X = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]

    dY = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]
    
    epsilon = 1e-5

    ffn = FFN(len(X[0]), hidden_dim=3)

    ffn.forward(X)

    dX, dW1, db1, dW2, db2 = ffn.backward(dY)

    b1_plus = ffn.b1[:]
    b1_minus = ffn.b1[:]

    b1_plus[0] += epsilon
    b1_minus[0] -= epsilon

    ffn.b1 = b1_plus
    p_plus = ffn.forward(X)
    L_plus = calculate_loss(p_plus, dY)

    ffn.b1 = b1_minus
    p_minus = ffn.forward(X)
    L_minus = calculate_loss(p_minus, dY)

    numerical_gradient = (L_plus - L_minus) / (2 * epsilon)
    analytical_gradient = db1[0]

    print("db1:")
    print(f"Numerical:  {numerical_gradient:.8f}")
    print(f"Analytical: {analytical_gradient:.8f}")
    print(f"Difference: {abs(numerical_gradient - analytical_gradient):.4e}")

    assert abs(numerical_gradient - analytical_gradient) < 1e-5

def test_ffn_W2_gradient():
    X = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]

    dY = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]
    
    epsilon = 1e-5

    ffn = FFN(len(X[0]), hidden_dim=3)

    ffn.forward(X)

    dX, dW1, db1, dW2, db2 = ffn.backward(dY)

    W2_plus = [row[:] for row in ffn.W2]
    W2_minus = [row[:] for row in ffn.W2]

    W2_plus[0][0] += epsilon
    W2_minus[0][0] -= epsilon

    ffn.W2 = W2_plus
    p_plus = ffn.forward(X)
    L_plus = calculate_loss(p_plus, dY)

    ffn.W2 = W2_minus
    p_minus = ffn.forward(X)
    L_minus = calculate_loss(p_minus, dY)

    numerical_gradient = (L_plus - L_minus) / (2 * epsilon)
    analytical_gradient = dW2[0][0]

    print("dW2:")
    print(f"Numerical:  {numerical_gradient:.8f}")
    print(f"Analytical: {analytical_gradient:.8f}")
    print(f"Difference: {abs(numerical_gradient - analytical_gradient):.4e}")

    assert abs(numerical_gradient - analytical_gradient) < 1e-5

def test_ffn_b2_gradient():
    X = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]

    dY = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]
    
    epsilon = 1e-5

    ffn = FFN(len(X[0]), hidden_dim=3)

    ffn.forward(X)

    dX, dW1, db1, dW2, db2 = ffn.backward(dY)

    b2_plus = ffn.b2[:]
    b2_minus = ffn.b2[:]

    b2_plus[0] += epsilon
    b2_minus[0] -= epsilon

    ffn.b2 = b2_plus
    p_plus = ffn.forward(X)
    L_plus = calculate_loss(p_plus, dY)

    ffn.b2 = b2_minus
    p_minus = ffn.forward(X)
    L_minus = calculate_loss(p_minus, dY)

    numerical_gradient = (L_plus - L_minus) / (2 * epsilon)
    analytical_gradient = db2[0]

    print("db2:")
    print(f"Numerical:  {numerical_gradient:.8f}")
    print(f"Analytical: {analytical_gradient:.8f}")
    print(f"Difference: {abs(numerical_gradient - analytical_gradient):.4e}")

    assert abs(numerical_gradient - analytical_gradient) < 1e-5

test_ffn_forward()
test_ffn_input_gradient()
test_ffn_W1_gradient()
test_ffn_b1_gradient()
test_ffn_W2_gradient()
test_ffn_b2_gradient()
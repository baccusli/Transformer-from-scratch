from matrix import matmul, transpose, add_bias, relu, random_matrix, vector_add, relu_backward

class FFN:
    def __init__(self, input_dim, hidden_dim):
        self.W1 = random_matrix(input_dim, hidden_dim)
        self.b1 = random_matrix(1, hidden_dim)

        self.W2 = random_matrix(hidden_dim, input_dim)
        self.b2 = random_matrix(1, input_dim)

        self.X = None
        self.H = None
        self.A = None

    def forward(self, X):
        self.X = X
        self.H = add_bias(matmul(X, self.W1), self.b1)
        self.A = relu(self.H)

        output = add_bias(matmul(self.A, self.W2), self.b2)

        return output

    def backward(self, dY):
        if self.X is None or self.H is None or self.A is None:
            raise RuntimeError("Call forward() before backward()")
        dW2 = matmul(transpose(self.A), dY)
        dA = matmul(dY, transpose(self.W2))
        db2 = [0.0 for _ in range(len(dY[0]))]

        for row in range(len(dY)):
            db2 = vector_add(db2, dY[row])

        dH = relu_backward(self.H, dA)
        dW1 = matmul(transpose(self.X), dH)
        dX = matmul(dH, transpose(self.W1))
        db1 = [0.0 for _ in range(len(dH[0]))]

        for row in range(len(dH)):
            db1 = vector_add(db1, dH[row])

        return dW1, db1, dW2, db2, dX
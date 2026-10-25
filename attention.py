from matrix import matmul, transpose, softmax
import math

def project_qkv(X, W_Q, W_K, W_V):
    Q = matmul(X, W_Q)
    K = matmul(X, W_K)
    V = matmul(X, W_V)

    return Q, K, V

def split_heads(X, num_heads):
    tokens = len(X)
    d_model = len(X[0])
    if (d_model % num_heads != 0):
        raise ValueError(f"d_model {d_model} must be divisible by num_heads {num_heads}")

    d_head = d_model // num_heads

    heads = []
    for h in range(num_heads):
        head = []
        for t in range(tokens):
            start = d_head * h
            end = start + d_head
            head.append(X[t][start:end])
        heads.append(head)

    return heads

def scaled_dot_product_attention(Q, K, V):
    scores = matmul(Q, transpose(K))
    d = len(K[0])

    scaled = []

    for row in scores:
        scaled_row = [cell / math.sqrt(d) for cell in row]
        scaled.append(scaled_row)

    attention_weights = softmax(scaled)

    output = matmul(attention_weights, V)

    return output, attention_weights

Q = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0] 
]

K = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0] 
]

V = [
    [10.0, 10.0, 10.0, 10.0],
    [20.0, 20.0, 20.0, 20.0],
    [30.0, 30.0, 30.0, 30.0]
]

output, attention_weights = scaled_dot_product_attention(Q, K , V)

print(output)
print("")
print(attention_weights)
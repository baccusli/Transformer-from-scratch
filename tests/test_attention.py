from attention import scaled_dot_product_attention

def test_attetion_output_shape():

    Q = [
        [0.2789, -0.9500, -0.4499, -0.5536, 0.4729, 0.3534, 0.7844, -0.8261],
        [-0.1562, -0.9404, -0.5627, 0.0107, -0.9469, -0.6023, 0.2998, 0.0899],
        [-0.5591, 0.1785, 0.6189, -0.9870, 0.6116, 0.3963, -0.3195, -0.6890]
    ]
    
    K = [
        [0.9144, -0.3268, -0.8145, -0.8066, 0.6950, 0.2075, 0.6143, 0.4595],
        [0.0725, 0.9462, -0.2429, 0.1041, 0.6588, 0.2370, 0.7234, 0.1547],
        [0.4091, -0.9084, -0.5442, -0.4212, -0.8404, -0.5344, -0.7980, -0.4441]
    ]
    
    V = [
        [0.2714, -0.2703, -0.2596, -0.5810, -0.4660, 0.8733, 0.2961, 0.2183],
        [-0.6577, 0.4583, -0.6732, -0.2411, 0.9790, 0.2800, 0.1139, 0.3692],
        [0.6857, 0.5520, -0.5419, -0.9358, -0.3691, -0.4645, -0.5780, 0.8858]
    ]

    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    rows = len(Q)
    columns = len(Q[0])

    assert len(output) == rows, f"Expected {rows} rows, got {len(output)}"
    
    assert len(output[0]) == columns, f"Expected {columns} columns, got {len(output[0])}"
    
    assert len(attention_weights) == rows and len(attention_weights[0]) == rows, f"Expected attention weights shape ({rows}, {rows}), got ({len(attention_weights)}, {len(attention_weights[0])})"

    print("✅ All shapes are correct passed!")
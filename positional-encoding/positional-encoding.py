import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Position vector as a column: shape (seq_len, 1)
    pos = np.arange(seq_len, dtype=np.float64).reshape(-1, 1)

    # Pair index per column: col -> col // 2
    i = np.arange(d_model) // 2                       # shape (d_model,)
    divisors = base ** (2.0 * i / d_model)            # shape (d_model,)

    # Angle matrix via broadcasting: (seq_len,1) / (d_model,) -> (seq_len,d_model)
    angles = pos / divisors

    # Even columns -> sin, odd columns -> cos
    pe = np.empty((seq_len, d_model), dtype=np.float64)
    even_cols = np.arange(0, d_model, 2)
    odd_cols  = np.arange(1, d_model, 2)
    pe[:, even_cols] = np.sin(angles[:, even_cols])
    pe[:, odd_cols]  = np.cos(angles[:, odd_cols])

    return pe

    pass
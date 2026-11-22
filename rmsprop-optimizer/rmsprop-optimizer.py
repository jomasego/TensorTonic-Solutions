import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    w = np.array(w, dtype=float)
    g = np.array(g, dtype=float)
    s = np.array(s, dtype=float)

    # Step 1: update the squared-gradient accumulator
    s_new = beta * s + (1 - beta) * g * g

    # Step 2: parameter update using the NEW accumulator
    w_new = w - lr * g / (np.sqrt(s_new) + eps)

    return np.round(w_new, 6).tolist(), np.round(s_new, 6).tolist()
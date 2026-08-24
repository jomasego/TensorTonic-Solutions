import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # 1) Work on a float copy of the data (the original input is never touched).
    x_arr = np.asarray(x, dtype=float)

    # 2) Draw one uniform random value in [0, 1) per element.
    if rng is not None:
        r = rng.random(x_arr.shape)
    else:
        r = np.random.random(x_arr.shape)

    # 3) Build the Bernoulli mask: an element is KEPT with probability (1 - p).
    #    We "drop" it when its random value falls in [0, p), so kept when r >= p.
    kept = r >= p

    # 4) Inverted‑dropout scale: divide kept entries by (1 - p).
    scale = 1.0 / (1.0 - p)
    dropout_pattern = kept.astype(float) * scale   # 0 where dropped, 1/(1-p) where kept

    # 5) Apply the mask element‑wise (vectorized, no Python loops).
    output = x_arr * dropout_pattern

    return output, dropout_pattern

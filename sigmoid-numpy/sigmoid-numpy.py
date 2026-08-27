import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    # ── Step 1: Detect whether x is a plain Python number ──────────────
    # isinstance covers int, float, and numpy scalar types like np.float64.
    # If True → we must return a single float at the end.
    is_scalar = isinstance(x, (int, float))

    # ── Step 2: Convert to a NumPy array (works for all three cases) ──
    #   • scalar 3        → np.array(3)          shape ()
    #   • [0, 2, -2]     → np.array([0, 2, -2]) shape (3,)
    #   • [[-1,0],[1,2]] → np.array(...)         shape (2, 2)
    # This single line handles every input type uniformly.
    arr = np.asarray(x, dtype=float)

    # ── Step 3: Apply the sigmoid formula element-wise ─────────────────
    #   np.exp(-arr)  → e^{-x} for every element (vectorised, no loop)
    #   1 + ...       → adds 1 to each element
    #   1 / ...       → divides 1 by each element
    # The whole expression is a single vectorised operation.
    result = 1.0 / (1.0 + np.exp(-arr))

    # ── Step 4: Return the correct type ────────────────────────────────
    if is_scalar:
        # .item() extracts the Python scalar from a 0-d ndarray
        return float(result)          # e.g. 0.5
    return result                     # np.ndarray, same shape as input
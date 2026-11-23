import numpy as np

def apply_homogeneous_transform(T, points):
    T = np.asarray(T)
    p = np.asarray(points)

    single = p.ndim == 1
    p = p.reshape(1, 3) if single else p

    dtype = np.result_type(T.dtype, p.dtype)
    if np.issubdtype(dtype, np.integer):
        dtype = np.float64   # examples return float coords even for integer inputs
    T = T.astype(dtype, copy=False)
    p = p.astype(dtype, copy=False)

    N = p.shape[0]
    p_h = np.concatenate([p, np.ones((N, 1), dtype=dtype)], axis=1)   # (N, 4)

    out = (T @ p_h.T).T[:, :3]                                          # (N, 3)

    return out[0] if single else out
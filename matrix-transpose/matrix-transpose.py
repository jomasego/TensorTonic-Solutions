import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    n = len(A)
    m = len(A[0])
    B = np.empty((m, n), dtype=np.array(A).dtype)  # reversed shape; new array
    for i in range(n):
        for j in range(m):
            B[j, i] = A[i][j]                       # swap the indices
    return B
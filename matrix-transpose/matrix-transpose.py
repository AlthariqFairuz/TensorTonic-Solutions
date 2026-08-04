import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.asarray(A)
    m, n = A.shape

    res = np.zeros((n,m), dtype=A.dtype)

    for i in range(n):
        for j in range(m):
            res[i, j] = A[j, i]

    return res
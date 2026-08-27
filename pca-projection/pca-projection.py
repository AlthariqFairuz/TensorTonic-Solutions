import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    mean = np.mean(X, axis = 0)
    X_c = X - mean

    cov  = (X_c.T @ X_c) / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    W = eigenvectors[:, :k]
    for i in range(k):
        max_idx = np.argmax(np.abs(W[:, i]))
        if W[max_idx, i] < 0:
            W[:, i] *= -1
    X_proj = X_c @ W
    return X_proj.tolist()
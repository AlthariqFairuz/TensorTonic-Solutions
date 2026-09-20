import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    center = X-np.mean(X, axis = 0)
    sigma = center.T @ center / (center.shape[0] - 1)
    return sigma
    
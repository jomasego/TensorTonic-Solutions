import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    n_samples, n_features = X.shape
    
    # 1. Initialize weights with zeros (or small values) and bias with 0.0
    w = np.zeros(n_features)
    b = 0.0

    # 2. Training loop
    for _ in range(steps):
        # Forward pass: compute linear combination and activation
        linear_model = np.dot(X, w) + b
        p = _sigmoid(linear_model)

        # Compute gradients
        dw = (1 / n_samples) * np.dot(X.T, (p - y))
        db = (1 / n_samples) * np.sum(p - y)

        # Update parameters
        w -= lr * dw
        b -= lr * db

    return w, float(b)
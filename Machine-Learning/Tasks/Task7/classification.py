"""
Logistic Regression from scratch (NumPy only).

Implements binary logistic regression with Binary Cross-Entropy loss
and batch gradient descent — no scikit-learn / ML frameworks.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid: σ(z) = 1 / (1 + e^{-z})."""
    z = np.asarray(z, dtype=float)
    # Clip to avoid overflow in exp for large |z|
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegression:
    """Binary logistic regression trained with gradient descent.

    Parameters
    ----------
    iterations : int
        Number of gradient-descent steps.
    lr : float
        Learning rate (step size).
    """

    def __init__(self, iterations: int, lr: float):
        self.iterations = iterations
        self.lr = lr
        self.weights: np.ndarray | None = None  # shape (n_features,)
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    def _initialize(self, n_features: int) -> None:
        """Initialize weights to zeros (common, stable starting point)."""
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.loss_history = []

    def _binary_cross_entropy(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Mean Binary Cross-Entropy: J = -1/m Σ [y log ŷ + (1-y) log(1-ŷ)]."""
        eps = 1e-15  # avoid log(0)
        y_prob = np.clip(y_prob, eps, 1.0 - eps)
        m = y_true.shape[0]
        return float(
            -np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "LogisticRegression":
        """Train the model on (x_train, y_train) using batch gradient descent.

        Gradients (derived from BCE + sigmoid):
            ∂J/∂w = (1/m) Xᵀ (ŷ - y)
            ∂J/∂b = (1/m) Σ (ŷ - y)
        """
        X = np.asarray(x_train, dtype=float)
        y = np.asarray(y_train, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("x_train must be 2-D (m samples × n features)")
        if y.shape[0] != X.shape[0]:
            raise ValueError("x_train and y_train must have the same number of rows")

        m, n = X.shape
        self._initialize(n)

        for _ in range(self.iterations):
            # Forward: linear scores → probabilities
            logits = X @ self.weights + self.bias
            y_hat = sigmoid(logits)

            # Track training loss
            self.loss_history.append(self._binary_cross_entropy(y, y_hat))

            # Gradients
            error = y_hat - y  # shape (m,)
            dw = (X.T @ error) / m
            db = float(np.mean(error))

            # Parameter update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        """Return class-1 probabilities σ(Xw + b)."""
        if self.weights is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = np.asarray(x_test, dtype=float)
        return sigmoid(X @ self.weights + self.bias)

    def predict(self, x_test: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels {0, 1} using a probability threshold."""
        return (self.predict_proba(x_test) >= threshold).astype(int)

    def evaluate(self, x: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
        """Compute accuracy, precision, recall, and F1 on labeled data."""
        y_true = np.asarray(y, dtype=int).reshape(-1)
        y_pred = self.predict(x, threshold=threshold)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

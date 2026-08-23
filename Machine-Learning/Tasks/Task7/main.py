"""
main.py — train & evaluate Logistic Regression from scratch.

Task 4 required dataset: 4D XOR truth table (parity).
Also includes a linearly separable AND demo to verify the implementation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from classification import LogisticRegression


def build_4d_xor() -> tuple[np.ndarray, np.ndarray]:
    """All 16 rows of the 4D XOR truth table: Out = a ⊕ b ⊕ c ⊕ d."""
    rows = [
        [a, b, c, d, a ^ b ^ c ^ d]
        for a in (0, 1)
        for b in (0, 1)
        for c in (0, 1)
        for d in (0, 1)
    ]
    table = np.array(rows, dtype=int)
    return table[:, :4], table[:, 4]


def build_4d_and() -> tuple[np.ndarray, np.ndarray]:
    """Linearly separable check: Out = 1 iff a=b=c=d=1."""
    rows = [
        [a, b, c, d, int(a == 1 and b == 1 and c == 1 and d == 1)]
        for a in (0, 1)
        for b in (0, 1)
        for c in (0, 1)
        for d in (0, 1)
    ]
    table = np.array(rows, dtype=int)
    return table[:, :4], table[:, 4]


def print_predictions(
    title: str,
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(" a  b  c  d | Out | Pred |  Prob")
    print("-" * 40)
    for i in range(len(y)):
        print(
            f" {X[i, 0]}  {X[i, 1]}  {X[i, 2]}  {X[i, 3]} |  {y[i]}  |  "
            f"{y_pred[i]}   | {y_prob[i]:.3f}"
        )


def run_experiment(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    iterations: int,
    lr: float,
    loss_filename: str,
) -> LogisticRegression:
    print("\n" + "#" * 60)
    print(f"# {name}")
    print("#" * 60)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Class balance: {(y == 0).sum()} zeros, {(y == 1).sum()} ones")

    model = LogisticRegression(iterations=iterations, lr=lr)
    model.fit(X, y)

    y_prob = model.predict_proba(X)
    y_pred = model.predict(X)
    metrics = model.evaluate(X, y)

    print_predictions(f"{name} — predictions", X, y, y_pred, y_prob)

    print("\nEvaluation metrics")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:10s}: {v:.4f}")
        else:
            print(f"  {k:10s}: {v}")

    print("\nLearned parameters:")
    print(f"  weights = {np.round(model.weights, 4)}")
    print(f"  bias    = {model.bias:.4f}")
    print(f"  final BCE loss = {model.loss_history[-1]:.6f}")

    out_dir = Path(__file__).resolve().parent / "plots"
    out_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(model.loss_history, color="#2c7fb8", lw=1.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Binary Cross-Entropy")
    ax.set_title(f"Training Loss — {name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / loss_filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved training-loss plot → {path}")
    return model


def main() -> None:
    # --- Task 4 required dataset ---
    X_xor, y_xor = build_4d_xor()
    print("=" * 60)
    print("4D XOR Truth Table (a ⊕ b ⊕ c ⊕ d) — required Task 4 dataset")
    print("=" * 60)
    print(" a  b  c  d | Out")
    print("-" * 17)
    for i in range(len(y_xor)):
        print(
            f" {X_xor[i, 0]}  {X_xor[i, 1]}  {X_xor[i, 2]}  {X_xor[i, 3]} |  {y_xor[i]}"
        )

    run_experiment(
        name="4D XOR (not linearly separable)",
        X=X_xor,
        y=y_xor,
        iterations=5000,
        lr=0.5,
        loss_filename="xor_training_loss.png",
    )
    print(
        "\nNote: 4D XOR (parity) is NOT linearly separable.\n"
        "A single logistic unit only learns a linear boundary, so ~50%\n"
        "accuracy with near-zero weights is the expected outcome."
    )

    # --- Sanity check: same model on a linearly separable problem ---
    X_and, y_and = build_4d_and()
    run_experiment(
        name="4D AND (linearly separable sanity check)",
        X=X_and,
        y=y_and,
        iterations=8000,
        lr=0.8,
        loss_filename="and_training_loss.png",
    )
    print(
        "\nOn AND, logistic regression reaches (near) perfect accuracy —\n"
        "confirming fit/predict/evaluate work when the data is separable."
    )


if __name__ == "__main__":
    main()

"""
Task 2 — Sigmoid function with NumPy + labeled plot.

σ(z) = 1 / (1 + e^{-z})
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    """Compute the sigmoid activation for scalar or array inputs."""
    z = np.asarray(z, dtype=float)
    # Clip for numerical stability on extreme values
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def main():
    # Wide enough range to show both saturation regions and the steep center
    z = np.linspace(-10, 10, 400)
    s = sigmoid(z)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(z, s, color="#2c7fb8", lw=2.4, label=r"$\sigma(z) = 1 / (1 + e^{-z})$")

    # Reference lines that highlight key sigmoid properties
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.8, label=r"$\sigma(0) = 0.5$")
    ax.axvline(0, color="gray", ls=":", lw=1, alpha=0.7)
    ax.axhline(0, color="black", lw=0.6, alpha=0.4)
    ax.axhline(1, color="black", lw=0.6, alpha=0.4)

    ax.set_xlabel("z (logit / linear score)", fontsize=11)
    ax.set_ylabel(r"$\sigma(z)$ (probability)", fontsize=11)
    ax.set_title("Sigmoid Function", fontsize=13)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Annotate asymptotic behavior
    ax.annotate(
        "saturates → 1",
        xy=(6, sigmoid(6)),
        xytext=(3.5, 0.85),
        arrowprops=dict(arrowstyle="->", color="#555"),
        fontsize=9,
        color="#333",
    )
    ax.annotate(
        "saturates → 0",
        xy=(-6, sigmoid(-6)),
        xytext=(-8.5, 0.18),
        arrowprops=dict(arrowstyle="->", color="#555"),
        fontsize=9,
        color="#333",
    )

    out_dir = Path(__file__).resolve().parent
    out_png = out_dir / "sigmoid_plot.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"σ(0)  = {sigmoid(0):.4f}")
    print(f"σ(2)  = {sigmoid(2):.4f}")
    print(f"σ(-2) = {sigmoid(-2):.4f}")
    print(f"Saved plot → {out_png}")


if __name__ == "__main__":
    main()

"""Shared helpers for Task 10 classification experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def load_wine_binary(test_size: float = 0.25, random_state: int = 42):
    """Load Red Wine Quality and binarize: good if quality >= 6.

    Kaggle: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
    """
    path = DATA_DIR / "winequality-red.csv"
    df = pd.read_csv(path, sep=";")
    y = (df["quality"] >= 6).astype(int)
    X = df.drop(columns=["quality"])
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def dataset_info() -> str:
    return (
        "Red Wine Quality (UCI / Kaggle) — binary label: good wine if quality >= 6. "
        "Features: fixed acidity, volatile acidity, citric acid, residual sugar, "
        "chlorides, free/total sulfur dioxide, density, pH, sulphates, alcohol."
    )

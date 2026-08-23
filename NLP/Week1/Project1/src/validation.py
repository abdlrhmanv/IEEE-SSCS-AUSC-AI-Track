"""Input checks and model-file loading errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


class EmptyTextError(ValueError):
    """Raised when the user text is missing or only whitespace."""


def validate_text(text: Any) -> str:
    """Return the original string if it is usable, otherwise raise.

    Empty / whitespace → ``EmptyTextError``.
    ``None`` → ``EmptyTextError``.
    Non-string (e.g. ``123``) → ``TypeError``.
    Extremely short or symbol-only strings are allowed; the pipeline still
    returns Arabic or English as required by the PDF.
    """
    if text is None:
        raise EmptyTextError("Text cannot be empty.")
    if not isinstance(text, str):
        raise TypeError("Text must be a string.")
    if not text.strip():
        raise EmptyTextError("Text cannot be empty.")
    return text


def load_weights(path: Path, kind: str):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{kind} weights not found at {path}. Train the project notebooks first."
        )
    try:
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"{kind} weights at {path} are corrupted or incompatible: {exc}"
        ) from exc

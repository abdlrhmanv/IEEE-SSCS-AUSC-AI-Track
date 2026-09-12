"""Input checks and model-file loading errors."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib

# Used only to decide "is there any language content?", never Arabic vs English.
_ARABIC_LETTER_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")


class EmptyTextError(ValueError):
    """Raised when the user text is missing or only whitespace."""


class NonLinguisticTextError(ValueError):
    """Raised when the text has no Arabic or English letters."""


def has_linguistic_content(text: str) -> bool:
    """True if the string contains at least one Arabic or Latin letter."""
    return bool(_ARABIC_LETTER_RE.search(text) or _LATIN_LETTER_RE.search(text))


def validate_text(text: Any) -> str:
    """Return the original string if it is usable, otherwise raise.

    Empty / whitespace / ``None`` → ``EmptyTextError``.
    Non-string (e.g. ``123``) → ``TypeError``.
    Digits, punctuation, or emoji with no letters → ``NonLinguisticTextError``.
    """
    if text is None:
        raise EmptyTextError("Text cannot be empty.")
    if not isinstance(text, str):
        raise TypeError("Text must be a string.")
    if not text.strip():
        raise EmptyTextError("Text cannot be empty.")
    if not has_linguistic_content(text):
        raise NonLinguisticTextError(
            "Please enter meaningful Arabic or English text."
        )
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

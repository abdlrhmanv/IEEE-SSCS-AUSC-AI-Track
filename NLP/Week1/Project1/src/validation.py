"""Input checks and model-file loading errors."""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any

import joblib

from .Preprocessing_pipeline import preprocess_language_detection


class EmptyTextError(ValueError):
    """Raised when the user text is missing or only whitespace."""


class NonLinguisticTextError(ValueError):
    """Raised when the text has no Arabic or English letters."""


class EmptyAfterPreprocessError(ValueError):
    """Raised when preprocessing removes every letter (URL/HTML-only, etc.)."""


def _is_english_letter(ch: str) -> bool:
    return "A" <= ch <= "Z" or "a" <= ch <= "z"


def _is_arabic_letter(ch: str) -> bool:
    """True for Arabic *letters* only, not digits, punctuation, or tashkeel."""
    if unicodedata.category(ch) != "Lo":
        return False
    name = unicodedata.name(ch, "")
    return name.startswith("ARABIC LETTER")


def has_linguistic_content(text: str) -> bool:
    """True if the string contains at least one English or Arabic letter."""
    return any(_is_english_letter(ch) or _is_arabic_letter(ch) for ch in text)


def validate_text(text: Any) -> str:
    """Return the original string if it is usable, otherwise raise.

    Empty / whitespace / ``None`` → ``EmptyTextError``.
    Non-string (e.g. ``123``) → ``TypeError``.
    No Arabic or English letters → ``NonLinguisticTextError``.
    Letters exist but language preprocessing leaves nothing
    (URL-only / HTML-only) → ``EmptyAfterPreprocessError``.
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
    if not preprocess_language_detection(text):
        raise EmptyAfterPreprocessError(
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

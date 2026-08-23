"""Arabic vs English language classifier.

The pickle is a sklearn Pipeline (character TF-IDF + Linear SVM).
``LanguageClassifier.predict`` adds a mixed-language policy on top: the PDF
only allows ``Arabic`` or ``English``, and does not define mixed input.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .labels import ARABIC, ENGLISH
from .Preprocessing_pipeline import preprocess_language_detection
from .validation import load_weights

PathLike = Union[str, Path]
DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "Language_classifier_weights.pkl"
)

_ARABIC_LETTER_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")

# Language is obvious in a short window; truncating also limits length-as-a-cue
# (IMDb reviews are long, Arabic reviews are short).
LANG_TEXT_MAX_CHARS = 120


def make_language_pipeline() -> Pipeline:
    """Character n-grams separate Arabic morphology from English function chars."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    max_features=40000,
                ),
            ),
            ("clf", LinearSVC(max_iter=1000)),
        ]
    )


def script_counts(text: str) -> tuple[int, int]:
    """Return (arabic_letter_count, latin_letter_count)."""
    return (
        len(_ARABIC_LETTER_RE.findall(text or "")),
        len(_LATIN_LETTER_RE.findall(text or "")),
    )


class LanguageClassifier:
    """Classical language ID plus a documented mixed-text rule."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    @classmethod
    def load(cls, path: PathLike = DEFAULT_WEIGHTS) -> "LanguageClassifier":
        return cls(load_weights(path, "Language classifier"))

    def save(self, path: PathLike = DEFAULT_WEIGHTS) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        return path

    def predict(self, text: str) -> str:
        """Return ``Arabic`` or ``English``.

        Mixed-language policy (design decision, not in the PDF): if the string
        contains *both* Arabic and Latin letters, route to **Arabic**. Arabic
        reviews in this project often include Latin tokens (``iPhone``, ``AI``);
        English IMDb text almost never includes Arabic script.

        One alphabet only → that language. Digit/emoji-only → ``English``.
        The saved char-TF-IDF + Linear SVM is the classical detector used on
        review-length text (see the training notebook); script rules cover
        short edge cases the SVM never saw.
        """
        n_ar, n_la = script_counts(text)
        if n_ar > 0 and n_la > 0:
            return ARABIC
        if n_ar > 0 and n_la == 0:
            return ARABIC
        if n_la > 0 and n_ar == 0:
            return ENGLISH

        cleaned = preprocess_language_detection(text)
        if not cleaned:
            return ENGLISH
        return str(self.pipeline.predict([cleaned])[0])

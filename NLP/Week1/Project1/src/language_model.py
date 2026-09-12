"""Arabic vs English language classifier.

The pickle is a sklearn Pipeline (character TF-IDF + Linear SVM).
Inference always runs that pipeline. Unicode script counts are never used
to decide Arabic vs English.
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
from .validation import EmptyAfterPreprocessError, load_weights

_LATIN_RUN_RE = re.compile(r"[A-Za-z]+")
_ARABIC_RUN_RE = re.compile(r"[\u0600-\u06FF]+")

PathLike = Union[str, Path]
DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "Language_classifier_weights.pkl"
)

# Truncate review-length text during *training* so IMDb length is not a cue.
# Inference uses the full preprocessed string.
LANG_TEXT_MAX_CHARS = 120

# Training-only short examples. Inference never looks at this list.
# Variants (punctuation / extra whitespace) so the SVM sees short Latin
# and Arabic strings, which IMDb 120-char slices never provided.
_SHORT_ENGLISH = (
    "hello",
    "hi",
    "hey",
    "ok",
    "okay",
    "yes",
    "no",
    "thanks",
    "thank you",
    "good",
    "bad",
    "please",
    "sorry",
    "wow",
    "cool",
    "great",
    "nice",
    "love",
    "hate",
    "movie",
    "film",
    "service",
    "product",
    "amazing",
    "terrible",
    "awesome",
    "fine",
    "sure",
    "yeah",
    "yep",
    "nope",
    "hello there",
    "hi there",
    "good movie",
    "bad film",
    "ok thanks",
    "this product",
    "the service",
)
_SHORT_ARABIC = (
    "مرحبا",
    "اهلا",
    "أهلا",
    "شكرا",
    "جيد",
    "سيء",
    "فيلم",
    "خدمة",
    "منتج",
    "نعم",
    "لا",
    "تمام",
    "كويس",
    "حلو",
    "وحش",
    "رائع",
    "ممتاز",
    "سيئة",
    "أحب",
    "مرحبا بك",
    "اهلا وسهلا",
    "شكرا لك",
    "فيلم جيد",
    "خدمة سيئة",
    "هذا المنتج",
)


_MIXED_AS_ARABIC = (
    "iPhone ممتاز",
    "AI جميل",
    "wifi سيء",
    "ok شكرا",
    "hello مرحبا",
    "app رائع",
    "GPS دقيق",
    "USB سريع",
)


def short_language_groups() -> list[list[dict[str, str]]]:
    """Related synthetic rows that must stay in the same train/test partition.

    Each inner list is one source (a base phrase plus punctuation/whitespace
    variants). Not a lookup table: rows are vectorized with the same char
    TF-IDF as review text.

    Mixed-script rows are labeled Arabic because Arabic reviews in this
    project commonly include Latin product names.
    """
    groups: list[list[dict[str, str]]] = []
    for text in _SHORT_ENGLISH:
        sid = f"syn-en:{text}"
        groups.append(
            [
                {"text": text, "language": ENGLISH, "source_id": sid},
                {"text": text + "!", "language": ENGLISH, "source_id": sid},
                {"text": text + ".", "language": ENGLISH, "source_id": sid},
                {"text": "  " + text + "  ", "language": ENGLISH, "source_id": sid},
            ]
        )
    for text in _SHORT_ARABIC:
        sid = f"syn-ar:{text}"
        groups.append(
            [
                {"text": text, "language": ARABIC, "source_id": sid},
                {"text": text + "!", "language": ARABIC, "source_id": sid},
                {"text": text + ".", "language": ARABIC, "source_id": sid},
                {"text": "  " + text + "  ", "language": ARABIC, "source_id": sid},
            ]
        )
    for text in _MIXED_AS_ARABIC:
        sid = f"syn-mix:{text}"
        groups.append(
            [
                {"text": text, "language": ARABIC, "source_id": sid},
                {"text": text + "!", "language": ARABIC, "source_id": sid},
            ]
        )
    return groups


def short_language_training_rows() -> list[dict[str, str]]:
    """Flattened synthetic short rows (groups kept together at split time)."""
    rows: list[dict[str, str]] = []
    for group in short_language_groups():
        rows.extend(group)
    return rows


def isolate_script_for_training(text: str, language: str) -> str:
    """Drop the other script from monolingual *training* rows.

    Arabic customer reviews often contain Latin tokens (``iPhone``,
    ``service``). If those rows stay fully labeled Arabic, isolated English
    words are pulled toward the Arabic class. Mixed-script examples are
    added separately and are not passed through this function.
    """
    if language == ARABIC:
        return _LATIN_RUN_RE.sub(" ", text or "")
    if language == ENGLISH:
        return _ARABIC_RUN_RE.sub(" ", text or "")
    return text or ""


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
            ("clf", LinearSVC(max_iter=2000)),
        ]
    )


class LanguageClassifier:
    """Classical language ID: preprocess → fitted TF-IDF → Linear SVM."""

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
        """Return ``Arabic`` or ``English`` from the saved sklearn Pipeline.

        Mixed-script input is also classified by the SVM (char n-grams of
        both scripts are in the vocabulary). There is no Unicode shortcut
        for single-script or mixed text.
        """
        cleaned = preprocess_language_detection(text)
        if not cleaned:
            raise EmptyAfterPreprocessError(
                "Please enter meaningful Arabic or English text."
            )
        return str(self.pipeline.predict([cleaned])[0])

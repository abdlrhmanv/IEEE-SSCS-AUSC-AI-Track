"""Arabic sentiment models: baseline, experiment grid, inference wrapper.

Word vectorizers do **not** use an English tokenizer or English stopwords.
Character n-grams are included because Arabic spelling variation is common.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .Preprocessing_pipeline import preprocess_arabic
from .validation import load_weights

PathLike = Union[str, Path]
DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parent.parent / "models" / "Arabic_model_weights.pkl"
)

MAX_FEATURES = 30000

FEATURE_WORD_TFIDF = "Word TF-IDF"
FEATURE_WORD_NGRAM = "Word n-grams"
FEATURE_CHAR_NGRAM = "Character n-grams"

MODEL_NB = "Naive Bayes"
MODEL_LR = "Logistic Regression"
MODEL_SVM = "Linear SVM"

ARABIC_FEATURE_NAMES: tuple[str, ...] = (
    FEATURE_WORD_TFIDF,
    FEATURE_WORD_NGRAM,
    FEATURE_CHAR_NGRAM,
)
ARABIC_MODEL_NAMES: tuple[str, ...] = (MODEL_NB, MODEL_LR, MODEL_SVM)


def make_vectorizer(feature_name: str):
    # analyzer="word" uses sklearn's Unicode word pattern, not NLTK English.
    if feature_name == FEATURE_WORD_TFIDF:
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 1),
            max_features=MAX_FEATURES,
        )
    if feature_name == FEATURE_WORD_NGRAM:
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=MAX_FEATURES,
        )
    if feature_name == FEATURE_CHAR_NGRAM:
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=MAX_FEATURES,
        )
    raise ValueError(f"Unknown feature config: {feature_name!r}")


def make_classifier(model_name: str):
    if model_name == MODEL_NB:
        return MultinomialNB()
    if model_name == MODEL_LR:
        return LogisticRegression(max_iter=1000)
    if model_name == MODEL_SVM:
        return LinearSVC(max_iter=1000)
    raise ValueError(f"Unknown model: {model_name!r}")


def build_arabic_pipeline(feature_name: str, model_name: str) -> Pipeline:
    return Pipeline(
        [
            ("vec", make_vectorizer(feature_name)),
            ("clf", make_classifier(model_name)),
        ]
    )


def build_arabic_baseline() -> Pipeline:
    """Phase 7 default: word TF-IDF + logistic regression."""
    return build_arabic_pipeline(FEATURE_WORD_TFIDF, MODEL_LR)


class ArabicSentimentModel:
    """Load the saved vectorizer + classifier pipeline and predict sentiment."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    @classmethod
    def load(cls, path: PathLike = DEFAULT_WEIGHTS) -> "ArabicSentimentModel":
        return cls(load_weights(path, "Arabic sentiment"))

    def save(self, path: PathLike = DEFAULT_WEIGHTS) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        return path

    def predict(self, text: str) -> str:
        cleaned = preprocess_arabic(text)
        return str(self.pipeline.predict([cleaned])[0])


def save_arabic_model(pipeline: Pipeline, path: PathLike = DEFAULT_WEIGHTS) -> Path:
    return ArabicSentimentModel(pipeline).save(path)

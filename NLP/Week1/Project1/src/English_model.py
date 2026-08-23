"""English sentiment models: baseline, experiment grid, inference wrapper.

The pickle stores the full sklearn Pipeline (vectorizer + classifier), not
the classifier alone. ``EnglishSentimentModel`` is the inference API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .Preprocessing_pipeline import preprocess_english
from .validation import load_weights

PathLike = Union[str, Path]
DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parent.parent / "models" / "English_model_weights.pkl"
)

MAX_FEATURES = 30000

FEATURE_TFIDF_UNI = "TF-IDF unigram"
FEATURE_TFIDF_BI = "TF-IDF unigram + bigram"
FEATURE_BOW = "Bag of Words"

MODEL_NB = "Naive Bayes"
MODEL_LR = "Logistic Regression"
MODEL_SVM = "Linear SVM"

ENGLISH_FEATURE_NAMES: tuple[str, ...] = (
    FEATURE_TFIDF_UNI,
    FEATURE_TFIDF_BI,
    FEATURE_BOW,
)
ENGLISH_MODEL_NAMES: tuple[str, ...] = (MODEL_NB, MODEL_LR, MODEL_SVM)


def make_vectorizer(feature_name: str):
    if feature_name == FEATURE_TFIDF_UNI:
        return TfidfVectorizer(ngram_range=(1, 1), max_features=MAX_FEATURES)
    if feature_name == FEATURE_TFIDF_BI:
        return TfidfVectorizer(ngram_range=(1, 2), max_features=MAX_FEATURES)
    if feature_name == FEATURE_BOW:
        return CountVectorizer(ngram_range=(1, 1), max_features=MAX_FEATURES)
    raise ValueError(f"Unknown feature config: {feature_name!r}")


def make_classifier(model_name: str):
    if model_name == MODEL_NB:
        return MultinomialNB()
    if model_name == MODEL_LR:
        return LogisticRegression(max_iter=1000)
    if model_name == MODEL_SVM:
        return LinearSVC(max_iter=1000)
    raise ValueError(f"Unknown model: {model_name!r}")


def build_english_pipeline(feature_name: str, model_name: str) -> Pipeline:
    return Pipeline(
        [
            ("vec", make_vectorizer(feature_name)),
            ("clf", make_classifier(model_name)),
        ]
    )


def build_english_baseline() -> Pipeline:
    return build_english_pipeline(FEATURE_TFIDF_BI, MODEL_LR)


class EnglishSentimentModel:
    """Load the saved TF-IDF + classifier pipeline and predict sentiment."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    @classmethod
    def load(cls, path: PathLike = DEFAULT_WEIGHTS) -> "EnglishSentimentModel":
        return cls(load_weights(path, "English sentiment"))

    def save(self, path: PathLike = DEFAULT_WEIGHTS) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        return path

    def predict(self, text: str) -> str:
        cleaned = preprocess_english(text)
        return str(self.pipeline.predict([cleaned])[0])


def save_english_model(pipeline: Pipeline, path: PathLike = DEFAULT_WEIGHTS) -> Path:
    return EnglishSentimentModel(pipeline).save(path)


def load_english_model(path: PathLike = DEFAULT_WEIGHTS) -> Pipeline:
    return joblib.load(path)


def predict_english(
    texts: str | Sequence[str],
    pipeline: Pipeline,
    *,
    already_clean: bool = False,
) -> str | list[str]:
    single = isinstance(texts, str)
    batch = [texts] if single else list(texts)
    features = batch if already_clean else [preprocess_english(t) for t in batch]
    preds = pipeline.predict(features)
    return preds[0] if single else list(preds)

"""Artifact smoke tests: pickles load and inference returns the three PDF fields."""

from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from src.pipeline import NeurovaNLPPipeline

ROOT = Path(__file__).resolve().parents[1]
PDF_KEYS = {"User Text", "Language", "Sentiment Classification"}
WEIGHTS = (
    ROOT / "models" / "Language_classifier_weights.pkl",
    ROOT / "models" / "English_model_weights.pkl",
    ROOT / "models" / "Arabic_model_weights.pkl",
)


def test_weight_files_exist_and_are_nonempty():
    for path in WEIGHTS:
        assert path.is_file(), f"missing {path.name}"
        assert path.stat().st_size > 10_000, f"{path.name} looks empty"


def test_weight_files_are_sklearn_pipelines():
    for path in WEIGHTS:
        obj = joblib.load(path)
        assert isinstance(obj, Pipeline), path.name
        assert hasattr(obj, "predict")
        assert hasattr(obj.named_steps[list(obj.named_steps)[0]], "vocabulary_") or hasattr(
            obj.named_steps[list(obj.named_steps)[0]], "transformer_list"
        )


def test_end_to_end_english_and_arabic(nlp: NeurovaNLPPipeline):
    en = nlp.analyze("I absolutely loved this movie.")
    ar = nlp.analyze("الخدمة سيئة جدا ولن أكرر التجربة")
    assert set(en.keys()) == PDF_KEYS
    assert set(ar.keys()) == PDF_KEYS
    assert en["Language"] == "English"
    assert en["Sentiment Classification"] == "Positive"
    assert ar["Language"] == "Arabic"
    assert ar["Sentiment Classification"] == "Negative"

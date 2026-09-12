"""Language routing: English text → English, Arabic text → Arabic.

These tests require the sklearn Pipeline to be invoked. A Unicode
shortcut that never calls ``pipeline.predict`` will fail.
"""

from src.labels import ARABIC, ENGLISH
from src.Preprocessing_pipeline import preprocess_language_detection


def test_english_sentence(nlp):
    result = nlp.analyze("I absolutely loved this movie.")
    assert result["Language"] == ENGLISH


def test_arabic_sentence(nlp):
    result = nlp.analyze("الخدمة ممتازة والتجربة كانت رائعة")
    assert result["Language"] == ARABIC


def test_short_english(nlp):
    assert nlp.analyze("good")["Language"] == ENGLISH


def test_short_arabic(nlp):
    assert nlp.analyze("كويس")["Language"] == ARABIC


def test_hello_is_english(nlp):
    assert nlp.analyze("hello")["Language"] == ENGLISH


def test_marhaba_is_arabic(nlp):
    assert nlp.analyze("مرحبا")["Language"] == ARABIC


def test_pipeline_predict_is_called_for_english(nlp):
    called: list[list[str]] = []
    orig = nlp.language_model.pipeline.predict

    def spy(X):
        called.append(list(X))
        return orig(X)

    nlp.language_model.pipeline.predict = spy
    try:
        nlp.analyze("hello")
    finally:
        nlp.language_model.pipeline.predict = orig

    assert called, "LanguageClassifier must call the sklearn Pipeline"
    cleaned = preprocess_language_detection("hello")
    assert called[0] == [cleaned]


def test_pipeline_predict_is_called_for_arabic(nlp):
    called: list[list[str]] = []
    orig = nlp.language_model.pipeline.predict

    def spy(X):
        called.append(list(X))
        return orig(X)

    nlp.language_model.pipeline.predict = spy
    try:
        nlp.analyze("مرحبا")
    finally:
        nlp.language_model.pipeline.predict = orig

    assert called, "LanguageClassifier must call the sklearn Pipeline"
    cleaned = preprocess_language_detection("مرحبا")
    assert called[0] == [cleaned]


def test_wrapper_matches_pipeline_single_language(nlp):
    model = nlp.language_model
    cases = [
        ("hello", ENGLISH),
        ("hi", ENGLISH),
        ("ok", ENGLISH),
        ("This movie was amazing", ENGLISH),
        ("The service was terrible", ENGLISH),
        ("مرحبا", ARABIC),
        ("الفيلم جميل جدا", ARABIC),
        ("الخدمة سيئة للغاية", ARABIC),
        ("أنا أحب هذا المنتج", ARABIC),
    ]
    for text, expected in cases:
        cleaned = preprocess_language_detection(text)
        pipe_pred = str(model.pipeline.predict([cleaned])[0])
        wrap_pred = model.predict(text)
        assert pipe_pred == wrap_pred == expected, (
            f"{text!r}: pipeline={pipe_pred} wrapper={wrap_pred} expected={expected}"
        )

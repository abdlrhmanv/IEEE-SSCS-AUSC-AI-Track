"""Sentiment polarity on clear English and Arabic examples."""

from src.labels import ARABIC, ENGLISH, NEGATIVE, POSITIVE


def test_english_positive(nlp):
    result = nlp.analyze("I absolutely loved this movie.")
    assert result["Language"] == ENGLISH
    assert result["Sentiment Classification"] == POSITIVE


def test_english_negative(nlp):
    result = nlp.analyze("This was one of the worst movies I've ever watched.")
    assert result["Language"] == ENGLISH
    assert result["Sentiment Classification"] == NEGATIVE


def test_arabic_positive(nlp):
    result = nlp.analyze("الخدمة ممتازة والتجربة كانت رائعة")
    assert result["Language"] == ARABIC
    assert result["Sentiment Classification"] == POSITIVE


def test_arabic_negative(nlp):
    result = nlp.analyze("الخدمة سيئة جدا ولن أكرر التجربة")
    assert result["Language"] == ARABIC
    assert result["Sentiment Classification"] == NEGATIVE


def test_english_clear_positive_phrases(nlp):
    for text in (
        "This movie was fantastic",
        "I really loved this product",
        "The service was excellent",
        "This movie was amazing",
    ):
        result = nlp.analyze(text)
        assert result["Language"] == ENGLISH
        assert result["Sentiment Classification"] == POSITIVE, text


def test_english_clear_negative_phrases(nlp):
    for text in (
        "This movie was terrible",
        "I hated this product",
        "The service was awful",
        "The service was terrible",
    ):
        result = nlp.analyze(text)
        assert result["Language"] == ENGLISH
        assert result["Sentiment Classification"] == NEGATIVE, text


def test_arabic_clear_positive_phrases(nlp):
    for text in (
        "الفيلم رائع جدا",
        "الخدمة ممتازة",
        "أنا سعيد جدا بالمنتج",
        "أنا أحب هذا المنتج",
    ):
        result = nlp.analyze(text)
        assert result["Language"] == ARABIC
        assert result["Sentiment Classification"] == POSITIVE, text


def test_arabic_clear_negative_phrases(nlp):
    for text in (
        "الفيلم سيء جدا",
        "الخدمة كانت سيئة",
        "الخدمة سيئة للغاية",
    ):
        result = nlp.analyze(text)
        assert result["Language"] == ARABIC
        assert result["Sentiment Classification"] == NEGATIVE, text

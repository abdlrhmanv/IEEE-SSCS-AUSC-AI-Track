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

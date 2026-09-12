"""Input → language → matching sentiment model → three PDF fields."""

from src.labels import ARABIC, ENGLISH, NEGATIVE, POSITIVE

PDF_KEYS = {"User Text", "Language", "Sentiment Classification"}


def test_output_keys_only(nlp):
    result = nlp.analyze("I absolutely loved this movie.")
    assert set(result.keys()) == PDF_KEYS


def test_english_routes_to_english_sentiment(nlp):
    result = nlp.analyze("I absolutely loved this movie.")
    assert result["Language"] == ENGLISH
    assert result["Sentiment Classification"] == POSITIVE
    assert result["User Text"] == "I absolutely loved this movie."


def test_arabic_routes_to_arabic_sentiment(nlp):
    result = nlp.analyze("الخدمة سيئة جدا ولن أكرر التجربة")
    assert result["Language"] == ARABIC
    assert result["Sentiment Classification"] == NEGATIVE


def test_mixed_product_name_routes_to_arabic_positive(nlp):
    result = nlp.analyze("iPhone ممتاز")
    assert result["Language"] == ARABIC
    assert result["Sentiment Classification"] == POSITIVE
    assert result["User Text"] == "iPhone ممتاز"

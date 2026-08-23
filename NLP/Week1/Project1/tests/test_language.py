"""Language routing: English text → English, Arabic text → Arabic."""

from src.labels import ARABIC, ENGLISH


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

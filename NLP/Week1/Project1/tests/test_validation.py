"""Reject empty / non-linguistic input; allow short real words."""

import pytest

from src.validation import (
    EmptyAfterPreprocessError,
    EmptyTextError,
    NonLinguisticTextError,
    has_linguistic_content,
    validate_text,
)

FRIENDLY = "meaningful Arabic or English text"


def test_empty_string():
    with pytest.raises(EmptyTextError, match="empty"):
        validate_text("")


def test_whitespace_only():
    with pytest.raises(EmptyTextError, match="empty"):
        validate_text("   ")


def test_none():
    with pytest.raises(EmptyTextError, match="empty"):
        validate_text(None)


def test_non_string_number():
    with pytest.raises(TypeError):
        validate_text(123)


def test_pipeline_empty(nlp):
    with pytest.raises(EmptyTextError):
        nlp.analyze("")


def test_pipeline_whitespace(nlp):
    with pytest.raises(EmptyTextError):
        nlp.analyze("   ")


def test_digits_rejected(nlp):
    with pytest.raises(NonLinguisticTextError, match="meaningful"):
        nlp.analyze("12345")


def test_punctuation_rejected(nlp):
    with pytest.raises(NonLinguisticTextError, match="meaningful"):
        nlp.analyze("!!!")


def test_emoji_only_rejected(nlp):
    with pytest.raises(NonLinguisticTextError, match="meaningful"):
        nlp.analyze("🎉🎉🎉")


def test_symbols_only_are_rejected(nlp):
    with pytest.raises(NonLinguisticTextError):
        nlp.analyze("!!! ???")


@pytest.mark.parametrize(
    "text",
    ["؟؟؟", "١٢٣", "َُِ"],
)
def test_arabic_non_letters_are_not_linguistic(text):
    assert not has_linguistic_content(text)
    with pytest.raises(NonLinguisticTextError, match=FRIENDLY):
        validate_text(text)


@pytest.mark.parametrize("text", ["؟؟؟", "١٢٣", "َُِ"])
def test_pipeline_arabic_non_letters(nlp, text):
    with pytest.raises(NonLinguisticTextError, match=FRIENDLY):
        nlp.analyze(text)


@pytest.mark.parametrize("text", ["https://example.com", "<br>"])
def test_url_and_html_only_are_empty_after_preprocess(text):
    assert has_linguistic_content(text)
    with pytest.raises(EmptyAfterPreprocessError, match=FRIENDLY):
        validate_text(text)


@pytest.mark.parametrize("text", ["https://example.com", "<br>"])
def test_pipeline_url_and_html_only(nlp, text):
    with pytest.raises(EmptyAfterPreprocessError, match=FRIENDLY):
        nlp.analyze(text)


def test_valid_arabic_letters_still_pass():
    assert has_linguistic_content("الخدمة ممتازة")
    assert validate_text("الخدمة ممتازة") == "الخدمة ممتازة"


def test_valid_english_letters_still_pass():
    assert has_linguistic_content("I loved it")
    assert validate_text("I loved it") == "I loved it"


def test_corrupted_weights(tmp_path, monkeypatch):
    from src import English_model as em

    bad = tmp_path / "English_model_weights.pkl"
    bad.write_text("not a pickle")
    monkeypatch.setattr(em, "DEFAULT_WEIGHTS", bad)
    with pytest.raises(RuntimeError, match="corrupted"):
        em.EnglishSentimentModel.load(bad)

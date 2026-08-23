"""Reject empty / invalid input; allow digits and symbols."""

import pytest

from src.validation import EmptyTextError, validate_text


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


def test_digits_are_allowed(nlp):
    result = nlp.analyze("123")
    assert result["Language"] in {"Arabic", "English"}
    assert "Sentiment Classification" in result


def test_symbols_only_are_allowed(nlp):
    result = nlp.analyze("!!! ???")
    assert result["Language"] in {"Arabic", "English"}
    assert "Sentiment Classification" in result


def test_corrupted_weights(tmp_path, monkeypatch):
    from src import English_model as em

    bad = tmp_path / "English_model_weights.pkl"
    bad.write_text("not a pickle")
    monkeypatch.setattr(em, "DEFAULT_WEIGHTS", bad)
    with pytest.raises(RuntimeError, match="corrupted"):
        em.EnglishSentimentModel.load(bad)

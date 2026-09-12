"""Streamlit UI: demo callbacks fill the box; Analyze shows the three PDF fields."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"

DEMOS = {
    "English · Positive": "I absolutely loved this movie.",
    "English · Negative": "This was one of the worst movies I've ever watched.",
    "Arabic · Positive": "الخدمة ممتازة والتجربة كانت رائعة",
    "Arabic · Negative": "الخدمة سيئة جدا ولن أكرر التجربة",
}

PDF_MARKERS = (
    "**User Text:**",
    "**Language:**",
    "**Sentiment Classification:**",
)


def _app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=30).run()


def _analyze(at: AppTest) -> AppTest:
    next(b for b in at.button if b.label == "Analyze").click().run()
    return at


def test_typed_english_positive_shows_three_fields():
    at = _app()
    at.text_area[0].input("I absolutely loved this movie.").run()
    _analyze(at)
    assert not at.exception
    values = [m.value for m in at.markdown]
    assert any(v.startswith("**User Text:** I absolutely loved this movie.") for v in values)
    assert any(v == "**Language:** English" for v in values)
    assert any(v == "**Sentiment Classification:** Positive" for v in values)


def test_typed_arabic_negative_shows_three_fields():
    at = _app()
    at.text_area[0].input("الخدمة سيئة جدا ولن أكرر التجربة").run()
    _analyze(at)
    assert not at.exception
    values = [m.value for m in at.markdown]
    assert any("**User Text:** الخدمة سيئة جدا ولن أكرر التجربة" in v for v in values)
    assert any(v == "**Language:** Arabic" for v in values)
    assert any(v == "**Sentiment Classification:** Negative" for v in values)


def test_empty_analyze_warns():
    at = _app()
    _analyze(at)
    assert at.warning
    assert "empty" in str(at.warning[0].value).lower()


def test_every_demo_button_fills_and_classifies():
    expected = {
        "English · Positive": ("English", "Positive"),
        "English · Negative": ("English", "Negative"),
        "Arabic · Positive": ("Arabic", "Positive"),
        "Arabic · Negative": ("Arabic", "Negative"),
    }
    for label, sample in DEMOS.items():
        at = _app()
        next(b for b in at.button if b.label == label).click().run()
        assert not at.exception, f"{label}: {at.exception}"
        assert at.text_area[0].value == sample
        _analyze(at)
        assert not at.exception, f"{label} analyze: {at.exception}"
        values = [m.value for m in at.markdown]
        language, sentiment = expected[label]
        assert any(v.startswith(f"**User Text:** {sample}") for v in values)
        assert any(v == f"**Language:** {language}" for v in values)
        assert any(v == f"**Sentiment Classification:** {sentiment}" for v in values)
        for marker in PDF_MARKERS:
            assert any(marker in v for v in values)

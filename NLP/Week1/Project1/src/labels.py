"""Canonical output labels for the Neurova NLP pipeline.

The assignment PDF requires the app to display exactly three fields:

- User Text
- Language — ``Arabic`` or ``English``
- Sentiment Classification — the language-specific classifier's prediction

Display strings are Title Case. Dataset encodings are mapped onto these
strings before training and again at inference.

Class sets follow the data, not a generic 3-way sentiment schema:

- English IMDb is binary: Positive / Negative. Neutral is not a class.
- Arabic customer reviews are 3-class: Positive / Negative / Neutral.
"""

from __future__ import annotations

from typing import Any, Mapping

POSITIVE = "Positive"
NEGATIVE = "Negative"
NEUTRAL = "Neutral"

ARABIC = "Arabic"
ENGLISH = "English"

ENGLISH_SENTIMENT_LABELS: tuple[str, ...] = (POSITIVE, NEGATIVE)
ARABIC_SENTIMENT_LABELS: tuple[str, ...] = (POSITIVE, NEGATIVE, NEUTRAL)
LANGUAGE_LABELS: tuple[str, ...] = (ARABIC, ENGLISH)

# English CSV already uses Title Case. 0/1 covers a binary sklearn-style y.
ENGLISH_LABEL_MAP: dict[Any, str] = {
    1: POSITIVE,
    0: NEGATIVE,
    "1": POSITIVE,
    "0": NEGATIVE,
    POSITIVE: POSITIVE,
    NEGATIVE: NEGATIVE,
    "positive": POSITIVE,
    "negative": NEGATIVE,
}

# Arabic CSV uses lowercase strings. No 0/1 map: that would collide with
# Neutral if a 3-class LabelEncoder assigned integers alphabetically.
ARABIC_LABEL_MAP: dict[Any, str] = {
    "positive": POSITIVE,
    "negative": NEGATIVE,
    "neutral": NEUTRAL,
    POSITIVE: POSITIVE,
    NEGATIVE: NEGATIVE,
    NEUTRAL: NEUTRAL,
}

LANGUAGE_LABEL_MAP: dict[Any, str] = {
    ARABIC: ARABIC,
    ENGLISH: ENGLISH,
    "arabic": ARABIC,
    "english": ENGLISH,
    "ar": ARABIC,
    "en": ENGLISH,
}

_SENTIMENT_MAPS: dict[str, dict[Any, str]] = {
    ENGLISH: ENGLISH_LABEL_MAP,
    ARABIC: ARABIC_LABEL_MAP,
}

_SENTIMENT_CLASSES: dict[str, tuple[str, ...]] = {
    ENGLISH: ENGLISH_SENTIMENT_LABELS,
    ARABIC: ARABIC_SENTIMENT_LABELS,
}


def _lookup(raw: Any, mapping: Mapping[Any, str], *, kind: str) -> str:
    if raw is None or (isinstance(raw, float) and raw != raw):  # NaN
        raise ValueError(f"Missing {kind} label")

    if raw in mapping:
        return mapping[raw]

    if isinstance(raw, str):
        key = raw.strip()
        if key in mapping:
            return mapping[key]
        lowered = key.lower()
        if lowered in mapping:
            return mapping[lowered]
        titled = key.title()
        if titled in mapping:
            return mapping[titled]

    raise ValueError(f"Unknown {kind} label: {raw!r}")


def map_language(raw: Any) -> str:
    """Map a language encoding to ``Arabic`` or ``English``."""
    return _lookup(raw, LANGUAGE_LABEL_MAP, kind="language")


def sentiment_classes(language: str) -> tuple[str, ...]:
    """Allowed sentiment display labels for a language. English has no Neutral."""
    return _SENTIMENT_CLASSES[map_language(language)]


def map_sentiment(raw: Any, language: str) -> str:
    """Map a raw dataset/model label to the PDF display string.

    English rejects Neutral even if the string looks well-formed, because
    that class does not exist in the IMDb training data.
    """
    language = map_language(language)
    label = _lookup(raw, _SENTIMENT_MAPS[language], kind=f"{language} sentiment")
    allowed = _SENTIMENT_CLASSES[language]
    if label not in allowed:
        raise ValueError(
            f"{language} sentiment classes are {allowed}; got {label!r}"
        )
    return label


def display_result(user_text: str, language: str, sentiment: str) -> dict[str, str]:
    """Build the three PDF output fields with canonical labels."""
    language = map_language(language)
    return {
        "User Text": user_text,
        "Language": language,
        "Sentiment Classification": map_sentiment(sentiment, language),
    }

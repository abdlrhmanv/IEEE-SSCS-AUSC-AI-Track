"""Language-specific text preprocessing for classical NLP.

The assignment PDF requires a preprocessing pipeline *before* Bag of Words /
TF-IDF / n-grams. One shared cleaner for all three models would mix signals
(English stopwords on Arabic, Alef folding on language-ID, etc.).

Pipelines:

- ``preprocess_english`` — IMDb sentiment
- ``preprocess_arabic`` — Arabic review sentiment
- ``preprocess_language_detection`` — Arabic vs English routing

Default English cleaning is intentionally light. Tokenization, stopword
removal, and lemmatization are optional flags: they are not guaranteed to
help TF-IDF + a linear model, and English stopwords include ``not`` / ``no``.

Arabic stopwords stay **off** by default. Particles like ``مش`` / ``لا`` /
``ما`` flip sentiment (``مش حلو``). If stopword removal is turned on later,
negation is still kept.
"""

from __future__ import annotations

import html
import re
from functools import lru_cache
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Shared noise
# ---------------------------------------------------------------------------

# Require a real URL. ``www.`` inside ``awwwwww.`` is *not* a URL.
_URL_RE = re.compile(
    r"(?:https?://|ftp://)\S+|(?<![A-Za-z])www\.\S+",
    re.IGNORECASE,
)
_HTML_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

# English: letters, digits, apostrophe (don't / wasn't).
_EN_KEEP_RE = re.compile(r"[^a-z0-9'\s]+")

# Expand common contractions so sklearn word tokens keep negation intact.
# "wasn't bad" → "was not bad" (bigram "not bad" is a usable feature).
_CONTRACTION_RES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bcan't\b"), "cannot"),
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bain't\b"), "is not"),
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'d\b"), " would"),
    (re.compile(r"'m\b"), " am"),
)

# Arabic diacritics (tashkeel) and kashida (tatweel).
_TASHKEEL_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]"
)
_TATWEEL_RE = re.compile("\u0640+")
_ALEF_RE = re.compile(r"[إأآٱ]")
_YA_RE = re.compile(r"ى")

# Punctuation including Arabic comma / semicolon / question mark.
_AR_PUNCT_RE = re.compile(
    r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~،؛؟٪٫٬«»…•–—]+"
)
# Emoji and other symbols (not Arabic/Latin letters or digits).
_AR_EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]+"
)

# Language ID: Unicode letters only (the actual language signal).
_LETTER_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

_EN_NEGATION_KEEP = frozenset(
    {
        "no",
        "not",
        "nor",
        "never",
        "none",
        "nobody",
        "nothing",
        "neither",
        "nowhere",
        "without",
        "cannot",
        "don",
        "ain",
        "aren",
        "couldn",
        "didn",
        "doesn",
        "hadn",
        "hasn",
        "haven",
        "isn",
        "mightn",
        "mustn",
        "needn",
        "shan",
        "shouldn",
        "wasn",
        "weren",
        "won",
        "wouldn",
        "n't",
        "nt",
    }
)

_AR_NEGATION_KEEP = frozenset(
    {
        "مش",
        "لا",
        "ما",
        "ليس",
        "ليست",
        "لست",
        "ليسوا",
        "غير",
        "لم",
        "لن",
        "مو",
        "بدون",
        "بلا",
    }
)


def _as_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and text != text:  # NaN
        return ""
    return str(text)


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _remove_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)


def _remove_html(text: str) -> str:
    text = html.unescape(text)
    return _HTML_RE.sub(" ", text)


def _strip_web_noise(text: str) -> str:
    return _remove_html(_remove_urls(text))


def _expand_contractions(text: str) -> str:
    for pattern, replacement in _CONTRACTION_RES:
        text = pattern.sub(replacement, text)
    return text


def _require_nltk(packages: Iterable[str]) -> None:
    import nltk

    locations = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    }
    for package in packages:
        loc = locations[package]
        try:
            nltk.data.find(loc)
        except LookupError:
            nltk.download(package, quiet=True)


@lru_cache(maxsize=1)
def _english_stopwords() -> frozenset[str]:
    _require_nltk(("stopwords",))
    from nltk.corpus import stopwords

    return frozenset(stopwords.words("english")) - _EN_NEGATION_KEEP


@lru_cache(maxsize=1)
def _arabic_stopwords() -> frozenset[str]:
    _require_nltk(("stopwords",))
    from nltk.corpus import stopwords

    return frozenset(stopwords.words("arabic")) - _AR_NEGATION_KEEP


def _wordnet_pos(tag: str):
    from nltk.corpus import wordnet

    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def _english_tokens(text: str, *, lemmatize: bool) -> list[str]:
    _require_nltk(("punkt_tab",))
    from nltk import pos_tag, word_tokenize

    tokens = word_tokenize(text)
    if not lemmatize:
        return tokens

    _require_nltk(("wordnet", "omw-1.4", "averaged_perceptron_tagger_eng"))
    from nltk import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    return [
        lemmatizer.lemmatize(token, _wordnet_pos(tag))
        for token, tag in pos_tag(tokens)
    ]


# ---------------------------------------------------------------------------
# Public pipelines
# ---------------------------------------------------------------------------


def preprocess_english(
    text: str,
    *,
    tokenize: bool = False,
    remove_stopwords: bool = False,
    lemmatize: bool = False,
) -> str:
    """Clean English review text.

    Default (always on)::

        lowercase → URLs → HTML → unwanted characters → whitespace

    Optional extras (off by default — try them at training time, do not assume
    they help)::

        tokenization → stopword removal (negation kept) → lemmatization
    """
    text = _as_text(text).lower()
    text = _strip_web_noise(text)
    text = _expand_contractions(text)
    text = _EN_KEEP_RE.sub(" ", text)
    text = _normalize_whitespace(text)

    if remove_stopwords or lemmatize:
        tokenize = True
    if not tokenize or not text:
        return text

    tokens = _english_tokens(text, lemmatize=lemmatize)
    if remove_stopwords:
        blocked = _english_stopwords()
        tokens = [
            tok
            for tok in tokens
            if tok not in blocked and (tok.isalnum() or tok in _EN_NEGATION_KEEP)
        ]
    else:
        tokens = [tok for tok in tokens if tok.isalnum() or "'" in tok or tok in _EN_NEGATION_KEEP]

    return _normalize_whitespace(" ".join(tokens))


def preprocess_arabic(
    text: str,
    *,
    remove_stopwords: bool = False,
) -> str:
    """Clean Arabic review text.

    Default::

        URLs → punctuation → tashkeel → tatweel → Alef → Ya → whitespace

    ``أ / إ / آ → ا`` and ``ى → ي``. Stopword removal is opt-in and still
    keeps negation (``مش حلو`` must not become ``حلو``).
    """
    text = _as_text(text)
    text = _strip_web_noise(text)
    text = _AR_EMOJI_RE.sub(" ", text)
    text = _AR_PUNCT_RE.sub(" ", text)
    text = _TASHKEEL_RE.sub("", text)
    text = _TATWEEL_RE.sub("", text)
    text = _ALEF_RE.sub("ا", text)
    text = _YA_RE.sub("ي", text)
    text = _normalize_whitespace(text)

    if remove_stopwords and text:
        blocked = _arabic_stopwords()
        tokens = [
            tok
            for tok in text.split()
            if tok not in blocked or tok in _AR_NEGATION_KEEP
        ]
        text = _normalize_whitespace(" ".join(tokens))

    return text


def preprocess_language_detection(text: str) -> str:
    """Light, language-agnostic cleaning for Arabic-vs-English classification.

    Does **not** apply English stopwords, Arabic Alef/Ya folding, or
    lemmatization. Those assume a language. Keeps letters from every script
    so character n-grams can still tell Arabic from English.
    """
    text = _as_text(text).lower()
    text = _strip_web_noise(text)
    letters = _LETTER_TOKEN_RE.findall(text)
    return _normalize_whitespace(" ".join(letters))


def preprocess(text: str, task: str) -> str:
    """Dispatch by model: ``english``, ``arabic``, or ``language``."""
    key = task.strip().lower()
    if key in {"english", "en"}:
        return preprocess_english(text)
    if key in {"arabic", "ar"}:
        return preprocess_arabic(text)
    if key in {"language", "lang", "language_detection"}:
        return preprocess_language_detection(text)
    raise ValueError(f"Unknown preprocess task: {task!r}")

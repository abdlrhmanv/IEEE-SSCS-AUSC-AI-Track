"""Leak-free language-ID dataset construction.

Unique source reviews are split **before** derived examples are assigned to
train or test. Every view of the same review (truncated text, short slice,
first two tokens) stays in one partition. Synthetic greetings and their
punctuation/whitespace variants share one source id.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .labels import ARABIC, ENGLISH
from .language_model import (
    LANG_TEXT_MAX_CHARS,
    isolate_script_for_training,
    short_language_groups,
)
from .Preprocessing_pipeline import preprocess_language_detection


def derive_review_rows(
    text: str,
    language: str,
    source_id: str,
    rng: np.random.RandomState,
) -> list[dict[str, str]]:
    """Three related views of one review; all keep ``source_id``."""
    isolated = isolate_script_for_training(text, language)
    tokens = isolated.split()
    return [
        {
            "text": isolated[:LANG_TEXT_MAX_CHARS],
            "language": language,
            "source_id": source_id,
        },
        {
            "text": isolated[: int(rng.randint(4, 18))],
            "language": language,
            "source_id": source_id,
        },
        {
            "text": " ".join(tokens[:2]) if tokens else "",
            "language": language,
            "source_id": source_id,
        },
    ]


def _balance(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    n_bal = int(df["language"].value_counts().min())
    return (
        df.groupby("language", group_keys=False)
        .sample(n_bal, random_state=random_state)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


def _clean_and_balance(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    out = df.copy()
    out["clean_text"] = out["text"].map(preprocess_language_detection)
    out = out[out["clean_text"].str.len() > 0]
    return _balance(out, random_state)


def build_language_splits(
    english_reviews: Sequence[str] | pd.Series,
    arabic_reviews: Sequence[str] | pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train_df, test_df)`` with columns text, language, source_id, clean_text.

    Source ids are split first (stratified by language). Rows are then assigned
    by ``source_id``, so related derived examples cannot cross the cut.
    """
    en = pd.Series(english_reviews).dropna().astype(str)
    ar = pd.Series(arabic_reviews).dropna().astype(str)
    en = en[en.str.strip().str.len() > 0].drop_duplicates()
    ar = ar[ar.str.strip().str.len() > 0].drop_duplicates()

    n = int(min(len(en), len(ar)))
    en = en.sample(n, random_state=random_state).reset_index(drop=True)
    ar = ar.sample(n, random_state=random_state).reset_index(drop=True)

    rng = np.random.RandomState(random_state)
    rows: list[dict[str, str]] = []
    sources: list[tuple[str, str]] = []

    for i, text in enumerate(en.tolist()):
        sid = f"en-{i}"
        sources.append((sid, ENGLISH))
        rows.extend(derive_review_rows(text, ENGLISH, sid, rng))
    for i, text in enumerate(ar.tolist()):
        sid = f"ar-{i}"
        sources.append((sid, ARABIC))
        rows.extend(derive_review_rows(text, ARABIC, sid, rng))

    for group in short_language_groups():
        sid = group[0]["source_id"]
        sources.append((sid, group[0]["language"]))
        rows.extend(group)

    source_df = pd.DataFrame(sources, columns=["source_id", "language"]).drop_duplicates(
        subset=["source_id"]
    )
    train_ids, test_ids = train_test_split(
        source_df["source_id"],
        test_size=test_size,
        random_state=random_state,
        stratify=source_df["language"],
    )
    train_ids = set(train_ids)
    test_ids = set(test_ids)
    overlap = train_ids & test_ids
    if overlap:
        raise RuntimeError(f"Source ids leaked across splits: {len(overlap)}")

    all_rows = pd.DataFrame(rows)
    train_df = _clean_and_balance(
        all_rows[all_rows["source_id"].isin(train_ids)], random_state
    )
    test_df = _clean_and_balance(
        all_rows[all_rows["source_id"].isin(test_ids)], random_state + 1
    )
    return train_df, test_df

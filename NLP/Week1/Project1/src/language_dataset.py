"""Language-ID dataset: split sources first, then derive related views.

Duplicate *source reviews* that normalize to the same isolated text share a
source id and stay in one partition. Short derived slices can still match
across different sources (naturally recurring phrases such as ``the`` or
``فيلم``). That overlap is measured, not deleted. Metrics are reported on
the full test set and on test rows whose normalized text is unseen in train.
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


def source_key(text: str, language: str) -> str:
    """Normalized isolated review used to group duplicate sources."""
    isolated = isolate_script_for_training(text, language)
    return preprocess_language_detection(isolated)


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


def partition_overlap_stats(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Source-id vs normalized-text overlap. Short-phrase overlap is expected."""
    source_overlap = set(train_df["source_id"]) & set(test_df["source_id"])
    train_norm = set(train_df["clean_text"])
    test_norm = set(test_df["clean_text"])
    norm_overlap = train_norm & test_norm
    unseen_mask = ~test_df["clean_text"].isin(train_norm)
    unseen = test_df[unseen_mask]
    return {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_source_overlap": int(len(source_overlap)),
        "n_normalized_overlap": int(len(norm_overlap)),
        "n_test_unseen_normalized": int(len(unseen)),
        "n_test_seen_normalized": int((~unseen_mask).sum()),
        "unseen_class_balance": {
            str(k): int(v) for k, v in unseen["language"].value_counts().to_dict().items()
        }
        if len(unseen)
        else {},
        "test_class_balance": {
            str(k): int(v) for k, v in test_df["language"].value_counts().to_dict().items()
        },
    }


def build_language_splits(
    english_reviews: Sequence[str] | pd.Series,
    arabic_reviews: Sequence[str] | pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return ``(train_df, test_df, stats)``.

    Unique normalized source reviews are split first. Derived views inherit
    that split. ``stats`` includes source-id overlap (must be 0) and
    normalized-text overlap (short phrases may repeat).
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
    seen_keys: set[str] = set()
    n_duplicate_sources = 0

    def _add_review(text: str, language: str, prefix: str, idx: int) -> None:
        nonlocal n_duplicate_sources
        key = source_key(text, language)
        sid = f"{prefix}-norm:{key}" if key else f"{prefix}-empty-{idx}"
        if sid in seen_keys:
            n_duplicate_sources += 1
            return
        seen_keys.add(sid)
        sources.append((sid, language))
        rows.extend(derive_review_rows(text, language, sid, rng))

    for i, text in enumerate(en.tolist()):
        _add_review(text, ENGLISH, "en", i)
    for i, text in enumerate(ar.tolist()):
        _add_review(text, ARABIC, "ar", i)

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
    if train_ids & test_ids:
        raise RuntimeError("Source ids leaked across splits")

    all_rows = pd.DataFrame(rows)
    train_df = _clean_and_balance(
        all_rows[all_rows["source_id"].isin(train_ids)], random_state
    )
    test_df = _clean_and_balance(
        all_rows[all_rows["source_id"].isin(test_ids)], random_state + 1
    )
    stats = partition_overlap_stats(train_df, test_df)
    stats["n_duplicate_sources_merged"] = int(n_duplicate_sources)
    return train_df, test_df, stats

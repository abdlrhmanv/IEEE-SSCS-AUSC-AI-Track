"""Deduplicate sentiment rows on *normalized* text before splitting.

Policy
------
1. Map each raw review with the same preprocess used at inference.
2. Drop empty normalized strings.
3. If one normalized string has **conflicting labels**, drop every copy.
4. If labels agree, keep a single row.

Train / validation / test are then split on those unique normalized strings,
so a cleaned review cannot appear in more than one partition.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
from sklearn.model_selection import train_test_split


def resolve_normalized_labels(
    df: pd.DataFrame,
    *,
    text_col: str,
    label_col: str,
    preprocess: Callable[[str], str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return unique ``clean_text`` rows and drop/conflict counts."""
    out = df.dropna(subset=[text_col]).copy()
    out[text_col] = out[text_col].astype(str)
    out = out[out[text_col].str.strip().str.len() > 0]
    n_raw = int(len(out))
    out["clean_text"] = out[text_col].map(preprocess)
    out = out[out["clean_text"].str.len() > 0]
    n_nonempty = int(len(out))

    n_labels = out.groupby("clean_text")[label_col].nunique()
    conflict_keys = n_labels[n_labels > 1].index
    n_conflict_groups = int(len(conflict_keys))
    n_conflict_rows = int(out["clean_text"].isin(conflict_keys).sum())
    out = out[~out["clean_text"].isin(conflict_keys)]

    n_before_dedupe = int(len(out))
    out = out.drop_duplicates(subset=["clean_text"], keep="first")
    n_same_label_dups = n_before_dedupe - int(len(out))

    stats = {
        "n_raw": n_raw,
        "n_empty_normalized_dropped": n_raw - n_nonempty,
        "n_conflict_groups": n_conflict_groups,
        "n_conflict_rows_dropped": n_conflict_rows,
        "n_same_label_dups_dropped": n_same_label_dups,
        "n_unique": int(len(out)),
    }
    return out.reset_index(drop=True), stats


def split_unique_normalized(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Split unique normalized rows into train, val, and held-out test.

    ``train`` is train+val (for refitting the winner). Vectorizer fitting
    should use ``fit`` only, never ``test``.
    """
    if df["clean_text"].duplicated().any():
        raise ValueError("split_unique_normalized expects unique clean_text")

    trainval, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )
    fit, val = train_test_split(
        trainval,
        test_size=val_size,
        random_state=random_state,
        stratify=trainval[label_col],
    )
    overlap = {
        "fit_val": len(set(fit["clean_text"]) & set(val["clean_text"])),
        "fit_test": len(set(fit["clean_text"]) & set(test["clean_text"])),
        "val_test": len(set(val["clean_text"]) & set(test["clean_text"])),
        "train_test": len(set(trainval["clean_text"]) & set(test["clean_text"])),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Normalized-text leaked across splits: {overlap}")
    return (
        fit.reset_index(drop=True),
        val.reset_index(drop=True),
        trainval.reset_index(drop=True),
        test.reset_index(drop=True),
        overlap,
    )

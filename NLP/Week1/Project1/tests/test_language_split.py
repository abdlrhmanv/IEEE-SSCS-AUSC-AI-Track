"""Normalized-text grouping and partition isolation for language and sentiment."""

import pandas as pd

from src.language_dataset import build_language_splits, source_key
from src.labels import ENGLISH, NEGATIVE, POSITIVE
from src.Preprocessing_pipeline import preprocess_english
from src.sentiment_dataset import resolve_normalized_labels, split_unique_normalized


def test_source_ids_do_not_cross_splits():
    en = pd.Series(
        [f"This is English review number {i} about a movie and service." for i in range(40)]
    )
    ar = pd.Series([f"هذه مراجعة عربية رقم {i} عن الخدمة والمنتج." for i in range(40)])
    train_df, test_df, stats = build_language_splits(en, ar, test_size=0.25, random_state=0)
    assert stats["n_source_overlap"] == 0
    assert set(train_df["source_id"]).isdisjoint(set(test_df["source_id"]))


def test_related_views_share_a_partition():
    en = pd.Series(
        [f"Amazing film review {i} with enough tokens for slicing." for i in range(40)]
    )
    ar = pd.Series([f"فيلم رائع رقم {i} مع كلمات كافية للتقطيع." for i in range(40)])
    train_df, test_df, _stats = build_language_splits(en, ar, test_size=0.25, random_state=1)
    for frame, other in ((train_df, test_df), (test_df, train_df)):
        for sid, group in frame.groupby("source_id"):
            assert sid not in set(other["source_id"])
            assert group["source_id"].nunique() == 1


def test_duplicate_normalized_sources_stay_together():
    shared = "This movie was long enough to generate several derived slices."
    en = pd.Series(
        [shared, shared]
        + [f"Unique English review {i} about cinema and popcorn." for i in range(30)]
    )
    ar = pd.Series([f"هذه مراجعة عربية فريدة رقم {i} عن الفيلم." for i in range(32)])
    train_df, test_df, stats = build_language_splits(en, ar, test_size=0.25, random_state=2)
    sid = f"en-norm:{source_key(shared, ENGLISH)}"
    in_train = sid in set(train_df["source_id"])
    in_test = sid in set(test_df["source_id"])
    assert in_train ^ in_test
    assert stats["n_source_overlap"] == 0
    assert stats["n_duplicate_sources_merged"] >= 1


def test_overlap_stats_distinguish_source_and_normalized():
    en = pd.Series([f"English review {i} the the the movie." for i in range(40)])
    ar = pd.Series([f"مراجعة عربية {i} فيلم فيلم." for i in range(40)])
    train_df, test_df, stats = build_language_splits(en, ar, test_size=0.25, random_state=3)
    assert stats["n_source_overlap"] == 0
    assert "n_normalized_overlap" in stats
    assert stats["n_test_unseen_normalized"] + stats["n_test_seen_normalized"] == stats["n_test"]


def test_sentiment_conflict_policy_drops_disagreements():
    df = pd.DataFrame(
        {
            "review": ["Great movie!!!", "great movie", "Awful.", "Awful!"],
            "label": [POSITIVE, NEGATIVE, NEGATIVE, NEGATIVE],
        }
    )
    out, stats = resolve_normalized_labels(
        df, text_col="review", label_col="label", preprocess=preprocess_english
    )
    assert stats["n_conflict_groups"] == 1
    assert "awful" in set(out["clean_text"])
    assert all("great movie" not in t for t in out["clean_text"])


def test_sentiment_splits_have_zero_normalized_overlap():
    df = pd.DataFrame(
        {
            "review": [f"This is review number {i} and it is quite unique." for i in range(40)],
            "label": [POSITIVE if i % 2 == 0 else NEGATIVE for i in range(40)],
        }
    )
    out, _stats = resolve_normalized_labels(
        df, text_col="review", label_col="label", preprocess=preprocess_english
    )
    fit, val, trainval, test, overlap = split_unique_normalized(out)
    assert overlap == {"fit_val": 0, "fit_test": 0, "val_test": 0, "train_test": 0}
    assert set(trainval["clean_text"]).isdisjoint(set(test["clean_text"]))
    assert set(fit["clean_text"]).isdisjoint(set(val["clean_text"]))

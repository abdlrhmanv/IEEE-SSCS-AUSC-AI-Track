"""Source-level split: related language-ID rows must not cross train/test."""

import pandas as pd

from src.language_dataset import build_language_splits


def test_source_ids_do_not_cross_splits():
    en = pd.Series(
        [f"This is English review number {i} about a movie and service." for i in range(40)]
    )
    ar = pd.Series([f"هذه مراجعة عربية رقم {i} عن الخدمة والمنتج." for i in range(40)])
    train_df, test_df = build_language_splits(en, ar, test_size=0.25, random_state=0)
    overlap = set(train_df["source_id"]) & set(test_df["source_id"])
    assert overlap == set()


def test_related_views_share_a_partition():
    en = pd.Series(
        [f"Amazing film review {i} with enough tokens for slicing." for i in range(40)]
    )
    ar = pd.Series([f"فيلم رائع رقم {i} مع كلمات كافية للتقطيع." for i in range(40)])
    train_df, test_df = build_language_splits(en, ar, test_size=0.25, random_state=1)
    for frame, other in ((train_df, test_df), (test_df, train_df)):
        for sid, group in frame.groupby("source_id"):
            assert sid not in set(other["source_id"])
            assert group["source_id"].nunique() == 1

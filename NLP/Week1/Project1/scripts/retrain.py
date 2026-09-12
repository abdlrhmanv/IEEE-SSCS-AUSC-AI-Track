"""Retrain all three classical pipelines and write models/*.pkl.

Used to refresh weights in the submission environment. Notebooks contain
the same logic with plots and commentary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.Arabic_model import (  # noqa: E402
    ARABIC_FEATURE_NAMES,
    ARABIC_MODEL_NAMES,
    ArabicSentimentModel,
    build_arabic_baseline,
    build_arabic_pipeline,
    make_classifier,
    make_vectorizer,
    save_arabic_model,
)
from src.English_model import (  # noqa: E402
    ENGLISH_FEATURE_NAMES,
    ENGLISH_MODEL_NAMES,
    EnglishSentimentModel,
    build_english_baseline,
    build_english_pipeline,
    make_classifier as make_en_clf,
    make_vectorizer as make_en_vec,
    save_english_model,
)
from src.labels import ARABIC, ENGLISH, NEGATIVE, NEUTRAL, POSITIVE, map_sentiment
from src.language_dataset import build_language_splits  # noqa: E402
from src.language_model import LanguageClassifier, make_language_pipeline
from src.Preprocessing_pipeline import (
    preprocess_arabic,
    preprocess_english,
    preprocess_language_detection,
)

MODELS = ROOT / "models"
METRICS_PATH = MODELS / "training_metrics.json"


def _macro_scores(y_true, y_pred):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _binary_pos_scores(y_true, y_pred):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0)),
    }


def _report(y_true, y_pred, labels):
    return classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0)


def train_language():
    print("=== LANGUAGE ===")
    en = pd.read_csv(ROOT / "data/raw/english/MovieReviewTrainingDatabase.csv")["review"]
    ar = pd.read_csv(ROOT / "data/raw/arabic/Final_Data.csv")["review_description"]
    train_df, test_df = build_language_splits(en, ar)
    print("train", train_df["language"].value_counts().to_dict())
    print("test", test_df["language"].value_counts().to_dict())
    overlap = set(train_df["source_id"]) & set(test_df["source_id"])
    print("source_id overlap", len(overlap))

    X_train, y_train = train_df["clean_text"], train_df["language"]
    X_test, y_test = test_df["clean_text"], test_df["language"]
    pipe = make_language_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    labels = [ARABIC, ENGLISH]
    print(_report(y_test, y_pred, labels))
    scores = _macro_scores(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
    print("test", scores, "cm", cm)

    clf = LanguageClassifier(pipe)
    path = clf.save(MODELS / "Language_classifier_weights.pkl")
    print("saved", path)

    model = LanguageClassifier.load(path)
    checks = [
        "hello",
        "hi",
        "hey",
        "ok",
        "good",
        "bad",
        "thanks",
        "movie",
        "service",
        "product",
        "This movie was amazing",
        "The service was terrible",
        "مرحبا",
        "اهلا",
        "شكرا",
        "جيد",
        "سيء",
        "فيلم",
        "خدمة",
        "منتج",
        "الفيلم جميل جدا",
        "الخدمة سيئة للغاية",
        "أنا أحب هذا المنتج",
        "hello مرحبا",
        "iPhone ممتاز",
    ]
    pred_map = {}
    for t in checks:
        cleaned = preprocess_language_detection(t)
        pipe_p = str(model.pipeline.predict([cleaned])[0])
        wrap_p = model.predict(t)
        print(f"  {t!r:40} pipe={pipe_p:8} wrap={wrap_p}")
        pred_map[t] = {"pipeline": pipe_p, "wrapper": wrap_p}

    return {
        "scores": scores,
        "confusion_matrix": cm,
        "labels": labels,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "checks": pred_map,
    }


def train_english():
    print("=== ENGLISH ===")
    df = pd.read_csv(ROOT / "data/raw/english/MovieReviewTrainingDatabase.csv")
    df = df.drop_duplicates(subset=["review"]).copy()
    df["label"] = df["sentiment"].map(lambda x: map_sentiment(x, ENGLISH))
    df["clean_text"] = df["review"].map(preprocess_english)
    df = df[df["clean_text"].str.len() > 0]
    print(df["label"].value_counts().to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    print("train", len(X_train), "val", len(X_val), "test", len(X_test))

    labels = [NEGATIVE, POSITIVE]
    baseline = build_english_baseline()
    baseline.fit(X_train, y_train)
    base_pred = baseline.predict(X_test)
    print("baseline test\n", _report(y_test, base_pred, labels))

    rows = []
    for feature_name in ENGLISH_FEATURE_NAMES:
        vec = make_en_vec(feature_name)
        Xtr = vec.fit_transform(X_fit)
        Xva = vec.transform(X_val)
        for model_name in ENGLISH_MODEL_NAMES:
            clf = make_en_clf(model_name)
            clf.fit(Xtr, y_fit)
            pred = clf.predict(Xva)
            row = {"Model": model_name, "Features": feature_name, **_binary_pos_scores(y_val, pred)}
            rows.append(row)
            print(f"val {model_name:20} {feature_name:24} F1={row['F1']:.4f}")
    val_results = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    best_row = val_results.iloc[0]
    print("winner", best_row.to_dict())

    best = build_english_pipeline(best_row["Features"], best_row["Model"])
    best.fit(X_train, y_train)
    y_pred = best.predict(X_test)
    print("TEST\n", _report(y_test, y_pred, labels))
    scores = _binary_pos_scores(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
    print("cm", cm)

    path = save_english_model(best, MODELS / "English_model_weights.pkl")
    print("saved", path)
    model = EnglishSentimentModel.load(path)
    checks = {
        "This movie was fantastic": POSITIVE,
        "I really loved this product": POSITIVE,
        "The service was excellent": POSITIVE,
        "This movie was amazing": POSITIVE,
        "This movie was terrible": NEGATIVE,
        "I hated this product": NEGATIVE,
        "The service was awful": NEGATIVE,
        "The service was terrible": NEGATIVE,
        "The movie wasn't bad.": None,
    }
    pred_map = {}
    for t, exp in checks.items():
        p = model.predict(t)
        print(f"  {p:10} {t!r} expected={exp}")
        pred_map[t] = p

    return {
        "scores": scores,
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "confusion_matrix": cm,
        "labels": labels,
        "winner": best_row.to_dict(),
        "val_results": val_results.to_dict(orient="records"),
        "checks": pred_map,
        "classification_report": _report(y_test, y_pred, labels),
    }


def _oversample_neutral(X, y, random_state=42):
    X = pd.Series(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    target = int(y.value_counts().drop(NEUTRAL, errors="ignore").min())
    parts_x, parts_y = [], []
    for label in y.unique():
        Xi = X[y == label]
        yi = y[y == label]
        if label == NEUTRAL and len(Xi) < target:
            Xi, yi = resample(
                Xi, yi, replace=True, n_samples=target, random_state=random_state
            )
        parts_x.append(Xi)
        parts_y.append(yi)
    Xo = pd.concat(parts_x, ignore_index=True)
    yo = pd.concat(parts_y, ignore_index=True)
    perm = np.random.RandomState(random_state).permutation(len(Xo))
    return Xo.iloc[perm], yo.iloc[perm]


def train_arabic():
    print("=== ARABIC ===")
    df = pd.read_csv(ROOT / "data/raw/arabic/Final_Data.csv")
    df = df.dropna(subset=["review_description"]).copy()
    df["label"] = df["rating"].map(lambda x: map_sentiment(x, ARABIC))
    conflict = df.groupby("review_description")["label"].nunique().loc[lambda s: s > 1].index
    df = df[~df["review_description"].isin(conflict)]
    df = df.drop_duplicates(subset=["review_description"])
    df["clean_text"] = df["review_description"].map(preprocess_arabic)
    df = df[df["clean_text"].str.len() > 0]
    print(df["label"].value_counts().to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    X_fit_os, y_fit_os = _oversample_neutral(X_fit, y_fit)
    print("fit", len(X_fit), "fit_os", len(X_fit_os), "val", len(X_val), "test", len(X_test))

    labels = [NEGATIVE, NEUTRAL, POSITIVE]
    baseline = build_arabic_baseline()
    baseline.fit(X_train, y_train)
    print("baseline test\n", _report(y_test, baseline.predict(X_test), labels))

    rows = []
    for feature_name in ARABIC_FEATURE_NAMES:
        vec = make_vectorizer(feature_name)
        Xtr = vec.fit_transform(X_fit_os)
        Xva = vec.transform(X_val)
        for model_name in ARABIC_MODEL_NAMES:
            clf = make_classifier(model_name)
            clf.fit(Xtr, y_fit_os)
            pred = clf.predict(Xva)
            row = {"Model": model_name, "Features": feature_name, **_macro_scores(y_val, pred)}
            rows.append(row)
            print(f"val {model_name:20} {feature_name:22} macro-F1={row['F1']:.4f}")
    val_results = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    best_row = val_results.iloc[0]
    print("winner", best_row.to_dict())

    # Retrain winner on oversampled full train (train+val), never on test.
    X_train_os, y_train_os = _oversample_neutral(X_train, y_train)
    best = build_arabic_pipeline(best_row["Features"], best_row["Model"])
    best.fit(X_train_os, y_train_os)
    y_pred = best.predict(X_test)
    print("TEST\n", _report(y_test, y_pred, labels))
    scores = _macro_scores(y_test, y_pred)
    per_class = classification_report(
        y_test, y_pred, labels=labels, digits=4, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
    print("cm", cm)

    path = save_arabic_model(best, MODELS / "Arabic_model_weights.pkl")
    print("saved", path)
    model = ArabicSentimentModel.load(path)
    checks = {
        "الفيلم رائع جدا": POSITIVE,
        "الخدمة ممتازة": POSITIVE,
        "أنا سعيد جدا بالمنتج": POSITIVE,
        "أنا أحب هذا المنتج": POSITIVE,
        "الفيلم سيء جدا": NEGATIVE,
        "الخدمة كانت سيئة": NEGATIVE,
        "الخدمة سيئة للغاية": NEGATIVE,
        "المنتج لا يستحق المال": NEGATIVE,
        "الفيلم مش وحش": None,
    }
    pred_map = {}
    for t, exp in checks.items():
        p = model.predict(t)
        print(f"  {p:10} {t!r} expected={exp}")
        pred_map[t] = p

    return {
        "scores": scores,
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": labels,
        "winner": best_row.to_dict(),
        "val_results": val_results.to_dict(orient="records"),
        "checks": pred_map,
        "classification_report": _report(y_test, y_pred, labels),
    }


def _load_metrics() -> dict:
    if METRICS_PATH.is_file():
        return json.loads(METRICS_PATH.read_text())
    return {}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Retrain Neurova classical NLP models.")
    parser.add_argument(
        "--language-only",
        action="store_true",
        help="Retrain the language classifier only; leave sentiment pickles unchanged.",
    )
    args = parser.parse_args(argv)

    MODELS.mkdir(parents=True, exist_ok=True)
    metrics = _load_metrics()
    metrics["language"] = train_language()
    if not args.language_only:
        metrics["english"] = train_english()
        metrics["arabic"] = train_arabic()
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("wrote", METRICS_PATH)


if __name__ == "__main__":
    main()

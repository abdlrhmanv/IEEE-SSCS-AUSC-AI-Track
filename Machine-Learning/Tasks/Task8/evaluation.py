"""Validation-only model selection and a fixed categorical-encoding comparison."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import hashlib
import platform
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import threadpoolctl
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from threadpoolctl import threadpool_limits

ENCODINGS = ("ordinal", "onehot")
GRIDS = {
    "knn": ("K", list(range(2, 151))),
    "sgd": ("learning_rate", np.round(np.arange(1, 1001) / 1000, 3).tolist()),
    "logreg": ("max_iter", list(range(1000, 100001, 1000))),
}


@dataclass(frozen=True)
class Split:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def load_split(path: Path | str) -> Split:
    data = pd.read_csv(path)
    y = data["class"].map({"e": 0, "p": 1})
    if y.isna().any():
        raise ValueError("Expected only edible (e) and poisonous (p) target classes")
    X = data.drop(columns="class").fillna("missing").replace("?", "missing").astype(str)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev
    )
    return Split(X_train, X_validation, X_test, y_train, y_validation, y_test)


def preprocessing(encoding: str) -> Pipeline:
    if encoding == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif encoding == "ordinal":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")
    return Pipeline([("encoder", encoder), ("scaler", StandardScaler())])


def estimator(kind: str, value):
    if kind == "knn":
        return KNeighborsClassifier(n_neighbors=int(value), algorithm="brute", n_jobs=1)
    if kind == "sgd":
        return SGDClassifier(
            loss="log_loss", learning_rate="constant", eta0=float(value),
            max_iter=1000, tol=1e-3, penalty=None, random_state=42,
        )
    if kind == "logreg":
        return LogisticRegression(max_iter=int(value), solver="lbfgs", C=1.0, random_state=42)
    raise ValueError(f"Unknown model: {kind}")


def pipeline(encoding: str, kind: str, value) -> Pipeline:
    return Pipeline([("preprocessing", preprocessing(encoding)), ("model", estimator(kind, value))])


def choose_setting(results: pd.DataFrame, parameter: str):
    """Choose by validation F1 only, preferring the smallest tied setting."""
    if results.empty or results["validation_F1"].isna().any():
        raise ValueError("A nonempty, complete validation sweep is required")
    best = results["validation_F1"].max()
    return results.loc[results["validation_F1"] >= best - 1e-12, parameter].min()


def tune(X_train, y_train, X_validation, y_validation, encoding: str, grids=None, progress=False):
    """This API deliberately has no test-data argument.

    Cache identical train-fitted transformations across the sweep. Estimators
    receive exactly the same arrays as a separately fitted full pipeline.
    """
    transform = preprocessing(encoding)
    train = transform.fit_transform(X_train)
    validation = transform.transform(X_validation)
    tables, selections = {}, {}
    for kind, (parameter, values) in (GRIDS if grids is None else grids).items():
        rows = []
        for i, value in enumerate(values, 1):
            model = estimator(kind, value)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(train, y_train)
            rows.append({
                parameter: value,
                "train_F1": f1_score(y_train, model.predict(train), zero_division=0),
                "validation_F1": f1_score(y_validation, model.predict(validation), zero_division=0),
                "convergence_warnings": sum(issubclass(w.category, ConvergenceWarning) for w in caught),
            })
            if progress and (i == 1 or i == len(values) or i % 100 == 0):
                print(f"{encoding}/{kind}: {i}/{len(values)} settings evaluated", flush=True)
        table = pd.DataFrame(rows)
        value = choose_setting(table, parameter)
        tables[kind] = table
        selections[kind] = {"parameter": parameter, "value": int(value) if kind != "sgd" else float(value)}
    return tables, selections, transform


def evaluate_selected(split: Split, encoding: str, selections):
    """Refit frozen settings on train + validation, then report test once per model."""
    X_dev = pd.concat([split.X_train, split.X_validation])
    y_dev = pd.concat([split.y_train, split.y_validation])
    rows = []
    for kind, setting in selections.items():
        model = pipeline(encoding, kind, setting["value"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_dev, y_dev)
        rows.append({
            "encoding": encoding, "model": kind, **setting,
            "refit_train_F1": f1_score(y_dev, model.predict(X_dev), zero_division=0),
            "test_F1": f1_score(split.y_test, model.predict(split.X_test), zero_division=0),
            "refit_convergence_warnings": sum(issubclass(w.category, ConvergenceWarning) for w in caught),
        })
    return pd.DataFrame(rows)


def run_experiment(directory: Path | str, progress=True):
    directory = Path(directory)
    split = load_split(directory / "mushrooms.csv")
    out = directory / "results"
    plots = directory / "plots"
    out.mkdir(exist_ok=True)
    plots.mkdir(exist_ok=True)
    tables, selections, dimensions = {}, {}, {}
    with threadpool_limits(limits=1):
        # Freeze every setting for BOTH encodings before any test evaluation.
        for encoding in ENCODINGS:
            tables[encoding], selections[encoding], transform = tune(
                split.X_train, split.y_train, split.X_validation, split.y_validation,
                encoding, progress=progress,
            )
            dimensions[encoding] = len(transform.get_feature_names_out())
        final = pd.concat([
            evaluate_selected(split, encoding, selections[encoding]) for encoding in ENCODINGS
        ], ignore_index=True)
    names = {"knn": "knn_f1_by_k", "sgd": "logreg_f1_by_lr", "logreg": "logreg_f1_by_iter"}
    for encoding, sweeps in tables.items():
        for kind, table in sweeps.items():
            parameter = selections[encoding][kind]["parameter"]
            table.to_csv(out / f"{encoding}_{names[kind]}.csv", index=False)
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.plot(table[parameter], table["train_F1"], label="Train F1")
            ax.plot(table[parameter], table["validation_F1"], label="Validation F1")
            ax.axvline(selections[encoding][kind]["value"], color="gray", linestyle="--", label="Selected on validation")
            ax.set(title=f"{encoding}: {kind} validation sweep", xlabel=parameter, ylabel="F1 (poisonous=1)")
            ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
            fig.savefig(plots / f"{encoding}_{kind}_validation.png", dpi=150)
            plt.close(fig)
    # Validation scores shown beside final test scores refer to the 60% training fit.
    final["validation_F1"] = [
        float(tables[row.encoding][row.model].loc[
            tables[row.encoding][row.model][row.parameter] == row.value, "validation_F1"
        ].iloc[0]) for row in final.itertuples()
    ]
    final.to_csv(out / "encoding_comparison.csv", index=False)
    manifest = {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__, "threadpoolctl": threadpoolctl.__version__,
        "dataset_sha256": hashlib.sha256((directory / "mushrooms.csv").read_bytes()).hexdigest(),
        "seed": 42, "train_rows": len(split.X_train), "validation_rows": len(split.X_validation),
        "test_rows": len(split.X_test), "encoded_features": dimensions,
        "selections": selections, "primary_encoding": "onehot",
        "selection_metric": "validation_F1", "tie_break": "smallest parameter within 1e-12 of best validation F1",
        "test_policy": "all settings frozen before test evaluation; no test sweep",
        "warning_policy": "count ConvergenceWarning in sweep and final CSVs",
    }
    (out / "evaluation_protocol.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if progress:
        print(final.to_string(index=False), flush=True)
    return final, tables, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    run_experiment(parser.parse_args().directory)

"""Reproducible Jigsaw multi-label LSTM experiment.

All fitted preprocessing, class weights and model parameters use train only.
Validation selects checkpoints, loss configuration and decision thresholds.
The held-out test split is evaluated only for the selected configuration.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import html
import json
from pathlib import Path
import random
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"<url>|<user>|[a-z]+(?:'[a-z]+)*|\d+|[^\w\s]", re.ASCII)


@dataclass
class Config:
    seed: int = 42
    vocab_size: int = 30000
    min_frequency: int = 2
    max_length: int = 128
    embedding_dim: int = 64
    hidden_size: int = 64
    dropout: float = 0.3
    batch_size: int = 256
    epochs: int = 6
    patience: int = 2
    learning_rate: float = 0.002
    weight_decay: float = 0.0001
    clip_norm: float = 1.0
    threads: int = 4


def seed_everything(seed: int, threads: int = 4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def clean_text(text: str) -> str:
    """Conservative normalization: keep negation, profanity and punctuation."""
    text = html.unescape(str(text)).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " <url> ", text)
    text = re.sub(r"(?<!\w)@[a-z0-9_]+", " <user> ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def build_vocabulary(token_lists, max_size: int, min_frequency: int):
    counts = Counter(token for tokens in token_lists for token in tokens)
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in counts.most_common():
        if count < min_frequency or len(vocab) >= max_size:
            break
        vocab[token] = len(vocab)
    return vocab


def encode(tokens, vocab: dict, max_length: int):
    # At least one token keeps empty/punctuation-only inputs valid at inference.
    return [vocab.get(token, 1) for token in tokens[:max_length]] or [1]


class CommentDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = np.asarray(targets, dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return torch.tensor(self.sequences[index], dtype=torch.long), torch.from_numpy(self.targets[index])


def collate_batch(batch):
    sequences, targets = zip(*batch)
    lengths = torch.tensor([len(x) for x in sequences], dtype=torch.long)
    return pad_sequence(sequences, batch_first=True, padding_value=0), lengths, torch.stack(targets)


class ToxicLSTM(nn.Module):
    """A causal LSTM using the final REAL token, never the final padded token."""
    def __init__(self, vocab_size, embedding_dim=64, hidden_size=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, len(LABELS))
        # Gate-wise initialization, rather than applying ReLU initialization to LSTM gates.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.embedding.weight[0].zero_()
        for name, parameter in self.lstm.named_parameters():
            if "weight_ih" in name:
                for gate in parameter.chunk(4, dim=0):
                    nn.init.xavier_uniform_(gate)
            elif "weight_hh" in name:
                for gate in parameter.chunk(4, dim=0):
                    nn.init.orthogonal_(gate)
            elif "bias" in name:
                nn.init.zeros_(parameter)
        # PyTorch adds two bias vectors: set only one forget-gate bias to one.
        with torch.no_grad():
            self.lstm.bias_ih_l0[hidden_size:2 * hidden_size].fill_(1.0)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, tokens, lengths):
        output, _ = self.lstm(self.embedding(tokens))
        last_real = output[torch.arange(output.size(0), device=output.device), lengths.to(output.device) - 1]
        return self.classifier(self.dropout(last_real))  # Six logits, no sigmoid during BCE training.


def metric_report(y_true, probabilities, thresholds=0.5):
    prediction = probabilities >= np.asarray(thresholds)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, prediction, average=None, zero_division=0)
    rows = []
    for index, label in enumerate(LABELS):
        p, r, _ = precision_recall_curve(y_true[:, index], probabilities[:, index])
        has_both = len(np.unique(y_true[:, index])) == 2
        rows.append({"label": label, "precision": float(precision[index]),
                     "recall": float(recall[index]), "f1": float(f1[index]),
                     "support": int(support[index]),
                     "roc_auc": float(roc_auc_score(y_true[:, index], probabilities[:, index])) if has_both else None,
                     "pr_auc": float(auc(r, p)) if has_both else None,
                     "average_precision": float(average_precision_score(y_true[:, index], probabilities[:, index])) if has_both else None})
    valid = [row for row in rows if row["roc_auc"] is not None]
    return {"per_label": rows, "micro_f1": float(f1_score(y_true, prediction, average="micro", zero_division=0)),
            "macro_f1": float(np.mean(f1)), "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)),
            "macro_roc_auc": float(np.mean([r["roc_auc"] for r in valid])) if valid else None,
            "macro_pr_auc": float(np.mean([r["pr_auc"] for r in valid])) if valid else None,
            "macro_average_precision": float(np.mean([r["average_precision"] for r in valid])) if valid else None,
            "subset_accuracy": float(np.mean(np.all(prediction == y_true, axis=1))),
            "hamming_loss": float(np.mean(prediction != y_true))}


def tune_thresholds(y_validation, validation_probabilities):
    grid = np.linspace(0.05, 0.95, 91)
    thresholds = []
    for j in range(len(LABELS)):
        scores = [f1_score(y_validation[:, j], validation_probabilities[:, j] >= t, zero_division=0) for t in grid]
        # In a tie prefer the threshold closest to 0.5, reducing arbitrary extremes.
        best = np.flatnonzero(np.isclose(scores, max(scores)))
        chosen = min(best, key=lambda k: abs(grid[k] - 0.5))
        thresholds.append(float(grid[chosen]))
    return thresholds


def prepare_data(csv_path: Path, config: Config, results: Path):
    raw = pd.read_csv(csv_path)
    required = ["id", "comment_text"] + LABELS
    missing_columns = sorted(set(required) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if raw[LABELS].isna().any().any() or not np.isin(raw[LABELS].to_numpy(), [0, 1]).all():
        raise ValueError("Targets must be six non-missing binary labels.")
    missing_text = int(raw.comment_text.isna().sum())
    data = raw.copy()
    data["clean_text"] = data.comment_text.fillna("").map(clean_text)
    empty = int((data.clean_text == "").sum())
    data = data.loc[data.clean_text != ""].copy()
    normalized_duplicates = int(data.duplicated("clean_text").sum())
    conflicting = int((data.groupby("clean_text", sort=False)[LABELS].nunique().max(axis=1) > 1).sum())
    # Keep normalized identical comments together by collapsing them before splitting.
    # When annotations disagree, union the positive labels; record this explicitly.
    data = data.groupby("clean_text", sort=False, as_index=False).agg(
        {"id": "first", "comment_text": "first", **{label: "max" for label in LABELS}})
    data["tokens"] = data.clean_text.map(tokenize)
    data["token_length"] = data.tokens.map(len)
    y = data[LABELS].to_numpy(dtype=np.float32)
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=config.seed)
    train_idx, hold_idx = next(splitter.split(np.zeros((len(data), 1)), y))
    second = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=config.seed)
    val_local, test_local = next(second.split(np.zeros((len(hold_idx), 1)), y[hold_idx]))
    indices = {"train": train_idx, "validation": hold_idx[val_local], "test": hold_idx[test_local]}
    text_sets = [set(data.iloc[idx].clean_text) for idx in indices.values()]
    assert not (text_sets[0] & text_sets[1] or text_sets[0] & text_sets[2] or text_sets[1] & text_sets[2])
    if any((y[idx].sum(axis=0) == 0).any() for idx in indices.values()):
        raise ValueError("Every split must contain positives for every label.")
    vocab = build_vocabulary(data.iloc[train_idx].tokens, config.vocab_size, config.min_frequency)
    sequences = [encode(tokens, vocab, config.max_length) for tokens in data.tokens]
    split_table = data[["id"] + LABELS].copy()
    split_table["split"] = ""
    for split, idx in indices.items():
        split_table.loc[idx, "split"] = split
    split_table.to_csv(results / "split_assignments.csv", index=False)
    save_json(results / "vocabulary.json", vocab)
    summary = {"raw_rows": len(raw), "raw_shape": list(raw.shape),
               "missing_text": missing_text, "empty_text_removed": empty,
               "duplicate_ids": int(raw.id.duplicated().sum()),
               "exact_text_duplicates": int(raw.comment_text.duplicated().sum()),
               "normalized_duplicates_collapsed": normalized_duplicates,
               "conflicting_normalized_groups": conflicting,
               "duplicate_policy": "Collapse normalized identical comments; union positive labels; retain first ID/text.",
               "usable_rows": len(data), "vocabulary_size": len(vocab),
               "source_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
               "label_counts_raw": {c: int(raw[c].sum()) for c in LABELS},
               "all_zero_fraction_raw": float((raw[LABELS].sum(axis=1) == 0).mean()),
               "multi_label_fraction_raw": float((raw[LABELS].sum(axis=1) > 1).mean()),
               "split_sizes": {name: len(idx) for name, idx in indices.items()},
               "split_positive_counts": {name: {label: int(y[idx, j].sum()) for j, label in enumerate(LABELS)} for name, idx in indices.items()},
               "train_token_length_quantiles": {str(q): float(data.iloc[train_idx].token_length.quantile(q)) for q in [0.5, 0.9, 0.95, 0.99]},
               "max_length": config.max_length,
               "truncated_fraction": {name: float((data.iloc[idx].token_length > config.max_length).mean()) for name, idx in indices.items()},
               "oov_token_fraction": {name: float(sum(token not in vocab for tokens in data.iloc[idx].tokens for token in tokens) / max(1, sum(len(tokens) for tokens in data.iloc[idx].tokens))) for name, idx in indices.items()}}
    save_json(results / "data_summary.json", summary)
    plot_eda(raw, data, indices, results, config.max_length)
    return data, y, sequences, indices, vocab, summary


def plot_eda(raw, data, indices, results, max_length=128):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    counts = raw[LABELS].sum().sort_values()
    axes[0].barh(counts.index, counts.values, color="#236a8a")
    axes[0].set(xlabel="Positive comments", title="Original data label imbalance")
    axes[1].hist(data.token_length.clip(upper=500), bins=60, color="#236a8a")
    axes[1].axvline(max_length, color="#c5533f", linestyle="--", label=f"{max_length}-token cap")
    axes[1].set(xlabel="Tokens (values above 500 grouped at 500)", ylabel="Comments", title="Normalized comment lengths")
    axes[1].legend(); fig.tight_layout(); fig.savefig(results / "eda_overview.png", dpi=170); plt.close(fig)
    co = raw[LABELS].to_numpy().T @ raw[LABELS].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 5.5)); im = ax.imshow(co, cmap="Blues")
    ax.set_xticks(range(6), LABELS, rotation=35, ha="right"); ax.set_yticks(range(6), LABELS)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{co[i,j]:,}", ha="center", va="center", fontsize=8,
                    color="white" if co[i,j] > co.max() * .5 else "black")
    ax.set_title("Label co-occurrence counts in original data")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(results / "label_cooccurrence.png", dpi=170); plt.close(fig)
    prevalences = pd.DataFrame({name: data.iloc[idx][LABELS].mean() for name, idx in indices.items()})
    prevalences.to_csv(results / "split_prevalence.csv")
    ax = prevalences.plot.bar(figsize=(10, 4), color=["#236a8a", "#51a98b", "#c5533f"])
    ax.set(ylabel="Positive fraction", title="Multi-label stratification audit")
    ax.figure.tight_layout(); ax.figure.savefig(results / "split_prevalence.png", dpi=170); plt.close(ax.figure)


def make_loader(sequences, y, idx, config, shuffle=False):
    generator = torch.Generator().manual_seed(config.seed)
    dataset = CommentDataset([sequences[i] for i in idx], y[idx])
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle,
                      collate_fn=collate_batch, num_workers=0, generator=generator)


@torch.no_grad()
def predict_loader(model, loader, device, criterion=None):
    model.eval(); probabilities = []; targets = []; total_loss = 0.0
    for tokens, lengths, target in loader:
        tokens, lengths, target = tokens.to(device), lengths.to(device), target.to(device)
        logits = model(tokens, lengths)
        if criterion is not None:
            total_loss += float(criterion(logits, target)) * len(target)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(target.cpu().numpy())
    return np.concatenate(targets), np.concatenate(probabilities), total_loss / len(loader.dataset)


def train_experiment(name, config, sequences, y, indices, vocab, output, device):
    seed_everything(config.seed, config.threads)
    train_loader = make_loader(sequences, y, indices["train"], config, shuffle=True)
    validation_loader = make_loader(sequences, y, indices["validation"], config)
    model = ToxicLSTM(len(vocab), config.embedding_dim, config.hidden_size, config.dropout).to(device)
    positives = y[indices["train"]].sum(axis=0)
    weights = (len(indices["train"]) - positives) / np.maximum(positives, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights, dtype=torch.float32, device=device)
                                    if name == "weighted_bce" else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=1, factor=0.5)
    best_score = -1.0; stale = 0; history = []; best_epoch = 0
    experiment_dir = output / name; experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ROOT / "models" / f"{name}.pt"
    for epoch in range(1, config.epochs + 1):
        start = time.perf_counter(); model.train(); loss_sum = 0.0; norms = []
        for batch, (tokens, lengths, target) in enumerate(train_loader, start=1):
            tokens, lengths, target = tokens.to(device), lengths.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tokens, lengths), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss.")
            loss.backward()
            norms.append(float(nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)))
            optimizer.step(); loss_sum += float(loss.detach()) * len(target)
            if batch % 100 == 0:
                print(f"{name} epoch {epoch}: batch {batch}/{len(train_loader)}, loss {loss_sum / min(batch * config.batch_size, len(train_loader.dataset)):.4f}", flush=True)
        y_val, p_val, val_loss = predict_loader(model, validation_loader, device, criterion)
        report = metric_report(y_val, p_val)
        score = report["macro_average_precision"]
        history.append({"epoch": epoch, "train_loss": loss_sum / len(train_loader.dataset),
                        "validation_loss": val_loss, "validation_macro_ap": score,
                        "validation_macro_f1_at_05": report["macro_f1"],
                        "validation_micro_f1_at_05": report["micro_f1"],
                        "gradient_norm_mean_before_clip": float(np.mean(norms)),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "elapsed_seconds": time.perf_counter() - start})
        pd.DataFrame(history).to_csv(experiment_dir / "history.csv", index=False)
        print(f"{name} epoch {epoch}: train={history[-1]['train_loss']:.4f}, val={val_loss:.4f}, macro AP={score:.4f}, {history[-1]['elapsed_seconds']:.1f}s", flush=True)
        if score > best_score + 1e-4:
            best_score = score; best_epoch = epoch; stale = 0
            torch.save({"model_state": model.state_dict(), "config": asdict(config), "vocabulary": vocab,
                        "labels": LABELS, "loss_configuration": name, "best_epoch": best_epoch}, checkpoint_path)
        else:
            stale += 1
        scheduler.step(score)
        if stale >= config.patience:
            print(f"Early stopping {name} at epoch {epoch}.", flush=True); break
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state"])
    y_val, p_val, _ = predict_loader(model, validation_loader, device, criterion)
    thresholds = tune_thresholds(y_val, p_val)
    report_05 = metric_report(y_val, p_val, 0.5)
    report_tuned = metric_report(y_val, p_val, thresholds)
    np.savez_compressed(experiment_dir / "validation_predictions.npz", y_true=y_val, probabilities=p_val)
    save_json(experiment_dir / "validation_metrics_at_05.json", report_05)
    save_json(experiment_dir / "validation_metrics_tuned.json", report_tuned)
    saved["thresholds"] = thresholds; saved["positive_weights"] = weights.tolist()
    torch.save(saved, checkpoint_path)
    plot_history(pd.DataFrame(history), experiment_dir, name)
    return {"experiment": name, "best_epoch": best_epoch, "epochs_run": len(history),
            "validation_macro_ap": report_tuned["macro_average_precision"],
            "validation_macro_pr_auc": report_tuned["macro_pr_auc"],
            "validation_macro_f1_at_05": report_05["macro_f1"],
            "validation_macro_f1_tuned": report_tuned["macro_f1"],
            "validation_micro_f1_tuned": report_tuned["micro_f1"],
            "thresholds": thresholds, "positive_weights": weights.tolist(),
            "training_seconds": float(sum(row["elapsed_seconds"] for row in history)),
            "parameters": sum(p.numel() for p in model.parameters())}


def plot_history(history, directory, name):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].plot(history.epoch, history.train_loss, label="Train")
    axes[0].plot(history.epoch, history.validation_loss, label="Validation")
    axes[0].set(title="Loss (within this configuration)", xlabel="Epoch"); axes[0].legend()
    axes[1].plot(history.epoch, history.validation_macro_ap, marker="o", color="#236a8a")
    axes[1].set(title="Validation macro average precision", xlabel="Epoch")
    axes[2].plot(history.epoch, history.validation_micro_f1_at_05, label="Micro F1")
    axes[2].plot(history.epoch, history.validation_macro_f1_at_05, label="Macro F1")
    axes[2].set(title="Validation F1 at threshold 0.5", xlabel="Epoch"); axes[2].legend()
    fig.suptitle(name.replace("_", " ")); fig.tight_layout()
    fig.savefig(directory / "training_curves.png", dpi=170); plt.close(fig)


def plot_test(y_true, probabilities, thresholds, results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for j, label in enumerate(LABELS):
        p, r, _ = precision_recall_curve(y_true[:, j], probabilities[:, j])
        fpr, tpr, _ = roc_curve(y_true[:, j], probabilities[:, j])
        axes[0].plot(r, p, label=f"{label} (AP {average_precision_score(y_true[:,j], probabilities[:,j]):.3f})")
        axes[1].plot(fpr, tpr, label=label)
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Held-out precision-recall curves", xlim=(0,1), ylim=(0,1))
    axes[1].plot([0,1],[0,1],"k--",alpha=.5)
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="Held-out ROC curves", xlim=(0,1), ylim=(0,1))
    for ax in axes: ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(results / "test_pr_roc_curves.png", dpi=180); plt.close(fig)
    predicted = probabilities >= np.asarray(thresholds)
    fig, axes = plt.subplots(2,3,figsize=(10,6))
    for j, ax in enumerate(axes.flat):
        cm = confusion_matrix(y_true[:,j], predicted[:,j], labels=[0,1])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for k in range(2):
                ax.text(k,i,f"{cm[i,k]:,}",ha="center",va="center",color="white" if cm[i,k] > cm.max()*.5 else "black")
        ax.set(title=LABELS[j], xlabel="Predicted", ylabel="Actual", xticks=[0,1], yticks=[0,1])
        ax.grid(False)
    fig.tight_layout(); fig.savefig(results / "test_confusion_matrices.png", dpi=170); plt.close(fig)


def save_error_analysis(data_test, y_true, probabilities, thresholds, results, max_length=128):
    prediction = probabilities >= np.asarray(thresholds)
    rows = []
    for j, label in enumerate(LABELS):
        fp = np.flatnonzero((prediction[:,j] == 1) & (y_true[:,j] == 0))
        fn = np.flatnonzero((prediction[:,j] == 0) & (y_true[:,j] == 1))
        for kind, positions in [("false_positive", fp[np.argsort(-probabilities[fp,j])][:5]),
                                ("false_negative", fn[np.argsort(probabilities[fn,j])][:5])]:
            for pos in positions:
                row = data_test.iloc[pos]
                rows.append({"id": row.id, "label": label, "error_type": kind,
                             "probability": float(probabilities[pos,j]), "threshold": thresholds[j],
                             "original_tokens": int(row.token_length), "was_truncated": bool(row.token_length > max_length),
                             "comment_excerpt": str(row.comment_text)[:800]})
    pd.DataFrame(rows).to_csv(results / "error_examples.csv", index=False)
    errors_per_sample = (prediction != y_true).sum(axis=1)
    truncated = data_test.token_length.to_numpy() > max_length
    stats = {"samples_with_any_error": int((errors_per_sample > 0).sum()),
             "total_test_samples": len(data_test),
             "error_fraction_truncated": float((errors_per_sample[truncated] > 0).mean()) if truncated.any() else None,
             "error_fraction_not_truncated": float((errors_per_sample[~truncated] > 0).mean()) if (~truncated).any() else None}
    save_json(results / "error_summary.json", stats)


def run(csv_path: Path, config: Config, output: Path | None = None):
    output = output or ROOT / "results"
    output.mkdir(parents=True, exist_ok=True); (ROOT / "models").mkdir(exist_ok=True)
    seed_everything(config.seed, config.threads)
    save_json(output / "config.json", asdict(config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; preparing {csv_path}", flush=True)
    data, y, sequences, indices, vocab, summary = prepare_data(csv_path, config, output)
    print(f"Data ready: {summary['split_sizes']}; vocabulary {len(vocab)}", flush=True)
    comparison = []
    for name in ["unweighted_bce", "weighted_bce"]:
        comparison.append(train_experiment(name, config, sequences, y, indices, vocab, output, device))
    save_json(output / "experiment_comparison.json", comparison)
    selected = max(comparison, key=lambda row: row["validation_macro_ap"])
    checkpoint = torch.load(ROOT / "models" / f"{selected['experiment']}.pt", map_location=device, weights_only=False)
    model = ToxicLSTM(len(vocab), config.embedding_dim, config.hidden_size, config.dropout).to(device)
    model.load_state_dict(checkpoint["model_state"])
    # Only now construct and evaluate the held-out test loader.
    test_loader = make_loader(sequences, y, indices["test"], config)
    y_test, p_test, _ = predict_loader(model, test_loader, device)
    test_05 = metric_report(y_test, p_test, 0.5)
    test_tuned = metric_report(y_test, p_test, selected["thresholds"])
    save_json(output / "test_metrics_at_05.json", test_05)
    save_json(output / "test_metrics_tuned.json", test_tuned)
    save_json(output / "all_negative_baseline.json", {
        "micro_f1": 0.0, "macro_f1": 0.0,
        "subset_accuracy": float((y_test.sum(axis=1) == 0).mean()),
        "hamming_loss": float(y_test.mean())})
    pd.DataFrame(test_tuned["per_label"]).to_csv(output / "test_per_label_metrics.csv", index=False)
    np.savez_compressed(output / "test_predictions.npz", ids=data.iloc[indices["test"]].id.to_numpy(dtype=str),
                        y_true=y_test, probabilities=p_test, thresholds=np.asarray(selected["thresholds"]))
    torch.save(checkpoint, ROOT / "models" / "selected_model.pt")
    plot_test(y_test, p_test, selected["thresholds"], output)
    save_error_analysis(data.iloc[indices["test"]], y_test, p_test, selected["thresholds"], output, config.max_length)
    final = {"status": "completed", "dataset": "Full original Jigsaw training CSV; normalized duplicates collapsed",
             "selected_experiment": selected["experiment"], "selection_rule": "Highest validation macro average precision",
             "threshold_rule": "Per-label F1 maximization on validation only, grid 0.05 to 0.95",
             "device": str(device), "torch_version": torch.__version__,
             "config": asdict(config), "data": summary, "experiments": comparison,
             "test_at_05": test_05, "test_tuned": test_tuned}
    save_json(output / "run_summary.json", final)
    print(f"Completed. Selected {selected['experiment']}; test macro F1={test_tuned['macro_f1']:.4f}, micro F1={test_tuned['micro_f1']:.4f}.", flush=True)
    return final


def predict_texts(texts, checkpoint_path: Path | None = None):
    checkpoint_path = checkpoint_path or ROOT / "models" / "selected_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]; vocab = checkpoint["vocabulary"]
    model = ToxicLSTM(len(vocab), config["embedding_dim"], config["hidden_size"], config["dropout"])
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    batch = [(torch.tensor(encode(tokenize(clean_text(t)), vocab, config["max_length"])), torch.zeros(6)) for t in texts]
    tokens, lengths, _ = collate_batch(batch)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(tokens, lengths)).numpy()
    return pd.DataFrame(probabilities, columns=LABELS), probabilities >= np.asarray(checkpoint["thresholds"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/train.csv")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run(args.data, Config(epochs=args.epochs, batch_size=args.batch_size, threads=args.threads))

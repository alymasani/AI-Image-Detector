"""Scoring a detector: accuracy, F1 and AUROC, plus averaging results over seeds.

"Fake" is the positive class throughout: label 1, and `prob_fake` is the model's probability
that an image is AI-generated.

Shared file formats (written by evaluation, read by the plots):
    results/predictions/<run>__<test_set>.csv   path, label, prob_fake
    results/metrics.csv                         run, model, seed, test_set, accuracy, f1, auroc
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

METRIC_NAMES = ("accuracy", "f1", "auroc")
METRICS_COLUMNS = ("run", "model", "seed", "test_set", *METRIC_NAMES)


def compute_metrics(
    labels: Sequence[int] | np.ndarray,
    prob_fake: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Accuracy and F1 at `threshold`, and AUROC (which doesn't depend on a threshold)."""
    y_true = np.asarray(labels)
    scores = np.asarray(prob_fake, dtype=float)
    if y_true.ndim != 1 or y_true.shape != scores.shape:
        raise ValueError(
            f"labels and prob_fake must be 1-D and the same length, got {y_true.shape} and "
            f"{scores.shape}"
        )
    if len(y_true) == 0:
        raise ValueError("Need at least one prediction")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("labels must be 0 (real) or 1 (fake)")
    if len(np.unique(y_true)) < 2:
        raise ValueError("AUROC needs both real and fake images in the labels")
    if not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("prob_fake must be probabilities between 0 and 1")

    y_pred = (scores >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, scores)),
    }


def read_predictions(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """(labels, prob_fake) from a predictions CSV."""
    df = pd.read_csv(path)
    return df["label"].to_numpy(), df["prob_fake"].to_numpy()


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of every metric over seeds, per model and test set.

    Returns one row per (model, test_set) with columns like accuracy_mean and accuracy_std.
    The std is 0 when there is only one seed.
    """
    grouped = metrics.groupby(["model", "test_set"], sort=False)[list(METRIC_NAMES)]
    summary = grouped.agg(["mean", "std"]).fillna(0.0)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["seeds"] = grouped.size()
    return summary.reset_index()

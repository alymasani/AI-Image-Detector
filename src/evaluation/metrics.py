"""Binary classification metrics with AI-generated (label 1) as the positive class.

All rates are computed from the confusion matrix with explicit zero-denominator handling: an
undefined metric is returned as ``NaN`` and the reason is listed in ``undefined``. AUROC uses the
continuous AI-class probability and is undefined when only one class is present.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from .. import LABEL_AI, LABEL_REAL


def _div(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")


def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, int]:
    """TN, FP, FN, TP with 1 = AI-generated as positive."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "tn": int(np.sum((y_true == LABEL_REAL) & (y_pred == LABEL_REAL))),
        "fp": int(np.sum((y_true == LABEL_REAL) & (y_pred == LABEL_AI))),
        "fn": int(np.sum((y_true == LABEL_AI) & (y_pred == LABEL_REAL))),
        "tp": int(np.sum((y_true == LABEL_AI) & (y_pred == LABEL_AI))),
    }


def compute_metrics(
    y_true: Sequence[int],
    prob_ai: Sequence[float],
    threshold: float = 0.5,
    loss: Optional[float] = None,
) -> Dict[str, Any]:
    """All evaluation metrics for one set of predictions.

    ``prediction = 1 if prob_ai >= threshold else 0``.
    """
    y_true = np.asarray(y_true, dtype=int)
    prob_ai = np.asarray(prob_ai, dtype=float)
    if y_true.shape != prob_ai.shape:
        raise ValueError("y_true and prob_ai must have the same shape")
    y_pred = (prob_ai >= threshold).astype(int)
    c = confusion_counts(y_true, y_pred)
    tn, fp, fn, tp = c["tn"], c["fp"], c["fn"], c["tp"]
    n = tn + fp + fn + tp
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    tnr = _div(tn, tn + fp)
    f1 = _div(2 * tp, 2 * tp + fp + fn)
    undefined = []
    if tp + fp == 0:
        undefined.append("precision: no samples predicted as AI")
    if tp + fn == 0:
        undefined.append("recall/FNR: no AI samples present")
    if tn + fp == 0:
        undefined.append("TNR/FPR: no real samples present")
    if 2 * tp + fp + fn == 0:
        undefined.append("f1: no positives predicted or present")
    if len(np.unique(y_true)) < 2:
        auroc = float("nan")
        undefined.append("auroc: only one class present")
    else:
        auroc = float(roc_auc_score(y_true, prob_ai))
    balanced = (recall + tnr) / 2 if not (math.isnan(recall) or math.isnan(tnr)) else float("nan")
    return {
        "n": n,
        "n_real": tn + fp,
        "n_ai": tp + fn,
        "threshold": threshold,
        "accuracy": _div(tp + tn, n),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "fpr": _div(fp, fp + tn),
        "fnr": _div(fn, fn + tp),
        "tnr": tnr,
        "balanced_accuracy": balanced,
        "loss": float(loss) if loss is not None else float("nan"),
        "confusion_matrix": [[tn, fp], [fn, tp]],   # rows = true (real, ai), cols = predicted (real, ai)
        **c,
        "undefined": undefined,
    }


def chance_baselines(y_true: Sequence[int], seed: int = 0) -> Dict[str, Dict[str, Any]]:
    """Reference baselines on a test set.

    * ``analytic_chance``: expected values of a classifier that guesses AI with probability 0.5
      independently of the image, on a balanced set: accuracy 0.5, precision 0.5 (= prevalence),
      recall 0.5, F1 0.5, FPR 0.5, AUROC 0.5. Exact expectations, not a sampled run.
    * ``random_scores``: one seeded realisation of uniform random scores (shows sampling noise).
    * ``always_ai``: majority/constant rule predicting AI for everything (accuracy 0.5,
      recall 1, precision 0.5, F1 0.667, FPR 1, AUROC 0.5 since all scores tie).
    """
    y = np.asarray(y_true, dtype=int)
    prev = float(np.mean(y == LABEL_AI))
    analytic = {
        "rule": "predict AI with probability 0.5 independent of input (expected values)",
        "accuracy": 0.5, "precision": prev, "recall": 0.5,
        "f1": _div(2 * prev * 0.5, prev + 0.5), "auroc": 0.5, "fpr": 0.5, "fnr": 0.5, "tnr": 0.5,
        "balanced_accuracy": 0.5,
    }
    rng = np.random.default_rng(seed)
    random_scores = compute_metrics(y, rng.uniform(size=len(y)))
    random_scores["rule"] = f"uniform random AI score, seed={seed}, threshold 0.5"
    always_ai = compute_metrics(y, np.ones(len(y)))
    always_ai["rule"] = "always predict AI (constant score 1.0)"
    return {"analytic_chance": analytic, "random_scores": random_scores, "always_ai": always_ai}


def degradation(familiar: Dict[str, Any], unseen: Dict[str, Any]) -> Dict[str, float]:
    """Familiar minus unseen. Accuracy drop in percentage points; positive = worse on unseen."""
    return {
        "accuracy_drop_pp": 100.0 * (familiar["accuracy"] - unseen["accuracy"]),
        "f1_drop": familiar["f1"] - unseen["f1"],
        "auroc_drop": familiar["auroc"] - unseen["auroc"],
        "fpr_change": unseen["fpr"] - familiar["fpr"],
        "fnr_change": unseen["fnr"] - familiar["fnr"],
    }

"""Bootstrap confidence intervals and paired tests over a FIXED trained model's test predictions.

These intervals quantify uncertainty from the finite test sample only. They say nothing about
variability across independent training runs (seeds), and they assume test images are
independent draws, which is optimistic (GenImage images of the same ImageNet class/prompt are
correlated). All resampling is stratified by class so every replicate stays balanced and AUROC
stays defined.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from .. import LABEL_AI, LABEL_REAL

METRICS = ("accuracy", "f1", "auroc")


def _metric_fns(threshold: float) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    def acc(y, p):
        return float(np.mean((p >= threshold).astype(int) == y))

    def f1(y, p):
        pred = (p >= threshold).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        den = 2 * tp + np.sum((pred == 1) & (y == 0)) + np.sum((pred == 0) & (y == 1))
        return float(2 * tp / den) if den else float("nan")

    def auroc(y, p):
        return float(roc_auc_score(y, p))

    return {"accuracy": acc, "f1": f1, "auroc": auroc}


def _stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for cls in (LABEL_REAL, LABEL_AI):
        idx = np.flatnonzero(y == cls)
        parts.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(parts)


def _summ(point: float, samples: np.ndarray, alpha: float) -> Dict[str, float]:
    s = samples[~np.isnan(samples)]
    lo, hi = (np.quantile(s, [alpha / 2, 1 - alpha / 2]) if len(s) else (np.nan, np.nan))
    return {"point": float(point), "ci_low": float(lo), "ci_high": float(hi), "n_valid_resamples": int(len(s))}


def bootstrap_ci(y: Sequence[int], prob_ai: Sequence[float], n_resamples: int = 2000, seed: int = 519,
                 threshold: float = 0.5, alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """Percentile bootstrap CIs for accuracy, F1 and AUROC of one model on one test set."""
    y, p = np.asarray(y, int), np.asarray(prob_ai, float)
    fns = _metric_fns(threshold)
    rng = np.random.default_rng(seed)
    boots = {m: np.empty(n_resamples) for m in METRICS}
    for b in range(n_resamples):
        idx = _stratified_indices(y, rng)
        for m in METRICS:
            boots[m][b] = fns[m](y[idx], p[idx])
    return {m: _summ(fns[m](y, p), boots[m], alpha) for m in METRICS}


def paired_model_difference(y: Sequence[int], prob_a: Sequence[float], prob_b: Sequence[float],
                            n_resamples: int = 2000, seed: int = 519, threshold: float = 0.5,
                            alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """CI of metric(A) - metric(B) when both models were evaluated on the SAME samples (paired)."""
    y, a, b_ = np.asarray(y, int), np.asarray(prob_a, float), np.asarray(prob_b, float)
    fns = _metric_fns(threshold)
    rng = np.random.default_rng(seed)
    boots = {m: np.empty(n_resamples) for m in METRICS}
    for i in range(n_resamples):
        idx = _stratified_indices(y, rng)
        for m in METRICS:
            boots[m][i] = fns[m](y[idx], a[idx]) - fns[m](y[idx], b_[idx])
    return {m: _summ(fns[m](y, a) - fns[m](y, b_), boots[m], alpha) for m in METRICS}


def cross_generator_difference(familiar: pd.DataFrame, unseen: pd.DataFrame, n_resamples: int = 2000,
                               seed: int = 519, threshold: float = 0.5,
                               alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """CI of metric(familiar) - metric(unseen) for ONE model under the shared-real protocol.

    Both frames are prediction exports (``sample_id``, ``true_label``, ``probability_ai``). The real
    photographs are the same samples in both sets, so each replicate draws ONE resample of the
    shared real images (used in both sets) and independent resamples of each set's AI images.
    This keeps the positive correlation induced by the shared reals.
    """
    fr, ur = familiar[familiar.true_label == LABEL_REAL], unseen[unseen.true_label == LABEL_REAL]
    if set(fr.sample_id) != set(ur.sample_id):
        raise ValueError("Real samples differ between the two sets; the shared-real resampling does not apply")
    ur = ur.set_index("sample_id").loc[fr.sample_id].reset_index()
    fa, ua = familiar[familiar.true_label == LABEL_AI], unseen[unseen.true_label == LABEL_AI]
    p_fr, p_ur = fr.probability_ai.to_numpy(), ur.probability_ai.to_numpy()
    p_fa, p_ua = fa.probability_ai.to_numpy(), ua.probability_ai.to_numpy()
    fns = _metric_fns(threshold)

    def metric_pair(ir, ifa, iua):
        out = {}
        yf = np.r_[np.zeros(len(ir), int), np.ones(len(ifa), int)]
        yu = np.r_[np.zeros(len(ir), int), np.ones(len(iua), int)]
        pf = np.r_[p_fr[ir], p_fa[ifa]]
        pu = np.r_[p_ur[ir], p_ua[iua]]
        for m in METRICS:
            out[m] = fns[m](yf, pf) - fns[m](yu, pu)
        return out

    point = metric_pair(np.arange(len(p_fr)), np.arange(len(p_fa)), np.arange(len(p_ua)))
    rng = np.random.default_rng(seed)
    boots = {m: np.empty(n_resamples) for m in METRICS}
    for b in range(n_resamples):
        ir = rng.integers(0, len(p_fr), len(p_fr))
        d = metric_pair(ir, rng.integers(0, len(p_fa), len(p_fa)), rng.integers(0, len(p_ua), len(p_ua)))
        for m in METRICS:
            boots[m][b] = d[m]
    return {m: _summ(point[m], boots[m], alpha) for m in METRICS}


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> Dict[str, Optional[float]]:
    """Exact (binomial) McNemar test on paired correct/incorrect outcomes of two models."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    n = only_a + only_b
    p = float(binomtest(only_a, n, 0.5).pvalue) if n else 1.0
    return {"a_correct_b_wrong": only_a, "a_wrong_b_correct": only_b, "discordant": n, "p_value": p}

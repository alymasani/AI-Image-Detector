"""Metric correctness against hand calculations and scikit-learn; statistics sanity checks."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.evaluation.metrics import chance_baselines, compute_metrics, degradation
from src.evaluation.statistics import bootstrap_ci, cross_generator_difference, mcnemar_exact, paired_model_difference


def test_hand_calculated_example():
    # 4 real, 4 AI. Predictions at 0.5: real -> [0,0,1,0] (1 FP); AI -> [1,1,0,1] (1 FN)
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    p = [0.1, 0.2, 0.7, 0.4, 0.9, 0.6, 0.3, 0.8]
    m = compute_metrics(y, p)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 1, 1, 3)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["precision"] == pytest.approx(3 / 4) and m["recall"] == pytest.approx(3 / 4)
    assert m["f1"] == pytest.approx(3 / 4) and m["fpr"] == pytest.approx(1 / 4) and m["fnr"] == pytest.approx(1 / 4)
    # AUROC: of 16 (real, AI) pairs the AI image scores higher in 4+3+2+4 = 13 -> 0.8125
    assert m["auroc"] == pytest.approx(13 / 16)
    assert m["confusion_matrix"] == [[3, 1], [1, 3]]


def test_matches_sklearn_on_random_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    m = compute_metrics(y, p)
    pred = (p >= 0.5).astype(int)
    assert m["accuracy"] == pytest.approx(accuracy_score(y, pred))
    assert m["precision"] == pytest.approx(precision_score(y, pred))
    assert m["recall"] == pytest.approx(recall_score(y, pred))
    assert m["f1"] == pytest.approx(f1_score(y, pred))
    assert m["auroc"] == pytest.approx(roc_auc_score(y, p))


def test_threshold_is_inclusive_and_auroc_uses_scores():
    m = compute_metrics([0, 1], [0.5, 0.5])
    assert m["fp"] == 1 and m["tp"] == 1 and m["auroc"] == 0.5


def test_zero_denominators_and_single_class():
    m = compute_metrics([0, 0, 0], [0.1, 0.2, 0.3])
    assert math.isnan(m["precision"]) and math.isnan(m["recall"]) and math.isnan(m["auroc"])
    assert m["accuracy"] == 1.0 and m["fpr"] == 0.0
    assert any("only one class" in u for u in m["undefined"])


def test_chance_baselines_and_degradation():
    y = np.array([0] * 50 + [1] * 50)
    b = chance_baselines(y)
    assert b["analytic_chance"]["accuracy"] == 0.5 and b["analytic_chance"]["auroc"] == 0.5
    assert b["always_ai"]["recall"] == 1.0 and b["always_ai"]["f1"] == pytest.approx(2 / 3)
    d = degradation({"accuracy": 0.9, "f1": 0.9, "auroc": 0.95, "fpr": 0.1, "fnr": 0.1},
                    {"accuracy": 0.6, "f1": 0.5, "auroc": 0.7, "fpr": 0.1, "fnr": 0.7})
    assert d["accuracy_drop_pp"] == pytest.approx(30.0) and d["auroc_drop"] == pytest.approx(0.25)


def test_bootstrap_deterministic_and_contains_point():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(100, int), np.ones(100, int)]
    p = np.clip(y * 0.4 + rng.random(200) * 0.6, 0, 1)
    a, b = bootstrap_ci(y, p, 200, seed=3), bootstrap_ci(y, p, 200, seed=3)
    assert a == b
    for v in a.values():
        assert v["ci_low"] <= v["point"] <= v["ci_high"]
    d = paired_model_difference(y, p, p, 100, seed=3)
    assert all(v["point"] == 0 and v["ci_low"] == 0 == v["ci_high"] for v in d.values())


def test_cross_generator_difference_identical_sets_is_zero():
    rng = np.random.default_rng(2)
    reals = pd.DataFrame({"sample_id": [f"r{i}" for i in range(50)], "true_label": 0, "probability_ai": rng.random(50) * 0.6})
    fakes = pd.DataFrame({"sample_id": [f"f{i}" for i in range(50)], "true_label": 1, "probability_ai": 0.4 + rng.random(50) * 0.6})
    fam = pd.concat([reals, fakes])
    out = cross_generator_difference(fam, fam.copy(), 100, seed=0)
    assert all(v["point"] == 0 for v in out.values())


def test_mcnemar():
    r = mcnemar_exact([True] * 10 + [False] * 0, [False] * 10)
    assert r["a_correct_b_wrong"] == 10 and r["p_value"] < 0.01
    assert mcnemar_exact([True, False], [True, False])["p_value"] == 1.0

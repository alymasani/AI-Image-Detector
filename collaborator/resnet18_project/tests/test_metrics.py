import numpy as np
import pandas as pd
import pytest

from detector.metrics import compute_metrics, read_predictions, summarize


def test_perfect_predictions():
    metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics == {"accuracy": 1.0, "f1": 1.0, "auroc": 1.0}


def test_inverted_predictions():
    metrics = compute_metrics([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    assert metrics == {"accuracy": 0.0, "f1": 0.0, "auroc": 0.0}


def test_random_predictions_score_about_half():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=20000)
    metrics = compute_metrics(labels, rng.random(20000))
    for value in metrics.values():
        assert value == pytest.approx(0.5, abs=0.02)


def test_hand_computed_example():
    # Predictions at 0.5 are [0, 1, 0, 1]: one real and one fake right, one of each wrong.
    # So accuracy 0.5, precision = recall = 0.5, F1 0.5. Of the 4 (fake, real) score pairs,
    # 3 rank the fake higher (0.4 < 0.6 is the miss), so AUROC is 0.75.
    metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
    assert metrics == pytest.approx({"accuracy": 0.5, "f1": 0.5, "auroc": 0.75})


def test_threshold_changes_accuracy_but_not_auroc():
    labels, scores = [0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9]
    strict = compute_metrics(labels, scores, threshold=0.7)
    assert strict["accuracy"] == 0.75  # only 0.9 is called fake now
    assert strict["auroc"] == compute_metrics(labels, scores)["auroc"]


def test_score_exactly_at_threshold_counts_as_fake():
    assert compute_metrics([0, 1], [0.2, 0.5])["accuracy"] == 1.0


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ([0, 1], [0.5], "same length"),
        ([], [], "at least one"),
        ([0, 2], [0.1, 0.9], "0 \\(real\\) or 1"),
        ([1, 1], [0.1, 0.9], "both real and fake"),
        ([0, 1], [0.1, 1.5], "between 0 and 1"),
    ],
)
def test_bad_input_is_rejected(labels, scores, message):
    with pytest.raises(ValueError, match=message):
        compute_metrics(labels, scores)


def test_read_predictions(tmp_path):
    path = tmp_path / "preds.csv"
    path.write_text("path,label,prob_fake\na.jpg,0,0.25\nb.jpg,1,0.75\n")
    labels, scores = read_predictions(path)
    assert labels.tolist() == [0, 1] and scores.tolist() == [0.25, 0.75]


def test_summarize_averages_over_seeds():
    metrics = pd.DataFrame(
        {
            "run": ["a", "b", "a", "b"],
            "model": ["cnn"] * 4,
            "seed": [0, 1, 0, 1],
            "test_set": ["test_sd14", "test_sd14", "test_midjourney", "test_midjourney"],
            "accuracy": [0.9, 0.8, 0.6, 0.6],
            "f1": [0.9, 0.7, 0.5, 0.7],
            "auroc": [0.95, 0.85, 0.7, 0.6],
        }
    )
    summary = summarize(metrics).set_index("test_set")
    assert summary.loc["test_sd14", "accuracy_mean"] == pytest.approx(0.85)
    assert summary.loc["test_sd14", "accuracy_std"] == pytest.approx(np.std([0.9, 0.8], ddof=1))
    assert summary.loc["test_midjourney", "accuracy_std"] == 0
    assert (summary["seeds"] == 2).all()


def test_summarize_single_seed_has_zero_std():
    metrics = pd.DataFrame(
        {"model": ["cnn"], "test_set": ["test_sd14"], "accuracy": [0.9], "f1": [0.9], "auroc": [1]}
    )
    assert summarize(metrics).loc[0, "auroc_std"] == 0

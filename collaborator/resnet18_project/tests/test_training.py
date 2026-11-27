import json

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import load_script

from detector.data import make_loaders
from detector.metrics import METRICS_COLUMNS, read_predictions, summarize
from detector.models import build_model
from detector.training import (
    HISTORY_COLUMNS,
    TEST_SETS,
    evaluate_run,
    fit,
    predict,
    upsert_metrics,
)

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


@pytest.fixture
def loaders(processed_data_dir):
    # A tiny stand-in dataset from conftest.py: every split has two real and two fake images.
    return make_loaders(processed_data_dir, batch_size=2, num_workers=0)


@pytest.fixture
def resnet():
    # pretrained=False keeps the tests from downloading the ImageNet weights.
    return build_model("resnet18", pretrained=False)


def test_fit_saves_best_checkpoint_and_history(tmp_path, loaders, resnet):
    run_dir = tmp_path / "runs" / "resnet18_seed0"
    history = fit(resnet, loaders, run_dir, epochs=2, device=CPU)

    assert tuple(history.columns) == HISTORY_COLUMNS
    assert history["epoch"].tolist() == [1, 2]
    assert history["val_auroc"].between(0, 1).all()
    pd.testing.assert_frame_equal(pd.read_csv(run_dir / "history.csv"), history)

    checkpoint = torch.load(run_dir / "best.pt", weights_only=True)
    best = history.loc[history["val_auroc"].idxmax()]
    assert checkpoint["epoch"] == best["epoch"]
    assert checkpoint["val_auroc"] == pytest.approx(best["val_auroc"])


def test_fit_rejects_zero_epochs(tmp_path, loaders, resnet):
    with pytest.raises(ValueError, match="epochs"):
        fit(resnet, loaders, tmp_path, epochs=0, device=CPU)


def test_predict_returns_probabilities_in_dataset_order(loaders, resnet):
    labels, prob_fake, loss = predict(resnet, loaders["test_sd14"], CPU)
    assert labels.tolist() == loaders["test_sd14"].dataset.labels
    assert prob_fake.shape == labels.shape
    assert ((prob_fake >= 0) & (prob_fake <= 1)).all()
    assert loss > 0


def test_evaluate_run_writes_predictions_and_metrics(tmp_path, loaders, resnet):
    fit(resnet, loaders, tmp_path / "runs" / "resnet18_seed0", epochs=1, device=CPU)
    metrics = evaluate_run(resnet, loaders, "resnet18_seed0", "resnet18", 0, tmp_path, device=CPU)

    assert tuple(metrics.columns) == METRICS_COLUMNS
    assert metrics["test_set"].tolist() == list(TEST_SETS)
    for test_set in TEST_SETS:
        predictions = pd.read_csv(tmp_path / "predictions" / f"resnet18_seed0__{test_set}.csv")
        dataset = loaders[test_set].dataset
        assert predictions["path"].tolist() == [path for path, _ in dataset.samples]
        labels, prob_fake = read_predictions(
            tmp_path / "predictions" / f"resnet18_seed0__{test_set}.csv"
        )
        assert labels.tolist() == dataset.labels
        assert ((prob_fake >= 0) & (prob_fake <= 1)).all()
    assert len(summarize(metrics)) == len(TEST_SETS)


def metric_rows(run: str, accuracy: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run": run,
                "model": "resnet18",
                "seed": 0,
                "test_set": t,
                "accuracy": accuracy,
                "f1": accuracy,
                "auroc": accuracy,
            }
            for t in TEST_SETS
        ],
        columns=METRICS_COLUMNS,
    )


def test_upsert_metrics_replaces_rows_of_the_same_run(tmp_path):
    path = tmp_path / "metrics.csv"
    upsert_metrics(metric_rows("cnn_seed0", 0.6), path)
    upsert_metrics(metric_rows("resnet18_seed0", 0.7), path)
    table = upsert_metrics(metric_rows("resnet18_seed0", 0.9), path)

    assert len(table) == 4
    on_disk = pd.read_csv(path)
    pd.testing.assert_frame_equal(on_disk, table.reset_index(drop=True))
    by_run = on_disk.groupby("run")["accuracy"].unique()
    assert by_run["cnn_seed0"].tolist() == [0.6]  # other runs are left alone
    assert by_run["resnet18_seed0"].tolist() == [0.9]


def test_train_script_end_to_end(tmp_path, processed_data_dir):
    train = load_script("train")
    args = [
        "--no-pretrained",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--num-workers",
        "0",
        "--data-dir",
        str(processed_data_dir),
        "--results-dir",
        str(tmp_path),
    ]
    train.main(args)
    train.main(args)  # running the same run again replaces its rows

    run_dir = tmp_path / "runs" / "resnet18_seed0"
    assert {p.name for p in run_dir.iterdir()} == {"best.pt", "history.csv", "config.json"}
    config = json.loads((run_dir / "config.json").read_text())
    assert config["run"] == "resnet18_seed0" and config["model"] == "resnet18"
    assert config["pretrained"] is False
    # Nothing that only describes this computer ends up in the saved settings.
    assert not {"data_dir", "results_dir", "num_workers"} & config.keys()

    metrics = pd.read_csv(tmp_path / "metrics.csv")
    assert metrics["run"].tolist() == ["resnet18_seed0"] * len(TEST_SETS)
    assert np.isfinite(metrics[["accuracy", "f1", "auroc"]].to_numpy()).all()

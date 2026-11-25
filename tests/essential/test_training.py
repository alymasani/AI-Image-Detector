"""Early stopping logic and an end-to-end CPU training/evaluation smoke test with resumption."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.training.early_stopping import EarlyStopping
from src.training.trainer import train_model
from src.utils import load_config

from .conftest import ROOT


def test_early_stopping_patience_and_min_delta():
    es = EarlyStopping(patience=2, min_delta=0.01)
    seq = [1.0, 0.8, 0.795, 0.81, 0.7]   # 0.795 is not a >0.01 improvement
    improved = [es.step(i + 1, v) for i, v in enumerate(seq[:4])]
    assert improved == [True, True, False, False]
    assert es.should_stop and es.best_epoch == 2 and es.best_loss == 0.8
    assert "2 epochs" in es.reason


def test_early_stopping_resets_on_improvement_and_nan():
    es = EarlyStopping(patience=2, min_delta=0.0)
    for i, v in enumerate([1.0, 1.1, 0.9, 1.0]):
        es.step(i + 1, v)
    assert not es.should_stop and es.best_epoch == 3
    es2 = EarlyStopping()
    es2.step(1, float("nan"))
    assert es2.should_stop and "NaN" in es2.reason


def _cfg(tmp_path: Path, manifest_dir: Path, root: Path, epochs: int, name: str = "t"):
    return load_config(ROOT / "configs" / "smoke.yaml", [
        f"data.root={root}", f"data.manifest_dir={manifest_dir}", "data.num_workers=0",
        f"output.runs_dir={tmp_path / 'runs'}", f"output.checkpoints_dir={tmp_path / 'ckpt'}",
        "training.device=cpu", "training.batch_size=8", f"training.max_epochs={epochs}",
        "training.early_stopping_patience=5", "training.log_every_batches=0", f"experiment_name={name}"])


def test_training_pipeline_checkpoint_and_resume(tmp_path, manifests, synthetic_root):
    mdir, _ = manifests
    s1 = train_model(_cfg(tmp_path, mdir, synthetic_root, epochs=1))
    assert s1["epochs_run"] == 1 and s1["best_epoch"] == 1 and s1["nan_free"]
    assert (tmp_path / "ckpt" / "t" / "best.pt").exists() and (tmp_path / "ckpt" / "t" / "last.pt").exists()
    s2 = train_model(_cfg(tmp_path, mdir, synthetic_root, epochs=2), resume=str(tmp_path / "ckpt" / "t" / "last.pt"))
    hist = pd.read_csv(tmp_path / "runs" / "t" / "history.csv")
    assert list(hist.epoch) == [1, 2] and s2["epochs_run"] == 2
    expected = {"train_loss", "val_loss", "train_accuracy", "val_accuracy", "val_precision", "val_recall", "val_f1",
                "val_auroc", "learning_rate", "epoch_seconds", "cumulative_seconds", "is_best"}
    assert expected <= set(hist.columns)


def test_evaluation_exports_predictions(tmp_path, manifests, synthetic_root, monkeypatch):
    from src import evaluate as ev

    mdir, _ = manifests
    train_model(_cfg(tmp_path, mdir, synthetic_root, epochs=1, name="e"))
    from .conftest import suite_config

    suite = suite_config()
    suite["dataset"]["manifest_dir"] = str(mdir)
    suite["dataset"]["root"] = str(synthetic_root)
    monkeypatch.setattr(ev, "merged_dataset_config", lambda _=None: suite)
    out = ev.evaluate_checkpoint(str(tmp_path / "ckpt" / "e" / "best.pt"), ["test_sd", "test_mj"],
                                 results_dir=str(tmp_path / "res"), device="cpu", num_workers=0)
    assert [o["test"] for o in out] == ["test_sd", "test_mj"]
    pred = pd.read_csv(tmp_path / "res" / "predictions" / "e__test_mj.csv")
    cols = {"sample_id", "model_name", "checkpoint_identifier", "dataset_split", "generator", "true_label",
            "predicted_label", "probability_real", "probability_ai", "correct"}
    assert cols <= set(pred.columns) and len(pred) == 12
    assert ((pred.probability_ai >= 0.5).astype(int) == pred.predicted_label).all()
    sd = pd.read_csv(tmp_path / "res" / "predictions" / "e__test_sd.csv")
    # shared-real protocol: identical real samples get identical scores in both test sets
    r1 = sd[sd.true_label == 0].set_index("sample_id").probability_ai.sort_index()
    r2 = pred[pred.true_label == 0].set_index("sample_id").probability_ai.sort_index()
    assert r1.index.equals(r2.index) and (r1 - r2).abs().max() < 1e-6

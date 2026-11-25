"""Evaluate a selected checkpoint on held-out manifests and export predictions + metrics.

The checkpoint must be the one chosen by Stable Diffusion VALIDATION loss (``best.pt``). Test
results are never used for model or threshold selection. Preprocessing is rebuilt from the
checkpoint's own training config with augmentation disabled.

Examples:
    python -m src.evaluate --checkpoint checkpoints/cnn/best.pt --tests test_sd test_mj
    python -m src.evaluate --checkpoint checkpoints/cnn/best.pt --tests test_sd test_mj --extra-jpeg 75
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import torch

from .data.create_splits import merged_dataset_config
from .data.dataset import make_dataset, make_loader, read_manifest
from .evaluation.metrics import compute_metrics
from .models.factory import display_name
from .training.checkpointing import load_model_for_inference
from .training.trainer import predict
from .utils import portable_path, resolve_data_root, resolve_path, select_device, sha256_file, \
    write_json


def checkpoint_identifier(path: Path, ckpt: Dict[str, Any]) -> str:
    name = ckpt["config"].get("experiment_name", ckpt["model_name"])
    return f"{name}/{path.name}@epoch{ckpt['epoch']}#{sha256_file(path)[:12]}"


def evaluation_tag(experiment: str, test: str, extra_jpeg: Optional[int]) -> str:
    return f"{experiment}__{test}" + (f"__jpeg{extra_jpeg}" if extra_jpeg else "")


def evaluate_checkpoint(
    checkpoint: str,
    tests: Sequence[str],
    data_root: Optional[str] = None,
    extra_jpeg: Optional[int] = None,
    results_dir: str = "results",
    device: str = "auto",
    batch_size: int = 64,
    num_workers: Optional[int] = None,
    dataset_config: Optional[str] = None,
    cache_dir: Optional[str] = "__from_checkpoint__",
) -> List[Dict[str, Any]]:
    """Evaluate one checkpoint on each named test manifest. Returns the metric summaries."""
    dcfg = merged_dataset_config(dataset_config)
    dev = select_device(device)
    ckpt_path = resolve_path(checkpoint)
    model, ckpt = load_model_for_inference(ckpt_path, dev)
    train_cfg = ckpt["config"]
    data_cfg = dict(train_cfg["data"])
    if cache_dir != "__from_checkpoint__":
        data_cfg["cache_dir"] = cache_dir
    root = resolve_data_root({"data": {"root": dcfg["dataset"]["root"]}}, data_root)
    manifest_dir = resolve_path(dcfg["dataset"]["manifest_dir"])
    experiment = train_cfg.get("experiment_name", ckpt["model_name"])
    ident = checkpoint_identifier(ckpt_path, ckpt)
    res = resolve_path(results_dir)
    workers = int(num_workers if num_workers is not None else data_cfg.get("num_workers", 0))
    use_amp = bool(train_cfg["training"].get("mixed_precision", False)) and dev.type == "cuda"
    summaries = []
    for test in tests:
        mpath = manifest_dir / dcfg["manifests"][test]
        df = read_manifest(mpath)
        ds = make_dataset(df, root, data_cfg, train=False, extra_jpeg_quality=extra_jpeg, return_index=True)
        loader = make_loader(ds, batch_size, shuffle=False, num_workers=workers, seed=0, device_type=dev.type)
        t0 = time.perf_counter()
        out = predict(model, loader, dev, use_amp)
        seconds = time.perf_counter() - t0
        order = out["indices"]
        rows = df.iloc[order].reset_index(drop=True)
        threshold = float(dcfg["evaluation"]["threshold"])
        preds = pd.DataFrame({
            "sample_id": rows["sample_id"],
            "model_name": display_name(ckpt["model_config"]),
            "experiment": experiment,
            "checkpoint_identifier": ident,
            "dataset_split": test,
            "generator": rows["generator"],
            "source_folder": rows["source_folder"],
            "true_label": rows["label"].astype(int),
            "predicted_label": (out["prob_ai"] >= threshold).astype(int),
            "probability_real": 1.0 - out["prob_ai"],
            "probability_ai": out["prob_ai"],
            "loss": out["loss_per_sample"],
            "extra_jpeg_quality": extra_jpeg if extra_jpeg else "",
        })
        assert (preds["true_label"].to_numpy() == out["labels"]).all(), "prediction/label misalignment"
        preds["correct"] = preds["true_label"] == preds["predicted_label"]
        tag = evaluation_tag(experiment, test, extra_jpeg)
        pred_path = res / "predictions" / f"{tag}.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(pred_path, index=False, float_format="%.6f")
        metrics = compute_metrics(preds["true_label"], preds["probability_ai"], threshold, loss=out["loss"])
        summary = {
            "experiment": experiment, "model": display_name(ckpt["model_config"]), "test": test,
            "extra_jpeg_quality": extra_jpeg, "checkpoint": portable_path(ckpt_path), "checkpoint_identifier": ident,
            "best_epoch": ckpt["epoch"], "manifest": portable_path(mpath), "manifest_sha256": sha256_file(mpath),
            "training_manifest_sha256": ckpt.get("manifest_sha256", {}),
            "preprocessing": {k: data_cfg.get(k) for k in ("image_size", "interpolation", "jpeg_quality", "normalize")},
            "device": str(dev), "eval_seconds": round(seconds, 2), "metrics": metrics,
            "predictions_file": portable_path(pred_path),
        }
        write_json(summary, res / "summaries" / f"eval_{tag}.json")
        print(f"{experiment:>20s} | {test:15s} | jpeg={extra_jpeg or '-':>3} | acc={metrics['accuracy']:.4f} "
              f"f1={metrics['f1']:.4f} auroc={metrics['auroc']:.4f} fpr={metrics['fpr']:.4f} n={metrics['n']}")
        summaries.append(summary)
    return summaries


@torch.no_grad()
def measure_latency(checkpoint: str, device: str = "auto", batch_sizes: Sequence[int] = (1, 32), repeats: int = 20,
                    results_dir: str = "results") -> Dict[str, Any]:
    """Model-only forward latency on random 224x224 input (excludes image decoding)."""
    dev = select_device(device)
    model, ckpt = load_model_for_inference(resolve_path(checkpoint), dev)
    out: Dict[str, Any] = {"experiment": ckpt["config"].get("experiment_name"), "device": str(dev), "results": []}
    for bs in batch_sizes:
        x = torch.randn(bs, 3, 224, 224, device=dev)
        for _ in range(3):
            model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elif dev.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elif dev.type == "mps":
            torch.mps.synchronize()
        ms = (time.perf_counter() - t0) / repeats * 1000
        out["results"].append({"batch_size": bs, "ms_per_batch": round(ms, 3), "ms_per_image": round(ms / bs, 3)})
    write_json(out, resolve_path(results_dir) / "summaries" / f"latency_{out['experiment']}.json")
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint on held-out test manifests")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tests", nargs="+", default=["test_sd", "test_mj"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--dataset-config", default=None, help="e.g. configs/smoke.yaml for synthetic manifests")
    p.add_argument("--extra-jpeg", type=int, default=None, help="additional JPEG quality (robustness E3)")
    p.add_argument("--results", default="results")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--latency", action="store_true", help="also measure model forward latency")
    args = p.parse_args(argv)
    evaluate_checkpoint(args.checkpoint, args.tests, args.data_root, args.extra_jpeg, args.results, args.device,
                        args.batch_size, args.num_workers, args.dataset_config)
    if args.latency:
        print(measure_latency(args.checkpoint, args.device, results_dir=args.results))


if __name__ == "__main__":
    main()

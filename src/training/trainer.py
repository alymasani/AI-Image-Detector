"""Shared training loop used by both models (explicit, no training framework)."""

from __future__ import annotations

import contextlib
import math
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data.dataset import label_counts, limit_per_class, make_dataset, make_loader, read_manifest
from ..evaluation.metrics import compute_metrics
from ..models.factory import build_model, display_name, initialization_source, parameter_counts
from ..utils import environment_report, git_commit, portable_path, resolve_data_root, resolve_path, \
    select_device, set_seed, sha256_file, write_json
from .checkpointing import load_checkpoint, restore_rng_state, save_checkpoint
from .early_stopping import EarlyStopping

OOM_HELP = ("CUDA out of memory. Reduce training.batch_size (e.g. 16) and set training.grad_accum_steps=2 to keep the "
            "effective batch size, or lower data.num_workers. Record the change: both models must use the same "
            "effective batch policy for a fair comparison. Resume with --resume <run>/last.pt.")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def autocast_context(device: torch.device, enabled: bool):
    """fp16 autocast on CUDA when enabled; a no-op context otherwise."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_grad_scaler(enabled: bool):
    if not enabled:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()  # older PyTorch


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool = False) -> Dict[str, Any]:
    """Run the model in eval mode without gradients.

    Returns labels, AI-class probabilities (softmax of the two logits, computed in float32),
    per-sample cross-entropy, and the dataset indices (manifest order when loader is unshuffled).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    labels, probs, losses, indices = [], [], [], []
    for batch in loader:
        images, y = batch[0], batch[1]
        idx = batch[2] if len(batch) > 2 else None
        images = images.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_context(device, use_amp):
            logits = model(images)
        logits = logits.float()
        losses.append(criterion(logits, y).cpu())
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
        labels.append(y.cpu())
        if idx is not None:
            indices.append(idx.cpu())
    out = {
        "labels": torch.cat(labels).numpy() if labels else np.array([], dtype=int),
        "prob_ai": torch.cat(probs).numpy() if probs else np.array([], dtype=float),
        "loss_per_sample": torch.cat(losses).numpy() if losses else np.array([], dtype=float),
    }
    out["loss"] = float(out["loss_per_sample"].mean()) if len(out["loss_per_sample"]) else float("nan")
    if indices:
        out["indices"] = torch.cat(indices).numpy()
    return out


def _cuda_memory(device: torch.device) -> Dict[str, Optional[float]]:
    if device.type != "cuda":
        return {"peak_allocated_gb": None, "peak_reserved_gb": None}
    return {"peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3)}


def train_model(
    cfg: Dict[str, Any],
    data_root: Optional[str] = None,
    resume: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Train one model according to ``cfg`` and return a run summary.

    Outputs: ``<runs_dir>/<experiment_name>/{config.json,history.csv,history.json,summary.json}`` and
    ``<checkpoints_dir>/<experiment_name>/{best.pt,last.pt}``.
    """
    data_cfg, train_cfg, model_cfg = cfg["data"], cfg["training"], cfg["model"]
    name = cfg.get("experiment_name", model_cfg["name"])
    run_dir = resolve_path(cfg["output"]["runs_dir"]) / name
    ckpt_dir = resolve_path(cfg["output"]["checkpoints_dir"]) / name
    if (ckpt_dir / "last.pt").exists() and not resume and not overwrite:
        raise FileExistsError(f"{ckpt_dir}/last.pt exists. Use --resume {ckpt_dir}/last.pt or --overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = int(train_cfg["seed"])
    set_seed(seed)
    device = select_device(train_cfg.get("device", "auto"))
    use_amp = bool(train_cfg.get("mixed_precision", False)) and device.type == "cuda"
    precision_mode = "cuda-fp16-autocast+GradScaler" if use_amp else "fp32"

    # ----- data
    root = resolve_data_root(cfg, data_root)
    manifest_dir = resolve_path(data_cfg["manifest_dir"])
    train_path, val_path = manifest_dir / data_cfg["train_manifest"], manifest_dir / data_cfg["val_manifest"]
    train_df = limit_per_class(read_manifest(train_path), data_cfg.get("train_limit"), seed)
    val_df = limit_per_class(read_manifest(val_path), data_cfg.get("val_limit"), seed)
    manifest_hashes = {train_path.name: sha256_file(train_path), val_path.name: sha256_file(val_path)}
    train_ds = make_dataset(train_df, root, data_cfg, train=True)
    val_ds = make_dataset(val_df, root, data_cfg, train=False)
    workers = int(data_cfg.get("num_workers", 0))
    bs = int(train_cfg["batch_size"])
    train_loader = make_loader(train_ds, bs, shuffle=True, num_workers=workers, seed=seed, device_type=device.type)
    val_loader = make_loader(val_ds, bs, shuffle=False, num_workers=workers, seed=seed, device_type=device.type)

    # ----- model / optimisation
    model = build_model(model_cfg).to(device)
    params = parameter_counts(model)
    criterion = nn.CrossEntropyLoss()
    if train_cfg.get("optimizer", "adam").lower() != "adam":
        raise ValueError("The proposal fixes the optimiser to Adam")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg.get("weight_decay", 0)))
    scaler = make_grad_scaler(use_amp)
    stopper = EarlyStopping(patience=int(train_cfg["early_stopping_patience"]),
                            min_delta=float(train_cfg["early_stopping_min_delta"]))
    accum = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    history: list = []
    start_epoch = 1
    cumulative = 0.0

    if resume:
        ckpt = load_checkpoint(resolve_path(resume), map_location=device)
        if ckpt["model_name"] != model_cfg["name"]:
            raise ValueError(f"Checkpoint model {ckpt['model_name']} != config model {model_cfg['name']}")
        for key, h in manifest_hashes.items():
            if ckpt.get("manifest_sha256", {}).get(key) not in (None, h):
                raise ValueError(f"Manifest {key} changed since the checkpoint was written; refusing to resume")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scaler is not None and ckpt.get("scaler_state"):
            scaler.load_state_dict(ckpt["scaler_state"])
        stopper.load_state_dict(ckpt["early_stopping"])
        history = ckpt.get("history") or []
        cumulative = history[-1]["cumulative_seconds"] if history else 0.0
        start_epoch = int(ckpt["epoch"]) + 1
        restore_rng_state(ckpt["rng_state"])
        train_loader.generator.manual_seed(seed + start_epoch)
        print(f"Resumed from {resume} at epoch {start_epoch}")

    real_n, ai_n = label_counts(train_df)
    vreal, vai = label_counts(val_df)
    print(f"[{name}] model={display_name(model_cfg)} device={device} precision={precision_mode} "
          f"params={params['total']:,} (trainable {params['trainable']:,})")
    print(f"[{name}] train={len(train_df)} (real {real_n}, ai {ai_n})  val={len(val_df)} (real {vreal}, ai {vai})  "
          f"batch={bs} accum={accum} workers={workers}")
    write_json({"config": cfg, "data_root": portable_path(root), "precision_mode": precision_mode}, run_dir / "config.json")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    log_every = int(train_cfg.get("log_every_batches", 100))
    max_epochs = int(train_cfg["max_epochs"])
    epoch = start_epoch - 1
    for epoch in range(start_epoch, max_epochs + 1):
        if stopper.should_stop:
            break
        t0 = time.perf_counter()
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        data_time = compute_time = 0.0
        optimizer.zero_grad(set_to_none=True)
        t_data = time.perf_counter()
        n_batches = len(train_loader)
        for b, (images, y) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            t_comp = time.perf_counter()
            data_time += t_comp - t_data
            try:
                with autocast_context(device, use_amp):
                    logits = model(images)
                    loss = criterion(logits.float(), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss {loss.item()} at epoch {epoch} batch {b}")
                scaled = loss / accum
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                if b % accum == 0 or b == n_batches:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise RuntimeError(OOM_HELP) from exc
                raise
            _sync(device)
            compute_time += time.perf_counter() - t_comp
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
            if log_every and b % log_every == 0:
                print(f"  epoch {epoch} batch {b}/{n_batches} running_loss={loss_sum / seen:.4f} "
                      f"elapsed={time.perf_counter() - t0:.0f}s")
            t_data = time.perf_counter()

        train_seconds = time.perf_counter() - t0
        val = predict(model, val_loader, device, use_amp)
        vm = compute_metrics(val["labels"], val["prob_ai"], 0.5, loss=val["loss"])
        improved = stopper.step(epoch, vm["loss"])
        epoch_seconds = time.perf_counter() - t0
        cumulative += epoch_seconds
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "train_accuracy": correct / max(seen, 1),
            "val_loss": vm["loss"], "val_accuracy": vm["accuracy"], "val_precision": vm["precision"],
            "val_recall": vm["recall"], "val_f1": vm["f1"], "val_auroc": vm["auroc"],
            "val_undefined": "; ".join(vm["undefined"]),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_seconds": round(train_seconds, 2), "train_compute_seconds": round(compute_time, 2),
            "train_data_wait_seconds": round(data_time, 2), "epoch_seconds": round(epoch_seconds, 2),
            "cumulative_seconds": round(cumulative, 2), "is_best": improved,
            **_cuda_memory(device),
        }
        history.append(row)
        ckpt_kwargs = dict(model=model, config=cfg, epoch=epoch, optimizer=optimizer, scaler=scaler,
                           early_stopping=stopper.state_dict(), history=history, manifest_hashes=manifest_hashes,
                           extra={"precision_mode": precision_mode, "device": str(device)})
        if improved:
            save_checkpoint(ckpt_dir / "best.pt", **ckpt_kwargs)
        save_checkpoint(ckpt_dir / "last.pt", **ckpt_kwargs)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        write_json(history, run_dir / "history.json")
        print(f"[{name}] epoch {epoch}/{max_epochs} train_loss={row['train_loss']:.4f} "
              f"train_acc={row['train_accuracy']:.4f} val_loss={vm['loss']:.4f} val_acc={vm['accuracy']:.4f} "
              f"val_f1={vm['f1']:.4f} val_auroc={vm['auroc']:.4f} time={epoch_seconds:.0f}s"
              f"{'  *best*' if improved else ''}")

    stop_reason = stopper.reason or (f"reached max_epochs={max_epochs}" if epoch >= max_epochs else "unknown")
    summary = {
        "experiment_name": name,
        "model": display_name(model_cfg),
        "model_name": model_cfg["name"],
        "initialization": initialization_source(model_cfg),
        "parameters": params,
        "device": str(device),
        "precision_mode": precision_mode,
        "best_epoch": stopper.best_epoch,
        "best_val_loss": stopper.best_loss,
        "epochs_run": len(history),
        "stop_reason": stop_reason,
        "total_training_seconds": round(cumulative, 2),
        "total_compute_seconds": round(sum(h["train_compute_seconds"] for h in history), 2),
        "peak_memory": _cuda_memory(device),
        "train_counts": {"real": real_n, "ai": ai_n},
        "val_counts": {"real": vreal, "ai": vai},
        "manifest_sha256": manifest_hashes,
        "best_checkpoint": portable_path(ckpt_dir / "best.pt"),
        "best_val_metrics": next((h for h in history if h["epoch"] == stopper.best_epoch), None),
        "environment": environment_report(),
        "git_commit": git_commit(),
        "nan_free": all(not math.isnan(h["train_loss"]) for h in history),
    }
    write_json(summary, run_dir / "summary.json")
    print(f"[{name}] done: best epoch {stopper.best_epoch} (val_loss {stopper.best_loss:.4f}); {stop_reason}")
    return summary

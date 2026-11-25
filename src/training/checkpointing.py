"""Checkpoint save/load for resuming training and for weights-only inference.

Checkpoints are loaded with ``torch.load(weights_only=False)`` because they contain the config,
RNG states and early-stopping state as plain Python objects. Only load checkpoints you created
or obtained from a trusted source (pickle can execute code).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn

from ..models.factory import build_model

CHECKPOINT_FORMAT = 1


def rng_state() -> Dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: Dict[str, Any],
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    early_stopping: Optional[Dict[str, Any]] = None,
    history: Optional[list] = None,
    manifest_hashes: Optional[Dict[str, str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomically write a checkpoint (write to ``.tmp`` then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "model_name": config["model"]["name"],
        "model_config": config["model"],
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "early_stopping": early_stopping,
        "history": history,
        "config": config,
        "seed": config.get("training", {}).get("seed"),
        "rng_state": rng_state(),
        "manifest_sha256": manifest_hashes or {},
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, map_location: Any = "cpu") -> Dict[str, Any]:
    """Load a full (trusted) checkpoint dictionary."""
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def load_model_for_inference(path: Path, device: Any = "cpu") -> tuple:
    """Rebuild the model from a checkpoint without optimizer/training objects.

    Pretrained-weight download is skipped (``pretrained=False``) because the checkpoint already
    contains every weight. Returns ``(model_in_eval_mode, checkpoint_dict)``.
    """
    ckpt = load_checkpoint(path, map_location="cpu")
    model_cfg = dict(ckpt["model_config"])
    if model_cfg.get("name") == "resnet18":
        model_cfg["pretrained"] = False
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt

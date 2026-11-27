"""Train a detector, keep its best epoch, and test it on the two test sets.

A "run" is one model trained with one seed, named like resnet18_seed0. It saves:

    results/runs/<run>/best.pt                  the weights from its best epoch
    results/runs/<run>/history.csv              loss and validation AUROC after each epoch
    results/predictions/<run>__<test_set>.csv   how likely the model thinks each test image is fake
    results/metrics.csv                         accuracy, F1 and AUROC, one row per run and test set

These are the files the plotting code in `detector.plots` reads.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from detector.data import REPO_ROOT
from detector.metrics import METRICS_COLUMNS, compute_metrics

RESULTS_DIR = REPO_ROOT / "results"
TEST_SETS = ("test_sd14", "test_midjourney")
HISTORY_COLUMNS = ("epoch", "train_loss", "val_loss", "val_auroc")
CHECKPOINT_NAME = "best.pt"


def set_seed(seed: int) -> None:
    """Fix every random number generator so running the same seed again gives the same result."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_device() -> torch.device:
    """The GPU if there is one, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast(device: torch.device):
    """On the GPU, do the math in 16-bit numbers, which roughly doubles the speed.

    We use bfloat16, which handles large and small values as well as normal 32-bit floats, so
    training stays stable without extra tricks. On the CPU nothing changes.
    """
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> float:
    """Go through the training images once, updating the model after every batch.

    Returns the average loss per image over the epoch.
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, seen = 0.0, 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device):
            loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(labels)
        seen += len(labels)
    return total_loss / seen


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the model on every image in a split without changing it.

    Returns three things, in the same order as the images in the loader:
    the true labels, the model's probability that each image is fake, and the average loss.
    """
    model.eval()
    all_labels, all_probs = [], []
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(device):
            logits = model(images)
        logits = logits.float()
        total_loss += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        all_probs.append(logits.softmax(dim=1)[:, 1].cpu())
        all_labels.append(labels.cpu())
    labels_arr = torch.cat(all_labels).numpy()
    return labels_arr, torch.cat(all_probs).numpy(), total_loss / len(labels_arr)


def fit(
    model: nn.Module,
    loaders: Mapping[str, DataLoader],
    run_dir: str | Path,
    epochs: int,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Train the model and save the weights from its best epoch.

    The optimizer is AdamW, and the learning rate starts at `lr` and smoothly drops to zero by
    the last batch (a cosine schedule), so the model takes smaller steps as it settles in.

    After each epoch we check the model on the validation images. If its AUROC is the best so
    far, the weights are saved to <run_dir>/best.pt. We pick the best epoch by validation
    AUROC, never by test results, so the test numbers stay honest.

    history.csv is saved after every epoch, so a run that crashes partway still leaves one.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    device = device or default_device()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(loaders["train"])
    )

    rows = []
    best_auroc = -1.0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, scheduler, device)
        labels, prob_fake, val_loss = predict(model, loaders["val"], device)
        val_auroc = compute_metrics(labels, prob_fake)["auroc"]
        rows.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auroc": val_auroc}
        )
        pd.DataFrame(rows, columns=HISTORY_COLUMNS).to_csv(run_dir / "history.csv", index=False)

        improved = val_auroc > best_auroc
        if improved:
            best_auroc = val_auroc
            torch.save(
                {"epoch": epoch, "val_auroc": val_auroc, "state_dict": model.state_dict()},
                run_dir / CHECKPOINT_NAME,
            )
        print(
            f"epoch {epoch}/{epochs}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
            f"val AUROC {val_auroc:.4f}{'  (best, saved)' if improved else ''}"
        )
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def load_best(model: nn.Module, run_dir: str | Path, device: torch.device) -> dict:
    """Put the saved best weights back into `model`. Returns which epoch they came from and
    that epoch's validation AUROC."""
    checkpoint = torch.load(Path(run_dir) / CHECKPOINT_NAME, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return {"epoch": checkpoint["epoch"], "val_auroc": checkpoint["val_auroc"]}


def evaluate_run(
    model: nn.Module,
    loaders: Mapping[str, DataLoader],
    run: str,
    model_name: str,
    seed: int,
    results_dir: str | Path = RESULTS_DIR,
    test_sets: Iterable[str] = TEST_SETS,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Test the run's best weights on each test set.

    Saves the model's prediction for every test image to results/predictions/, and returns
    accuracy, F1 and AUROC for each test set as rows ready for results/metrics.csv.
    """
    device = device or default_device()
    results_dir = Path(results_dir)
    load_best(model, results_dir / "runs" / run, device)
    predictions_dir = results_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for test_set in test_sets:
        loader = loaders[test_set]
        labels, prob_fake, _ = predict(model, loader, device)
        # Test images are never shuffled, so the predictions come back in the same order as the
        # file list, and we can match each prediction to its image.
        paths = [path for path, _ in loader.dataset.samples]
        pd.DataFrame({"path": paths, "label": labels, "prob_fake": prob_fake}).to_csv(
            predictions_dir / f"{run}__{test_set}.csv", index=False
        )
        rows.append(
            {
                "run": run,
                "model": model_name,
                "seed": seed,
                "test_set": test_set,
                **compute_metrics(labels, prob_fake),
            }
        )
    return pd.DataFrame(rows, columns=METRICS_COLUMNS)


def upsert_metrics(rows: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Add a run's results to the metrics file ("upsert" means update or insert).

    If the file already has rows for the same run, they are replaced, so training a run again
    updates its numbers instead of listing it twice. Rows for other runs, such as the CNN's,
    are left alone. Returns the whole table as saved.
    """
    path = Path(path)
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[~existing["run"].isin(rows["run"])]
        table = pd.concat([existing, rows], ignore_index=True)
    else:
        table = rows
    table = table[list(METRICS_COLUMNS)]
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table

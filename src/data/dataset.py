"""Manifest-driven PyTorch dataset and DataLoader construction shared by both models.

Images are loaded on demand (never the whole dataset in RAM). Corrupt-file policy: manifests only
contain files that decoded successfully when they were built; if a file becomes unreadable later the
dataset raises ``CorruptImageError`` naming the file. It never substitutes another image or label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .. import LABEL_AI, LABEL_REAL
from ..utils import seed_worker
from .preprocessing import CorruptImageError, augment, cache_path_for, jpeg_roundtrip, load_rgb, standardizer, \
    to_normalized_tensor


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read a manifest CSV with string ids/classes (no float coercion of hashes or blank classes)."""
    df = pd.read_csv(path, dtype={"sample_id": str, "imagenet_class": str, "sha256": str, "phash": str},
                     keep_default_na=False, na_values=[])
    df["label"] = df["label"].astype(int)
    return df


def label_counts(df: pd.DataFrame) -> Tuple[int, int]:
    """(real, ai) counts."""
    return int((df.label == LABEL_REAL).sum()), int((df.label == LABEL_AI).sum())


def limit_per_class(df: pd.DataFrame, limit: Optional[int], seed: int) -> pd.DataFrame:
    """Seeded subsample of at most ``limit`` rows per class (pilot runs); manifest order is preserved."""
    if not limit:
        return df
    parts = [g.sample(n=min(int(limit), len(g)), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sort_index()


class ManifestImageDataset(Dataset):
    """Items are ``(image tensor 3x224x224 float32, label int64)`` (plus the row index if requested)."""

    def __init__(self, df: pd.DataFrame, root: str | Path, data_cfg: Dict[str, Any], train: bool,
                 extra_jpeg_quality: Optional[int] = None, return_index: bool = False):
        self.paths = df["relative_path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.shas = df["sha256"].tolist() if "sha256" in df else [None] * len(df)
        self.root = Path(root)
        self.cfg = dict(data_cfg)
        self.train = train
        self.extra_jpeg_quality = extra_jpeg_quality
        self.return_index = return_index
        self.standardize = standardizer(self.cfg)
        cache = self.cfg.get("cache_dir")
        self.cache_dir = Path(cache) if cache else None

    def __len__(self) -> int:
        return len(self.paths)

    def _standardized(self, i: int) -> Image.Image:
        if self.cache_dir is not None and self.shas[i]:
            cached = cache_path_for(self.cache_dir, self.cfg, self.shas[i])
            if cached.exists():
                return load_rgb(cached)
        return self.standardize(load_rgb(self.root / self.paths[i]))

    def __getitem__(self, i: int):
        try:
            img = self._standardized(i)
        except CorruptImageError as exc:
            raise CorruptImageError(f"{exc} (manifest row {i}; fix or rebuild the manifest)") from exc
        if self.extra_jpeg_quality:
            img = jpeg_roundtrip(img, self.extra_jpeg_quality)   # E3 robustness: AFTER standardisation
        if self.train:
            img = augment(img, self.cfg.get("augment", {}) or {})
        x = to_normalized_tensor(img, self.cfg.get("normalize", "imagenet"))
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return (x, y, torch.tensor(i)) if self.return_index else (x, y)


def make_dataset(df: pd.DataFrame, root: str | Path, data_cfg: Dict[str, Any], train: bool,
                 extra_jpeg_quality: Optional[int] = None, return_index: bool = False) -> ManifestImageDataset:
    cfg = dict(data_cfg)
    if cfg.get("cache_dir"):
        from ..utils import resolve_path

        cfg["cache_dir"] = str(resolve_path(cfg["cache_dir"]))
    return ManifestImageDataset(df, root, cfg, train, extra_jpeg_quality, return_index)


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0, seed: int = 0,
                device_type: str = "cpu") -> DataLoader:
    """Seeded DataLoader. Validation/test loaders are unshuffled (manifest order).

    Worker RNGs derive from the loader's generator (``seed_worker``), so augmentation is reproducible.
    On Windows/macOS workers are spawned processes; the dataset holds only picklable lists.
    """
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(ds, batch_size=int(batch_size), shuffle=shuffle, num_workers=int(num_workers),
                      generator=g, worker_init_fn=seed_worker if num_workers else None,
                      pin_memory=device_type == "cuda", persistent_workers=num_workers > 0,
                      prefetch_factor=4 if num_workers else None, drop_last=False)

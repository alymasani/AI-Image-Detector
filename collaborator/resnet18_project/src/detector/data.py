"""Loading the preprocessed GenImage splits for training and evaluation.

Images are expected to be preprocessed already (scripts/preprocess.py), so every image on disk
is 224x224 and the dataset only has to turn it into a normalized tensor.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

DATA_DIR_ENV = "GENIMAGE_DATA_DIR"
REPO_ROOT = Path(__file__).resolve().parents[2]

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SPLITS = ("train", "val", "test_sd14", "test_midjourney")
FORMATS = ("jpeg", "png")
CLASS_NAMES = ("real", "fake")  # label 0 = real, 1 = fake


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """The data folder: the given path, else $GENIMAGE_DATA_DIR, else the repo's data/."""
    if data_dir:
        return Path(data_dir).resolve()
    if os.environ.get(DATA_DIR_ENV):
        return Path(os.environ[DATA_DIR_ENV]).resolve()
    return REPO_ROOT / "data"


def raw_manifest(data_dir: Path, split: str) -> Path:
    return data_dir / "manifests" / f"{split}.csv"


def processed_manifest(data_dir: Path, fmt: str, split: str) -> Path:
    return data_dir / "manifests" / f"processed_{fmt}_{split}.csv"


def center_crop_resize(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Resize the shorter side to `size`, then crop the center `size` x `size` square.

    Real photos come in many aspect ratios while generated images are square, so squashing
    straight to a square would distort only the real photos and hand the model a shortcut.
    """
    img = img.convert("RGB")
    width, height = img.size
    scale = size / min(width, height)
    new_width = max(size, round(width * scale))
    new_height = max(size, round(height * scale))
    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    left = (new_width - size) // 2
    top = (new_height - size) // 2
    return img.crop((left, top, left + size, top + size))


def build_transforms(train: bool) -> v2.Compose:
    """PIL image to a normalized float tensor. Training only adds a random horizontal flip,
    nothing that would blur or re-compress the image and change its generator artifacts."""
    steps: list[v2.Transform] = [v2.ToImage()]
    if train:
        steps.append(v2.RandomHorizontalFlip())
    steps += [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return v2.Compose(steps)


def read_manifest(manifest: Path, limit: int | None = None) -> list[tuple[str, int]]:
    """(path, label) pairs from a manifest. `limit` keeps the first limit // 2 of each class."""
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = [(row["path"], int(row["label"])) for row in csv.DictReader(f)]
    if limit is None:
        return rows
    per_class = limit // 2
    kept: list[tuple[str, int]] = []
    taken = [0, 0]
    for path, label in rows:
        if taken[label] < per_class:
            kept.append((path, label))
            taken[label] += 1
    return kept


class GenImageDataset(Dataset):
    """One split of preprocessed images. Items are (3x224x224 float tensor, label)."""

    def __init__(
        self,
        manifest: Path,
        data_dir: Path,
        transform: v2.Compose | None = None,
        limit: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.samples = read_manifest(Path(manifest), limit)
        self.transform = transform or build_transforms(train=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        rel_path, label = self.samples[index]
        path = self.data_dir / rel_path
        with Image.open(path) as img:
            img = img.convert("RGB")
        if img.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"{path} is {img.size[0]}x{img.size[1]}, expected {IMAGE_SIZE}x{IMAGE_SIZE}. "
                "Use the processed manifests (run scripts/preprocess.py first)."
            )
        return self.transform(img), label

    @property
    def labels(self) -> list[int]:
        return [label for _, label in self.samples]


def make_loaders(
    data_dir: str | Path | None = None,
    fmt: str = "jpeg",
    batch_size: int = 64,
    num_workers: int = 4,
    limit: int | None = None,
    seed: int = 0,
) -> dict[str, DataLoader]:
    """DataLoaders for every split. Only train is shuffled (and flipped), with a seeded order."""
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {FORMATS}, got {fmt!r}")
    root = resolve_data_dir(data_dir)
    loaders = {}
    for split in SPLITS:
        train = split == "train"
        dataset = GenImageDataset(
            processed_manifest(root, fmt, split),
            root,
            transform=build_transforms(train=train),
            limit=limit,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            generator=torch.Generator().manual_seed(seed) if train else None,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
    return loaders

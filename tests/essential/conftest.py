"""Shared fixtures: a small SYNTHETIC GenImage-shaped tree and manifests built from it.

Counts are chosen so every candidate in the SD1.4 val pools is consumed, which guarantees the
grayscale, RGBA and corrupt edge-case files are actually processed by the split builder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.create_splits import build_manifests, merged_dataset_config  # noqa: E402
from src.data.synthetic import make_synthetic_genimage  # noqa: E402

SPLITS = {"train_per_class": 8, "val_per_class": 5, "test_familiar_per_class": 6, "test_unseen_ai": 6,
          "test_unseen_native_real": 6, "seed": 42, "near_duplicate_hamming": 4}


def suite_config():
    cfg = merged_dataset_config()
    cfg["splits"].update(SPLITS)
    return cfg


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    return make_synthetic_genimage(tmp_path_factory.mktemp("genimage"), per_folder=10, seed=1)


@pytest.fixture(scope="session")
def manifests(synthetic_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("manifests")
    meta = build_manifests(suite_config(), synthetic_root, out)
    return out, meta


@pytest.fixture()
def data_cfg():
    return {"image_size": 224, "interpolation": "bilinear", "jpeg_quality": 95, "normalize": "imagenet",
            "augment": {"hflip": True, "random_crop_padding": 8}, "cache_dir": None}

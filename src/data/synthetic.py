"""SYNTHETIC GenImage-shaped fixtures for software tests only. Never report metrics computed on them.

Layout mirrors the real dataset: ``<root>/<generator folder>/{train,val}/{ai,nature}/``. "Real" images
are JPEGs of varied size with smooth random colour fields; "AI" images are square PNGs with a fine
periodic pattern, so the classes are trivially separable (a pipeline test, not a detector test).
Edge cases are planted in the SD 1.4 val pools: a grayscale real photo, an RGBA AI image and a
corrupt file. Two SD val real photos are also copied byte-for-byte into the Midjourney real folder
to exercise exact-duplicate exclusion.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .create_splits import merged_dataset_config


def _real(rng: np.random.Generator, w: int, h: int) -> Image.Image:
    coarse = rng.integers(0, 256, size=(4, 4, 3), dtype=np.uint8)
    img = Image.fromarray(coarse).resize((w, h), Image.Resampling.BICUBIC)
    noise = rng.normal(0, 6, size=(h, w, 3))
    return Image.fromarray(np.clip(np.asarray(img, dtype=float) + noise, 0, 255).astype(np.uint8))


def _ai(rng: np.random.Generator, size: int) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size]
    period = rng.integers(6, 10)
    pattern = 127 + 120 * np.sign(np.sin(2 * np.pi * xx / period) * np.sin(2 * np.pi * yy / period))
    tint = rng.uniform(0.6, 1.0, size=3)
    return Image.fromarray(np.clip(pattern[..., None] * tint, 0, 255).astype(np.uint8))


def make_synthetic_genimage(root: str | Path, per_folder: int = 40, seed: int = 0) -> Path:
    """Create (or recreate) the synthetic tree under ``root`` and return ``root``."""
    root = Path(root)
    cfg = merged_dataset_config()["dataset"]
    rng = np.random.default_rng(seed)
    for gen, ai_size in (("sdv14", 256), ("midjourney", 320)):
        base = root / cfg["generators"][gen]["folder"]
        if base.exists():
            shutil.rmtree(base)
        for part in ("train", "val"):
            ai_dir, real_dir = base / part / cfg["ai_subdir"], base / part / cfg["real_subdir"]
            ai_dir.mkdir(parents=True)
            real_dir.mkdir(parents=True)
            for i in range(per_folder):
                w, h = int(rng.integers(200, 360)), int(rng.integers(180, 300))
                _real(rng, w, h).save(real_dir / f"n{1440764 + i:08d}_{part}_{i:04d}.JPEG", quality=int(rng.integers(75, 96)))
                _ai(rng, ai_size).save(ai_dir / f"{i:03d}_{gen}_{part}_{i:05d}.png")
    sd_val = root / cfg["generators"]["sdv14"]["folder"] / "val"
    _real(rng, 260, 200).convert("L").save(sd_val / cfg["real_subdir"] / "gray_0001.png")
    rgba = _ai(rng, 256).convert("RGBA")
    rgba.putalpha(200)
    rgba.save(sd_val / cfg["ai_subdir"] / "rgba_0001.png")
    (sd_val / cfg["ai_subdir"] / "corrupt_0001.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    mj_real = root / cfg["generators"]["midjourney"]["folder"] / "val" / cfg["real_subdir"]
    for src in sorted((sd_val / cfg["real_subdir"]).glob("*.JPEG"))[:2]:
        shutil.copyfile(src, mj_real / f"copy_{src.name}")
    return root

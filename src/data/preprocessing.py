"""Shared image preprocessing, identical for both classes, all generators and both models.

Order (``configs/cnn.yaml -> data``):
1. Safe decode; convert to RGB. Alpha is composited on white; grayscale/palette images are expanded.
2. Deterministic spatial standardisation: resize the SHORT side to 224 (PIL bilinear, which
   antialiases when downscaling), then centre-crop 224 x 224. Aspect ratio is preserved.
3. Standardised JPEG re-encoding at a fixed quality (95, 4:2:0 chroma subsampling, PIL/libjpeg) for
   EVERY image, so real photos (ImageNet JPEGs) and generated images (PNGs) end up in the same
   container. This does not remove compression history already present in the source photos.
4. [training only] Modest augmentation: random horizontal flip; reflect-pad 8 px and random 224 crop
   (a small translation without resampling).
5. Float tensor in [0, 1], ImageNet mean/std normalisation.

Steps 1-3 are deterministic and can be cached on disk (``cache_path_for``); the cached file is the
exact JPEG byte stream produced in step 3, so cached and on-the-fly loading are bit-identical.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image, ImageFile

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INTERPOLATIONS = {"bilinear": Image.Resampling.BILINEAR, "bicubic": Image.Resampling.BICUBIC,
                  "lanczos": Image.Resampling.LANCZOS}
JPEG_SUBSAMPLING = "4:2:0"

ImageFile.LOAD_TRUNCATED_IMAGES = False   # truncated files must fail, never be silently padded


class CorruptImageError(RuntimeError):
    """An image file could not be decoded. The dataset never substitutes another image."""


def load_rgb(path: str | Path) -> Image.Image:
    """Decode an image file to RGB. Transparent pixels are composited onto white."""
    try:
        with Image.open(path) as im:
            im.load()
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                rgba = im.convert("RGBA")
                canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                return Image.alpha_composite(canvas, rgba).convert("RGB")
            return im.convert("RGB")
    except Exception as exc:  # PIL raises many exception types for corrupt data
        raise CorruptImageError(f"Cannot decode image {path}: {type(exc).__name__}: {exc}") from exc


def standardize_spatial(img: Image.Image, size: int = 224, interpolation: str = "bilinear") -> Image.Image:
    """Resize the short side to ``size`` and centre-crop ``size x size`` (aspect ratio preserved)."""
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = max(size, round(w * scale)), max(size, round(h * scale))
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), INTERPOLATIONS[interpolation])
    left, top = (nw - size) // 2, (nh - size) // 2
    return img.crop((left, top, left + size, top + size))


def jpeg_bytes(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), subsampling=JPEG_SUBSAMPLING)
    return buf.getvalue()


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    """Encode as JPEG at ``quality`` and decode again."""
    with Image.open(io.BytesIO(jpeg_bytes(img, quality))) as im:
        return im.convert("RGB")


class StandardizeImage:
    """Deterministic steps 2-3: spatial standardisation then optional fixed-quality JPEG re-encoding."""

    def __init__(self, size: int = 224, interpolation: str = "bilinear", jpeg_quality: Optional[int] = 95):
        self.size, self.interpolation, self.jpeg_quality = int(size), interpolation, jpeg_quality

    def __call__(self, img: Image.Image) -> Image.Image:
        img = standardize_spatial(img.convert("RGB"), self.size, self.interpolation)
        return jpeg_roundtrip(img, self.jpeg_quality) if self.jpeg_quality else img

    def encoded(self, img: Image.Image) -> bytes:
        """The standardised image as a file: the JPEG stream itself, or lossless PNG when JPEG is off."""
        img = standardize_spatial(img.convert("RGB"), self.size, self.interpolation)
        if self.jpeg_quality:
            return jpeg_bytes(img, self.jpeg_quality)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def standardizer(data_cfg: Dict[str, Any]) -> StandardizeImage:
    return StandardizeImage(int(data_cfg.get("image_size", 224)), data_cfg.get("interpolation", "bilinear"),
                            data_cfg.get("jpeg_quality"))


def standardized_bytes(path: str | Path, data_cfg: Dict[str, Any]) -> bytes:
    """Cache file content for one source image (decoding it yields exactly the on-the-fly result)."""
    return standardizer(data_cfg).encoded(load_rgb(path))


def cache_key(data_cfg: Dict[str, Any]) -> str:
    q = data_cfg.get("jpeg_quality")
    return f"s{int(data_cfg.get('image_size', 224))}_{data_cfg.get('interpolation', 'bilinear')}_" \
           f"{'q' + str(int(q)) if q else 'nojpeg'}"


def cache_path_for(cache_dir: str | Path, data_cfg: Dict[str, Any], sha256: str) -> Path:
    """Where the standardised version of the image with this SHA-256 lives in the cache."""
    ext = "jpg" if data_cfg.get("jpeg_quality") else "png"
    return Path(cache_dir) / cache_key(data_cfg) / sha256[:2] / f"{sha256}.{ext}"


def augment(img: Image.Image, aug_cfg: Dict[str, Any]) -> Image.Image:
    """Training-only augmentation driven by torch's RNG (seeded per DataLoader worker)."""
    if aug_cfg.get("hflip", False) and torch.rand(1).item() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    pad = int(aug_cfg.get("random_crop_padding", 0) or 0)
    if pad > 0:
        arr = np.asarray(img)
        h, w = arr.shape[:2]
        padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        top = int(torch.randint(0, 2 * pad + 1, (1,)).item())
        left = int(torch.randint(0, 2 * pad + 1, (1,)).item())
        img = Image.fromarray(padded[top:top + h, left:left + w])
    return img


def to_normalized_tensor(img: Image.Image, normalize: Optional[str] = "imagenet") -> torch.Tensor:
    """HWC uint8 RGB -> CHW float32, scaled to [0, 1] and normalised with ImageNet statistics."""
    x = torch.from_numpy(np.array(img, dtype=np.uint8, copy=True)).permute(2, 0, 1).float().div_(255.0)
    if normalize == "imagenet":
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        x = (x - mean) / std
    elif normalize not in (None, "none"):
        raise ValueError(f"Unknown normalisation {normalize!r}")
    return x

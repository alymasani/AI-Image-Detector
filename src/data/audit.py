"""Dataset bias audit and a metadata-only shortcut baseline (run before training).

Statistics are broken down by split, class and generator: counts, resolution, aspect ratio, container
format, colour mode, file size, bytes per pixel and estimated JPEG quality (from the luminance
quantisation table, IJG scaling). Exclusions (unreadable, exact duplicates) and near-duplicate counts
come from ``split_metadata.json``.

The metadata baseline is a logistic regression trained on the TRAIN manifest using only file metadata
(width, height, aspect ratio, log file size, bytes/pixel, format, colour mode, JPEG quality); never
filenames, folders or generator labels. High accuracy means the raw files carry a shortcut that
preprocessing must remove. When a standardised-image cache exists, a second baseline uses only the
size of the standardised 224px JPEG (the residual signal that survives our preprocessing).
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from ..utils import load_config, read_json, resolve_path, write_json
from .create_splits import merged_dataset_config
from .dataset import read_manifest
from .preprocessing import cache_path_for

# Standard IJG luminance quantisation table (quality 50), natural (row-major) order.
IJG_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
                     92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=float)


def estimate_jpeg_quality(path: str) -> float:
    """IJG quality estimate from the luminance table; NaN for non-JPEG files."""
    try:
        with Image.open(path) as im:
            if im.format != "JPEG" or not getattr(im, "quantization", None):
                return float("nan")
            q = np.array(im.quantization[0], dtype=float)
    except Exception:
        return float("nan")
    # Pillow returns the table in natural order; scale factor S relative to the quality-50 table.
    s = float(np.mean(q * 100.0 / IJG_LUMA))
    quality = (200.0 - s) / 2.0 if s <= 100 else 5000.0 / s
    return float(np.clip(round(quality, 1), 1, 100))


def load_all_manifests(manifest_dir: Path, names: Dict[str, str]) -> pd.DataFrame:
    frames = [read_manifest(manifest_dir / n).assign(manifest=k) for k, n in names.items()
              if (manifest_dir / n).exists()]
    if not frames:
        raise FileNotFoundError(f"No manifests in {manifest_dir}; run prepare_data.py manifests first")
    df = pd.concat(frames, ignore_index=True)
    df["class_name"] = np.where(df.label == 1, "ai", "real")
    df["aspect_ratio"] = df.width / df.height
    df["bytes_per_pixel"] = df.file_size / (df.width * df.height)
    return df


def metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric metadata features only (no identifiers)."""
    x = pd.DataFrame({
        "width": df.width.astype(float), "height": df.height.astype(float), "aspect_ratio": df.aspect_ratio,
        "log_file_size": np.log1p(df.file_size.astype(float)), "bytes_per_pixel": df.bytes_per_pixel,
        "jpeg_quality": df.jpeg_quality.fillna(0.0),
    })
    for fmt in ("JPEG", "PNG"):
        x[f"format_{fmt}"] = (df.file_format == fmt).astype(float)
    for mode in ("RGB", "L", "RGBA", "CMYK", "P"):
        x[f"mode_{mode}"] = (df["mode"] == mode).astype(float)
    return x


def shortcut_baseline(df: pd.DataFrame, features: pd.DataFrame, name: str) -> List[Dict[str, Any]]:
    """Logistic regression fit on the train manifest, scored on every other manifest."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = df.manifest == "train"
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(features[train], df.label[train])
    rows = []
    for manifest in df.manifest.unique():
        m = (df.manifest == manifest).to_numpy()
        p = clf.predict_proba(features[m])[:, 1]
        y = df.label[m].to_numpy()
        rows.append({"baseline": name, "evaluated_on": manifest, "n": int(m.sum()),
                     "accuracy": float(accuracy_score(y, p >= 0.5)),
                     "auroc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan"),
                     "features": ", ".join(features.columns)})
    return rows


def run_audit(cfg: Dict[str, Any], root: Path, results: Path, workers: int = 4,
              cache_dir: Optional[Path] = None, data_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    manifest_dir = resolve_path(cfg["dataset"]["manifest_dir"])
    df = load_all_manifests(manifest_dir, cfg["manifests"])
    files = df.drop_duplicates("relative_path")
    paths = [str(root / p) for p in files.relative_path]
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        qualities = list(ex.map(estimate_jpeg_quality, paths, chunksize=64))
    df = df.merge(pd.DataFrame({"relative_path": files.relative_path, "jpeg_quality": qualities}),
                  on="relative_path", how="left")
    tables, figures = results / "tables", results / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    keys = ["manifest", "class_name", "generator"]
    summary = df.groupby(keys).agg(
        images=("sample_id", "size"), width_median=("width", "median"), width_min=("width", "min"),
        width_max=("width", "max"), height_median=("height", "median"), height_min=("height", "min"),
        height_max=("height", "max"), aspect_ratio_mean=("aspect_ratio", "mean"),
        square_fraction=("aspect_ratio", lambda a: float((a == 1).mean())),
        file_kb_mean=("file_size", lambda s: s.mean() / 1024), bytes_per_pixel_mean=("bytes_per_pixel", "mean"),
        jpeg_quality_median=("jpeg_quality", "median"),
        imagenet_classes_known=("imagenet_class", lambda c: int((c != "").sum())),
        imagenet_classes_distinct=("imagenet_class", lambda c: int(c[c != ""].nunique())),
    ).reset_index()
    summary.to_csv(tables / "dataset_summary.csv", index=False)
    df.groupby(keys + ["file_format"]).size().rename("images").reset_index() \
        .to_csv(tables / "dataset_formats.csv", index=False)
    df.groupby(keys + ["mode"]).size().rename("images").reset_index() \
        .to_csv(tables / "dataset_color_modes.csv", index=False)

    baseline = shortcut_baseline(df, metadata_features(df), "raw-file metadata")
    if cache_dir is not None and data_cfg is not None:
        sizes = [cache_path_for(cache_dir, data_cfg, s) for s in df.sha256]
        if all(p.exists() for p in sizes):
            feats = pd.DataFrame({"standardized_jpeg_bytes": [p.stat().st_size for p in sizes]})
            baseline += shortcut_baseline(df, feats, "standardised 224px JPEG size")
    pd.DataFrame(baseline).to_csv(tables / "metadata_baseline.csv", index=False)

    meta_path = manifest_dir / "split_metadata.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    excl = pd.Series([e["reason"] for e in meta.get("exclusions", [])], dtype=str).value_counts().to_dict()
    from ..evaluation.visualization import plot_dataset_audit

    # Group by manifest name (shared test_mj reals carry split=test_sd), so replace the row-level column.
    plot_dataset_audit(df.drop(columns=["split"], errors="ignore").rename(columns={"manifest": "split"}), figures)
    out = {"images": int(len(df)), "unique_files": int(files.shape[0]), "exclusions": excl,
           "near_duplicate_pairs": meta.get("near_duplicate_pairs"),
           "near_duplicate_pairs_across_leakage_pairs": meta.get("near_duplicate_pairs_across_leakage_pairs"),
           "metadata_baseline": baseline}
    write_json(out, results / "summaries" / "dataset_audit.json")
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Dataset bias audit + metadata shortcut baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--results", default="results")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--train-config", default="configs/cnn.yaml")
    args = p.parse_args(argv)
    cfg = merged_dataset_config(args.config)
    root = resolve_path(args.data_root or os.environ.get("GENIMAGE_ROOT") or cfg["dataset"]["root"])
    data_cfg = load_config(resolve_path(args.train_config))["data"]
    out = run_audit(cfg, root, resolve_path(args.results), args.workers, resolve_path(args.cache_dir), data_cfg)
    print("Exclusions:", out["exclusions"], "| near-duplicate pairs:", out["near_duplicate_pairs"])
    print(pd.DataFrame(out["metadata_baseline"]).drop(columns="features").to_string(index=False))


if __name__ == "__main__":
    main()

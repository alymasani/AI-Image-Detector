"""Dataset preparation entry point.

Subcommands (run from the repository root):
    instructions  print the official GenImage acquisition steps and licence
    check         verify the extracted folder layout and count images
    manifests     build deterministic train/val/test manifests (+ leakage report)
    leakage       re-run the leakage checks on existing manifests
    audit         dataset bias audit tables/figures + metadata-only shortcut baseline
    cache         write the standardised 224px JPEG-q95 images to a cache folder (optional speed-up)
    synthetic     create SYNTHETIC GenImage-shaped fixtures under .dev/ for smoke tests

Examples:
    python scripts/prepare_data.py check --data-root /path/to/genimage
    python scripts/prepare_data.py manifests --data-root /path/to/genimage --workers 8
    python scripts/prepare_data.py audit --data-root /path/to/genimage
    python scripts/prepare_data.py cache --data-root /path/to/genimage --cache-dir data/cache
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import audit, create_splits, download  # noqa: E402
from src.data.create_splits import check_leakage, merged_dataset_config  # noqa: E402
from src.data.dataset import read_manifest  # noqa: E402
from src.data.preprocessing import cache_path_for, standardized_bytes  # noqa: E402
from src.data.synthetic import make_synthetic_genimage  # noqa: E402
from src.utils import load_config, resolve_path, write_json  # noqa: E402


def _cache_one(args):
    src, dst, data_cfg = args
    dst = Path(dst)
    if dst.exists():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_bytes(standardized_bytes(Path(src), data_cfg))
    tmp.replace(dst)
    return 1


def build_cache(cfg_path: str, data_root: Path, cache_dir: Path, workers: int, dataset_config: str = None) -> None:
    """Pre-compute standardised images for every manifest row (deterministic steps only)."""
    data_cfg = load_config(cfg_path)["data"]
    suite = merged_dataset_config(dataset_config)
    mdir = resolve_path(suite["dataset"]["manifest_dir"])
    jobs = {}
    for name in suite["manifests"].values():
        if (mdir / name).exists():
            for rel, sha in read_manifest(mdir / name)[["relative_path", "sha256"]].itertuples(index=False):
                jobs[sha] = (str(data_root / rel), str(cache_path_for(cache_dir, data_cfg, sha)), data_cfg)
    print(f"Caching {len(jobs)} images to {cache_dir} (key depends on size/interpolation/jpeg_quality)")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        written = sum(ex.map(_cache_one, jobs.values(), chunksize=32))
    print(f"Wrote {written} new files. Use with:  --set data.cache_dir={cache_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["instructions", "check", "manifests", "leakage", "audit", "cache", "synthetic"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--config", default=None, help="dataset override config (e.g. configs/smoke.yaml)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--results", default="results")
    p.add_argument("--train-config", default="configs/cnn.yaml", help="preprocessing settings for `cache`")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--per-folder", type=int, default=40, help="synthetic images per folder")
    args = p.parse_args()

    passthrough = (["--config", args.config] if args.config else []) + \
                  (["--data-root", args.data_root] if args.data_root else [])
    if args.command in ("instructions", "check"):
        download.main([args.command] + passthrough)
    elif args.command == "manifests":
        create_splits.main(passthrough + ["--workers", str(args.workers)])
    elif args.command == "leakage":
        cfg = merged_dataset_config(args.config)
        mdir = resolve_path(cfg["dataset"]["manifest_dir"])
        rep = check_leakage(mdir, cfg["manifests"])
        write_json(rep, mdir / "leakage_report.json")
        print("Leakage check:", "PASSED" if rep["passed"] else "FAILED", *rep["problems"], sep="\n  ")
        raise SystemExit(0 if rep["passed"] else 1)
    elif args.command == "audit":
        audit.main(passthrough + ["--results", args.results])
    elif args.command == "cache":
        cfg = merged_dataset_config(args.config)
        root = resolve_path(args.data_root or os.environ.get("GENIMAGE_ROOT") or cfg["dataset"]["root"])
        build_cache(args.train_config, root, resolve_path(args.cache_dir), args.workers, args.config)
    elif args.command == "synthetic":
        cfg = merged_dataset_config(args.config or "configs/smoke.yaml")
        root = resolve_path(args.data_root or cfg["dataset"]["root"])
        print("SYNTHETIC fixtures (software testing only):", make_synthetic_genimage(root, args.per_folder))


if __name__ == "__main__":
    main()

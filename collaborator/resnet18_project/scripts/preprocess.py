"""Turn the downloaded GenImage images into small, model-ready 224x224 copies.

Every image is resized so its shorter side is 224 and then center-cropped to 224x224, so
nothing gets stretched. Two copies are saved:

    processed/jpeg/<split>/<real|fake>/<name>.jpg   re-encoded as JPEG quality 95
    processed/png/<split>/<real|fake>/<name>.png    lossless, for the JPEG ablation

In GenImage every real photo is a JPEG and every generated image is a PNG, so a detector can
cheat by spotting compression. Re-encoding everything the same way removes that shortcut.

Each format/split gets a manifest, manifests/processed_<format>_<split>.csv, which is what the
training code reads. The originals under raw/ are never modified.

The data folder is $GENIMAGE_DATA_DIR, or the repo's data/ if that isn't set.

Usage:
    uv run python scripts/preprocess.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from detector.data import (
    CLASS_NAMES,
    FORMATS,
    IMAGE_SIZE,
    SPLITS,
    center_crop_resize,
    processed_manifest,
    raw_manifest,
    resolve_data_dir,
)

EXTENSIONS = {"jpeg": ".jpg", "png": ".png"}
MANIFEST_FIELDS = ("path", "label", "subset", "source_generator", "raw_path")

Task = tuple[str, list[tuple[str, str]], int]  # (raw image, [(format, output)], jpeg quality)


def save_image(img: Image.Image, target: Path, fmt: str, quality: int) -> None:
    """Save atomically, so an interrupted run never leaves a half-written image behind."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    if fmt == "jpeg":
        img.save(tmp, format="JPEG", quality=quality, subsampling="4:2:0")
    else:
        img.save(tmp, format="PNG")
    tmp.replace(target)


def process_one(task: Task) -> None:
    raw_path, outputs, quality = task
    pending = [(fmt, Path(out)) for fmt, out in outputs if not Path(out).is_file()]
    if not pending:
        return
    with Image.open(raw_path) as img:
        processed = center_crop_resize(img, IMAGE_SIZE)
    for fmt, target in pending:
        save_image(processed, target, fmt, quality)


def check_settings(data_dir: Path, formats: list[str], quality: int, overwrite: bool) -> None:
    """Refuse to mix images made with different settings in one folder.

    Existing files are skipped on re-runs, so changing e.g. the JPEG quality would otherwise
    silently keep the old images.
    """
    for fmt in formats:
        settings = {"size": IMAGE_SIZE, "resize": "shorter-side bicubic + center crop"}
        if fmt == "jpeg":
            settings |= {"quality": quality, "subsampling": "4:2:0"}
        settings_file = data_dir / "processed" / fmt / "settings.json"
        if settings_file.is_file() and not overwrite:
            existing = json.loads(settings_file.read_text(encoding="utf-8"))
            if existing != settings:
                raise SystemExit(
                    f"{settings_file.parent} was made with {existing}, not {settings}. "
                    "Delete that folder or pass --overwrite."
                )
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def plan_split(
    data_dir: Path, split: str, formats: list[str]
) -> tuple[list[tuple[str, list[tuple[str, str]]]], dict[str, list[dict[str, str]]]]:
    """Work out every output path for one split, plus the manifest rows for each format."""
    manifest = raw_manifest(data_dir, split)
    if not manifest.is_file():
        raise SystemExit(f"Missing {manifest}. Run scripts/download_genimage.py first.")
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    jobs: list[tuple[str, list[tuple[str, str]]]] = []
    manifests: dict[str, list[dict[str, str]]] = {fmt: [] for fmt in formats}
    seen: set[Path] = set()
    for row in rows:
        name = Path(row["path"]).stem
        class_name = CLASS_NAMES[int(row["label"])]
        outputs = []
        for fmt in formats:
            rel = Path("processed", fmt, split, class_name, name + EXTENSIONS[fmt])
            if rel in seen:
                raise SystemExit(f"Two images in {split} would both be saved as {rel}")
            seen.add(rel)
            outputs.append((fmt, str(data_dir / rel)))
            manifests[fmt].append(
                {
                    "path": rel.as_posix(),
                    "label": row["label"],
                    "subset": row["subset"],
                    "source_generator": row["source_generator"],
                    "raw_path": row["path"],
                }
            )
        jobs.append((str(data_dir / row["path"]), outputs))
    return jobs, manifests


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def remove_outputs(jobs: list[tuple[str, list[tuple[str, str]]]]) -> None:
    for _, outputs in jobs:
        for _, out in outputs:
            Path(out).unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS))
    parser.add_argument("--quality", type=int, default=95, choices=range(1, 101), metavar="1-100")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate every image instead of skipping ones that already exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = resolve_data_dir()
    formats = list(dict.fromkeys(args.formats))
    check_settings(data_dir, formats, args.quality, args.overwrite)

    plans = {split: plan_split(data_dir, split, formats) for split in SPLITS}
    tasks: list[Task] = [
        (raw, outputs, args.quality) for jobs, _ in plans.values() for raw, outputs in jobs
    ]
    if args.overwrite:
        for jobs, _ in plans.values():
            remove_outputs(jobs)

    progress = tqdm(total=len(tasks), desc="Preprocessing", unit="img")
    with progress:
        if args.workers <= 1:
            for task in tasks:
                process_one(task)
                progress.update()
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for _ in pool.map(process_one, tasks, chunksize=64):
                    progress.update()

    # Manifests are written last, so one only exists once all of its images do.
    for split, (_, manifests) in plans.items():
        for fmt, rows in manifests.items():
            write_manifest(processed_manifest(data_dir, fmt, split), rows)

    print(f"\nSaved to {data_dir / 'processed'}")
    for fmt in formats:
        total_bytes = 0
        counts = []
        for split, (_, manifests) in plans.items():
            rows = manifests[fmt]
            total_bytes += sum((data_dir / r["path"]).stat().st_size for r in rows)
            counts.append(f"{split} {len(rows)}")
        print(f"  {fmt:<5} {total_bytes / 1e9:5.2f} GB   " + ", ".join(counts))


if __name__ == "__main__":
    main()

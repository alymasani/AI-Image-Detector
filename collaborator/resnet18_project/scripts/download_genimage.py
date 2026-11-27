"""Download the GenImage subsets this project uses from Hugging Face.

Everything comes from the nebula/GenImage-arrow mirror of GenImage, pinned to one revision so
every teammate ends up with exactly the same images.

    subset           source                                real    fake
    train            data/train/stable_diffusion_v_1_4    20000   20000
    val              data/train/stable_diffusion_v_1_4     2500    2500
    test_sd14        data/test/stable_diffusion_v_1_4      6000    6000
    test_midjourney  data/test/Midjourney                  6000    6000

Usage:
    uv run python scripts/download_genimage.py
    uv run python scripts/download_genimage.py --keep-shards   # keep the ~25 GB Arrow cache
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from PIL import Image
from tqdm import tqdm

from detector.data import resolve_data_dir

REPO_ID = "nebula/GenImage-arrow"
REVISION = "3f4b9f921a673be09a93b335ed728cea0c6ecf33"

SD14_TRAIN = "data/train/stable_diffusion_v_1_4"
SD14_TEST = "data/test/stable_diffusion_v_1_4"
MIDJOURNEY_TEST = "data/test/Midjourney"

CLASS_NAMES = ("real", "fake")  # label 0 = real, 1 = fake, same as upstream
MANIFEST_FIELDS = ("path", "label", "subset", "source_generator", "source_path", "sha256")
DOWNLOAD_ATTEMPTS = 6


@dataclass
class Quota:
    """How many images of each class a subset needs, and the rows it has collected so far."""

    subset: str
    per_class: int
    rows: list[dict[str, str | int]] = field(default_factory=list)
    counts: list[int] = field(default_factory=lambda: [0, 0])

    def wants(self, label: int) -> bool:
        return self.counts[label] < self.per_class

    @property
    def full(self) -> bool:
        return not self.wants(0) and not self.wants(1)


def may_share(subset_a: str, subset_b: str) -> bool:
    """Whether the same image is allowed to appear in both subsets.

    The two test sets may draw real photos from the same ImageNet pool, which is harmless since
    neither is trained on. Any other repeat would leak between splits or duplicate an image.
    """
    return subset_a != subset_b and subset_a.startswith("test_") and subset_b.startswith("test_")


def fetch(filename: str, shard_dir: Path) -> Path:
    """Download one file from the dataset repo, retrying on flaky connections."""
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            return Path(
                hf_hub_download(
                    REPO_ID,
                    filename,
                    repo_type="dataset",
                    revision=REVISION,
                    local_dir=shard_dir,
                )
            )
        except (EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError):
            raise
        except Exception as exc:  # connection resets, timeouts; partial downloads resume
            if attempt == DOWNLOAD_ATTEMPTS:
                raise
            wait = 2**attempt
            tqdm.write(f"Downloading {filename} failed ({exc}), retrying in {wait}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def fetch_shards(leaf: str, shard_dir: Path, delete: bool) -> Iterator[Path]:
    """Yield a leaf's Arrow shards one at a time, downloading each only when it is needed."""
    state = json.loads(fetch(f"{leaf}/state.json", shard_dir).read_text(encoding="utf-8"))
    for entry in state["_data_files"]:
        path = fetch(f"{leaf}/{entry['filename']}", shard_dir)
        try:
            yield path
        finally:
            if delete:
                path.unlink(missing_ok=True)


def iter_rows(shard: Path) -> Iterator[tuple[str, bytes, int]]:
    """Yield (image_path, image_bytes, label) for every row of an Arrow stream shard."""
    with pa.OSFile(str(shard), "rb") as source:
        for batch in pa.ipc.open_stream(source):
            paths = batch.column("image_path").to_pylist()
            images = batch.column("image").to_pylist()
            labels = batch.column("label").to_pylist()
            yield from zip(paths, images, labels, strict=True)


def is_valid_image(data: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
    except Exception:  # Pillow raises a variety of types for truncated or corrupt files
        return False
    return True


def write_bytes(target: Path, data: bytes) -> None:
    """Write a file atomically, skipping it if an earlier run already wrote it."""
    if target.is_file() and target.stat().st_size == len(data):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(target)


def extract_leaf(
    shards: Iterable[Path],
    quotas: list[Quota],
    source_generator: str,
    data_dir: Path,
    claimed: dict[str, str],
    stats: Counter[str],
) -> None:
    """Fill the quotas, in order, with rows from a leaf's shards.

    Each row goes to the first quota that still needs its class, so with [train, val] the first
    20000 images of a class go to train and the next 2500 to val. Reading stops (and so does
    downloading) as soon as every quota is full.

    `claimed` maps the sha256 of every image saved so far to its subset and is shared across
    leaves so the same image can never land in two splits.
    """
    image_root = data_dir / "raw" / "genimage"
    used_targets: set[Path] = set()
    total = sum(q.per_class for q in quotas) * len(CLASS_NAMES)
    progress = tqdm(total=total, desc=source_generator, unit="img")
    with progress:
        for shard in shards:
            with closing(iter_rows(shard)) as rows:
                for source_path, image_bytes, label in rows:
                    quota = next((q for q in quotas if q.wants(label)), None)
                    if quota is None:
                        continue
                    if not image_bytes or not is_valid_image(image_bytes):
                        stats["unreadable images skipped"] += 1
                        continue

                    digest = hashlib.sha256(image_bytes).hexdigest()
                    owner = claimed.get(digest)
                    if owner is not None and not may_share(owner, quota.subset):
                        stats["duplicate images skipped"] += 1
                        continue
                    claimed.setdefault(digest, quota.subset)

                    target = image_root / quota.subset / CLASS_NAMES[label] / Path(source_path).name
                    if target in used_targets:
                        raise ValueError(f"Two different images map to {target}")
                    used_targets.add(target)
                    write_bytes(target, image_bytes)

                    quota.rows.append(
                        {
                            "path": target.relative_to(data_dir).as_posix(),
                            "label": label,
                            "subset": quota.subset,
                            "source_generator": source_generator,
                            "source_path": source_path,
                            "sha256": digest,
                        }
                    )
                    quota.counts[label] += 1
                    progress.update()

                    if all(q.full for q in quotas):
                        return


def write_manifest(quota: Quota, data_dir: Path) -> Path:
    manifest = data_dir / "manifests" / f"{quota.subset}.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(quota.rows)
    return manifest


def prune_stale_files(quota: Quota, data_dir: Path) -> int:
    """Delete files in a subset folder that are not in its manifest (left over from older runs)."""
    keep = {data_dir / str(row["path"]) for row in quota.rows}
    subset_dir = data_dir / "raw" / "genimage" / quota.subset
    removed = 0
    for path in subset_dir.rglob("*"):
        if path.is_file() and path not in keep:
            path.unlink()
            removed += 1
    return removed


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train-per-class", type=positive_int, default=20000)
    parser.add_argument("--val-per-class", type=positive_int, default=2500)
    parser.add_argument(
        "--test-per-class",
        type=positive_int,
        default=6000,
        help="Images per class for each test set. 6000 is the whole official test set.",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep the downloaded Arrow shards (~25 GB) instead of deleting each one after use.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = resolve_data_dir()
    shard_dir = data_dir / "raw" / "genimage-arrow"

    plan = [
        (SD14_TRAIN, [Quota("train", args.train_per_class), Quota("val", args.val_per_class)]),
        (SD14_TEST, [Quota("test_sd14", args.test_per_class)]),
        (MIDJOURNEY_TEST, [Quota("test_midjourney", args.test_per_class)]),
    ]
    claimed: dict[str, str] = {}
    stats: Counter[str] = Counter()

    for leaf, quotas in plan:
        with closing(fetch_shards(leaf, shard_dir, delete=not args.keep_shards)) as shards:
            extract_leaf(shards, quotas, Path(leaf).name, data_dir, claimed, stats)
        for quota in quotas:
            write_manifest(quota, data_dir)
            stats["stale files removed"] += prune_stale_files(quota, data_dir)

    print(f"\nSaved to {data_dir}")
    for _, quotas in plan:
        for q in quotas:
            real, fake = q.counts
            note = "" if q.full else f"  <- short of {q.per_class} per class"
            print(f"  {q.subset:<16} real {real:>6}  fake {fake:>6}{note}")
    for name, count in sorted(stats.items()):
        if count:
            print(f"  {name}: {count}")


if __name__ == "__main__":
    main()

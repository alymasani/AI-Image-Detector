"""Selective GenImage download from the pinned Hugging Face mirror ``nebula/GenImage-arrow``.

Adapted from the collaborator's ``scripts/download_genimage.py`` (Group 38 ResNet-18 project,
``collaborator/resnet18_project/``): same mirror, same pinned revision, same retrying shard fetch and
Arrow stream reader. Two changes make it fit the shared split builder (``create_splits.py``):

1. Files are written in the OFFICIAL GenImage layout, ``<root>/<generator folder>/{train,val}/{ai,nature}/``.
   The mirror's ``test`` split is GenImage's ``val`` partition (6,000 images per class); its ``train`` split
   is GenImage's ``train`` partition.
2. Instead of keeping the first N rows (the mirror is sorted by ImageNet class, so the first 20k rows
   cover only ~120 of the 1,000 classes), every row whose seeded key
   ``sha256(f"{seed}:{pool}:{filename}")`` falls below the pool's sampling fraction is kept. These are
   exactly the leading candidates of the split builder's permutation, so a uniform random sample over
   all classes is downloaded in one streaming pass. Every shard is still read once to record the
   folder's complete file list (``_official_listing.txt``) and is deleted after use, so disk use stays
   at a few GB beyond the kept images.

Usage:
    python -m src.data.genimage_hf --data-root data/genimage --workers 3
Re-running resumes: finished shards are recorded under ``<root>/_hf_progress`` and skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .create_splits import OFFICIAL_LISTING, candidate_key, key_fraction, merged_dataset_config, pool_plan
from ..utils import resolve_path, write_json

REPO_ID = "nebula/GenImage-arrow"
REVISION = "3f4b9f921a673be09a93b335ed728cea0c6ecf33"   # pinned: every machine gets identical files
# (mirror leaf, generator key in configs/experiments.yaml, official GenImage partition)
LEAVES = [
    ("data/train/stable_diffusion_v_1_4", "sdv14", "train"),
    ("data/test/stable_diffusion_v_1_4", "sdv14", "val"),
    ("data/test/Midjourney", "midjourney", "val"),
]
MIRROR_CLASS = {0: "real", 1: "fake"}
ATTEMPTS = 6


def fetch(filename: str, cache_dir: Path) -> Path:
    """Download one file of the pinned revision, retrying flaky connections (from the collaborator's code)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError

    for attempt in range(1, ATTEMPTS + 1):
        try:
            return Path(hf_hub_download(REPO_ID, filename, repo_type="dataset", revision=REVISION,
                                        local_dir=cache_dir))
        except (EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError):
            raise
        except Exception as exc:  # connection resets/timeouts; partial downloads resume
            if attempt == ATTEMPTS:
                raise
            print(f"  fetching {filename} failed ({exc}); retry in {2 ** attempt}s", flush=True)
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def iter_rows(shard: Path) -> Iterator[Dict[str, Any]]:
    """Rows of one Arrow IPC stream shard (``image_path``, ``image`` bytes, ``label``, optional ``md5``)."""
    import pyarrow as pa

    with pa.OSFile(str(shard), "rb") as source:
        for batch in pa.ipc.open_stream(source):
            cols = {n: batch.column(n).to_pylist() for n in batch.schema.names if n in ("image_path", "image",
                                                                                           "label", "md5")}
            for i in range(batch.num_rows):
                yield {k: v[i] for k, v in cols.items()}


def sampling_fractions(cfg: Dict[str, Any], generator: str, partition: str, counts: Dict[str, int],
                       margin: float, extra: Dict[str, int]) -> Dict[str, Tuple[str, float]]:
    """mirror class name -> (pool id, fraction of the pool to download).

    The fraction covers the builder's ``need`` plus a margin for unreadable/duplicate files.
    """
    d = cfg["dataset"]
    out = {}
    for spec in pool_plan(cfg):
        if spec["generator"] != generator or spec["partition"] != partition:
            continue
        mirror_cls = "fake" if spec["subdir"].endswith(d["ai_subdir"]) else "real"
        total = max(1, int(counts.get(mirror_cls, 0)))
        want = spec["need"] * (1 + margin) + (50 if spec["need"] else 0) + extra.get(spec["pool"], 0)
        # val partitions (6k/class) are kept whole: they are small, and the Midjourney real photos can be
        # byte-identical to SD val reals, so the native-real pool may skip many exact duplicates.
        out[mirror_cls] = (spec["pool"], 1.0 if partition == "val" else min(1.0, want / total))
    return out


def process_shard(shard_name: str, leaf: str, root: Path, folder: str, partition: str,
                  fractions: Dict[str, Tuple[str, float]], seed: int, d: Dict[str, Any],
                  progress_dir: Path, cache_dir: Path) -> Dict[str, Any]:
    """Fetch one shard, keep the selected images, record the full listing, delete the shard."""
    marker = progress_dir / f"{leaf.replace('/', '__')}__{shard_name}.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    subdir = {"fake": d["ai_subdir"], "real": d["real_subdir"]}
    for attempt in (1, 2):
        path = fetch(f"{leaf}/{shard_name}", cache_dir)
        listing: Dict[str, List[str]] = {"fake": [], "real": []}
        kept = 0
        try:
            for row in iter_rows(path):
                parts = row["image_path"].replace("\\", "/").split("/")
                name, folder_cls = parts[-1], parts[-2]
                cls = MIRROR_CLASS[int(row["label"])]
                if folder_cls != subdir[cls]:
                    raise ValueError(f"label {row['label']} contradicts folder of {row['image_path']}")
                listing[cls].append(name)
                pool, fraction = fractions.get(cls, ("", 0.0))
                if fraction <= 0 or key_fraction(candidate_key(seed, pool, name)) >= fraction:
                    continue
                data = row["image"]
                if row.get("md5") and hashlib.md5(data).hexdigest() != row["md5"]:
                    raise ValueError(f"md5 mismatch for {row['image_path']}")
                target = root / folder / partition / subdir[cls] / name
                if not (target.exists() and target.stat().st_size == len(data)):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp = target.with_name(target.name + ".part")
                    tmp.write_bytes(data)
                    tmp.replace(target)
                kept += 1
            break
        except (OSError, ValueError) as exc:
            if attempt == 2 or "contradicts" in str(exc):
                raise
            print(f"  {shard_name}: {exc}; re-downloading once", flush=True)
            path.unlink(missing_ok=True)
    path.unlink(missing_ok=True)
    result = {"shard": shard_name, "kept": kept, "listing": listing}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(result), encoding="utf-8")
    return result


def download(root: Path, cfg: Dict[str, Any], workers: int = 3, margin: float = 0.03,
             extra: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Download every leaf; write listings and ``_provenance.json``. Returns a summary."""
    d = cfg["dataset"]
    seed = int(cfg["splits"]["seed"])
    extra = extra or {}
    progress_dir, cache_dir = root / "_hf_progress", root / "_hf_cache"
    summary: Dict[str, Any] = {"source": f"https://huggingface.co/datasets/{REPO_ID}", "revision": REVISION,
                               "retrieved": date.today().isoformat(), "seed": seed, "margin": margin, "leaves": {}}
    for leaf, gen, partition in LEAVES:
        folder = d["generators"][gen]["folder"]
        bundle = json.loads(fetch(f"{leaf}/bundle_manifest.json", cache_dir).read_text(encoding="utf-8"))
        state = json.loads(fetch(f"{leaf}/state.json", cache_dir).read_text(encoding="utf-8"))
        shards = [e["filename"] for e in state["_data_files"]]
        fractions = sampling_fractions(cfg, gen, partition, bundle["counts"], margin, extra)
        print(f"{leaf}: {len(shards)} shards, counts {bundle['counts']}, fractions "
              f"{ {k: round(v[1], 4) for k, v in fractions.items()} }", flush=True)
        t0 = time.time()
        results = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(process_shard, s, leaf, root, folder, partition, fractions, seed, d, progress_dir,
                              cache_dir) for s in shards]
            for n, fut in enumerate(as_completed(futs), 1):
                results.append(fut.result())
                if n % 5 == 0 or n == len(shards):
                    print(f"  {leaf}: {n}/{len(shards)} shards, kept {sum(r['kept'] for r in results)} "
                          f"({time.time() - t0:.0f}s)", flush=True)
        for cls, sub in (("fake", d["ai_subdir"]), ("real", d["real_subdir"])):
            names = sorted({n for r in results for n in r["listing"][cls]})
            if len(names) != int(bundle["counts"].get(cls, 0)):
                raise RuntimeError(f"{leaf}/{cls}: listed {len(names)} files, bundle says {bundle['counts'][cls]}")
            out = root / folder / partition / sub
            out.mkdir(parents=True, exist_ok=True)
            (out / OFFICIAL_LISTING).write_text("\n".join(names) + "\n", encoding="utf-8")
        summary["leaves"][leaf] = {"generator": gen, "partition": partition, "folder": folder,
                                   "counts": bundle["counts"], "kept": sum(r["kept"] for r in results),
                                   "fractions": {k: v[1] for k, v in fractions.items()}}
    write_json(summary, root / "_provenance.json")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-root", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--workers", type=int, default=3, help="shards downloaded in parallel")
    p.add_argument("--margin", type=float, default=0.03, help="extra fraction for unreadable/duplicate files")
    p.add_argument("--extra", nargs="*", default=[], help="pool=N extra images, e.g. sdv14/train/ai=500")
    args = p.parse_args(argv)
    cfg = merged_dataset_config(args.config)
    root = resolve_path(args.data_root or os.environ.get("GENIMAGE_ROOT") or cfg["dataset"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    extra = {k: int(v) for k, v in (e.split("=", 1) for e in args.extra)}
    print(json.dumps(download(root, cfg, args.workers, args.margin, extra), indent=2))


if __name__ == "__main__":
    main()

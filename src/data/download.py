"""Dataset acquisition instructions, disk check and layout check.

Two supported routes, both ending in the official GenImage folder layout
``<root>/<generator folder>/{train,val}/{ai,nature}/`` that ``create_splits`` reads:

1. Automated (recommended): ``python -m src.data.genimage_hf`` downloads only the needed images from
   the pinned Hugging Face mirror (see that module).
2. Manual: download the official split archives from the GenImage Google Drive / Baidu links and
   extract them under ``<root>`` (part ids in ``configs/genimage_parts.json``).
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..utils import resolve_path
from .create_splits import OFFICIAL_LISTING, merged_dataset_config

INSTRUCTIONS = """\
GenImage (Zhu et al., NeurIPS 2023)  https://github.com/GenImage-Dataset/GenImage
Licence: CC BY-NC-SA 4.0 (non-commercial research/teaching only). Do not commit images.

Automated (selective, ~20 GB kept; ~105 GB streamed and discarded shard by shard):
    python -m src.data.genimage_hf --data-root {root} --workers 4
  Source: Hugging Face mirror nebula/GenImage-arrow at a pinned revision (each row's MD5 is verified
  where the mirror provides it). Re-run to resume.

Manual (official archives, 324 GB for both generators):
  Google Drive https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS
  (folders stable_diffusion_v_1_4 and Midjourney; part ids in configs/genimage_parts.json).
  Join and extract, e.g. `zip -s 0 imagenet_ai_0419_sdv4.zip --out full.zip && unzip full.zip`, so that
  {root}/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/{{train,val}}/{{ai,nature}} exists (likewise
  Midjourney/imagenet_midjourney). Adjust dataset.generators.*.folder in configs/experiments.yaml if
  your extracted folder names differ (`python scripts/prepare_data.py check` lists candidates).

Then:  python scripts/prepare_data.py manifests --data-root {root}
"""


def disk_report(path: Path) -> Dict[str, float]:
    probe = path if path.exists() else next((p for p in path.parents if p.exists()), Path("."))
    usage = shutil.disk_usage(probe)
    return {"free_gb": round(usage.free / 1024**3, 1), "total_gb": round(usage.total / 1024**3, 1)}


def count_images(folder: Path, extensions: Sequence[str]) -> int:
    exts = {e.lower() for e in extensions}
    if not folder.is_dir():
        return 0
    return sum(1 for e in os.scandir(folder) if e.is_file() and Path(e.name).suffix.lower() in exts)


def discover_generator_folders(root: Path, max_depth: int = 3) -> List[str]:
    """Folders under ``root`` that contain ``train/`` or ``val/`` with ``ai``/``nature`` children."""
    found = []
    if not root.is_dir():
        return found
    for dirpath, dirnames, _ in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth >= max_depth:
            dirnames[:] = []
        if any((Path(dirpath) / part / sub).is_dir() for part in ("train", "val") for sub in ("ai", "nature")):
            found.append(Path(dirpath).relative_to(root).as_posix())
            dirnames[:] = []
    return sorted(found)


def check_layout(root: Path, dataset_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Count images in every configured folder; ``ready`` when each folder has images."""
    root = Path(root)
    report: Dict[str, Any] = {"root": str(root), "exists": root.is_dir(), "disk": disk_report(root), "folders": {}}
    ok = root.is_dir()
    for gen, g in dataset_cfg["generators"].items():
        for part in ("train", "val"):
            for sub in (dataset_cfg["ai_subdir"], dataset_cfg["real_subdir"]):
                folder = root / g["folder"] / part / sub
                listing = folder / OFFICIAL_LISTING
                n = count_images(folder, dataset_cfg["image_extensions"])
                report["folders"][f"{gen}/{part}/{sub}"] = {
                    "images_present": n,
                    "listed_in_source": sum(1 for _ in listing.open(encoding="utf-8")) if listing.exists() else None,
                }
                # Midjourney train is only an optional top-up source and may legitimately be absent.
                if n == 0 and not (gen == "midjourney" and part == "train"):
                    ok = False
    report["ready"] = ok
    if not ok:
        report["candidate_generator_folders"] = discover_generator_folders(root)
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="GenImage acquisition instructions and layout check")
    p.add_argument("command", choices=["instructions", "check"])
    p.add_argument("--config", default=None)
    p.add_argument("--data-root", default=None)
    args = p.parse_args(argv)
    cfg = merged_dataset_config(args.config)
    root = resolve_path(args.data_root or os.environ.get("GENIMAGE_ROOT") or cfg["dataset"]["root"])
    if args.command == "instructions":
        print(INSTRUCTIONS.format(root=root))
        return
    rep = check_layout(root, cfg["dataset"])
    print(f"Dataset root: {root}  ({'READY' if rep['ready'] else 'NOT READY'}; free disk {rep['disk']['free_gb']} GB)")
    for k, v in rep["folders"].items():
        listed = f" / {v['listed_in_source']} listed in source" if v["listed_in_source"] is not None else ""
        print(f"  {k:28s} {v['images_present']:7d} images{listed}")
    if not rep["ready"]:
        print("Candidate generator folders found:", rep.get("candidate_generator_folders") or "none")
        print("Run `python scripts/prepare_data.py instructions` for acquisition steps.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

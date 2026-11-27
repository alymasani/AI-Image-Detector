"""Deterministic train/val/test manifests with leakage checks.

Selection procedure (documented in docs/DATASET.md):

* Every candidate *pool* is one GenImage folder, e.g. ``sdv14/train/ai``. Its full file list comes
  from ``_official_listing.txt`` when present (written by the downloader, it lists every file of
  the folder in the source dataset, downloaded or not) or otherwise from the files on disk.
* Candidates are ordered by ``sha256(f"{seed}:{pool}:{filename}")``. This is a seeded random
  permutation that can be computed for one file without seeing the others, which lets the
  downloader fetch exactly the leading candidates in a single streaming pass
  (``src/data/genimage_hf.py``).
* Candidates are accepted in that order, skipping undecodable files and exact (SHA-256) duplicates of
  any image already selected anywhere. Both skips are logged. If a listed candidate has not been
  downloaded, the pool STOPS there (it never skips ahead) and the shortfall is reported, so a partial
  download can never produce a different selection than a complete one.
* SD 1.4 ``val`` is split into consecutive blocks: the first block is validation, the next block
  the familiar test set. ``test_mj`` reuses the exact real rows of ``test_sd`` (shared-real protocol)
  with Midjourney AI images; ``test_mj_native`` pairs the same Midjourney AI images with real photos
  from the Midjourney folder (secondary evaluation).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yaml
from PIL import Image

from .. import LABEL_AI, LABEL_REAL
from ..utils import PROJECT_ROOT, deep_update, load_config, resolve_path, sha256_file, write_json

OFFICIAL_LISTING = "_official_listing.txt"
SUITE_CONFIG = PROJECT_ROOT / "configs" / "experiments.yaml"
SUITE_KEYS = ("dataset", "splits", "manifests", "evaluation", "experiments")
MANIFEST_COLUMNS = ["sample_id", "relative_path", "label", "generator", "source_folder", "split",
                    "source_partition", "imagenet_class", "file_format", "width", "height", "mode", "file_size",
                    "sha256", "phash"]
# Pairs whose overlap would be leakage. test_sd<->test_mj (shared reals) and test_mj<->test_mj_native
# (shared Midjourney AI images) overlap by design and only between test sets.
LEAKAGE_PAIRS = [("train", "val"), ("train", "test_sd"), ("train", "test_mj"), ("train", "test_mj_native"),
                 ("val", "test_sd"), ("val", "test_mj"), ("val", "test_mj_native")]


# --------------------------------------------------------------------------- configuration


def merged_dataset_config(override: Optional[str] = None) -> Dict[str, Any]:
    """``configs/experiments.yaml`` with the dataset-related keys of ``override`` (e.g. smoke.yaml) merged in."""
    with open(SUITE_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if override:
        over = load_config(resolve_path(override))
        cfg = deep_update(cfg, {k: v for k, v in over.items() if k in SUITE_KEYS})
    return cfg


def pool_plan(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The candidate pools in the order the split builder consumes them.

    ``need`` is the number of images the builder takes from the pool (before exclusions).
    """
    d, s = cfg["dataset"], cfg["splits"]
    ai, real = d["ai_subdir"], d["real_subdir"]
    familiar = int(s["val_per_class"]) + int(s["test_familiar_per_class"])

    def spec(gen: str, part: str, sub: str, label: int, need: int) -> Dict[str, Any]:
        return {"pool": f"{gen}/{part}/{sub}", "generator": gen, "partition": part, "subdir": f"{part}/{sub}",
                "label": label, "need": need}

    return [
        spec("sdv14", "train", ai, LABEL_AI, int(s["train_per_class"])),
        spec("sdv14", "train", real, LABEL_REAL, int(s["train_per_class"])),
        spec("sdv14", "val", ai, LABEL_AI, familiar),
        spec("sdv14", "val", real, LABEL_REAL, familiar),
        spec("midjourney", "val", ai, LABEL_AI, int(s["test_unseen_ai"])),
        spec("midjourney", "train", ai, LABEL_AI, 0),          # top-up only, if midjourney/val/ai is short
        spec("midjourney", "val", real, LABEL_REAL, int(s["test_unseen_native_real"])),
    ]


def candidate_key(seed: int, pool: str, name: str) -> str:
    """Sort key of one candidate: a seeded hash, i.e. a permutation computable file by file."""
    return hashlib.sha256(f"{seed}:{pool}:{name}".encode("utf-8")).hexdigest()


def key_fraction(key: str) -> float:
    """The key mapped to [0, 1). Ordering by key and thresholding this value agree."""
    return int(key[:16], 16) / 2.0**64


# --------------------------------------------------------------------------- image inspection


WNID = re.compile(r"^(n\d{8})_")
CLASS_INDEX = re.compile(r"^(\d{1,4})_")


def imagenet_class(filename: str) -> str:
    """ImageNet class from a filename when it is encoded there, else ''.

    Real train photos are ``n01440764_123.JPEG`` (WordNet id). GenImage generated images start with the
    class index, e.g. ``337_sdv4_00153.png``. ImageNet *val* photos (``ILSVRC2012_val_*``) carry no class.
    """
    m = WNID.match(filename)
    if m:
        return m.group(1)
    m = CLASS_INDEX.match(filename)
    return str(int(m.group(1))) if m else ""


def inspect_image(path: str) -> Dict[str, Any]:
    """Decode one file fully and collect metadata. Returns ``{"error": ...}`` if it can't be decoded."""
    import imagehash

    try:
        data = Path(path).read_bytes()
        with Image.open(io.BytesIO(data)) as im:
            fmt, mode, (w, h) = im.format, im.mode, im.size
            im.load()
            rgb = im.convert("RGB")
        return {"sha256": hashlib.sha256(data).hexdigest(), "file_format": (fmt or "").upper(), "mode": mode,
                "width": w, "height": h, "file_size": len(data), "phash": str(imagehash.phash(rgb))}
    except Exception as exc:  # PIL raises many types for truncated/corrupt files; all mean "unreadable"
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


class Selector:
    """Walks seeded pool permutations, accepting readable, non-duplicate images."""

    def __init__(self, root: Path, seed: int, workers: int = 1):
        self.root = Path(root)
        self.seed = int(seed)
        self.workers = max(1, int(workers))
        self.pool_sizes: Dict[str, int] = {}
        self.listed_pools: Dict[str, bool] = {}
        self.not_downloaded: List[str] = []
        self.exclusions: List[Dict[str, Any]] = []
        self.selected_sha: Dict[str, str] = {}
        self._info: Dict[str, Dict[str, Any]] = {}

    def permutation(self, pool: str, folder_rel: str, extensions: Sequence[str]) -> List[str]:
        """Relative paths of a pool's candidates in selection order."""
        folder = self.root / folder_rel
        listing = folder / OFFICIAL_LISTING
        exts = {e.lower() for e in extensions}
        if listing.exists():
            names = [n.strip() for n in listing.read_text(encoding="utf-8").splitlines() if n.strip()]
            self.listed_pools[pool] = True
        elif folder.is_dir():
            names = [p.name for p in folder.iterdir() if p.is_file() and not p.name.startswith("_")]
            self.listed_pools[pool] = False
        else:
            names = []
            self.listed_pools[pool] = False
        names = sorted(n for n in set(names) if Path(n).suffix.lower() in exts)
        self.pool_sizes[pool] = len(names)
        return [f"{folder_rel}/{n}" for n in sorted(names, key=lambda n: candidate_key(self.seed, pool, n))]

    def _inspect_many(self, rels: Sequence[str]) -> None:
        todo = [r for r in rels if r not in self._info]
        paths = [str(self.root / r) for r in todo]
        if self.workers > 1 and len(todo) > 8:
            with ProcessPoolExecutor(max_workers=self.workers) as ex:
                infos = list(ex.map(inspect_image, paths, chunksize=16))
        else:
            infos = [inspect_image(p) for p in paths]
        self._info.update(zip(todo, infos))

    def take(self, order: Sequence[str], start: int, n: int, base_row: Dict[str, Any],
             split: str) -> Tuple[List[Dict[str, Any]], int]:
        """Accept up to ``n`` images from ``order[start:]``. Returns (rows, index to continue from)."""
        rows: List[Dict[str, Any]] = []
        i = start
        while len(rows) < n and i < len(order):
            rel = order[i]
            if rel not in self._info:
                if not (self.root / rel).exists():
                    if rel not in self.not_downloaded:
                        self.not_downloaded.append(rel)
                    break
                # Inspect the next chunk of present files in parallel (once per chunk, not per image);
                # acceptance below stays strictly sequential, so the selection does not depend on chunking.
                chunk = []
                for cand in order[i:i + max(32, n - len(rows) + 16)]:
                    if not (self.root / cand).exists():
                        break
                    chunk.append(cand)
                self._inspect_many(chunk)
            info = self._info.pop(rel)
            i += 1
            if "error" in info:
                self.exclusions.append({"relative_path": rel, "reason": "unreadable", "detail": info["error"]})
                continue
            dup = self.selected_sha.get(info["sha256"])
            if dup is not None:
                self.exclusions.append({"relative_path": rel, "reason": "exact_duplicate", "detail": dup})
                continue
            self.selected_sha[info["sha256"]] = rel
            row = {**base_row, **info, "relative_path": rel, "split": split,
                   "sample_id": hashlib.sha256(rel.encode("utf-8")).hexdigest()[:16],
                   "imagenet_class": imagenet_class(Path(rel).name)}
            rows.append(row)
        return rows, i


# --------------------------------------------------------------------------- manifests


def _frame(rows: Iterable[Dict[str, Any]], split: str) -> pd.DataFrame:
    df = pd.DataFrame([dict(r, split=split) for r in rows])
    if df.empty:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    return df[MANIFEST_COLUMNS]


def build_manifests(cfg: Dict[str, Any], root: Path, out_dir: Path, workers: int = 1) -> Dict[str, Any]:
    """Select images, write ``<out_dir>/{train,val,test_sd,test_mj,test_mj_native}.csv`` and metadata.

    Returns the metadata dict that is also written to ``split_metadata.json``.
    """
    root, out_dir = Path(root), Path(out_dir)
    d, s = cfg["dataset"], cfg["splits"]
    seed = int(s["seed"])
    sel = Selector(root, seed, workers)
    exts = d["image_extensions"]
    plan = {p["pool"]: p for p in pool_plan(cfg)}
    orders = {name: sel.permutation(name, f"{d['generators'][p['generator']]['folder']}/{p['subdir']}", exts)
              for name, p in plan.items()}
    shortfalls: List[Dict[str, Any]] = []

    def base(p: Dict[str, Any]) -> Dict[str, Any]:
        return {"label": p["label"], "generator": "real" if p["label"] == LABEL_REAL else p["generator"],
                "source_folder": p["generator"], "source_partition": p["partition"]}

    def take(pool: str, start: int, n: int, split: str) -> Tuple[List[Dict[str, Any]], int]:
        rows, nxt = sel.take(orders[pool], start, n, base(plan[pool]), split)
        if len(rows) < n:
            shortfalls.append({"pool": pool, "split": split, "wanted": n, "got": len(rows),
                               "pool_size": sel.pool_sizes[pool]})
        return rows, nxt

    ai, real = d["ai_subdir"], d["real_subdir"]
    sd_ai_tr, sd_real_tr = f"sdv14/train/{ai}", f"sdv14/train/{real}"
    sd_ai_va, sd_real_va = f"sdv14/val/{ai}", f"sdv14/val/{real}"
    mj_ai, mj_ai_top, mj_real = f"midjourney/val/{ai}", f"midjourney/train/{ai}", f"midjourney/val/{real}"
    n_tr, n_va, n_te = int(s["train_per_class"]), int(s["val_per_class"]), int(s["test_familiar_per_class"])

    train = take(sd_real_tr, 0, n_tr, "train")[0] + take(sd_ai_tr, 0, n_tr, "train")[0]
    val_real, i_real = take(sd_real_va, 0, n_va, "val")
    val_ai, i_ai = take(sd_ai_va, 0, n_va, "val")
    test_real = take(sd_real_va, i_real, n_te, "test_sd")[0]
    test_ai = take(sd_ai_va, i_ai, n_te, "test_sd")[0]
    n_mj = int(s["test_unseen_ai"])
    mj_rows, _ = sel.take(orders[mj_ai], 0, n_mj, base(plan[mj_ai]), "test_mj")
    if len(mj_rows) < n_mj and orders[mj_ai_top]:   # documented top-up from Midjourney train/ai (never trained on)
        mj_rows += sel.take(orders[mj_ai_top], 0, n_mj - len(mj_rows), base(plan[mj_ai_top]), "test_mj")[0]
    if len(mj_rows) < n_mj:
        shortfalls.append({"pool": mj_ai, "split": "test_mj", "wanted": n_mj, "got": len(mj_rows),
                           "pool_size": sel.pool_sizes[mj_ai] + sel.pool_sizes[mj_ai_top]})
    native_real = take(mj_real, 0, int(s["test_unseen_native_real"]), "test_mj_native")[0]

    names = cfg["manifests"]
    frames = {
        "train": _frame(train, "train"),
        "val": _frame(val_real + val_ai, "val"),
        "test_sd": _frame(test_real + test_ai, "test_sd"),
        "test_mj": _frame(test_real + mj_rows, "test_mj"),
        "test_mj_native": _frame(native_real + mj_rows, "test_mj_native"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_sha = {}
    for key, df in frames.items():
        path = out_dir / names[key]
        df.to_csv(path, index=False, lineterminator="\n")
        manifest_sha[names[key]] = sha256_file(path)

    near = near_duplicates(frames, int(s.get("near_duplicate_hamming", 4)))
    near.to_csv(out_dir / "near_duplicates.csv", index=False, lineterminator="\n")
    provenance_file = root / "_provenance.json"
    meta = {
        "seed": seed,
        "selection": "seeded sha256(seed:pool:filename) order; skip unreadable + exact duplicates; stop at "
                     "listed-but-missing candidates",
        "splits_config": s,
        "generator_folders": {k: v["folder"] for k, v in d["generators"].items()},
        "source_provenance": (pd.read_json(provenance_file, typ="series").to_dict()
                              if provenance_file.exists() else "local files (no _provenance.json)"),
        "pool_sizes": sel.pool_sizes,
        "pools_with_full_listing": sel.listed_pools,
        "counts": {k: {"real": int((df.label == LABEL_REAL).sum()), "ai": int((df.label == LABEL_AI).sum())}
                   for k, df in frames.items()},
        "manifest_sha256": manifest_sha,
        "exclusions": sel.exclusions,
        "shortfalls": shortfalls,
        "stopped_at_not_downloaded": sel.not_downloaded,
        "near_duplicate_pairs": int(len(near)),
        "near_duplicate_pairs_across_leakage_pairs": int(near.leakage_pair.sum()) if len(near) else 0,
    }
    write_json(meta, out_dir / "split_metadata.json")
    return meta


def near_duplicates(frames: Dict[str, pd.DataFrame], max_hamming: int) -> pd.DataFrame:
    """Candidate near-duplicate pairs by 64-bit pHash Hamming distance <= ``max_hamming``.

    Band bucketing: the 64 bits are cut into ``max_hamming + 1`` bands; two hashes within the distance
    agree exactly on at least one band (pigeonhole), so only pairs sharing a band value are compared.
    Pairs that are the same image (same sha256, e.g. the intentionally shared test reals) are skipped.
    """
    rows = pd.concat([f.assign(manifest=k) for k, f in frames.items() if len(f)], ignore_index=True) \
        if any(len(f) for f in frames.values()) else pd.DataFrame()
    cols = ["a_sample_id", "a_manifest", "a_path", "b_sample_id", "b_manifest", "b_path", "hamming", "leakage_pair"]
    if rows.empty:
        return pd.DataFrame(columns=cols)
    rows = rows.drop_duplicates(subset=["sha256", "manifest"]).reset_index(drop=True)
    bits = [int(h, 16) for h in rows.phash]
    bands = max_hamming + 1
    width = -(-64 // bands)
    candidates = set()
    for b in range(bands):
        buckets: Dict[int, List[int]] = {}
        for idx, v in enumerate(bits):
            buckets.setdefault((v >> (b * width)) & ((1 << width) - 1), []).append(idx)
        for members in buckets.values():
            if len(members) > 1:
                candidates.update(itertools.combinations(members, 2))
    leak = {frozenset(p) for p in LEAKAGE_PAIRS}
    sha, sid, man, rel = (rows[c].tolist() for c in ("sha256", "sample_id", "manifest", "relative_path"))
    out = []
    for i, j in sorted(candidates):
        if sha[i] == sha[j]:
            continue
        dist = bin(bits[i] ^ bits[j]).count("1")
        if dist <= max_hamming:
            out.append({"a_sample_id": sid[i], "a_manifest": man[i], "a_path": rel[i],
                        "b_sample_id": sid[j], "b_manifest": man[j], "b_path": rel[j],
                        "hamming": dist, "leakage_pair": frozenset((man[i], man[j])) in leak})
    return pd.DataFrame(out, columns=cols)


# --------------------------------------------------------------------------- leakage checks


def check_leakage(manifest_dir: Path, names: Dict[str, str]) -> Dict[str, Any]:
    """Run every leakage / consistency check on existing manifests. ``passed`` is False on any problem."""
    from .dataset import read_manifest

    manifest_dir = Path(manifest_dir)
    dfs = {k: read_manifest(manifest_dir / v) for k, v in names.items() if (manifest_dir / v).exists()}
    problems: List[str] = []
    info: Dict[str, Any] = {"counts": {}}
    for k, df in dfs.items():
        n_real, n_ai = int((df.label == LABEL_REAL).sum()), int((df.label == LABEL_AI).sum())
        info["counts"][k] = {"real": n_real, "ai": n_ai}
        for col in ("sample_id", "relative_path", "sha256"):
            dup = int(df[col].duplicated().sum())
            if dup:
                problems.append(f"{dup} duplicate {col} values within {k}")
        if n_real != n_ai:
            problems.append(f"{k} is not class-balanced (real {n_real}, ai {n_ai})")
        if df.relative_path.str.startswith("/").any() or df.relative_path.str.contains(":", regex=False).any():
            problems.append(f"{k} contains absolute paths")
        bad_gen = df[((df.label == LABEL_REAL) & (df.generator != "real")) |
                     ((df.label == LABEL_AI) & (df.generator == "real"))]
        if len(bad_gen):
            problems.append(f"{len(bad_gen)} rows in {k} whose label contradicts the generator (label inversion?)")
        bad_dir = df[((df.label == LABEL_REAL) & df.relative_path.str.contains("/ai/", regex=False)) |
                     ((df.label == LABEL_AI) & df.relative_path.str.contains("/nature/", regex=False))]
        if len(bad_dir):
            problems.append(f"{len(bad_dir)} rows in {k} whose label contradicts the source folder")
    for a, b in LEAKAGE_PAIRS:
        if a in dfs and b in dfs:
            for col in ("sample_id", "relative_path", "sha256"):
                shared = set(dfs[a][col]) & set(dfs[b][col])
                if shared:
                    problems.append(f"{len(shared)} shared {col} between {a} and {b}")
    for k in ("train", "val"):
        if k in dfs:
            n = int(((dfs[k].generator == "midjourney") | (dfs[k].source_folder == "midjourney")).sum())
            info[f"midjourney_in_{k}"] = n
            if n:
                problems.append(f"{n} Midjourney rows in {k}")
    for k, gen in (("val", "sdv14"), ("test_sd", "sdv14"), ("test_mj", "midjourney"), ("test_mj_native", "midjourney")):
        if k in dfs:
            wrong = int(((dfs[k].label == LABEL_AI) & (dfs[k].generator != gen)).sum())
            if wrong:
                problems.append(f"{wrong} AI rows in {k} not from {gen}")
    if "test_sd" in dfs and "test_mj" in dfs:
        r_sd = set(dfs["test_sd"].loc[dfs["test_sd"].label == LABEL_REAL, "sample_id"])
        r_mj = set(dfs["test_mj"].loc[dfs["test_mj"].label == LABEL_REAL, "sample_id"])
        info["shared_real_identical"] = r_sd == r_mj
        if r_sd != r_mj:
            problems.append("test_mj real rows differ from test_sd real rows (shared-real protocol broken)")
    return {"passed": not problems, "problems": problems, **info}


# --------------------------------------------------------------------------- CLI


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Build deterministic manifests + leakage report")
    p.add_argument("--config", default=None, help="dataset override config, e.g. configs/smoke.yaml")
    p.add_argument("--data-root", default=None)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    cfg = merged_dataset_config(args.config)
    root = resolve_path(args.data_root or os.environ.get("GENIMAGE_ROOT") or cfg["dataset"]["root"])
    out = resolve_path(cfg["dataset"]["manifest_dir"])
    meta = build_manifests(cfg, root, out, args.workers)
    rep = check_leakage(out, cfg["manifests"])
    write_json(rep, out / "leakage_report.json")
    for k, c in meta["counts"].items():
        print(f"  {k:15s} real {c['real']:6d}  ai {c['ai']:6d}")
    print(f"  excluded: {len(meta['exclusions'])}  near-duplicate pairs: {meta['near_duplicate_pairs']} "
          f"(across leakage pairs: {meta['near_duplicate_pairs_across_leakage_pairs']})")
    for sf in meta["shortfalls"]:
        print(f"  SHORTFALL {sf}")
    if meta["stopped_at_not_downloaded"]:
        print(f"  stopped at not-downloaded candidates: {meta['stopped_at_not_downloaded'][:5]}")
    print("Leakage check:", "PASSED" if rep["passed"] else "FAILED", *rep["problems"], sep="\n  ")
    raise SystemExit(0 if rep["passed"] and not meta["shortfalls"] else 1)


if __name__ == "__main__":
    main()

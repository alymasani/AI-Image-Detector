# Dataset: GenImage (SD 1.4 + Midjourney subsets)

## Source and licence
- Paper: Zhu et al., *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image*, NeurIPS 2023 (arXiv:2306.08571).
- Official repository: https://github.com/GenImage-Dataset/GenImage. Official downloads: Google Drive
  https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS and Baidu Yunpan (see the repository README).
- Licence: **CC BY-NC-SA 4.0, non-commercial research/teaching only**. No dataset image is committed to this repository.

Facts verified from the official archives (2026-11-25): the Drive folders `stable_diffusion_v_1_4` (96.4 GB, 30 split-zip
parts) and `Midjourney` (227.5 GB, 71 parts) each contain `{train,val}/{ai,nature}`, 162,000 train and 6,000 val images
per class. SD images are 512² PNG, Midjourney 1024² PNG, real images are ImageNet JPEGs. No checksums are published.

## Acquisition (what we actually used)
`src/data/genimage_hf.py` downloads only the needed images from the Hugging Face mirror
[`nebula/GenImage-arrow`](https://huggingface.co/datasets/nebula/GenImage-arrow), **pinned to revision
`3f4b9f921a673be09a93b335ed728cea0c6ecf33`** so every machine gets identical files. The mirror and the pinned revision were
chosen by our ResNet-18 collaborator (`collaborator/resnet18_project/scripts/download_genimage.py`). Our downloader reuses their
fetch/retry and Arrow-reading code. The mirror's `train`/`test` splits are GenImage's `train`/`val` partitions with the same
per-class counts (SD 1.4 train 162,000 real / 161,997 AI; val 6,000 / 6,000; Midjourney val 6,000 / 6,000). The mirror
stores an MD5 per training image, which the downloader verifies.
```bash
python -m src.data.genimage_hf --data-root data/genimage --workers 4    # resumable; ~105 GB streamed, ~20 GB kept
python scripts/prepare_data.py check
```
How it works. Every shard is streamed once and deleted after use. For each row the downloader:
1. records the file name in the folder's complete listing (`_official_listing.txt`, the mirror's full file list), and
2. keeps the image if its seeded key `sha256("42:<pool>:<filename>")` falls below the pool's sampling fraction
   (train pools: `(need × 1.03 + 50) / pool size` ≈ 12.7%; val pools: 100%).

The split builder orders candidates by the **same** key, so the downloaded files are exactly the leading candidates of
its permutation. If the builder ever reaches a listed candidate that was not downloaded, it stops that pool and reports
a shortfall. It never skips ahead, so a partial download cannot change the selection. Files are written in the official
layout (`<root>/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/{train,val}/{ai,nature}/…` and
`<root>/Midjourney/imagenet_midjourney/val/…`), so a manual download of the official archives works with the same code.
`<root>/_provenance.json` records the source, revision, date and sampling fractions.

**Why not "first N rows"?** The mirror is sorted by ImageNet class. Taking the first 20,000 rows per class (what the
collaborator's original script does) yields training images from only ~125 of the 1,000 ImageNet classes. Hash sampling
spreads the training set over all classes.

Manual alternative (official archives): download every part of both Drive folders, join and extract (`zip -s 0
imagenet_ai_0419_sdv4.zip --out full.zip && unzip full.zip`) under `<root>/stable_diffusion_v_1_4/` and
`<root>/Midjourney/`. Part ids are in `configs/genimage_parts.json`. Without `_official_listing.txt` files the builder
permutes the files present on disk, which gives the same result when the folders are complete.

## Splits (`configs/experiments.yaml → splits`, seed 42)

| Manifest | Real (0) | AI (1) | Source |
|---|---|---|---|
| `train.csv` | 20,000 | 20,000 | SD1.4 `train/nature`, `train/ai` |
| `val.csv` | 2,000 | 2,000 | SD1.4 `val/` (first block of each pool's permutation) |
| `test_sd.csv` | 2,000 | 2,000 | SD1.4 `val/` (next block, disjoint from val) |
| `test_mj.csv` | **same 2,000 as test_sd** | 2,000 | Midjourney `val/ai` (primary unseen test) |
| `test_mj_native.csv` | 2,000 | same 2,000 MJ | Midjourney `val/nature` (secondary) |

Actual counts, exclusions and pool sizes: `manifests/split_metadata.json`. Built 2026-11-26 from the pinned mirror:
every manifest has exactly the requested counts (no shortfalls and no candidate missing from the download). 52,000
distinct image files are used (56,000 manifest rows, because the 2,000 test reals and 2,000 Midjourney AI images appear in
two test manifests each). Three images were skipped as exact SHA-256 duplicates of an already selected image:
- one train real duplicated another train real;
- one SD val real was byte-identical to a selected train real;
- one Midjourney val real was byte-identical to a selected SD val real.

The leakage check **passed**. pHash found 8 near-duplicate pairs (Hamming = 4). Six of them cross a leakage pair (train↔val
or train↔test_mj). Those six are 4 distinct image pairs, because two Midjourney images appear in both test_mj and
test_mj_native. We inspected all of them visually: each pair shows clearly different content (e.g. a scoreboard and a
jellyfish). They are pHash collisions between dark, low-detail images, not copies. They are listed in
`manifests/near_duplicates.csv` and kept.

- GenImage has no separate test partition, so its `val` partition is split deterministically into our validation and
  familiar-test sets. Midjourney is used only in test manifests.
- **Shared-real protocol (primary).** `test_mj` reuses exactly the real rows of `test_sd`, so the familiar-vs-unseen
  comparison changes only the generator of the fake images. The overlap is between the two *test* sets only. The secondary
  `test_mj_native` uses the Midjourney folder's own real images, minus any byte-identical copy of an image already selected.
- Sampling: each pool's candidates are ordered by the seeded key, then accepted in order. Undecodable files and exact
  SHA-256 duplicates of any already-selected image are skipped and logged in `split_metadata.json → exclusions`.
  Shortfalls are recorded in `→ shortfalls`. Counts are never reduced silently.
- ImageNet class: parsed from file names where they encode it (train reals `n01440764_*.JPEG` → WordNet id; generated
  files `337_sdv4_*.png` → class index). ImageNet *val* reals (`ILSVRC2012_val_*`) carry no class, so it is left blank rather
  than invented. Hash sampling is uniform over the class-sorted pools, so classes are represented in proportion to the
  source. Coverage is reported in `results/tables/dataset_summary.csv` (`imagenet_classes_distinct`).

## Manifest fields
`sample_id` (first 16 hex digits of SHA-256 of the relative path), `relative_path`, `label`, `generator` (real|sdv14|midjourney),
`source_folder`, `split`, `source_partition`, `imagenet_class` (blank when unknown), `file_format`, `width`, `height`, `mode`,
`file_size`, `sha256`, `phash`. Paths are relative to the dataset root. The manifests are portable and committed under
`manifests/`. Checkpoints store the SHA-256 of the manifests they were trained on, and resuming refuses changed manifests.

## Leakage prevention (`create_splits.check_leakage`, run automatically → `manifests/leakage_report.json`)
Duplicate sample ids / paths / SHA-256 within a manifest; any shared sample id, path or SHA-256 across train↔val,
train↔tests, val↔tests; Midjourney rows in train/val; class balance; generator/label and folder/label consistency (catches
label inversion); shared-real identity between `test_sd` and `test_mj`. Exact duplicates across folders (the same ImageNet
photo stored under two generators) are excluded at selection time.
Near-duplicates: 64-bit DCT pHash, Hamming ≤ 4, found with band bucketing (no all-pairs comparison), written to
`manifests/near_duplicates.csv` with a `leakage_pair` flag. They are reported, not removed automatically.

## Bias audit
`python scripts/prepare_data.py audit` → `results/tables/dataset_summary.csv`, `dataset_formats.csv`,
`dataset_color_modes.csv`, `metadata_baseline.csv`, `results/figures/dataset_audit.{pdf,png}`,
`results/summaries/dataset_audit.json`. JPEG quality is estimated from luminance quantisation tables (IJG scaling).
The **metadata baseline** (logistic regression on width, height, aspect ratio, log file size, bytes/pixel, format, colour
mode and JPEG quality; never filenames or folders) measures how much shortcut signal the raw files carry. A second baseline
uses only the byte size of the standardised 224 px JPEG, the residual signal left after our preprocessing.

Results (2026-11-26, `results/tables/metadata_baseline.csv`):

**Raw source files: the metadata baseline is perfect (accuracy 1.000, AUROC 1.000 on every manifest).**
- Every real image is an ImageNet JPEG (median 500×375, estimated quality ≈ 96, ~0.6 bytes/pixel).
- Every AI image is a square PNG: SD 1.4 is 512², Midjourney is 1024² (with some at 512²).

A model trained on raw files could therefore solve the task from the container alone. This is exactly the shortcut that the
shared resize + JPEG-q95 standardisation removes.

**After standardisation the remaining file-size signal is weak.**

| Manifest | AUROC |
|---|---|
| train | 0.569 |
| val | 0.542 |
| test_sd | 0.554 |
| test_mj | 0.289 |

On Midjourney the size signal reverses:

| Standardised JPEG, median size | Real | AI |
|---|---|---|
| SD 1.4 (train) | 22.0 KB | 23.8 KB |
| Midjourney (test_mj) | 22.4 KB | 17.9 KB |

Any size shortcut learned on SD would therefore hurt rather than help on the unseen generator.

ImageNet class coverage: 1,000 distinct classes in each training class. SD val AI images span ~910 classes and Midjourney AI
images 915. ImageNet val reals carry no class label.

# Dataset: GenImage (SD 1.4 + Midjourney subsets)

## Source and licence
- Paper: Zhu et al., *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image*, NeurIPS 2023 (arXiv:2306.08571).
- Official repository: https://github.com/GenImage-Dataset/GenImage
- Downloads (from the official README): Google Drive
  https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS, Baidu Yunpan
  https://pan.baidu.com/s/1i0OFqYN5i6oFAxeK6bIwRQ (code `ztf1`).
- Licence: **CC BY-NC-SA 4.0, non-commercial research/teaching only**. Dataset images are not committed to this repo.

Facts verified from official sources (2026-11-25):
- The Drive root contains folders `ADM, BigGAN, glide, Midjourney, stable_diffusion_v_1_4, stable_diffusion_v_1_5, VQDM, wukong`.
  The README tree names the SD folder `Stable Diffusion V1.4`.
- Each generator folder has `train/{ai,nature}` and `val/{ai,nature}`. `nature` = real ImageNet photographs.
- Paper: per class, each generator has 162 training and 6 test images → about **162,000 train / 6,000 val
  AI images per generator**. SD V1.4 outputs are 512×512; Midjourney (V5) outputs are 1024×1024.
- **Not published:** archive sizes, archive format/part names, checksums. So the download isn't scripted (and Drive
  quotas make scripted downloads unreliable), and there's no checksum to validate against.

## Acquisition (manual)
```bash
python scripts/prepare_data.py instructions          # prints the steps above
# download ONLY Midjourney/ and stable_diffusion_v_1_4/ and extract under one root, then:
export GENIMAGE_ROOT=/path/to/genimage
python scripts/prepare_data.py check                 # counts images; lists candidate folders if paths differ
```
If extraction produces a wrapper directory (e.g. `stable_diffusion_v_1_4/imagenet_ai_xxxx/`), set
`dataset.generators.sdv14.folder` / `midjourney.folder` in `configs/experiments.yaml` to the relative path that
`check` reports. **Disk:** the SD 1.4 training partition is ~324k images (162k PNG at 512² plus 162k ImageNet JPEGs).
Its size isn't published; budget tens of GB. After manifests exist, `prepare_data.py cache` writes the 52k selected
images as 224 px JPEGs (well under 1 GB), and the raw archives could then be moved off the machine.

## Splits (`configs/experiments.yaml → splits`, seed 42)

| Manifest | Real (0) | AI (1) | Source |
|---|---|---|---|
| `train.csv` | 20,000 | 20,000 | SD1.4 `train/nature`, `train/ai` |
| `val.csv` | 2,000 | 2,000 | SD1.4 `val/` (first block of a seeded permutation) |
| `test_sd.csv` | 2,000 | 2,000 | SD1.4 `val/` (next block: disjoint from val) |
| `test_mj.csv` | **same 2,000 as test_sd** | 2,000 | Midjourney `val/ai` (primary unseen test) |
| `test_mj_native.csv` | 2,000 | same 2,000 MJ | Midjourney `val/nature` (secondary) |

- GenImage has no separate test partition, so its `val` partition is split deterministically into our validation and
  familiar-test sets. Midjourney `train/ai` is used **only** to top up the unseen test set if `val/ai` is short. It is
  never used in training.
- **Shared-real protocol (primary).** `test_mj` reuses exactly the real rows of `test_sd`, so the familiar-vs-unseen
  comparison changes only the generator of the fake images. This overlap is between the two *test* sets only. The
  secondary `test_mj_native` uses the Midjourney folder's own real images, minus any byte-identical copy of an
  already-selected image.
- Sampling: per pool, sorted file list → seeded permutation (seed derived from `seed` + pool name) → accept in order,
  skipping undecodable files and exact SHA-256 duplicates of any already-selected image (logged in
  `manifests/split_metadata.json`). Stratification by ImageNet class isn't applied: class ids are parseable only from
  ImageNet-train-style real filenames (`n########_*.JPEG`), not from val reals or generated files. Class distributions
  are reported where available instead of being invented.
- If a pool is too small, the shortfall is recorded in `split_metadata.json → shortfalls` and printed. Counts are
  never silently reduced.

## Manifest fields
`sample_id, relative_path, label, generator (real|sdv14|midjourney), source_folder, split, source_partition,
imagenet_class (blank when unknown), file_format, width, height, mode, file_size, sha256, phash`. Paths are relative
to the dataset root. `split_metadata.json` records seed, split config, candidate pool sizes, per-manifest SHA-256,
exclusions and shortfalls. Checkpoints store the SHA-256 of the manifests they were trained on, and resuming refuses
changed manifests.

## Leakage prevention (`create_splits.check_leakage`, run automatically)
Duplicate sample ids / paths / SHA-256 within a manifest; any shared sample id, path or SHA-256 across
train↔val, train↔tests, val↔tests; Midjourney rows in train/val; class balance; generator/label consistency (catches
label inversion); shared-real identity between `test_sd` and `test_mj`. Exact duplicates across generator folders
(e.g. the same ImageNet photo in both generator folders) are excluded at selection time.
Near-duplicates: 64-bit DCT pHash, Hamming ≤ 4, found with band bucketing (no all-pairs comparison) and written to
`manifests/near_duplicates.csv` for review. They're reported, not auto-removed.

## Bias audit
`python scripts/prepare_data.py audit` → `results/tables/dataset_summary.csv`, `dataset_formats.csv`,
`dataset_color_modes.csv`, `metadata_baseline.csv`, `results/figures/dataset_audit.{pdf,png}`. JPEG quality is
estimated from luminance quantisation tables (validated on PIL-encoded files: true 50/75/90/95 → 50.0/74.7/90.1/95.0).
The **metadata baseline** (logistic regression on width, height, aspect ratio, log file size, bytes/pixel, format,
colour mode and estimated JPEG quality, but never filenames or folders) measures how much shortcut signal the raw files
carry. Expected risk (Grommelt et al., 2024): real = ImageNet JPEGs of varied size; SD = 512² PNG; Midjourney =
1024² PNG. Our preprocessing removes the container/size cues (fixed 224 crop, uniform JPEG q95), but **not**
(a) different downscaling factors (ImageNet ~500→224 vs MJ 1024→224) or (b) the JPEG history of real photos.
Actual GenImage audit results: **PENDING (dataset not yet available)**.

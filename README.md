# Spotting the Unseen Generator: How Well Do Deep Learning Detectors Generalize?

ENSF 519 (Applied Deep Learning), Group 38: Charbel Maroun, Himla Rahman, Aly Masani.

AI-generated images are getting harder to tell apart from photographs. Detectors can learn generator-specific artefacts
that fail on generators they never saw. For newsroom fact-checking teams (an **assistive screening** use, not proof of
authenticity) we ask:

> How well does a CNN trained from scratch generalise to images from an unseen generator (Midjourney), compared with an
> ImageNet-pretrained, fully fine-tuned ResNet-18, when both are trained on Stable Diffusion 1.4 vs real photos?

- **Input:** 224×224 RGB image. **Output:** two logits → P(AI). Labels: **0 = real, 1 = AI-generated (positive)**.
- **Model A (ours):** custom four-block CNN, 1,189,794 parameters, trained from scratch.
- **Model B:** Torchvision ResNet-18 with ImageNet weights, new 2-way head, all 11,177,538 parameters fine-tuned
  (*reference implementation*; the collaborator's version is to be integrated, see `docs/ARCHITECTURE.md`).
- **Metrics:** accuracy, precision, recall, F1, AUROC, FPR/FNR, balanced accuracy, confusion matrices; familiar
  (SD 1.4) vs unseen (Midjourney) degradation; 50% chance baseline; bootstrap CIs and McNemar tests.

## Current status
Software pipeline implemented and verified on synthetic fixtures (31 essential tests pass; see
`docs/VALIDATION_REPORT.md`). **No GenImage results yet:** the dataset must be downloaded first. All result tables
currently show `PENDING`.

## Repository
```
configs/      cnn.yaml, resnet18.yaml, smoke.yaml, experiments.yaml (dataset layout, splits, experiment suite)
src/data/     download (instructions/check), create_splits (manifests + leakage), preprocessing, dataset, audit, synthetic
src/models/   cnn.py (Model A), resnet18.py (Model B), factory.py
src/training/ trainer.py, early_stopping.py, checkpointing.py
src/evaluation/ metrics.py, statistics.py, robustness.py, visualization.py
src/          train.py, evaluate.py, experiment.py, utils.py
scripts/      prepare_data.py, run_experiments.py, generate_report_assets.py, reproduce.py, hpc/train.slurm
tests/essential/   submission tests        docs/   design, dataset, protocol, validation, report guide, rubric
manifests/    portable split manifests (committed once built)     results/  tables, figures, summaries, predictions
```
`data/`, `checkpoints/`, `runs/`, `.dev/` and `.venv/` are git-ignored.

## Setup
Python ≥ 3.9 (verified with 3.9.6). From the repository root:
```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
# Optional, for an NVIDIA GPU: install the matching CUDA build first, e.g.
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-dev.txt  # or requirements.txt without test tools
python scripts/reproduce.py status   # dependencies, device (CUDA/MPS/CPU), dataset, checkpoints
```
The device is chosen automatically (CUDA → Apple MPS → CPU); override with `--set training.device=cpu`.

## Quick verification (no GPU, no dataset; a few minutes)
```bash
python scripts/reproduce.py tests    # essential tests
python scripts/reproduce.py smoke    # SYNTHETIC end-to-end run in .dev/ (not GenImage results)
```

## Dataset (GenImage, CC BY-NC-SA 4.0, non-commercial)
Only about 53k of the 2.7M images are needed. They're downloaded selectively (~17 GB) from the official split
archives via HTTP range requests, and each file is CRC-verified. Details: `docs/DATASET.md`.
```bash
python -m src.data.genimage_remote --data-root data/genimage --workers 16   # resumable; re-run to retry failures
export GENIMAGE_ROOT=data/genimage                     # or pass --data-root
python scripts/prepare_data.py check                   # verifies layout and counts
python scripts/prepare_data.py manifests --workers 8   # deterministic splits + leakage report
python scripts/prepare_data.py audit                   # bias statistics + metadata-only shortcut baseline
python scripts/prepare_data.py cache --cache-dir data/cache   # optional: pre-standardised 224px images
```
To use the cache in training, add `--set data.cache_dir=data/cache` (bit-identical to on-the-fly preprocessing).

## Training
```bash
python scripts/run_experiments.py E1_cnn E2_resnet18 --stage pilot          # ~1k images, no test evaluation
python scripts/run_experiments.py E1_cnn E2_resnet18 --stage intermediate   # ~5k images
python scripts/run_experiments.py E1_cnn         # full Custom CNN (40k) + held-out evaluation + JPEG robustness
python scripts/run_experiments.py E2_resnet18    # full ResNet-18
# equivalent low-level commands:
python -m src.train --config configs/cnn.yaml
python -m src.train --config configs/resnet18.yaml --set training.batch_size=16 training.grad_accum_steps=2
python -m src.train --config configs/cnn.yaml --resume checkpoints/cnn/last.pt
```
Optional experiments: `E4_cnn_no_jpeg`, `E6_cnn_no_aug`, `E7_*` (seeds 123/2026), `E8_resnet18_scratch`.
HPC: `sbatch scripts/hpc/train.slurm E1_cnn` (edit the placeholder partition/modules first).

## Evaluation and report assets
```bash
python -m src.evaluate --checkpoint checkpoints/cnn/best.pt --tests test_sd test_mj test_mj_native --latency
python -m src.evaluate --checkpoint checkpoints/cnn/best.pt --tests test_sd test_mj --extra-jpeg 75
python scripts/generate_report_assets.py               # results/tables/*.csv|md, results/figures/*.pdf|png
```
Evaluation-only reproduction of the main table from supplied checkpoints:
```bash
python scripts/reproduce.py checkpoints fetch    # download (once URLs exist) + SHA-256 verification
python scripts/reproduce.py evaluate             # both models, all test sets, robustness, latency
python scripts/reproduce.py report
```

## Checkpoints
Checkpoints are too large for Git (`checkpoints/` is ignored). After training, run
`python scripts/reproduce.py checkpoints register` to record SHA-256 hashes in `checkpoints_manifest.json`. Upload
the files, then add each `url`. **Status: no trained checkpoints yet; hosting PENDING.**

## Hardware
The CNN is small (1.2 M params); ResNet-18 should fit in a few GB of GPU memory at batch 32 with AMP (estimate, not measured). Measured on an
Apple M4 (MPS), compute-only throughput was 87 img/s (CNN) and 118 img/s (ResNet-18), i.e. roughly 6–8 min of
compute per 40k-image epoch before data loading. An RTX 3080 or TALC GPU will be faster. Disk: the SD 1.4 training
archives total 324 GB; the selective download needs ~17 GB.

## Known limitations
Architecture and initialisation are confounded in the main comparison; GenImage reals are ImageNet photos (overlap with
ResNet pretraining data); JPEG re-encoding doesn't remove prior compression history; 224 px downscaling removes some
high-frequency artefacts; one unseen generator only; balanced 50% prevalence is unrealistic for newsrooms; outputs are
uncalibrated probabilities. See `docs/ARCHITECTURE.md` and `docs/EXPERIMENT_PROTOCOL.md`.

## References and attribution
See `docs/ATTRIBUTION.md` (GenImage, ResNet, Torchvision weights, Wang et al. 2020, Grommelt et al. 2024, libraries,
and disclosure of AI coding assistance).

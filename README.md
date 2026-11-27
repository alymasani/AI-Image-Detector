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
  (the collaborator's construction, trained under the same protocol as the CNN; their original project is kept in
  `collaborator/resnet18_project/`, see `docs/ARCHITECTURE.md`).
- **Metrics:** accuracy, precision, recall, F1, AUROC, FPR/FNR, balanced accuracy, confusion matrices; familiar
  (SD 1.4) vs unseen (Midjourney) degradation; 50% chance baseline; bootstrap CIs and McNemar tests.

## Current status
All experiments (E0–E8) have been run on GenImage on an RTX 3080 (2026-11-26). The 35 essential tests pass. See
`docs/VALIDATION_REPORT.md` for what was run and `docs/REPORT_GUIDE.md` for the evidence summary.

Held-out results, primary seed-42 runs, threshold 0.5 (`results/tables/main_results.md`):

| Model | SD 1.4 acc / AUROC | Midjourney (unseen) acc / AUROC |
|---|---|---|
| Custom CNN | 0.897 / 0.961 | 0.514 / 0.465 |
| ResNet-18 | 0.963 / 0.994 | 0.608 / 0.727 |
| Chance | 0.500 / 0.500 | 0.500 / 0.500 |

Both detectors fail to generalise to Midjourney. Only the pretrained ResNet-18 keeps an above-chance ranking. A ResNet-18
trained from scratch (E8) behaves like the CNN, so the difference comes from pretraining.

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
Python ≥ 3.9. Verified with 3.12.16 on Windows 11 with CUDA 12.6, and with 3.9.6 on macOS (MPS). From the repository root:
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
Only 52,000 images are used. They're downloaded selectively from a Hugging Face mirror pinned to a fixed revision,
chosen by our collaborator. About 105 GB is streamed and about 23 GB kept. Rows are selected by the same seeded hash
order that the split builder uses. Details: `docs/DATASET.md`.
```bash
python -m src.data.genimage_hf --data-root data/genimage --workers 4   # resumable; re-run to retry failures
export GENIMAGE_ROOT=data/genimage                     # or pass --data-root
python scripts/prepare_data.py check                   # verifies layout and counts
python scripts/prepare_data.py manifests --workers 8   # deterministic splits + leakage report (~2 min)
python scripts/prepare_data.py audit                   # bias statistics + metadata-only shortcut baseline
python scripts/prepare_data.py cache --workers 16      # pre-standardised 224px images in data/cache (~1 min)
```
The configs read `data.cache_dir: data/cache`. Missing cache files are computed on the fly, bit-identically. The
committed `manifests/` let you skip the manifests step and check your download against them.

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
the files, then add each `url`. **Status:** `cnn/best.pt` and `resnet18/best.pt` are trained and their SHA-256 hashes are
registered. Hosting URLs are still PENDING.

## Hardware
Measured on an RTX 3080 (10 GB), fp16 AMP, batch 32, 4 loader workers:

| Model | Epoch time | Full run | Peak GPU memory |
|---|---|---|---|
| CNN | ≈49 s | 10 min | 1.0 GB |
| ResNet-18 | ≈32 s | 6 min | 0.6 GB |

The 7 optional runs took about 1.5 h. Disk: about 23 GB for the images plus 6 GB for both caches.

## Known limitations
Architecture and initialisation are confounded in the main comparison; GenImage reals are ImageNet photos (overlap with
ResNet pretraining data); JPEG re-encoding doesn't remove prior compression history; 224 px downscaling removes some
high-frequency artefacts; one unseen generator only; balanced 50% prevalence is unrealistic for newsrooms; outputs are
uncalibrated probabilities. See `docs/ARCHITECTURE.md` and `docs/EXPERIMENT_PROTOCOL.md`.

## References and attribution
See `docs/ATTRIBUTION.md` (GenImage, ResNet, Torchvision weights, Wang et al. 2020, Grommelt et al. 2024, libraries,
and disclosure of AI coding assistance).

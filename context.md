# context.md: project state for future sessions (updated 2026-11-25)

## Objective
ENSF 519 Group 38 (Charbel Maroun, Himla Rahman, Aly Masani). Due **2026-12-04 23:59**. Compare a custom four-block
CNN (from scratch; Aly's primary responsibility) with an ImageNet-pretrained fully fine-tuned ResNet-18 (assigned to
another member) at detecting AI-generated images. Train on GenImage SD 1.4 vs real; test on held-out SD 1.4 (familiar)
and Midjourney (unseen). Full spec: `ENSF 519.docx` in repo root (untracked). Proposal PDF is outside the repo.

## Non-negotiables
GenImage only; 20k real + 20k SD1.4 train; 224×224 RGB; labels 0 real / 1 AI; disjoint splits; Midjourney test-only;
consistent JPEG re-encoding; CNN: conv/ReLU/pool, CE, Adam, early stopping, no pretrained weights; ResNet-18:
ImageNet weights, new 2-way head, full fine-tune; metrics acc/F1/AUROC + degradation + 50% baseline. Never fabricate
results; don't tune on test sets; don't push without permission.

## Environment (this machine)
Apple M4, 16 GB RAM, macOS, **no CUDA**, MPS available, ~29.6 GB free disk. Local venv `.venv/` (Python 3.9.6, torch
2.8.0, torchvision 0.23.0; exact freeze in `.dev/pip-freeze-verified.txt`). ResNet-18 ImageNet weights are cached in
`~/.cache/torch/hub`.

## Architecture (see docs/ARCHITECTURE.md)
`src/data` (download/check, create_splits + leakage, preprocessing, dataset, audit, synthetic) · `src/models`
(cnn.py 1,189,794 params; resnet18.py reference 11,177,538) · `src/training` (trainer, early stopping on val loss,
checkpointing with resume + manifest hashes) · `src/evaluation` (metrics, bootstrap/McNemar, robustness, figures) ·
`src/train.py`, `src/evaluate.py`, `src/experiment.py` (run suite + report assets) · `scripts/` wrappers incl.
`reproduce.py` (status/tests/smoke/manifests/train/evaluate/report/checkpoints) and `hpc/train.slurm` (placeholders).
Configs: `configs/cnn.yaml`, `resnet18.yaml` (inherits cnn), `smoke.yaml`, `experiments.yaml` (dataset folders,
split counts, manifest names, evaluation settings, experiment ids E1/E2/E4/E6/E7/E8).

Preprocessing: RGB (alpha → white) → resize short side 224 bilinear → centre crop → JPEG q95 → [train: hflip, reflect
pad 8 + random crop] → ImageNet normalisation. Optional cache `prepare_data.py cache` (bit-identical).
Splits: train from SD1.4 train; val + test_sd from consecutive blocks of one permutation of SD1.4 val; test_mj = test_sd
reals + 2k MJ val AI (shared-real protocol); test_mj_native secondary. Exact-dup exclusion by SHA-256; pHash
near-dups reported in `manifests/near_duplicates.csv`.

## Dataset status: **BLOCKER**
Not downloaded. Official Drive folders: `stable_diffusion_v_1_4`, `Midjourney` (README tree says `Stable Diffusion
V1.4`); each has train/val × ai/nature; ~162k train / ~6k val AI per generator (paper); SD 512², MJ 1024².
Archive sizes/format/checksums are not published. SD1.4 train is probably larger than the free disk here (estimate). `prepare_data.py check`
discovers the actual extracted folder and the `dataset.generators.*.folder` values must be set to match.
No manifests exist yet (`manifests/` is empty).

## Verified so far (docs/VALIDATION_REPORT.md)
- `pytest tests/essential`: 31 passed, 1 skipped (CUDA).
- `reproduce.py smoke` (synthetic): full pipeline incl. both models on MPS, evaluation, robustness, report: OK.
- Tiny-overfit (random labels, 32 imgs): train acc 1.0 by step 40. Pretrained weights verified.
- MPS throughput estimate: CNN 87 img/s, ResNet 118 img/s (compute only).
Not verified: anything on real data, CUDA AMP, Windows multi-worker loading, SLURM script.

## Results
None. `results/tables/main_results.md` shows PENDING everywhere. No checkpoints (`checkpoints_manifest.json` = {}).
`results/tables/cnn_architecture.csv` and `parameter_counts.csv` are real (computed).

## Collaborator integration
No collaborator code on any branch as of 2026-11-25. `src/models/resnet18.py` is a clearly labelled reference.
Integration steps are in docs/ARCHITECTURE.md.

## Known issues / limitations
Architecture and init confounded; ImageNet overlap with ResNet pretraining; JPEG history not removed; downscaling
factors differ per source (SD 512→224, MJ 1024→224, ImageNet ~500→224); bootstrap ≠ seed variance; ImageNet class
stratification not possible for val reals/AI files.

## Next commands (in order)
```bash
source .venv/bin/activate
export GENIMAGE_ROOT=/path/to/extracted/genimage
python scripts/prepare_data.py check            # then set dataset.generators.*.folder if needed
python scripts/prepare_data.py manifests --workers 8
python scripts/prepare_data.py audit
python scripts/prepare_data.py cache --cache-dir data/cache        # optional speed-up
python -m src.train --config configs/cnn.yaml --set data.train_limit=16 data.val_limit=16 data.augment.hflip=false data.augment.random_crop_padding=0 model.dropout=0 training.max_epochs=30 training.early_stopping_patience=30 experiment_name=cnn_tiny
python scripts/run_experiments.py E1_cnn E2_resnet18 --stage pilot
python scripts/run_experiments.py E1_cnn E2_resnet18 --stage intermediate
python scripts/run_experiments.py E1_cnn E2_resnet18
python scripts/generate_report_assets.py
python scripts/reproduce.py checkpoints register
```
Then update docs/VALIDATION_REPORT.md, RUBRIC_CHECKLIST.md, REPORT_GUIDE.md and this file with actual results.

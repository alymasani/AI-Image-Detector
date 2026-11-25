# Validation report

All entries record commands actually executed on 2026-11-25 on the development machine below. Synthetic-fixture
results verify **software behaviour only** and are not GenImage results.

## Environment (measured by `python -m src.utils`)
Apple M4 (10 CPU cores), 16 GB RAM, macOS (Darwin 24.6.0), Python 3.9.6, torch 2.8.0, torchvision 0.23.0,
numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, scipy 1.13.1, Pillow 11.3.0, ImageHash 4.3.2, matplotlib 3.9.4.
CUDA: **not available**. MPS: available (used for smoke training). Free disk ≈ 29.6 GB.
The intended RTX 3080 / TALC environments were **not** available, so CUDA mixed precision is implemented but
**not executed**.

## Essential tests (`tests/essential`, retained in the repository)
Command: `.venv/bin/python -m pytest -q tests/essential` → **31 passed, 1 skipped, 0 failed** (≈11 s).
Skipped: `test_cnn_cuda_forward` (no CUDA on this machine).

| File | Covers |
|---|---|
| `test_data.py` (11) | manifest counts/labels/relative paths; corrupt file excluded + logged; gray/RGBA kept and converted; leakage checks pass; injected val→test leak is detected; split generation reproducible (identical manifest SHA-256); RGBA→white compositing; corrupt image raises; tensor shape/dtype/label; deterministic eval vs stochastic train transforms; aspect-preserving crop; cache bit-identical to on-the-fly |
| `test_models.py` (9) | CNN output N×2 for batch 1/3/8, finite; exactly 4 blocks, no softmax, 1,189,794 params; finite loss, all params get nonzero grads and change after Adam step; eval deterministic / dropout active in train; ResNet-18 2-way head, 11,177,538 trainable, frozen variant trains only fc; checkpoint round-trip equality; CUDA forward (skipped) |
| `test_metrics.py` (8) | hand-calculated confusion/acc/P/R/F1/FPR/FNR/AUROC (AUROC 13/16); agreement with scikit-learn; inclusive 0.5 threshold; zero-denominator and single-class handling; chance baselines + degradation arithmetic; bootstrap determinism; cross-generator CI = 0 for identical sets; McNemar |
| `test_training.py` (4) | early stopping (patience, min_delta, reset, NaN); CPU train 1 epoch → best/last checkpoints → resume to epoch 2 with full history columns; evaluation export columns, threshold consistency, shared real images scored identically in test_sd and test_mj |

Fix made during testing: the hand-calculated AUROC expectation in `test_metrics.py` was wrong (14/16); recomputed
pairwise as 13/16 = 0.8125, which matches scikit-learn. The implementation was correct.

## Stage A: synthetic end-to-end smoke (`python scripts/reproduce.py smoke`)
Synthetic GenImage-shaped tree (40 images/folder + gray/RGBA/corrupt edge cases) → manifests (train 24+24, val/test
8+8) → leakage check **PASSED** → CNN and ResNet-18 (ImageNet weights downloaded and verified) trained 3 epochs on MPS
→ evaluation on test_sd/test_mj/test_mj_native → extra-JPEG q50 robustness → latency → report assets (8 figures,
all tables). Completed in ≈65 s. The synthetic classes are trivially separable (100% accuracy), as designed for a
pipeline test.
Bug found and fixed: the report generator crashed when no evaluation existed yet (plot guard added). Legend overlap in
the main comparison figure was fixed.

## Development diagnostics (`.dev/tests/dev_diagnostics.py`, git-ignored)
- **Tiny-overfit (Phase 3):** 32 pure-noise images with random balanced labels, dropout 0, no augmentation, Adam 1e-3,
  MPS: eval-mode training accuracy 0.50 at step 20 → **1.00 from step 40** to 200 (loss ≈ 0). The model, loss,
  gradients and optimiser can memorise. (The step-20 lag is BatchNorm running statistics catching up.)
  Stage B (32 *real* GenImage images) is **PENDING** on the dataset.
- **Pretrained ResNet-18:** loaded `conv1.weight` equals the Torchvision hub file `resnet18-f37072fd.pth`; all
  parameters `requires_grad=True`.
- **Throughput (MPS, batch 32, 224 px, synthetic tensors, compute only):** CNN 87 img/s, ResNet-18 118 img/s.
  **Estimates:** ≈ 7.7 min (CNN) and ≈ 5.7 min (ResNet) of compute per 40k-image epoch, excluding data loading. The
  CNN is slower despite fewer parameters because its first block runs 32-channel 3×3 convolutions at full 224×224
  resolution.
- **JPEG quality estimator:** PIL-encoded q=50/75/90/95 → estimated 50.0/74.7/90.1/95.0.
- **Generator-folder discovery:** a nested `stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/val/{ai,nature}` tree was
  found and suggested by `prepare_data.py check`.
- Lint: `ruff check --select F,E9 src scripts` clean after removing 5 unused imports.

## Not yet verified (blocked on dataset / hardware)
GenImage acquisition, real manifests, leakage audit on real data, bias audit, Stage B–E training, all held-out
results, CUDA AMP path, multi-worker loading on Windows, TALC job script.

## Final audit
Pending: performed after real experiments (Phase 14).

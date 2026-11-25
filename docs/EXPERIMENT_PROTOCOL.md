# Experiment protocol (fixed before any test evaluation)

## Preprocessing (both models, both classes, all generators)
1. Decode fully; convert to RGB (grayscale replicated; transparency composited onto white).
2. Resize the shorter side to 224 px (PIL bilinear, antialiased), centre-crop 224×224.
3. Re-encode as JPEG quality 95 (4:2:0, PIL/libjpeg) and decode. This comes **after** spatial standardisation and
   **before** augmentation, so every training image has exactly one standard compression applied at the input
   resolution.
4. Training only: random horizontal flip; reflect-pad 8 px + random 224 crop (≤ 8 px translation, no resampling).
   No colour jitter, blur, CutMix, MixUp or erasing (these may destroy forensic cues).
5. ToTensor, ImageNet normalisation (mean 0.485/0.456/0.406, std 0.229/0.224/0.225). ResNet-18's pretrained weights
   expect it. For the scratch CNN it is a harmless fixed affine rescaling that centres inputs, and using it for both
   means identical inputs for both models.

Validation/test: steps 1–3 + 5, fully deterministic, no TTA. **Limitation:** step 3 doesn't erase compression traces
already present in the source files (ImageNet JPEGs vs generator PNGs).

## Training (initial settings; changes may be made using SD 1.4 validation only and must be logged below)

| | Custom CNN (E1) | ResNet-18 (E2) |
|---|---|---|
| init | random (Kaiming/Xavier) | ImageNet IMAGENET1K_V1, new fc |
| optimiser | Adam, lr 1e-3, weight_decay 1e-4 (coupled L2) | Adam, lr 1e-4, weight_decay 1e-4 |
| batch | 32 | 32 |
| epochs | ≤ 20, early stopping on val loss, patience 4, min_delta 1e-4 | same |
| dropout | 0.3 (head) | none (standard ResNet) |
| precision | fp16 autocast + GradScaler on CUDA, fp32 elsewhere (recorded per run) | same |
| seed | 42 | 42 |

Model selection: the checkpoint with the lowest **SD 1.4 validation loss** (`best.pt`). Decision threshold: P(AI) ≥ 0.5.
Any newsroom-oriented threshold must be chosen on validation predictions only.
If GPU memory forces a smaller batch, use `grad_accum_steps` to keep the effective batch at 32 for **both** models, and
record it.

### Hyperparameter change log
| Date | Model | Change | Reason (validation evidence) |
|---|---|---|---|
| – | – | none yet | – |

## Staged execution (Phase 6)
| Stage | Command | Purpose |
|---|---|---|
| A synthetic smoke | `python scripts/reproduce.py smoke` | software path end to end (done, see VALIDATION_REPORT) |
| B tiny real | `python -m src.train --config configs/cnn.yaml --set data.train_limit=16 data.val_limit=16 data.augment.hflip=false data.augment.random_crop_padding=0 model.dropout=0 training.max_epochs=30 training.early_stopping_patience=30 experiment_name=cnn_tiny` | CNN memorises 32 real images |
| C pilot | `python scripts/run_experiments.py E1_cnn E2_resnet18 --stage pilot` | 1k train images, loss/time/memory |
| D intermediate | `python scripts/run_experiments.py E1_cnn E2_resnet18 --stage intermediate` | 5k train images, lr sanity, runtime estimate |
| E full | `python scripts/run_experiments.py E1_cnn E2_resnet18` | primary models + held-out evaluation + E3 |

Pilot/intermediate stages never evaluate test sets.

## Experiments
| ID | Priority | What | How |
|---|---|---|---|
| E0 | mandatory | chance baseline | analytic expectations (acc 0.5, AUROC 0.5, P=R=F1=0.5 for random guessing; also "always AI": recall 1, F1 0.667, FPR 1) |
| E1 | mandatory | Custom CNN | `run_experiments.py E1_cnn` |
| E2 | mandatory | ResNet-18 fine-tuned | `run_experiments.py E2_resnet18` |
| E3 | high | JPEG robustness | extra JPEG q=90/75/50 on standardised test images, no retraining (run automatically after E1/E2) |
| E5 | high | failure cases | deterministic selection by confidence (tie-break sample_id) per model × test → `results/tables/failure_cases_*.csv`; image grids in git-ignored `results/figures/local/` |
| E4 | next | JPEG-standardisation shortcut | `run_experiments.py E4_cnn_no_jpeg` (same manifests/seed/protocol, `jpeg_quality=null`); reduced scope allowed with `--set data.train_limit=N`, documented |
| E6 | optional | no augmentation | `E6_cnn_no_aug` |
| E7 | optional | seeds 123, 2026 | `E7_*` ids; mean ± std only over completed runs |
| E8 | optional | ResNet-18 from scratch | `E8_resnet18_scratch` (lr 1e-3) |

Primary configurations and checkpoints are frozen before the first test evaluation. Optional experiments never replace
the primary baselines.

## Statistics
Stratified percentile bootstrap (2,000 resamples, seed 519) for accuracy/F1/AUROC; paired resampling for CNN − ResNet
differences on the same images; for familiar − unseen differences one resample of the **shared** real images is used
in both sets (preserves the dependence); exact McNemar test for paired correctness. These intervals reflect test-sample
uncertainty for fixed checkpoints, not training-seed variance, and they assume independent images (optimistic for
GenImage). Results are exploratory: no multiple-comparison correction.

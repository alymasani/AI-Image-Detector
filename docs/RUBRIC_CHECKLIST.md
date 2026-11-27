# Rubric checklist (status as of 2026-11-26; no grade is implied)

Statuses: VERIFIED · IMPLEMENTED_NOT_VERIFIED · PARTIALLY_COMPLETE · PENDING_EXTERNAL_INPUT · NOT_STARTED

## Group report (16)
| Criterion | Max | Requirements | Implementation evidence / files | Verification | Status | Outstanding |
|---|---|---|---|---|---|---|
| A1 Abstract & intro | 1 | motivation, task, I/O, metrics, findings | REPORT_GUIDE.md (evidence summary) | real results available | PENDING_EXTERNAL_INPUT | human-written text |
| A2 Background | 1 | 2–3 related approaches | ATTRIBUTION.md references | references checked against proposal | PARTIALLY_COMPLETE | write section |
| A3 Data | 2 | source, sizes, labels, splits, preprocessing, augmentation, leakage, bias | DATASET.md (actual counts, leakage, audit), `split_metadata.json`, `leakage_report.json`, `dataset_audit.png`, `metadata_baseline.csv` | real data, leakage PASSED | VERIFIED (evidence) | write section |
| A4 Methods | 4 | both model specs, original vs reused, training config, early stopping, reproducibility | ARCHITECTURE.md, EXPERIMENT_PROTOCOL.md, `cnn_architecture.csv`, `parameter_counts.csv`, ATTRIBUTION.md | model tests, full runs | VERIFIED (evidence) | write section |
| A5 Experiments & results | 4 | fair protocol, chance baseline, held-out eval, main table, params, times, curves, supporting experiments | `main_results.md`, 8 figures, `computational_comparison.csv`, `generalization_drop.csv`, `jpeg_robustness.csv`, `seed_repeatability.csv`, `all_evaluations.csv` (E4/E6/E7/E8) | E0–E8 completed on RTX 3080 | VERIFIED (evidence) | write section |
| A6 Analysis & conclusion | 3 | explanations, failure cases, limitations, recommendations | `failure_cases_*.csv`, `statistics.json` (bootstrap, McNemar), REPORT_GUIDE findings + newsroom list | real results | PARTIALLY_COMPLETE | inspect failure-case images; write section |
| A7 Response to feedback | 1 | address TA feedback | template in REPORT_GUIDE.md | – | PENDING_EXTERNAL_INPUT | actual TA feedback |

## Code (10)
| Criterion | Max | Requirements | Evidence / files | Verification | Status | Outstanding |
|---|---|---|---|---|---|---|
| B1 Completeness | 2 | code, data instructions, checkpoints | src/, scripts/, DATASET.md, `checkpoints_manifest.json` (SHA-256) | full pipeline run | PARTIALLY_COMPLETE | upload checkpoints and add URLs |
| B2 Organisation & README | 2 | structure, README | README.md, docs/ | commands run on Windows/CUDA | VERIFIED | – |
| B3 Reproducibility | 3 | configs, seeds, portable paths, commands, traceable results, main-table reproduction | configs/*.yaml, pinned dataset revision, manifests + hashes, `reproduce.py`, manifest rebuild byte-identical | rebuild identical; re-evaluation reproduced in-memory metrics | VERIFIED (except hosted checkpoints) | commit `manifests/`; publish checkpoints |
| B4 Code quality | 2 | readable, documented | docstrings, type hints, ruff clean | 35 essential tests | IMPLEMENTED_NOT_VERIFIED | group review |
| B5 Attribution | 1 | datasets, models, weights, code, AI | docs/ATTRIBUTION.md (collaborator integration, HF mirror, AI assistance) | – | VERIFIED (content) | group confirms collaborator credit wording |

## Individual reflection (4 per student)
Contribution log in REPORT_GUIDE.md. Each student writes their own reflection: NOT_STARTED (not automatable).

## Proposal commitments
| Commitment | Status |
|---|---|
| GenImage SD 1.4 train/val/test, Midjourney unseen test only | VERIFIED (manifests, leakage report) |
| 40k balanced training set | VERIFIED (20,000 + 20,000) |
| 224×224 RGB, binary labels, JPEG re-encoding | VERIFIED |
| Four-block CNN from scratch, ReLU, pooling, CE, Adam, early stopping | VERIFIED (trained, early-stopped at epoch 12) |
| ResNet-18 ImageNet, new binary head, full fine-tune | VERIFIED (collaborator's construction, trained under the shared protocol) |
| Accuracy, F1, AUROC, degradation, 50% chance baseline | VERIFIED (main_results, generalization_drop) |

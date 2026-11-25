# Rubric checklist (status as of 2026-11-25; no grade is implied)

Statuses: VERIFIED · IMPLEMENTED_NOT_VERIFIED · PARTIALLY_COMPLETE · PENDING_EXTERNAL_INPUT · NOT_STARTED

## Group report (16)
| Criterion | Max | Requirements | Implementation evidence / files | Verification | Status | Outstanding |
|---|---|---|---|---|---|---|
| A1 Abstract & intro | 1 | motivation, task, I/O, metrics, findings | REPORT_GUIDE.md | – | PENDING_EXTERNAL_INPUT | real results; human-written text |
| A2 Background | 1 | 2–3 related approaches | ATTRIBUTION.md references | references checked against proposal | PARTIALLY_COMPLETE | write section |
| A3 Data | 2 | source, sizes, labels, splits, preprocessing, augmentation, leakage, bias | DATASET.md, EXPERIMENT_PROTOCOL.md, `create_splits.py`, `audit.py` | synthetic tests | PARTIALLY_COMPLETE | download GenImage; real counts + audit |
| A4 Methods | 4 | both model specs, original vs reused, training config, early stopping, reproducibility | ARCHITECTURE.md, configs/, `cnn_architecture.csv`, `parameter_counts.csv` | model tests | PARTIALLY_COMPLETE | collaborator ResNet details; write section |
| A5 Experiments & results | 4 | fair protocol, chance baseline, held-out eval, main table, params, times, curves, supporting experiments | `experiment.py`, `evaluate.py`, report generator | synthetic smoke only | PENDING_EXTERNAL_INPUT | full training + evaluation |
| A6 Analysis & conclusion | 3 | explanations, failure cases, limitations, recommendations | failure-case selection, statistics, REPORT_GUIDE newsroom list | synthetic | PENDING_EXTERNAL_INPUT | real results |
| A7 Response to feedback | 1 | address TA feedback | template in REPORT_GUIDE.md | – | PENDING_EXTERNAL_INPUT | actual TA feedback |

## Code (10)
| Criterion | Max | Requirements | Evidence / files | Verification | Status | Outstanding |
|---|---|---|---|---|---|---|
| B1 Completeness | 2 | code, data instructions, checkpoints | src/, scripts/, DATASET.md | tests + smoke | PARTIALLY_COMPLETE | trained checkpoints + hosting link |
| B2 Organisation & README | 2 | structure, README | README.md, docs/ | commands run in smoke | IMPLEMENTED_NOT_VERIFIED | re-check README after real runs |
| B3 Reproducibility | 3 | configs, seeds, portable paths, commands, traceable results, main-table reproduction | configs/*.yaml, manifest hashes in checkpoints, `reproduce.py`, eval-only workflow | split reproducibility test; smoke | PARTIALLY_COMPLETE | real manifests committed; checkpoints published |
| B4 Code quality | 2 | readable, documented | docstrings, type hints, ruff clean | ruff F/E9 | IMPLEMENTED_NOT_VERIFIED | group review |
| B5 Attribution | 1 | datasets, models, weights, code, AI | docs/ATTRIBUTION.md | – | PARTIALLY_COMPLETE | collaborator credit after integration |

## Individual reflection (4 per student)
Contribution log in REPORT_GUIDE.md. Each student writes their own reflection: NOT_STARTED (not automatable).

## Proposal commitments
| Commitment | Status |
|---|---|
| GenImage SD 1.4 train/val/test, Midjourney unseen test only | implemented; data PENDING |
| 40k balanced training set | implemented; data PENDING |
| 224×224 RGB, binary labels, JPEG re-encoding | VERIFIED (synthetic) |
| Four-block CNN from scratch, ReLU, pooling, CE, Adam, early stopping | VERIFIED (tests) |
| ResNet-18 ImageNet, new binary head, full fine-tune | VERIFIED as reference impl.; collaborator version PENDING |
| Accuracy, F1, AUROC, degradation, 50% chance baseline | VERIFIED (tests); real results PENDING |

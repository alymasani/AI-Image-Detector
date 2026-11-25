# Project plan (deadline: 2026-12-04 23:59)

| Phase | Objective | Status (2026-11-25) |
|---|---|---|
| 0 | Repository, environment, gitignore | done |
| 1 | Dataset acquisition, manifests, leakage, audit | code done + tested on synthetic; **GenImage download blocked** |
| 2 | Shared preprocessing + loaders | done, tested |
| 3 | Custom CNN | done; tests + tiny-overfit passed; real tiny test pending |
| 4 | ResNet-18 integration | reference impl. done; collaborator code pending |
| 5 | Training infrastructure | done; CUDA AMP path not executed (no GPU) |
| 6 | Staged training A–E | A done; B–E pending data |
| 7 | Evaluation infrastructure | done, tested |
| 8 | Experiments E0–E8 | implemented; pending data |
| 9 | Statistics | done, tested |
| 10 | Figures/tables | done (from exported results) |
| 11 | Interpretation | pending results |
| 12 | Reproducibility | `reproduce.py`, README done; checkpoint hosting pending |
| 13 | Report support | guide/checklist done; writing pending |
| 14 | Final audit | pending results |

## Critical path to submission
1. Obtain `stable_diffusion_v_1_4` + `Midjourney` from the official Drive (needs a machine with enough disk, e.g.
   external drive, a teammate's PC with the RTX 3080, or TALC scratch).
2. `prepare_data.py check` → `manifests` → `audit` → (optional) `cache`; commit `manifests/`.
3. Stage B tiny real test, Stage C/D pilots (≈1 h on M4 or less on an RTX 3080).
4. Full E1 + E2 (estimated 2–4 h each on the M4 incl. data loading, less on CUDA; estimates, not measurements).
5. Freeze configs, evaluate, E3 robustness (automatic), `generate_report_assets.py`.
6. E4 (reduced scope if time), optional E6–E8.
7. Publish checkpoints (e.g. a GitHub Release or a shared drive), record SHA-256, write the report.

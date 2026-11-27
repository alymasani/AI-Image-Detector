# Project plan (deadline: 2026-12-04 23:59)

| Phase | Objective | Status (2026-11-26) |
|---|---|---|
| 0 | Repository, environment, gitignore | done (`.gitignore` fixed: `/data/` no longer hides `src/data/`) |
| 1 | Dataset acquisition, manifests, leakage, audit | done: pinned HF mirror, 52,000 images, leakage PASSED, audit done |
| 2 | Shared preprocessing + loaders | done, tested; cache built |
| 3 | Custom CNN | done; Stage B memorised 32 real images |
| 4 | ResNet-18 integration | done: collaborator's `build_resnet18`, their project kept under `collaborator/` |
| 5 | Training infrastructure | done; CUDA AMP verified on RTX 3080 |
| 6 | Staged training A–E | done |
| 7 | Evaluation infrastructure | done |
| 8 | Experiments E0–E8 | done (E0–E8, E7 = 3 seeds per model) |
| 9 | Statistics | done (bootstrap, paired differences, McNemar) |
| 10 | Figures/tables | done (8 figures) |
| 11 | Interpretation | evidence summary in REPORT_GUIDE.md; failure-case images still to inspect |
| 12 | Reproducibility | done except checkpoint hosting |
| 13 | Report support | guide/checklist done; writing by the group |
| 14 | Final audit | done (VALIDATION_REPORT.md) |

## Remaining before submission
1. Commit `manifests/` together with the code changes (needs the group's go-ahead).
2. Upload `checkpoints/cnn/best.pt` and `checkpoints/resnet18/best.pt` (e.g. a GitHub Release or a shared drive), and
   add their URLs to `checkpoints_manifest.json`. The SHA-256 hashes are already recorded.
3. Look at the failure-case images. `results/figures/local/` is git-ignored; select rows from
   `results/tables/failure_cases_*.csv`.
4. Write the report (4 pages) from REPORT_GUIDE.md, plus the individual reflections and A7 once TA feedback exists.

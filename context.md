# context.md: project state for future sessions (updated 2026-11-26)

## Objective
ENSF 519 Group 38 (Charbel Maroun, Himla Rahman, Aly Masani). Due **2026-12-04 23:59**.

The project compares a custom four-block CNN (from scratch; Aly's primary responsibility) with an ImageNet-pretrained,
fully fine-tuned ResNet-18 (the collaborator's model) at detecting AI-generated images. Both are trained on GenImage
SD 1.4 vs real photos and tested on held-out SD 1.4 (familiar) and Midjourney (unseen). Full spec: `ENSF 519.docx` in
the repo root (untracked).

## Non-negotiables
- Data and preprocessing:
  - GenImage only; 20k real + 20k SD 1.4 for training.
  - 224×224 RGB; labels 0 real / 1 AI.
  - Disjoint splits; Midjourney used only for testing; consistent JPEG re-encoding.
- Models:
  - CNN: from scratch.
  - ResNet-18: ImageNet weights, new 2-way head, full fine-tune.
- Metrics: accuracy / F1 / AUROC, degradation, 50% baseline.
- Integrity: never fabricate results; never tune on test sets (the primary configs are FROZEN now); don't push or
  commit without permission.

## Environments
**Windows PC (all real results).**
- Hardware: RTX 3080 10 GB, 20 CPUs.
- Software: `uv` venv `.venv` (Python 3.12.16, torch 2.14.1+cu126).
- Run Python as `.venv/Scripts/python.exe`.
- Smart App Control occasionally blocks a SciPy/scikit-learn DLL the first time it loads. Re-running the command works.

**Mac M4.** Used for development only (MPS, no CUDA).

## Data (done)
Source: HF mirror `nebula/GenImage-arrow`, revision `3f4b9f92…`, downloaded by `src/data/genimage_hf.py` into
`data/genimage` (23 GB, git-ignored), using seeded hash sampling.

`manifests/` (built, leakage PASSED, untracked until the user approves a commit):

| Manifest | Size |
|---|---|
| train | 20k + 20k |
| val | 2k + 2k |
| test_sd | 2k + 2k |
| test_mj | test_sd reals + 2k MJ AI |
| test_mj_native | 2k + 2k |

- 3 exact-duplicate exclusions. 8 pHash pairs, inspected visually: not copies.
- Caches: `data/cache/s224_bilinear_q95` and `.../s224_bilinear_nojpeg` (52k files each).
- Audit: the raw metadata baseline has AUROC 1.0 (real = JPEG, AI = PNG). The standardised file-size baseline gives
  0.55 on test_sd and 0.29 on test_mj.

## Results (done; `results/`, `docs/REPORT_GUIDE.md` evidence summary)
Primary seed 42:

| Model | SD acc / AUROC | Midjourney acc / AUROC |
|---|---|---|
| CNN | 0.897 / 0.961 | 0.514 / 0.465 |
| ResNet-18 | 0.963 / 0.994 | 0.608 / 0.727 |

Seeds (mean ± std over 3):
- CNN: SD 0.930 ± 0.029, Midjourney AUROC 0.565 ± 0.089.
- ResNet-18: SD 0.963 ± 0.001, Midjourney AUROC 0.721 ± 0.010.

Supporting experiments:
- E4 (CNN, no JPEG): SD 0.970, Midjourney 0.647 / AUROC 0.789. Probably a format shortcut (not tested).
- E6 (CNN, no augmentation): SD 0.951, Midjourney 0.534.
- E8 (ResNet-18 from scratch): SD 0.888, Midjourney 0.508 / AUROC 0.463. The advantage comes from pretraining.

E3 JPEG robustness: FPR stays roughly flat and FNR rises.

Checkpoints:
- Local in `checkpoints/<experiment>/best.pt`.
- SHA-256 hashes in `checkpoints_manifest.json` (cnn `cfe71097…`, resnet18 `31bb3092…`).
- No hosting URL yet.

## Code changes on 2026-11-26 (see docs/VALIDATION_REPORT.md §B)
- `src/data/`: new package with `genimage_hf.py` (downloader), `create_splits.py` (listing-aware selector), plus
  audit, preprocessing and dataset modules. `.gitignore` uses `/data/`.
- `src/models/resnet18.py`: now the collaborator's construction. `collaborator/resnet18_project/` holds their project,
  unmodified. `src/experiment.collaborator_table()` summarises their 3 seeds.
- Bug fixes:
  - a quadratic selector loop (249 min → 2 min, byte-identical output);
  - the audit plot's duplicate `split` column;
  - prediction CSV precision (`%.6f` → `%.9g`), after which all checkpoints were re-evaluated.
- Essential tests: 35 passed (CUDA included).

## Remaining (needs the group)
1. Commit (user approval needed): code, docs, `manifests/`, `results/` tables and figures, and `collaborator/`.
   Never commit `data/`, `checkpoints/`, `runs/`, `.dev/`, `.venv/` or `ENSF 519.docx`.
2. Upload the checkpoints and set their URLs in `checkpoints_manifest.json`.
3. Inspect the failure-case images (`results/tables/failure_cases_*.csv`).
4. Write the report, the reflections, and A7 once TA feedback exists.
5. Confirm the collaborator's name in REPORT_GUIDE.md and ATTRIBUTION.md.

## Re-running anything
```bash
.venv/Scripts/python.exe scripts/reproduce.py status
.venv/Scripts/python.exe -m pytest tests/essential -q
.venv/Scripts/python.exe scripts/run_experiments.py E1_cnn E2_resnet18   # skips training if runs/<exp>/summary.json exists; re-evaluates
.venv/Scripts/python.exe scripts/reproduce.py report
```

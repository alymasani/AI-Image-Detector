# Validation report

Every entry records a command that was actually executed. Synthetic-fixture results verify **software behaviour only**
and are not GenImage results. Two machines were used:
- the Apple M4 development machine on 2026-11-25 (§A);
- the Windows RTX 3080 machine that produced every real result, on 2026-11-26 (§B).

---------------------------------------------------------------------------------------------------------------------
## B. Windows / RTX 3080 (2026-11-26): real data, all reported results

### Environment (`python -m src.utils`)
| Item | Value |
|---|---|
| OS | Windows 11 (10.0.26300) |
| CPU | Intel, 20 logical CPUs |
| GPU | NVIDIA GeForce RTX 3080, 10 GB, CUDA 12.6 |
| Software | Python 3.12.16, torch 2.14.1+cu126, torchvision 0.29.1+cu126 |
| Environment | `uv`-managed `.venv` |

**Windows note.** Smart App Control twice blocked a freshly installed SciPy/scikit-learn DLL ("An Application Control
policy has blocked this file"):
- `scipy ... _dierckx`, during the first test run;
- `sklearn.linear_model._sgd_fast`, during the first audit.

Both runs succeeded when re-run unchanged. If it happens, re-run the command.

### Essential tests
`python -m pytest tests/essential -q` → **35 passed, 0 skipped** (≈10 s). This now includes `test_cnn_cuda_forward` (CUDA)
and the two Hugging Face downloader tests. They use a synthetic Arrow shard and check:
- the downloader keeps exactly the leading candidates of the split builder's permutation;
- it writes the official layout and the listing files;
- it resumes correctly.

The suite was re-run after the code changes listed below (final run in the "Final audit" section).

### Stage A smoke on CUDA
`python scripts/reproduce.py smoke` passed on CUDA (fp16 autocast + GradScaler) in about 2 min, producing 7 figures. This was the first execution of the CUDA mixed-precision path.

### Data acquisition and manifests
`python -m src.data.genimage_hf --data-root data/genimage --workers 4` downloaded from the pinned mirror revision
`3f4b9f92…` and kept 23 GB:

| Folder | Images kept |
|---|---|
| SD 1.4 train/ai | 20,531 |
| SD 1.4 train/nature | 20,840 |
| SD 1.4 val (each class) | 6,000 |
| Midjourney val (each class) | 6,000 |

`prepare_data.py check` → READY.

`prepare_data.py manifests --workers 16` produced every manifest at exactly the requested counts:
- train: 20,000 + 20,000
- val, test_sd, test_mj and test_mj_native: 2,000 + 2,000 each

There were no shortfalls, and no candidate was missing from the download. Three images were excluded as exact
duplicates. Eight pHash near-duplicate pairs were found; six cross a leakage pair, and all of them were inspected
visually: they are different images, not copies (DATASET.md). **The leakage check PASSED.**

**Bug found and fixed (performance only).** The first manifest build took **249 min**.
- Cause: `Selector.take` rebuilt the look-ahead chunk, with one `exists()` call per file, for every *accepted* image, so
  the work grew quadratically.
- Fix: the chunk is now built only when the next candidate has not been inspected yet.
- Result: a rebuild into a scratch folder took **1 min 50 s**, and all five manifests and `near_duplicates.csv` are
  **byte-identical** to the original build.

The near-duplicate loop also stopped using per-row `iloc`.

### Cache and audit
`prepare_data.py cache` built two caches of 52,000 files each:

| Cache | Size | Build time |
|---|---|---|
| q95 | 1.3 GB | 63 s |
| no-JPEG (for E4) | 4.4 GB | 103 s |

`prepare_data.py audit` results:
- **Raw metadata baseline:** accuracy and AUROC 1.000 on every manifest. Every real image is a JPEG and every AI image
  is a PNG.
- **Standardised-JPEG-size baseline:** AUROC 0.554 on test_sd and 0.289 on test_mj (the size signal reverses on
  Midjourney).

**Bug found and fixed.** The audit figure crashed with `Grouper for 'split' not 1-dimensional`. Renaming `manifest` to
`split` created a second `split` column, so the row-level column is now dropped first.

### Staged training
| Stage | Result |
|---|---|
| B: 32 real images (`.dev/tests/stage_b_tiny_overfit.py`) | Augmentation off, dropout 0, Adam 1e-3. Eval-mode accuracy 0.94 at step 20, **1.00 from step 40**, loss 1e-5 at step 300 (19 s) |
| C: pilot (1k train / 1k val) | Both models train. CNN best val loss 0.438, ResNet 0.314. 80 s in total |
| D: intermediate (5k train / 4k val) | CNN best val loss 0.397 (epoch 6), ResNet 0.235 (epoch 2). About 1,100 img/s with 4 loader workers, so `num_workers` stayed at 4 |
| E: full (40k train) | CNN early-stopped at epoch 12 (best 8). ResNet-18 early-stopped at epoch 11 (best 7). See `computational_comparison.csv` |

### Held-out evaluation
- Ran automatically after Stage E, together with E3 JPEG robustness and latency.
- Optional experiments E4, E6, E7 (×4) and E8 were run to completion; none failed.

**Bug found and fixed (reporting precision).** Prediction CSVs were written with `"%.6f"`.
- Effect: every probability below 5e-7 was rounded to 0. The AUROC that `statistics.json` recomputes from the CSVs
  therefore differed slightly from the in-memory AUROC (ResNet-18 test_mj: 0.7262 vs 0.7272).
- Fix: probabilities are now written with `"%.9g"`, which round-trips float32 exactly.
- All 9 checkpoints were **re-evaluated without retraining**. The bootstrap point estimates now equal the main-table
  values exactly for all six primary rows.

### Report assets
- `python scripts/reproduce.py report` produced 8 figures and the result tables. There are no PENDING rows.
- `reproduce.py checkpoints register` recorded the SHA-256 hashes of `checkpoints/cnn/best.pt` (`cfe71097…`) and
  `checkpoints/resnet18/best.pt` (`31bb3092…`).

---------------------------------------------------------------------------------------------------------------------
## A. Apple M4 development machine (2026-11-25): synthetic only

Environment: Apple M4, macOS, Python 3.9.6, torch 2.8.0 (MPS, no CUDA).
- `pytest tests/essential` → 31 passed, 1 skipped (CUDA).
- `reproduce.py smoke` → full synthetic pipeline on MPS in about 65 s.

Fixes made at that stage:
- a wrong hand-computed AUROC expectation in a test (the implementation was correct);
- a report-generator crash when no evaluations existed yet.

Development diagnostics (`.dev/`, git-ignored):
- a random-label tiny-overfit check reached 1.00 by step 40;
- the pretrained ResNet-18 weights match the Torchvision hub file `resnet18-f37072fd.pth`;
- the JPEG quality estimator reads PIL q50/75/90/95 as 50.0/74.7/90.1/95.0;
- `ruff check --select F,E9` is clean.

## Not verified
- The TALC/SLURM job script (`scripts/hpc/train.slurm`) was never submitted.
- Checkpoint hosting URLs are not set (files exist locally only).

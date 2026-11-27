# Spotting the Unseen Generator

ENSF 519 Group 38.
We train a CNN from scratch and a pretrained ResNet-18 to tell real images from AI-generated ones, then test how well they work on a generator they never saw.

## Setup

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync
```

## Get the dataset

The dataset is about 25 GB, so keep it in a folder outside OneDrive.
Tell the scripts where that folder is once (PowerShell, then open a new terminal):

```powershell
setx GENIMAGE_DATA_DIR C:\data\genimage
```

Then download it:

```bash
uv run python scripts/download_genimage.py
```

This pulls the images from [GenImage](https://huggingface.co/datasets/nebula/GenImage-arrow).
If it stops partway, run it again.

| Split | Images from | Real | Fake |
|---|---|---|---|
| train | Stable Diffusion 1.4 | 20,000 | 20,000 |
| val | Stable Diffusion 1.4 | 2,500 | 2,500 |
| test_sd14 | Stable Diffusion 1.4 | 6,000 | 6,000 |
| test_midjourney | Midjourney (unseen) | 6,000 | 6,000 |

Images are saved to `raw/genimage/<split>/<real|fake>/`.
Each split also gets a list of its images in `manifests/<split>.csv`, with labels 0 for real and 1 for fake.

Don't commit the images to git.

## Preprocess the images

```bash
uv run python scripts/preprocess.py
```

This makes small 224x224 copies of every image (center-cropped, never stretched) in `processed/jpeg/` and `processed/png/`.
The JPEG copy is what we train on.
Saving every image the same way stops the model from cheating, since in the original data real photos are JPEGs and AI images are PNGs.

The JPEG copy is only about 2 GB, so instead of downloading 25 GB you can get a zip of `processed/jpeg/` and `manifests/` from a teammate and unzip it into your data folder.

## Train ResNet-18

```bash
uv run python scripts/train.py --seed 0
```

This fine-tunes the pretrained ResNet-18 on the SD 1.4 training images for 8 epochs and keeps the epoch with the best validation AUROC.
That checkpoint is then scored on `test_sd14` (the generator it trained on) and `test_midjourney` (one it never saw).
Run it with `--seed 1` and `--seed 2` as well, so the results can be averaged over seeds.
Add `--epochs 1 --limit 64` for a quick check that everything works before a full run.
The CNN trains with the same script by adding `--model cnn`, and its runs are saved as `cnn_seed0` and so on.
Since the CNN starts from scratch rather than from ImageNet weights, it will likely need a higher `--lr` and more `--epochs` than the defaults, which are set for ResNet-18.

Each run is named like `resnet18_seed0` and writes:

| File | What's in it |
|---|---|
| `results/runs/<run>/best.pt` | Weights from the best epoch |
| `results/runs/<run>/history.csv` | Loss and validation AUROC per epoch |
| `results/predictions/<run>__<test_set>.csv` | The model's probability that each test image is fake |
| `results/metrics.csv` | Accuracy, F1 and AUROC for every run and test set |

Running the same run again replaces its rows in `results/metrics.csv` instead of adding duplicates.
The Evaluation section of `final_project.ipynb` reads these files and draws the figures into `results/figures/`.

## Run the tests

```bash
uv run pytest
```

import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from PIL import Image

from detector.data import CLASS_NAMES, DATA_DIR_ENV, SPLITS

matplotlib.use("Agg")  # render figures off-screen, no windows popping up during tests

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
RAW_FIELDS = ("path", "label", "subset", "source_generator", "source_path", "sha256")

RED = (255, 0, 0)
GREEN = (0, 255, 0)


def load_script(name: str):
    """Import scripts/<name>.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def striped(width: int, height: int, horizontal: bool) -> Image.Image:
    """Green image with 50px red bands on the two long ends.

    A center crop never reaches the bands; squashing the image to a square would keep them.
    """
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:] = GREEN
    if horizontal:
        arr[:, :50] = RED
        arr[:, -50:] = RED
    else:
        arr[:50, :] = RED
        arr[-50:, :] = RED
    return Image.fromarray(arr)


def sample_images(split_index: int) -> list[tuple[str, int, Image.Image, str]]:
    """(name, label, image, format) for one split: two odd-shaped real JPEGs, two square PNGs."""
    rng = np.random.default_rng(split_index)
    noise = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    return [
        (f"wide_{split_index}.JPEG", 0, striped(400, 200, horizontal=True), "JPEG"),
        (f"tall_{split_index}.JPEG", 0, striped(200, 400, horizontal=False), "JPEG"),
        (f"white_{split_index}.png", 1, Image.new("RGB", (512, 512), (255, 255, 255)), "PNG"),
        (f"noise_{split_index}.png", 1, Image.fromarray(noise), "PNG"),
    ]


@pytest.fixture
def raw_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tiny copy of the downloaded dataset layout: raw images plus raw manifests."""
    data_dir = tmp_path / "genimage"
    monkeypatch.setenv(DATA_DIR_ENV, str(data_dir))
    (data_dir / "manifests").mkdir(parents=True)
    for index, split in enumerate(SPLITS):
        rows = []
        for name, label, img, fmt in sample_images(index):
            rel = Path("raw", "genimage", split, CLASS_NAMES[label], name)
            (data_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            save_options = {"quality": 100} if fmt == "JPEG" else {}
            img.save(data_dir / rel, format=fmt, **save_options)
            rows.append(
                {
                    "path": rel.as_posix(),
                    "label": label,
                    "subset": split,
                    "source_generator": "stable_diffusion_v_1_4",
                    "source_path": f"upstream/{name}",
                    "sha256": "0" * 64,
                }
            )
        with (data_dir / "manifests" / f"{split}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    return data_dir


@pytest.fixture
def preprocess():
    return load_script("preprocess")


@pytest.fixture
def processed_data_dir(raw_data_dir: Path, preprocess) -> Path:
    preprocess.main(["--workers", "1"])
    return raw_data_dir

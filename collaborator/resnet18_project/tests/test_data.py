from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import striped
from PIL import Image

from detector import data
from detector.data import (
    GenImageDataset,
    build_transforms,
    center_crop_resize,
    make_loaders,
    processed_manifest,
    raw_manifest,
    read_manifest,
    resolve_data_dir,
)


def test_resolve_data_dir_precedence(monkeypatch, tmp_path):
    monkeypatch.delenv(data.DATA_DIR_ENV, raising=False)
    assert resolve_data_dir() == data.REPO_ROOT / "data"

    monkeypatch.setenv(data.DATA_DIR_ENV, str(tmp_path / "from_env"))
    assert resolve_data_dir() == tmp_path / "from_env"
    assert resolve_data_dir(tmp_path / "explicit") == tmp_path / "explicit"


@pytest.mark.parametrize(
    "img",
    [striped(400, 200, horizontal=True), striped(200, 400, horizontal=False)],
    ids=["wide", "tall"],
)
def test_center_crop_resize_keeps_the_middle(img):
    out = np.asarray(center_crop_resize(img))
    assert out.shape == (224, 224, 3)
    assert out[..., 0].max() < 30  # none of the red end bands made it in


def test_center_crop_resize_handles_small_and_non_rgb_images():
    assert center_crop_resize(Image.new("L", (100, 150))).size == (224, 224)
    assert center_crop_resize(Image.new("RGBA", (300, 300))).mode == "RGB"


def test_dataset_returns_normalized_tensors(processed_data_dir):
    dataset = GenImageDataset(
        processed_manifest(processed_data_dir, "png", "train"), processed_data_dir
    )
    assert len(dataset) == 4
    assert dataset.labels == [0, 0, 1, 1]

    image, label = dataset[2]  # the all-white fake image
    assert image.shape == (3, 224, 224) and image.dtype == torch.float32
    assert label == 1
    expected = (1 - torch.tensor(data.IMAGENET_MEAN)) / torch.tensor(data.IMAGENET_STD)
    torch.testing.assert_close(image.mean(dim=(1, 2)), expected)


def test_dataset_rejects_unprocessed_images(processed_data_dir):
    dataset = GenImageDataset(raw_manifest(processed_data_dir, "train"), processed_data_dir)
    with pytest.raises(ValueError, match="preprocess"):
        dataset[0]


def test_limit_keeps_classes_balanced(processed_data_dir):
    rows = read_manifest(processed_manifest(processed_data_dir, "jpeg", "val"), limit=2)
    assert sorted(label for _, label in rows) == [0, 1]


def test_only_training_transform_is_random():
    img = striped(224, 224, horizontal=True)
    img.paste((0, 0, 255), (0, 0, 20, 224))  # make it asymmetric so a flip is visible
    eval_tf, train_tf = build_transforms(train=False), build_transforms(train=True)
    assert all(torch.equal(eval_tf(img), eval_tf(img)) for _ in range(5))
    torch.manual_seed(0)
    outputs = {train_tf(img)[2, 0, 0].item() for _ in range(20)}
    assert len(outputs) == 2  # flipped and not flipped both showed up


def test_make_loaders_builds_every_split(processed_data_dir):
    def train_order(seed):
        loaders = make_loaders(processed_data_dir, batch_size=4, num_workers=0, seed=seed)
        return loaders, [label for _, labels in loaders["train"] for label in labels.tolist()]

    loaders, order = train_order(seed=0)
    assert set(loaders) == {"train", "val", "test_sd14", "test_midjourney"}
    images, labels = next(iter(loaders["test_midjourney"]))
    assert images.shape == (4, 3, 224, 224)
    assert labels.tolist() == [0, 0, 1, 1]  # evaluation keeps the manifest order
    assert train_order(seed=0)[1] == order


def test_make_loaders_rejects_unknown_format(processed_data_dir):
    with pytest.raises(ValueError, match="fmt"):
        make_loaders(Path(processed_data_dir), fmt="webp")

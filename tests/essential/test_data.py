"""Dataset, preprocessing, label and leakage-prevention tests (synthetic fixtures)."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src import LABEL_AI, LABEL_REAL
from src.data.create_splits import build_manifests, check_leakage
from src.data.dataset import make_dataset, read_manifest
from src.data.preprocessing import CorruptImageError, StandardizeImage, cache_path_for, load_rgb, standardized_bytes

from .conftest import suite_config

NAMES = suite_config()["manifests"]


def test_manifest_counts_labels_and_generators(manifests):
    out, meta = manifests
    assert not meta["shortfalls"]
    train = read_manifest(out / NAMES["train"])
    assert (train.label == LABEL_REAL).sum() == 8 and (train.label == LABEL_AI).sum() == 8
    # Label convention: real photos (nature folders) = 0, generated (ai folders) = 1.
    for name in NAMES.values():
        df = read_manifest(out / name)
        assert set(df[df.relative_path.str.contains("/nature/")].label) <= {LABEL_REAL}
        assert set(df[df.relative_path.str.contains("/ai/")].label) <= {LABEL_AI}
        assert not df.relative_path.str.startswith("/").any(), "manifest paths must be relative"


def test_corrupt_files_excluded_and_logged(manifests):
    out, meta = manifests
    reasons = {e["relative_path"].split("/")[-1]: e["reason"] for e in meta["exclusions"]}
    assert reasons.get("corrupt_0001.png") == "unreadable"
    allrows = pd.concat([read_manifest(out / n) for n in NAMES.values()])
    assert not allrows.relative_path.str.contains("corrupt").any()
    # The grayscale and RGBA edge cases were consumed and kept.
    assert allrows["mode"].isin(["L"]).any() and allrows["mode"].isin(["RGBA"]).any()


def test_leakage_checks_pass_and_shared_reals(manifests):
    out, _ = manifests
    rep = check_leakage(out, NAMES)
    assert rep["passed"], rep["problems"]
    assert rep["shared_real_identical"] is True
    assert rep["midjourney_in_train"] == 0 and rep["midjourney_in_val"] == 0


def test_leakage_check_detects_injected_overlap(manifests, tmp_path):
    out, _ = manifests
    for n in NAMES.values():
        shutil.copy(out / n, tmp_path / n)
    val = read_manifest(tmp_path / NAMES["val"])
    test = read_manifest(tmp_path / NAMES["test_sd"])
    leaked = pd.concat([test, val.iloc[[0]]])       # a validation image placed in the test set
    leaked.to_csv(tmp_path / NAMES["test_sd"], index=False)
    rep = check_leakage(tmp_path, NAMES)
    assert not rep["passed"]
    assert any("val and test_sd" in p for p in rep["problems"])


def test_split_generation_is_reproducible(synthetic_root, manifests, tmp_path):
    out, meta = manifests
    meta2 = build_manifests(suite_config(), synthetic_root, tmp_path)
    assert meta2["manifest_sha256"] == meta["manifest_sha256"]


def test_rgb_conversion_of_gray_and_rgba(tmp_path):
    g = tmp_path / "g.png"
    Image.new("L", (40, 30), 128).save(g)
    a = tmp_path / "a.png"
    Image.new("RGBA", (40, 30), (255, 0, 0, 0)).save(a)   # fully transparent red
    assert load_rgb(g).mode == "RGB"
    rgba = load_rgb(a)
    assert rgba.mode == "RGB" and rgba.getpixel((0, 0)) == (255, 255, 255)  # composited on white


def test_corrupt_image_raises(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\nnot a png")
    with pytest.raises(CorruptImageError):
        load_rgb(bad)


def test_dataset_tensor_shapes_dtypes_and_labels(manifests, synthetic_root, data_cfg):
    out, _ = manifests
    df = read_manifest(out / NAMES["val"])
    ds = make_dataset(df, synthetic_root, data_cfg, train=False)
    x, y = ds[0]
    assert x.shape == (3, 224, 224) and x.dtype == torch.float32
    assert y.dtype == torch.long and int(y) == int(df.label.iloc[0])
    batch = torch.stack([ds[i][0] for i in range(4)])
    assert batch.shape == (4, 3, 224, 224)


def test_eval_transform_is_deterministic_and_train_is_not(manifests, synthetic_root, data_cfg):
    out, _ = manifests
    df = read_manifest(out / NAMES["val"])
    ev = make_dataset(df, synthetic_root, data_cfg, train=False)
    assert torch.equal(ev[1][0], ev[1][0])
    tr = make_dataset(df, synthetic_root, data_cfg, train=True)
    torch.manual_seed(0)
    samples = [tr[1][0] for _ in range(6)]
    assert any(not torch.equal(samples[0], s) for s in samples[1:])


def test_standardize_preserves_aspect_and_crops(tmp_path):
    im = Image.fromarray((np.random.default_rng(0).random((300, 600, 3)) * 255).astype("uint8"))
    out = StandardizeImage(224, "bilinear", 95)(im)
    assert out.size == (224, 224) and out.mode == "RGB"


def test_cache_is_bit_identical_to_on_the_fly(manifests, synthetic_root, data_cfg, tmp_path):
    out, _ = manifests
    df = read_manifest(out / NAMES["test_sd"]).head(3)
    cfg = dict(data_cfg, cache_dir=str(tmp_path / "cache"))
    for rel, sha in df[["relative_path", "sha256"]].itertuples(index=False):
        dst = cache_path_for(tmp_path / "cache", cfg, sha)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(standardized_bytes(synthetic_root / rel, cfg))
    a = make_dataset(df, synthetic_root, data_cfg, train=False)
    b = make_dataset(df, synthetic_root, cfg, train=False)
    for i in range(len(df)):
        assert torch.equal(a[i][0], b[i][0])


def test_selective_download_keeps_exactly_the_leading_candidates(tmp_path):
    """Rows the downloader keeps (key below the pool fraction) are a prefix of the builder's permutation."""
    from src.data.create_splits import OFFICIAL_LISTING, Selector, candidate_key, key_fraction
    from src.data.genimage_hf import sampling_fractions

    cfg = suite_config()
    cfg["splits"]["train_per_class"] = 150
    folder = tmp_path / "gen" / "train" / "ai"
    folder.mkdir(parents=True)
    names = [f"{i % 1000:03d}_sdv4_{i:05d}.png" for i in range(2000)]
    (folder / OFFICIAL_LISTING).write_text("\n".join(names))
    pool, fraction = sampling_fractions(cfg, "sdv14", "train", {"fake": 2000, "real": 2000}, 0.03, {})["fake"]
    assert pool == "sdv14/train/ai" and 0 < fraction < 1
    seed = int(cfg["splits"]["seed"])
    kept = {f"gen/train/ai/{n}" for n in names if key_fraction(candidate_key(seed, pool, n)) < fraction}
    order = Selector(tmp_path, seed).permutation(pool, "gen/train/ai", [".png"])
    assert set(order[: len(kept)]) == kept and len(kept) >= 150


def test_downloader_writes_official_layout_and_listing(tmp_path, monkeypatch):
    """One synthetic Arrow shard -> selected files in <folder>/<partition>/<ai|nature>, full listing, md5 checked."""
    import hashlib
    import io

    import pyarrow as pa

    from src.data import genimage_hf
    from src.data.create_splits import candidate_key, key_fraction

    rows = {"image_path": [], "image": [], "label": [], "md5": []}
    for i in range(20):
        label = i % 2
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (i, 0, 0)).save(buf, format="PNG")
        rows["image_path"].append(f"stable_diffusion_v_1_4/train/{'ai' if label else 'nature'}/{i:03d}.png")
        rows["image"].append(buf.getvalue())
        rows["label"].append(label)
        rows["md5"].append(hashlib.md5(buf.getvalue()).hexdigest())
    shard = tmp_path / "shard.arrow"
    table = pa.table(rows)
    with pa.OSFile(str(shard), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    monkeypatch.setattr(genimage_hf, "fetch", lambda name, cache: shutil.copy(shard, tmp_path / "copy.arrow")
                        and tmp_path / "copy.arrow")
    d = suite_config()["dataset"]
    fractions = {"fake": ("sdv14/train/ai", 0.5), "real": ("sdv14/train/nature", 1.0)}
    root = tmp_path / "root"
    res = genimage_hf.process_shard("s0", "leaf", root, "SD", "train", fractions, 7, d, tmp_path / "prog",
                                    tmp_path / "cache")
    assert sorted(res["listing"]["fake"]) == [f"{i:03d}.png" for i in range(1, 20, 2)]
    written_ai = sorted(p.name for p in (root / "SD" / "train" / "ai").iterdir())
    expected_ai = sorted(n for n in res["listing"]["fake"]
                         if key_fraction(candidate_key(7, "sdv14/train/ai", n)) < 0.5)
    assert written_ai == expected_ai
    assert len(list((root / "SD" / "train" / "nature").iterdir())) == 10
    assert not (tmp_path / "copy.arrow").exists()          # shard deleted after use
    assert genimage_hf.process_shard("s0", "leaf", root, "SD", "train", fractions, 7, d, tmp_path / "prog",
                                     tmp_path / "cache") == json_roundtrip(res)   # resume skips finished shard


def json_roundtrip(obj):
    import json

    return json.loads(json.dumps(obj))


def test_partial_pool_stops_at_missing_candidate(tmp_path):
    """With an official listing, a not-yet-downloaded candidate stops the pool (never skipped)."""
    from src.data.create_splits import OFFICIAL_LISTING, Selector

    folder = tmp_path / "gen" / "val" / "ai"
    folder.mkdir(parents=True)
    names = [f"{i:03d}.png" for i in range(10)]
    (folder / OFFICIAL_LISTING).write_text("\n".join(names))
    sel = Selector(tmp_path, seed=1)
    order = sel.permutation("p", "gen/val/ai", [".png"])
    for rel in order[:3]:
        Image.new("RGB", (8, 8), (int(rel[-7:-4]), 0, 0)).save(tmp_path / rel)
    rows, _ = sel.take(order, 0, 5, {"label": 1}, "x")
    assert len(rows) == 3 and sel.not_downloaded == [order[3]] and sel.pool_sizes["p"] == 10

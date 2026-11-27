import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import SCRIPTS_DIR
from PIL import Image

from detector.data import DATA_DIR_ENV, FORMATS, SPLITS, processed_manifest, raw_manifest


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def processed_bytes(data_dir: Path) -> dict[str, bytes]:
    files = sorted((data_dir / "processed").rglob("*.*"))
    return {
        p.relative_to(data_dir).as_posix(): p.read_bytes() for p in files if p.suffix != ".json"
    }


def test_writes_224px_images_and_manifests(processed_data_dir):
    for fmt in FORMATS:
        for split in SPLITS:
            raw_rows = read_csv(raw_manifest(processed_data_dir, split))
            rows = read_csv(processed_manifest(processed_data_dir, fmt, split))
            assert [r["raw_path"] for r in rows] == [r["path"] for r in raw_rows]
            assert [r["label"] for r in rows] == [r["label"] for r in raw_rows]
            for row in rows:
                with Image.open(processed_data_dir / row["path"]) as img:
                    assert img.size == (224, 224)
                    assert img.mode == "RGB"
                    assert img.format == {"jpeg": "JPEG", "png": "PNG"}[fmt]
                assert row["path"].startswith(f"processed/{fmt}/{split}/")


@pytest.mark.parametrize("name", ["wide_0", "tall_0"])
def test_odd_shaped_images_are_cropped_not_stretched(processed_data_dir, name):
    path = processed_data_dir / "processed" / "png" / "train" / "real" / f"{name}.png"
    arr = np.asarray(Image.open(path))
    # The red bands sit outside the center crop, so not a single red-ish pixel should survive.
    assert arr[..., 0].max() < 30
    assert arr[..., 1].min() > 225


def test_rerun_skips_and_overwrite_is_byte_identical(processed_data_dir, preprocess):
    before = processed_bytes(processed_data_dir)
    some_file = processed_data_dir / next(iter(before))
    mtime = some_file.stat().st_mtime_ns

    preprocess.main(["--workers", "1"])
    assert some_file.stat().st_mtime_ns == mtime  # untouched on a plain re-run

    preprocess.main(["--workers", "1", "--overwrite"])
    assert some_file.stat().st_mtime_ns != mtime
    assert processed_bytes(processed_data_dir) == before


def test_refuses_to_mix_settings(processed_data_dir, preprocess):
    args = ["--workers", "1", "--quality", "80"]
    with pytest.raises(SystemExit, match="--overwrite"):
        preprocess.main(args)

    preprocess.main([*args, "--overwrite"])
    settings = json.loads((processed_data_dir / "processed/jpeg/settings.json").read_text())
    assert settings["quality"] == 80


def test_missing_download_is_reported(tmp_path, monkeypatch, preprocess):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    with pytest.raises(SystemExit, match="download_genimage"):
        preprocess.main(["--workers", "1", "--formats", "jpeg"])


def test_runs_in_parallel_from_the_command_line(raw_data_dir, preprocess):
    # Worker processes only work when the script is run as a program, so run it for real.
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "preprocess.py"),
            "--workers",
            "2",
            "--formats",
            "jpeg",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    rows = read_csv(processed_manifest(raw_data_dir, "jpeg", "test_midjourney"))
    assert len(rows) == 4
    assert all((raw_data_dir / r["path"]).is_file() for r in rows)
    assert not (raw_data_dir / "processed" / "png").exists()

import csv
import importlib.util
import io
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_genimage.py"
spec = importlib.util.spec_from_file_location("download_genimage", SCRIPT)
dg = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dg
spec.loader.exec_module(dg)


def png_bytes(seed: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (seed % 256, seed // 256 % 256, 7)).save(buf, format="PNG")
    return buf.getvalue()


def write_shard(path: Path, rows: list[tuple[str, bytes, int]]) -> Path:
    schema = pa.schema([("image_path", pa.string()), ("image", pa.binary()), ("label", pa.int64())])
    table = pa.table([list(col) for col in zip(*rows, strict=True)], schema=schema)
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(sink, schema) as writer:
        writer.write_table(table, max_chunksize=3)
    return path


def interleaved_rows(prefix: str, n: int, start_seed: int) -> list[tuple[str, bytes, int]]:
    rows = []
    for i in range(n):
        label = i % 2
        folder = "ai" if label else "nature"
        rows.append((f"{prefix}/{folder}/{i:04d}.png", png_bytes(start_seed + i), label))
    return rows


def run_leaf(tmp_path, shards, quotas, claimed=None, stats=None):
    claimed = {} if claimed is None else claimed
    stats = Counter() if stats is None else stats
    dg.extract_leaf(shards, quotas, "gen", tmp_path, claimed, stats)
    return claimed, stats


def test_fills_quotas_in_order_and_stops_early(tmp_path):
    rows = interleaved_rows("sd/train", 40, 0)
    shard_a = write_shard(tmp_path / "a.arrow", rows[:20])
    shard_b = write_shard(tmp_path / "b.arrow", rows[20:])
    consumed = []

    def shards():
        for shard in (shard_a, shard_b):
            consumed.append(shard)
            yield shard

    train, val = dg.Quota("train", 6), dg.Quota("val", 2)
    run_leaf(tmp_path, shards(), [train, val])

    assert train.counts == [6, 6] and val.counts == [2, 2]
    assert consumed == [shard_a]  # 16 rows were enough, so the second shard is never fetched
    train_ids = {int(Path(r["source_path"]).stem) for r in train.rows}
    val_ids = {int(Path(r["source_path"]).stem) for r in val.rows}
    # Train takes the first rows of each class, val the ones right after.
    assert train_ids == set(range(12)) and val_ids == {12, 13, 14, 15}
    for row in train.rows + val.rows:
        saved = tmp_path / row["path"]
        assert saved.parent.name == dg.CLASS_NAMES[row["label"]]
        assert saved.read_bytes() == png_bytes(int(Path(row["source_path"]).stem))


def test_skips_corrupt_and_duplicate_images(tmp_path):
    rows = interleaved_rows("sd/train", 10, 0)
    rows.insert(0, ("sd/train/ai/broken.png", b"not an image", 1))
    rows.insert(1, ("sd/train/nature/empty.png", b"", 0))
    rows.insert(2, ("sd/train/nature/copy.png", rows[2][1], 0))  # same bytes as 0000.png
    shard = write_shard(tmp_path / "a.arrow", rows)

    train = dg.Quota("train", 5)
    _, stats = run_leaf(tmp_path, [shard], [train])

    assert train.counts == [5, 5]
    assert stats == Counter({"unreadable images skipped": 2, "duplicate images skipped": 1})
    assert "sd/train/nature/0000.png" not in {r["source_path"] for r in train.rows}


def test_image_in_train_is_never_reused_for_test(tmp_path):
    train_rows = interleaved_rows("sd/train", 4, 0)
    test_rows = interleaved_rows("sd/test", 6, 100)
    test_rows[0] = ("sd/test/nature/leak.png", train_rows[0][1], 0)
    train = dg.Quota("train", 2)
    test = dg.Quota("test_sd14", 2)

    claimed, stats = run_leaf(tmp_path, [write_shard(tmp_path / "tr.arrow", train_rows)], [train])
    run_leaf(tmp_path, [write_shard(tmp_path / "te.arrow", test_rows)], [test], claimed, stats)

    assert test.full
    assert stats["duplicate images skipped"] == 1
    assert "sd/test/nature/leak.png" not in {r["source_path"] for r in test.rows}


def test_test_sets_may_share_real_photos(tmp_path):
    shared = png_bytes(999)
    sd_rows = [("sd/test/nature/x.png", shared, 0), ("sd/test/ai/y.png", png_bytes(1), 1)]
    mj_rows = [("mj/test/nature/x.png", shared, 0), ("mj/test/ai/z.png", png_bytes(2), 1)]
    sd, mj = dg.Quota("test_sd14", 1), dg.Quota("test_midjourney", 1)

    claimed, stats = run_leaf(tmp_path, [write_shard(tmp_path / "sd.arrow", sd_rows)], [sd])
    run_leaf(tmp_path, [write_shard(tmp_path / "mj.arrow", mj_rows)], [mj], claimed, stats)

    assert sd.full and mj.full
    assert stats["duplicate images skipped"] == 0


def test_short_leaf_leaves_quota_unfilled(tmp_path):
    shard = write_shard(tmp_path / "a.arrow", interleaved_rows("sd/test", 6, 0))
    test = dg.Quota("test_sd14", 5)
    run_leaf(tmp_path, [shard], [test])
    assert test.counts == [3, 3] and not test.full


def test_manifest_and_prune(tmp_path):
    shard = write_shard(tmp_path / "a.arrow", interleaved_rows("sd/train", 8, 0))
    train = dg.Quota("train", 2)
    run_leaf(tmp_path, [shard], [train])
    stale = tmp_path / "raw" / "genimage" / "train" / "real" / "old.png"
    stale.write_bytes(b"left over from a bigger run")

    manifest = dg.write_manifest(train, tmp_path)
    removed = dg.prune_stale_files(train, tmp_path)

    assert removed == 1 and not stale.exists()
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["label"] for r in rows] == ["0", "1", "0", "1"]
    assert all((tmp_path / r["path"]).is_file() for r in rows)
    assert set(rows[0]) == set(dg.MANIFEST_FIELDS)


def test_rejects_non_positive_counts():
    with pytest.raises(SystemExit):
        dg.parse_args(["--train-per-class", "0"])

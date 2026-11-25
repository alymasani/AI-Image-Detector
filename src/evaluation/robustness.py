"""JPEG compression robustness (experiment E3).

The main pipeline already re-encodes every image at JPEG quality 95 (dataset standardisation, so
both classes share one compression setting). E3 is different: trained models are NOT retrained;
held-out images receive an ADDITIONAL compression at quality 90 / 75 / 50 after standardisation,
simulating social-media recompression. Because preprocessing is deterministic, both models see
byte-identical transformed images.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from ..utils import read_json


def run_robustness(checkpoints: Sequence[str], tests: Sequence[str], qualities: Sequence[int], **eval_kwargs) -> None:
    """Evaluate every checkpoint on every test set at each additional JPEG quality."""
    from ..evaluate import evaluate_checkpoint

    for ckpt in checkpoints:
        for q in qualities:
            evaluate_checkpoint(ckpt, tests, extra_jpeg=int(q), **eval_kwargs)


def robustness_table(summaries_dir: Path, experiments: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Collect baseline (no extra JPEG) and extra-JPEG evaluation summaries into one long table."""
    rows: List[dict] = []
    for path in sorted(Path(summaries_dir).glob("eval_*.json")):
        s = read_json(path)
        if experiments and s["experiment"] not in experiments:
            continue
        m = s["metrics"]
        rows.append({
            "experiment": s["experiment"], "model": s["model"], "test": s["test"],
            "extra_jpeg_quality": s["extra_jpeg_quality"] or 100,  # 100 = no additional compression
            "accuracy": m["accuracy"], "f1": m["f1"], "auroc": m["auroc"], "fpr": m["fpr"], "fnr": m["fnr"],
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["experiment", "test", "extra_jpeg_quality"], ascending=[True, True, False])
    return df

"""Run experiments from configs/experiments.yaml (train, then evaluate full-stage runs).

Examples:
    python scripts/run_experiments.py E1_cnn --stage pilot          # Stage C, no test evaluation
    python scripts/run_experiments.py E1_cnn E2_resnet18 --stage intermediate
    python scripts/run_experiments.py E1_cnn E2_resnet18            # full primary experiments + E3 robustness
    python scripts/run_experiments.py E4_cnn_no_jpeg --set data.train_limit=5000   # reduced-scope E4
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment import main  # noqa: E402

if __name__ == "__main__":
    main(["run"] + sys.argv[1:])

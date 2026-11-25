"""Generate all report tables and figures from exported results (results/summaries, results/predictions).

Missing results are reported as PENDING; nothing is filled in by hand.

    python scripts/generate_report_assets.py [--results results] [--data-root /path/to/genimage]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment import main  # noqa: E402

if __name__ == "__main__":
    main(["report"] + sys.argv[1:])

"""Train a model from a config.

Examples:
    python -m src.train --config configs/cnn.yaml
    python -m src.train --config configs/resnet18.yaml --data-root /scratch/genimage
    python -m src.train --config configs/cnn.yaml --set data.train_limit=500 experiment_name=cnn_pilot
    python -m src.train --config configs/cnn.yaml --resume checkpoints/cnn/last.pt
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .training.trainer import train_model
from .utils import load_config


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Train the custom CNN or ResNet-18 detector")
    p.add_argument("--config", required=True)
    p.add_argument("--data-root", default=None, help="GenImage root (else $GENIMAGE_ROOT, else data.root)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. training.batch_size=16")
    p.add_argument("--resume", default=None, help="path to a last.pt checkpoint to resume from")
    p.add_argument("--overwrite", action="store_true", help="allow overwriting an existing run's checkpoints")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.set)
    train_model(cfg, data_root=args.data_root, resume=args.resume, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

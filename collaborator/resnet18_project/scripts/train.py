"""Train a detector on Stable Diffusion 1.4 images and test it on seen and unseen generators.

ResNet-18 is the default. The CNN uses the same script with --model cnn.

What it does, in order:
1. Trains on the SD 1.4 training images, checking validation AUROC after every epoch.
2. Keeps the weights from the epoch with the best validation AUROC.
3. Tests those weights on test_sd14 (the generator it trained on) and test_midjourney
   (a generator it never saw). The gap between the two is what the project is about.

Each run is named <model>_seed<seed>, for example resnet18_seed0, and saves:

    results/runs/<run>/        best.pt (weights), history.csv (per epoch), config.json (settings)
    results/predictions/       the model's prediction for every test image
    results/metrics.csv        accuracy, F1 and AUROC (a rerun replaces its old rows)

Images are read from --data-dir, or $GENIMAGE_DATA_DIR if that's not given, or the repo's data/.

Usage:
    uv run python scripts/train.py --seed 0
    uv run python scripts/train.py --seed 0 --epochs 1 --limit 64   # quick check, ~1 minute
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from detector.data import FORMATS, make_loaders
from detector.models import MODEL_NAMES, build_model
from detector.training import (
    RESULTS_DIR,
    default_device,
    evaluate_run,
    fit,
    set_seed,
    upsert_metrics,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model", choices=MODEL_NAMES, default="resnet18", help="which model to train"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed; train seeds 0, 1 and 2 to average"
    )
    parser.add_argument("--epochs", type=int, default=8, help="passes over the training images")
    parser.add_argument("--lr", type=float, default=1e-4, help="starting learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="how strongly to keep weights small"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="images per training step")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="background processes that load images"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="use only this many images per split, half real and half fake (for quick checks)",
    )
    parser.add_argument(
        "--format",
        choices=FORMATS,
        default="jpeg",
        dest="fmt",
        help="which preprocessed copy of the images to use",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="where the dataset is")
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR, help="where to save the results"
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="start ResNet-18 from random weights instead of the ImageNet ones",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run = f"{args.model}_seed{args.seed}"
    run_dir = args.results_dir / "runs" / run
    device = default_device()
    print(f"Run {run} on {device}")

    set_seed(args.seed)
    loaders = make_loaders(
        args.data_dir,
        fmt=args.fmt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
        seed=args.seed,
    )
    model = build_model(args.model, pretrained=args.pretrained)

    # Save the settings next to the weights, so we can always tell how a run was made. Settings
    # that only describe this computer (folders, loader processes) don't change the result, so
    # they're left out, which keeps the file the same for everyone who reruns it.
    run_dir.mkdir(parents=True, exist_ok=True)
    machine_only = {"data_dir", "results_dir", "num_workers"}
    config = {k: v for k, v in vars(args).items() if k not in machine_only}
    (run_dir / "config.json").write_text(json.dumps({"run": run, **config}, indent=2) + "\n")

    fit(
        model,
        loaders,
        run_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )
    # Test the best epoch (not the last one) on both test sets.
    metrics = evaluate_run(
        model, loaders, run, args.model, args.seed, args.results_dir, device=device
    )
    upsert_metrics(metrics, args.results_dir / "metrics.csv")
    print(
        metrics.drop(columns=["run", "model", "seed"]).to_string(index=False, float_format="%.4f")
    )


if __name__ == "__main__":
    main()

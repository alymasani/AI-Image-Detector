"""Unified reproduction workflow.

    python scripts/reproduce.py status        # deps, environment, dataset, manifests, checkpoints, results
    python scripts/reproduce.py tests         # essential verification tests (tests/essential)
    python scripts/reproduce.py smoke         # SYNTHETIC end-to-end run in .dev/ (CPU-friendly, ~minutes)
    python scripts/reproduce.py manifests --data-root D   # build + leakage-check manifests from GenImage
    python scripts/reproduce.py train --data-root D       # full primary experiments E1 + E2 (GPU hours)
    python scripts/reproduce.py evaluate --data-root D    # evaluation-only from supplied checkpoints
    python scripts/reproduce.py report --data-root D      # tables + figures from exported results
    python scripts/reproduce.py checkpoints register      # record SHA-256 of local best.pt files
    python scripts/reproduce.py checkpoints fetch         # download (if URLs are set) and verify checksums

The smoke workflow verifies SOFTWARE only. Its numbers are not GenImage results.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED = ["torch", "torchvision", "numpy", "PIL", "sklearn", "matplotlib", "pandas", "yaml", "scipy", "imagehash"]
SMOKE_RESULTS = ".dev/temporary_results/results"
SMOKE_CKPTS = ".dev/temporary_results/checkpoints"


def check_dependencies() -> bool:
    ok = True
    for mod in REQUIRED:
        try:
            m = importlib.import_module(mod)
            print(f"  ok       {mod} {getattr(m, '__version__', '')}")
        except ImportError:
            ok = False
            print(f"  MISSING  {mod}  (pip install -r requirements.txt)")
    return ok


def status(data_root: str = None) -> None:
    from src.data.create_splits import merged_dataset_config
    from src.data.download import check_layout
    from src.utils import environment_report, resolve_path

    print("Dependencies:")
    if not check_dependencies():
        return
    print("Environment:")
    for k, v in environment_report().items():
        print(f"  {k}: {v}")
    cfg = merged_dataset_config()
    root = resolve_path(data_root or os.environ.get("GENIMAGE_ROOT") or cfg["dataset"]["root"])
    lay = check_layout(root, cfg["dataset"])
    print(f"Dataset root {root}: {'READY' if lay['ready'] else 'NOT AVAILABLE'} (free disk {lay['disk']['free_gb']} GB)")
    mdir = resolve_path(cfg["dataset"]["manifest_dir"])
    for key, name in cfg["manifests"].items():
        print(f"  manifest {name:20s} {'present' if (mdir / name).exists() else 'missing'}")
    for exp in ("cnn", "resnet18"):
        ck = ROOT / "checkpoints" / exp / "best.pt"
        print(f"  checkpoint {exp:10s} {'present' if ck.exists() else 'missing'}")
    rs = ROOT / "results" / "summaries" / "report_status.json"
    print(f"  report status: {rs.read_text() if rs.exists() else 'not generated'}")


def run_tests() -> int:
    return subprocess.call([sys.executable, "-m", "pytest", "-q", "tests/essential"], cwd=ROOT)


def smoke() -> None:
    """Synthetic fixtures -> manifests -> leakage check -> train CNN + ResNet-18 -> evaluate -> robustness -> report."""
    from src.data.create_splits import build_manifests, check_leakage, merged_dataset_config
    from src.data.synthetic import make_synthetic_genimage
    from src.evaluate import evaluate_checkpoint, measure_latency
    from src.experiment import generate_report_assets
    from src.training.trainer import train_model
    from src.utils import load_config, resolve_path

    cfg_path = "configs/smoke.yaml"
    suite = merged_dataset_config(cfg_path)
    root = resolve_path(suite["dataset"]["root"])
    mdir = resolve_path(suite["dataset"]["manifest_dir"])
    print("== SYNTHETIC smoke test (software verification only) ==")
    make_synthetic_genimage(root, per_folder=40)
    meta = build_manifests(suite, root, mdir)
    leak = check_leakage(mdir, suite["manifests"])
    print("manifest counts:", meta["counts"], "| excluded:", [e["reason"] for e in meta["exclusions"]])
    print("leakage:", "PASSED" if leak["passed"] else leak["problems"])
    if not leak["passed"]:
        raise SystemExit(1)
    runs = {
        "cnn": [],
        "resnet18": ["model.name=resnet18", "model.pretrained=true", "model.freeze_backbone=false",
                     "training.learning_rate=0.0001"],
    }
    for name, extra in runs.items():
        cfg = load_config(ROOT / cfg_path, extra + [f"experiment_name={name}"])
        train_model(cfg, overwrite=True)
        best = f"{SMOKE_CKPTS}/{name}/best.pt"
        evaluate_checkpoint(best, ["test_sd", "test_mj", "test_mj_native"], results_dir=SMOKE_RESULTS,
                            dataset_config=cfg_path, batch_size=16, num_workers=0)
        evaluate_checkpoint(best, ["test_sd", "test_mj"], extra_jpeg=50, results_dir=SMOKE_RESULTS,
                            dataset_config=cfg_path, batch_size=16, num_workers=0)
        measure_latency(best, results_dir=SMOKE_RESULTS, repeats=3)
    st = generate_report_assets(SMOKE_RESULTS, dataset_config=cfg_path, runs_dir=".dev/temporary_results/runs")
    print("smoke report:", st)
    print(f"Smoke outputs in {SMOKE_RESULTS} (synthetic; do NOT report).")


CKPT_MANIFEST = ROOT / "checkpoints_manifest.json"


def checkpoints(action: str) -> int:
    """Register or fetch+verify the primary checkpoints listed in checkpoints_manifest.json.

    URLs are filled in by hand ONLY after a file has actually been uploaded somewhere.
    """
    import json
    import urllib.request

    from src.utils import sha256_file

    entries = json.loads(CKPT_MANIFEST.read_text()) if CKPT_MANIFEST.exists() else {}
    if action == "register":
        for exp in ("cnn", "resnet18"):
            path = ROOT / "checkpoints" / exp / "best.pt"
            if path.exists():
                e = entries.setdefault(exp, {"url": None})
                e.update(path=f"checkpoints/{exp}/best.pt", sha256=sha256_file(path), experiment=exp,
                         represents="E1 primary" if exp == "cnn" else "E2 primary")
        CKPT_MANIFEST.write_text(json.dumps(entries, indent=2) + "\n")
        print(json.dumps(entries, indent=2))
        return 0
    if not entries:
        print("No checkpoints registered in checkpoints_manifest.json (training/upload PENDING)")
        return 1
    bad = 0
    for exp, e in entries.items():
        path = ROOT / e["path"]
        if not path.exists():
            if not e.get("url"):
                print(f"{exp}: missing locally and no download URL recorded (PENDING upload)")
                bad += 1
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"{exp}: downloading {e['url']}")
            urllib.request.urlretrieve(e["url"], path)
        ok = sha256_file(path) == e["sha256"]
        print(f"{exp}: {'checksum OK' if ok else 'CHECKSUM MISMATCH'}  {e['path']}")
        bad += not ok
    return 1 if bad else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["status", "tests", "smoke", "manifests", "train", "evaluate", "report",
                                        "checkpoints"])
    p.add_argument("action", nargs="?", default="fetch", choices=["register", "fetch"],
                   help="for `checkpoints` only")
    p.add_argument("--data-root", default=None)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    dr = ["--data-root", args.data_root] if args.data_root else []
    if args.command == "status":
        status(args.data_root)
    elif args.command == "tests":
        raise SystemExit(run_tests())
    elif args.command == "smoke":
        smoke()
    elif args.command == "manifests":
        raise SystemExit(subprocess.call([sys.executable, "scripts/prepare_data.py", "manifests", *dr,
                                          "--workers", str(args.workers)], cwd=ROOT))
    elif args.command == "train":
        from src.experiment import run_experiments

        run_experiments(["E1_cnn", "E2_resnet18"], "full", args.data_root)
    elif args.command == "evaluate":
        from src.data.create_splits import merged_dataset_config
        from src.evaluate import evaluate_checkpoint, measure_latency
        from src.evaluation.robustness import run_robustness

        ev = merged_dataset_config()["evaluation"]
        for exp in ("cnn", "resnet18"):
            ck = ROOT / "checkpoints" / exp / "best.pt"
            if not ck.exists():
                print(f"checkpoint {ck} missing; see docs/CHECKPOINTS section of README")
                continue
            evaluate_checkpoint(str(ck), ev["primary_tests"] + ev["secondary_tests"], data_root=args.data_root)
            run_robustness([str(ck)], ev["primary_tests"], ev["robustness_jpeg_qualities"], data_root=args.data_root)
            measure_latency(str(ck))
    elif args.command == "checkpoints":
        raise SystemExit(checkpoints(args.action))
    elif args.command == "report":
        from src.experiment import generate_report_assets

        print(generate_report_assets("results", data_root=args.data_root))


if __name__ == "__main__":
    main()

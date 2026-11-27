"""Experiment orchestration (E0-E8) and report-asset generation from exported results.

Staged training (Phase 6) is expressed through ``--stage``:
    pilot         ~1,000 training images (500/class), 1,000 val images, no test evaluation
    intermediate  ~5,000 training images (2,500/class), full val, no test evaluation
    full          the finalized 40,000-image manifest, followed by held-out evaluation
Pilot/intermediate runs are suffixed (e.g. ``cnn_pilot``) so they never overwrite primary runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .data.create_splits import merged_dataset_config
from .data.dataset import read_manifest
from .evaluation.metrics import chance_baselines, degradation
from .evaluation.robustness import robustness_table
from .evaluation.statistics import bootstrap_ci, cross_generator_difference, mcnemar_exact, paired_model_difference
from .utils import PROJECT_ROOT, load_config, read_json, resolve_path, write_json

STAGES = {
    "pilot": {"data.train_limit": 500, "data.val_limit": 500},
    "intermediate": {"data.train_limit": 2500, "data.val_limit": None},
    "full": {},
}
PENDING = "PENDING"


# --------------------------------------------------------------------------- running


def experiment_config(exp_id: str, extra_overrides: Sequence[str] = (), stage: str = "full",
                      suite: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    suite = suite or merged_dataset_config()
    spec = suite["experiments"][exp_id]
    overrides = list(spec.get("overrides", [])) + list(extra_overrides)
    cfg = load_config(PROJECT_ROOT / spec["config"], overrides)
    for key, value in STAGES[stage].items():
        cfg = load_config_override(cfg, key, value)
    if stage != "full":
        cfg["experiment_name"] = f"{cfg['experiment_name']}_{stage}"
    return cfg


def load_config_override(cfg: Dict[str, Any], dotted: str, value: Any) -> Dict[str, Any]:
    node = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
    return cfg


def run_experiments(exp_ids: Sequence[str], stage: str = "full", data_root: Optional[str] = None,
                    overrides: Sequence[str] = (), evaluate: bool = True, robustness: bool = True,
                    resume: bool = True) -> List[Dict[str, Any]]:
    """Train (or resume) each experiment, then evaluate full-stage runs on the held-out sets."""
    from .evaluate import evaluate_checkpoint, measure_latency
    from .evaluation.robustness import run_robustness
    from .training.trainer import train_model

    suite = merged_dataset_config()
    out = []
    for exp_id in exp_ids:
        cfg = experiment_config(exp_id, overrides, stage, suite)
        ckpt_dir = resolve_path(cfg["output"]["checkpoints_dir"]) / cfg["experiment_name"]
        last = ckpt_dir / "last.pt"
        summary_path = resolve_path(cfg["output"]["runs_dir"]) / cfg["experiment_name"] / "summary.json"
        if summary_path.exists():
            print(f"[{exp_id}] already trained ({summary_path}); skipping training")
            summary = read_json(summary_path)
        else:
            summary = train_model(cfg, data_root=data_root, resume=str(last) if (resume and last.exists()) else None)
        out.append(summary)
        if stage != "full" or not evaluate:
            continue
        best = ckpt_dir / "best.pt"
        ev = suite["evaluation"]
        tests = list(ev["primary_tests"]) + list(ev["secondary_tests"])
        evaluate_checkpoint(str(best), tests, data_root=data_root)
        measure_latency(str(best))
        if robustness and exp_id in ("E1_cnn", "E2_resnet18"):
            run_robustness([str(best)], ev["primary_tests"], ev["robustness_jpeg_qualities"], data_root=data_root)
    return out


# --------------------------------------------------------------------------- report assets


def primary_experiments(suite: Dict[str, Any]) -> Dict[str, str]:
    """experiment_name -> display model name for the two mandatory primary experiments."""
    from .models.factory import display_name

    out = {}
    for exp_id in ("E1_cnn", "E2_resnet18"):
        cfg = experiment_config(exp_id, suite=suite)
        out[cfg["experiment_name"]] = display_name(cfg["model"])
    return out


def _load_eval(summaries: Path, experiment: str, test: str, extra_jpeg: Optional[int] = None) -> Optional[dict]:
    tag = f"{experiment}__{test}" + (f"__jpeg{extra_jpeg}" if extra_jpeg else "")
    path = summaries / f"eval_{tag}.json"
    return read_json(path) if path.exists() else None


def _load_predictions(results: Path, experiment: str, test: str) -> Optional[pd.DataFrame]:
    path = results / "predictions" / f"{experiment}__{test}.csv"
    return pd.read_csv(path, dtype={"sample_id": str}) if path.exists() else None


def main_results_table(results: Path, suite: Dict[str, Any], stats: Dict[str, Any]) -> pd.DataFrame:
    """The proposal's main table, built only from exported results; missing cells are PENDING."""
    summaries = results / "summaries"
    manifest_dir = resolve_path(suite["dataset"]["manifest_dir"])
    tests = list(suite["evaluation"]["primary_tests"]) + list(suite["evaluation"]["secondary_tests"])
    rows = []
    for test in tests:
        mpath = manifest_dir / suite["manifests"][test]
        if mpath.exists():
            base = chance_baselines(read_manifest(mpath)["label"].to_numpy())["analytic_chance"]
            rows.append({"model": "Chance baseline", "experiment": "E0", "test": test, "status": "analytic",
                         **{k: base[k] for k in ("accuracy", "precision", "recall", "f1", "auroc", "fpr")}})
        else:
            rows.append({"model": "Chance baseline", "experiment": "E0", "test": test, "status": PENDING})
    for experiment, model in primary_experiments(suite).items():
        for test in tests:
            s = _load_eval(summaries, experiment, test)
            if s is None:
                rows.append({"model": model, "experiment": experiment, "test": test,
                             "status": f"{PENDING}: no evaluation of {experiment} on {test}"})
                continue
            m = s["metrics"]
            row = {"model": model, "experiment": experiment, "test": test, "status": "evaluated",
                   "checkpoint": s["checkpoint_identifier"],
                   **{k: m[k] for k in ("accuracy", "precision", "recall", "f1", "auroc", "fpr", "fnr",
                                        "balanced_accuracy", "n")}}
            ci = stats.get("bootstrap", {}).get(experiment, {}).get(test, {})
            for metric, v in ci.items():
                row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = v["ci_low"], v["ci_high"]
            rows.append(row)
    return pd.DataFrame(rows)


def degradation_table(results: Path, suite: Dict[str, Any], stats: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    summaries = results / "summaries"
    for experiment, model in primary_experiments(suite).items():
        fam, uns = _load_eval(summaries, experiment, "test_sd"), _load_eval(summaries, experiment, "test_mj")
        if fam is None or uns is None:
            rows.append({"model": model, "experiment": experiment, "status": PENDING})
            continue
        d = degradation(fam["metrics"], uns["metrics"])
        ci = stats.get("cross_generator", {}).get(experiment, {})
        row = {"model": model, "experiment": experiment, "status": "evaluated", **d}
        for metric, v in ci.items():
            row[f"{metric}_drop_ci_low"], row[f"{metric}_drop_ci_high"] = v["ci_low"], v["ci_high"]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_statistics(results: Path, suite: Dict[str, Any]) -> Dict[str, Any]:
    """Bootstrap CIs, paired model differences, cross-generator CIs and McNemar tests."""
    ev = suite["evaluation"]
    n, seed, thr = int(ev["bootstrap_resamples"]), int(ev["bootstrap_seed"]), float(ev["threshold"])
    prim = primary_experiments(suite)
    tests = list(ev["primary_tests"]) + list(ev["secondary_tests"])
    out: Dict[str, Any] = {"n_resamples": n, "seed": seed, "threshold": thr, "bootstrap": {}, "cross_generator": {},
                           "paired_models": {}, "mcnemar": {},
                           "notes": "Stratified percentile bootstrap over test images of FIXED checkpoints; not "
                                    "training-seed variability. Multiple comparisons are uncorrected (exploratory)."}
    preds: Dict[str, Dict[str, pd.DataFrame]] = {}
    for exp in prim:
        for test in tests:
            df = _load_predictions(results, exp, test)
            if df is None:
                continue
            preds.setdefault(exp, {})[test] = df
            out["bootstrap"].setdefault(exp, {})[test] = bootstrap_ci(df.true_label, df.probability_ai, n, seed, thr)
        if exp in preds and {"test_sd", "test_mj"} <= set(preds[exp]):
            out["cross_generator"][exp] = cross_generator_difference(preds[exp]["test_sd"], preds[exp]["test_mj"],
                                                                     n, seed, thr)
    exps = list(prim)
    if len(exps) == 2 and all(e in preds for e in exps):
        a, b = exps
        for test in tests:
            if test in preds[a] and test in preds[b]:
                da = preds[a][test].set_index("sample_id")
                db = preds[b][test].set_index("sample_id").loc[da.index]
                key = f"{a}_minus_{b}"
                out["paired_models"].setdefault(key, {})[test] = paired_model_difference(
                    da.true_label, da.probability_ai, db.probability_ai, n, seed, thr)
                out["mcnemar"].setdefault(f"{a}_vs_{b}", {})[test] = mcnemar_exact(da.correct, db.correct)
    return out


def select_failure_cases(df: pd.DataFrame, k: int = 6) -> pd.DataFrame:
    """Deterministic selection: sort by confidence, tie-break by sample_id."""
    d = df.copy()
    d["margin"] = (d.probability_ai - 0.5).abs()
    cats = {
        "correct_real": (d[(d.true_label == 0) & (d.predicted_label == 0)], ["probability_ai", "sample_id"], [True, True]),
        "correct_ai": (d[(d.true_label == 1) & (d.predicted_label == 1)], ["probability_ai", "sample_id"], [False, True]),
        "false_positive": (d[(d.true_label == 0) & (d.predicted_label == 1)], ["probability_ai", "sample_id"], [False, True]),
        "false_negative": (d[(d.true_label == 1) & (d.predicted_label == 0)], ["probability_ai", "sample_id"], [True, True]),
        "near_threshold": (d, ["margin", "sample_id"], [True, True]),
    }
    parts = []
    for cat, (sub, by, asc) in cats.items():
        parts.append(sub.sort_values(by, ascending=asc).head(k).assign(category=cat))
    return pd.concat(parts, ignore_index=True)


def computational_table(suite: Dict[str, Any], results: Path, runs_dir: Path) -> pd.DataFrame:
    rows = []
    for experiment, model in primary_experiments(suite).items():
        s_path = runs_dir / experiment / "summary.json"
        if not s_path.exists():
            rows.append({"model": model, "experiment": experiment, "status": PENDING})
            continue
        s = read_json(s_path)
        env = s.get("environment", {})
        lat_path = results / "summaries" / f"latency_{experiment}.json"
        lat = read_json(lat_path) if lat_path.exists() else None
        rows.append({
            "model": model, "experiment": experiment, "status": "trained", "initialization": s["initialization"],
            "total_parameters": s["parameters"]["total"], "trainable_parameters": s["parameters"]["trainable"],
            "best_epoch": s["best_epoch"], "epochs_run": s["epochs_run"],
            "training_seconds": s["total_training_seconds"], "compute_seconds": s["total_compute_seconds"],
            "hardware": env.get("gpu_name") or env.get("cpu_name") or env.get("machine"), "device": s["device"],
            "precision": s["precision_mode"], "peak_gpu_allocated_gb": s["peak_memory"]["peak_allocated_gb"],
            "latency_ms_per_image_bs32": next((r["ms_per_image"] for r in lat["results"] if r["batch_size"] == 32), None)
            if lat else None,
            "latency_device": lat["device"] if lat else None,
            "train_images": s["train_counts"]["real"] + s["train_counts"]["ai"],
        })
    return pd.DataFrame(rows)


def seeds_table(results: Path, runs_dir: Path) -> pd.DataFrame:
    """Mean/std across completed seed repeats (E7). Only groups with >= 2 completed runs get a std."""
    groups = {"Custom CNN": ["cnn", "cnn_seed123", "cnn_seed2026"],
              "ResNet-18": ["resnet18", "resnet18_seed123", "resnet18_seed2026"]}
    rows = []
    for model, exps in groups.items():
        for test in ("test_sd", "test_mj"):
            vals = [read_json(p)["metrics"] for e in exps
                    if (p := results / "summaries" / f"eval_{e}__{test}.json").exists()]
            row = {"model": model, "test": test, "completed_runs": len(vals)}
            for m in ("accuracy", "f1", "auroc"):
                arr = np.array([v[m] for v in vals], dtype=float)
                row[f"{m}_mean"] = float(arr.mean()) if len(arr) else np.nan
                row[f"{m}_std"] = float(arr.std(ddof=1)) if len(arr) >= 2 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


COLLABORATOR_RESULTS = PROJECT_ROOT / "collaborator" / "resnet18_project" / "results"


def collaborator_table(results_dir: Path = COLLABORATOR_RESULTS) -> pd.DataFrame:
    """The collaborator's own ResNet-18 runs (their split, their recipe), mean/std over their seeds.

    Their protocol differs from E2: AdamW + cosine LR for a fixed 8 epochs, best epoch by validation
    AUROC, bicubic resizing, no random-crop augmentation, train/val = the first rows of the class-sorted
    mirror, test = all 6,000/class of SD 1.4 val and Midjourney val with Midjourney's own reals. The
    numbers are reported as supplementary evidence and are NOT comparable cell-by-cell with E1/E2.
    """
    path = results_dir / "metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    m = pd.read_csv(path)
    rows = []
    for (model, test), g in m.groupby(["model", "test_set"], sort=False):
        row = {"source": "collaborator/resnet18_project", "model": model, "test_set": test,
               "seeds": ",".join(str(s) for s in sorted(g.seed)), "runs": len(g)}
        for k in ("accuracy", "f1", "auroc"):
            row[f"{k}_mean"] = float(g[k].mean())
            row[f"{k}_std"] = float(g[k].std(ddof=1)) if len(g) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def all_evaluations_table(summaries: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(summaries.glob("eval_*.json")):
        s = read_json(p)
        m = s["metrics"]
        rows.append({"experiment": s["experiment"], "model": s["model"], "test": s["test"],
                     "extra_jpeg_quality": s["extra_jpeg_quality"], "best_epoch": s["best_epoch"],
                     "train_jpeg_quality": s["preprocessing"].get("jpeg_quality"),
                     **{k: m[k] for k in ("n", "accuracy", "precision", "recall", "f1", "auroc", "fpr", "fnr", "loss")}})
    return pd.DataFrame(rows)


def _markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_(no data)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def model_tables(tables: Path) -> None:
    """Exact CNN architecture table and parameter counts for both models (no training needed)."""
    from .models.cnn import CustomCNN, architecture_table
    from .models.factory import build_model, initialization_source, parameter_counts

    pd.DataFrame(architecture_table(CustomCNN())).to_csv(tables / "cnn_architecture.csv", index=False)
    rows = []
    for cfg in ({"name": "cnn", "dropout": 0.3}, {"name": "resnet18", "pretrained": True, "freeze_backbone": False}):
        # Counts do not depend on weight values, so ResNet is built without downloading weights here.
        model = build_model(dict(cfg, pretrained=False) if cfg["name"] == "resnet18" else cfg)
        rows.append({"model": cfg["name"], "initialization": initialization_source(cfg), **parameter_counts(model)})
    pd.DataFrame(rows).to_csv(tables / "parameter_counts.csv", index=False)


def generate_report_assets(results_dir: str = "results", data_root: Optional[str] = None,
                           dataset_config: Optional[str] = None, runs_dir: Optional[str] = None,
                           failure_images: bool = True) -> Dict[str, Any]:
    """Build every table and figure that exists data for; report what is still pending."""
    from .evaluation import visualization as viz
    from .utils import resolve_data_root

    suite = merged_dataset_config(dataset_config)
    results = resolve_path(results_dir)
    tables, figures, summaries = results / "tables", results / "figures", results / "summaries"
    tables.mkdir(parents=True, exist_ok=True)
    runs = resolve_path(runs_dir or load_config(PROJECT_ROOT / "configs" / "cnn.yaml")["output"]["runs_dir"])
    prim = primary_experiments(suite)

    model_tables(tables)
    stats = compute_statistics(results, suite)
    write_json(stats, summaries / "statistics.json")
    main = main_results_table(results, suite, stats)
    main.to_csv(tables / "main_results.csv", index=False)
    show = main[[c for c in ("model", "test", "accuracy", "precision", "recall", "f1", "auroc", "fpr", "status")
                 if c in main.columns]]
    (tables / "main_results.md").write_text(_markdown(show))
    deg = degradation_table(results, suite, stats)
    deg.to_csv(tables / "generalization_drop.csv", index=False)
    comp = computational_table(suite, results, runs)
    comp.to_csv(tables / "computational_comparison.csv", index=False)
    rob = robustness_table(summaries, experiments=list(prim))
    rob.to_csv(tables / "jpeg_robustness.csv", index=False)
    allev = all_evaluations_table(summaries)
    allev.to_csv(tables / "all_evaluations.csv", index=False)
    seeds_table(results, runs).to_csv(tables / "seed_repeatability.csv", index=False)
    collab = collaborator_table()
    if not collab.empty:
        collab.to_csv(tables / "collaborator_resnet18_runs.csv", index=False)

    histories = {}
    for exp, model in prim.items():
        h = runs / exp / "history.csv"
        if h.exists():
            histories[model] = pd.read_csv(h)
    viz.plot_training_curves(histories, figures)
    evaluated = main[main.status == "evaluated"]
    viz.plot_main_comparison(main, figures)
    viz.plot_degradation(deg[deg.status == "evaluated"] if "status" in deg else deg, figures)
    viz.plot_robustness(rob, figures)
    if not comp.empty and (comp.status == "trained").any():
        viz.plot_model_comparison(comp[comp.status == "trained"], figures)
    roc, cms = {}, {}
    for exp, model in prim.items():
        for test in ("test_sd", "test_mj"):
            df = _load_predictions(results, exp, test)
            if df is None:
                continue
            roc.setdefault(test, {})[model] = df
            s = _load_eval(summaries, exp, test)
            cms.setdefault(model, {})[test] = s["metrics"]["confusion_matrix"]
            sel = select_failure_cases(df)
            manifest = read_manifest(resolve_path(suite["dataset"]["manifest_dir"]) / suite["manifests"][test])
            sel = sel.merge(manifest[["sample_id", "relative_path"]], on="sample_id", how="left")
            sel.drop(columns=["relative_path"]).to_csv(tables / f"failure_cases_{exp}_{test}.csv", index=False)
            if failure_images:
                root = resolve_data_root({"data": {"root": suite["dataset"]["root"]}}, data_root)
                if root.exists():
                    viz.plot_failure_cases(sel, root, figures / "local", name=f"failure_cases_{exp}_{test}")
    viz.plot_roc_curves(roc, figures)
    viz.plot_confusion_matrices(cms, figures)

    status = {
        "evaluated_rows": int(len(evaluated)),
        "pending_rows": main[main.status.astype(str).str.startswith(PENDING)][["model", "test"]].to_dict("records"),
        "trained_primary_models": [r["model"] for _, r in comp.iterrows() if r.get("status") == "trained"],
        "figures": sorted(p.name for p in figures.glob("*.png")),
    }
    write_json(status, summaries / "report_status.json")
    return status


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Run experiments or build report assets")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="train + evaluate experiments")
    r.add_argument("experiments", nargs="+", help="ids from configs/experiments.yaml, e.g. E1_cnn E2_resnet18")
    r.add_argument("--stage", choices=list(STAGES), default="full")
    r.add_argument("--data-root", default=None)
    r.add_argument("--set", nargs="*", default=[])
    r.add_argument("--no-robustness", action="store_true")
    g = sub.add_parser("report", help="build tables/figures from exported results")
    g.add_argument("--results", default="results")
    g.add_argument("--data-root", default=None)
    g.add_argument("--dataset-config", default=None)
    g.add_argument("--runs-dir", default=None)
    args = p.parse_args(argv)
    if args.cmd == "run":
        run_experiments(args.experiments, args.stage, args.data_root, args.set, robustness=not args.no_robustness)
    else:
        print(generate_report_assets(args.results, args.data_root, args.dataset_config, args.runs_dir))


if __name__ == "__main__":
    main()

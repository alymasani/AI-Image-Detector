"""Report figures generated directly from exported results (never from hand-entered numbers).

Every figure is saved as both PDF (vector, for the report) and 300-dpi PNG.
Consistent encoding: colour = model, hatch/position = test generator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MODEL_COLORS = {"Custom CNN": "#1f77b4", "ResNet-18": "#d95f02", "ResNet-18 (scratch)": "#7570b3",
                "Chance baseline": "#7f7f7f"}
TEST_LABELS = {"test_sd": "SD 1.4 (familiar)", "test_mj": "Midjourney (unseen)",
               "test_mj_native": "Midjourney (native reals)"}
CLASS_COLORS = {"real": "#4daf4a", "ai": "#e41a1c"}

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": 100, "axes.spines.top": False,
                     "axes.spines.right": False})


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _color(model: str) -> str:
    return MODEL_COLORS.get(model, "#333333")


# --------------------------------------------------------------------------- dataset


def plot_dataset_audit(df: pd.DataFrame, out_dir: Path) -> None:
    """Class counts per split, container formats, resolutions and bytes/pixel by class x generator."""
    df = df.copy()
    df["group"] = df["class_name"] + " / " + df["generator"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    counts = df.groupby(["split", "class_name"]).size().unstack(fill_value=0)
    counts.plot.bar(ax=axes[0, 0], color=[CLASS_COLORS.get(c, "#999") for c in counts.columns], rot=0)
    axes[0, 0].set(title="Images per split and class", xlabel="Split", ylabel="Images")
    fmt = df.groupby(["group", "file_format"]).size().unstack(fill_value=0)
    fmt.div(fmt.sum(1), axis=0).plot.barh(stacked=True, ax=axes[0, 1])
    axes[0, 1].set(title="Source file format", xlabel="Fraction of images", ylabel="")
    axes[0, 1].legend(title="Format", loc="lower right")
    for name, g in df.groupby("group"):
        axes[1, 0].scatter(g["width"], g["height"], s=4, alpha=0.4, label=name)
    axes[1, 0].set(title="Source resolution", xlabel="Width (px)", ylabel="Height (px)")
    axes[1, 0].legend(markerscale=4)
    groups = sorted(df["group"].unique())
    axes[1, 1].boxplot([df.loc[df.group == g, "bytes_per_pixel"] for g in groups], showfliers=False)
    axes[1, 1].set_xticks(range(1, len(groups) + 1))
    axes[1, 1].set_xticklabels(groups, rotation=20, ha="right")
    axes[1, 1].set(title="Compression (file bytes per pixel)", ylabel="Bytes / pixel")
    save_figure(fig, out_dir, "dataset_audit")


# --------------------------------------------------------------------------- training


def plot_training_curves(histories: Dict[str, pd.DataFrame], out_dir: Path, name: str = "training_curves") -> None:
    """Loss and accuracy vs epoch for each model; the best (selected) epoch is marked."""
    if not histories:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for model, h in histories.items():
        c = _color(model)
        axes[0].plot(h.epoch, h.train_loss, "--", color=c, label=f"{model} train")
        axes[0].plot(h.epoch, h.val_loss, "-", color=c, label=f"{model} val")
        axes[1].plot(h.epoch, h.train_accuracy, "--", color=c, label=f"{model} train")
        axes[1].plot(h.epoch, h.val_accuracy, "-", color=c, label=f"{model} val")
        best = h[h.is_best.astype(bool)]
        if not best.empty:
            b = best.iloc[-1]
            axes[0].plot(b.epoch, b.val_loss, "*", ms=12, color=c)
            axes[1].plot(b.epoch, b.val_accuracy, "*", ms=12, color=c)
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss (★ = selected checkpoint)")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy (SD 1.4 validation)")
    axes[1].legend(loc="lower right", ncol=1)
    save_figure(fig, out_dir, name)


# --------------------------------------------------------------------------- results


def plot_main_comparison(table: pd.DataFrame, out_dir: Path, tests: Sequence[str] = ("test_sd", "test_mj")) -> None:
    """Accuracy / F1 / AUROC per model on familiar vs unseen generator, with chance line and 95% CIs."""
    if "accuracy" not in table.columns:
        return
    t = table[table.test.isin(tests) & (table.model != "Chance baseline") & table.accuracy.notna()]
    if t.empty:
        return
    models = list(dict.fromkeys(t.model))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    width = 0.8 / len(tests)
    for ax, metric in zip(axes, ("accuracy", "f1", "auroc")):
        for j, test in enumerate(tests):
            for i, model in enumerate(models):
                r = t[(t.model == model) & (t.test == test)]
                if r.empty or pd.isna(r[metric].iloc[0]):
                    continue
                v = float(r[metric].iloc[0])
                lo, hi = r.get(f"{metric}_ci_low"), r.get(f"{metric}_ci_high")
                err = None
                if lo is not None and not pd.isna(lo.iloc[0]):
                    err = [[v - float(lo.iloc[0])], [float(hi.iloc[0]) - v]]
                ax.bar(i + (j - (len(tests) - 1) / 2) * width, v, width, color=_color(model),
                       hatch="" if j == 0 else "//", edgecolor="white", yerr=err, capsize=3,
                       label=TEST_LABELS.get(test, test) if i == 0 else None)
        ax.axhline(0.5, color="#555", ls=":", lw=1, label="Chance (0.5)" if metric == "accuracy" else None)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.0)
        ax.set_title(metric.upper() if metric == "auroc" else metric.capitalize())
    axes[0].set_ylabel("Score on held-out test set")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.06), frameon=False)
    save_figure(fig, out_dir, "main_comparison")


def plot_degradation(deg: pd.DataFrame, out_dir: Path) -> None:
    """Familiar-minus-unseen drop per model (accuracy in percentage points; F1/AUROC x100)."""
    if deg.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 3.2))
    metrics = [("accuracy_drop_pp", "Accuracy (pp)"), ("f1_drop_x100", "F1 (x100)"), ("auroc_drop_x100", "AUROC (x100)")]
    d = deg.copy()
    d["f1_drop_x100"] = d["f1_drop"] * 100
    d["auroc_drop_x100"] = d["auroc_drop"] * 100
    width = 0.8 / max(len(d), 1)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.bar(np.arange(3) + (i - (len(d) - 1) / 2) * width, [r[m] for m, _ in metrics], width,
               color=_color(r["model"]), label=r["model"])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylabel("Familiar − unseen (higher = larger drop)")
    ax.set_title("Generalisation drop: SD 1.4 → Midjourney")
    ax.legend()
    save_figure(fig, out_dir, "generalization_drop")


def plot_roc_curves(predictions: Dict[str, Dict[str, pd.DataFrame]], out_dir: Path) -> None:
    """ROC curves per test set; ``predictions[test][model] = prediction DataFrame``."""
    from sklearn.metrics import roc_auc_score, roc_curve

    tests = [t for t in ("test_sd", "test_mj") if t in predictions]
    if not tests:
        return
    fig, axes = plt.subplots(1, len(tests), figsize=(4.2 * len(tests), 3.8), squeeze=False)
    for ax, test in zip(axes[0], tests):
        for model, df in predictions[test].items():
            fpr, tpr, _ = roc_curve(df.true_label, df.probability_ai)
            auc = roc_auc_score(df.true_label, df.probability_ai)
            ax.plot(fpr, tpr, color=_color(model), label=f"{model} (AUROC {auc:.3f})")
        ax.plot([0, 1], [0, 1], ":", color="#555", label="Chance (0.5)")
        ax.set(xlabel="False-positive rate (real flagged as AI)", ylabel="True-positive rate",
               title=TEST_LABELS.get(test, test), xlim=(0, 1), ylim=(0, 1.01))
        ax.legend(loc="lower right", fontsize=8)
    save_figure(fig, out_dir, "roc_curves")


def plot_confusion_matrices(cms: Dict[str, Dict[str, list]], out_dir: Path) -> None:
    """``cms[model][test] = [[tn, fp], [fn, tp]]`` -> grid of row-normalised matrices with counts."""
    models = list(cms)
    tests = [t for t in ("test_sd", "test_mj") if any(t in cms[m] for m in models)]
    if not models or not tests:
        return
    fig, axes = plt.subplots(len(models), len(tests), figsize=(3.3 * len(tests), 3.0 * len(models)), squeeze=False)
    for i, model in enumerate(models):
        for j, test in enumerate(tests):
            ax = axes[i, j]
            if test not in cms[model]:
                ax.axis("off")
                continue
            cm = np.array(cms[model][test], dtype=float)
            norm = cm / cm.sum(1, keepdims=True).clip(min=1)
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for (r, c), v in np.ndenumerate(cm):
                ax.text(c, r, f"{int(v)}\n({norm[r, c]:.1%})", ha="center", va="center",
                        color="white" if norm[r, c] > 0.6 else "black", fontsize=9)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Real", "AI"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Real", "AI"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{model}\n{TEST_LABELS.get(test, test)}", fontsize=9)
    save_figure(fig, out_dir, "confusion_matrices")


def plot_robustness(table: pd.DataFrame, out_dir: Path) -> None:
    """Accuracy, AUROC and FPR vs additional JPEG quality (100 = no extra compression)."""
    if table.empty or table.extra_jpeg_quality.nunique() < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    styles = {"test_sd": "-o", "test_mj": "--s", "test_mj_native": ":^"}
    for (model, test), g in table.groupby(["model", "test"]):
        g = g.sort_values("extra_jpeg_quality", ascending=False)
        for ax, metric in zip(axes, ("accuracy", "auroc", "fpr")):
            ax.plot(g.extra_jpeg_quality.astype(str), g[metric], styles.get(test, "-o"), color=_color(model),
                    label=f"{model}, {TEST_LABELS.get(test, test)}")
    for ax, (metric, label) in zip(axes, (("accuracy", "Accuracy"), ("auroc", "AUROC"), ("fpr", "False-positive rate"))):
        ax.set(xlabel="Additional JPEG quality (100 = none)", ylabel=label, ylim=(0, 1))
        if metric != "fpr":
            ax.axhline(0.5, color="#555", ls=":", lw=1)
    axes[0].legend(fontsize=7, loc="lower left")
    save_figure(fig, out_dir, "jpeg_robustness")


def plot_failure_cases(selection: pd.DataFrame, data_root: Path, out_dir: Path, image_size: int = 224,
                       name: str = "failure_cases") -> Optional[Path]:
    """Grid of selected predictions with true label, predicted label and P(AI).

    Uses the dataset images, so the output stays in a git-ignored folder (licence of ImageNet
    photos and GenImage outputs does not clearly allow redistribution in the repository).
    """
    from ..data.preprocessing import load_rgb, standardize_spatial

    if selection.empty:
        return None
    cats = list(dict.fromkeys(selection.category))
    per = selection.groupby("category").size().max()
    fig, axes = plt.subplots(len(cats), per, figsize=(1.9 * per, 2.2 * len(cats)), squeeze=False)
    for i, cat in enumerate(cats):
        rows = selection[selection.category == cat].reset_index(drop=True)
        for j in range(per):
            ax = axes[i, j]
            ax.axis("off")
            if j >= len(rows):
                continue
            r = rows.iloc[j]
            try:
                ax.imshow(standardize_spatial(load_rgb(data_root / r.relative_path), image_size))
            except Exception as exc:
                ax.text(0.5, 0.5, f"unavailable\n{type(exc).__name__}", ha="center", va="center", fontsize=7)
            t = "AI" if r.true_label == 1 else "Real"
            pr = "AI" if r.predicted_label == 1 else "Real"
            ax.set_title(f"T:{t} P:{pr}\nP(AI)={r.probability_ai:.2f}", fontsize=7,
                         color="green" if t == pr else "red")
        axes[i, 0].text(-0.08, 0.5, cat.replace("_", "\n"), transform=axes[i, 0].transAxes, ha="right",
                        va="center", fontsize=8)
    save_figure(fig, out_dir, name)
    return out_dir / f"{name}.png"


def plot_model_comparison(comp: pd.DataFrame, out_dir: Path) -> None:
    """Parameters (log scale) and training time per model."""
    if comp.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    colors = [_color(m) for m in comp.model]
    axes[0].bar(comp.model, comp.total_parameters, color=colors)
    axes[0].set_yscale("log")
    axes[0].set(ylabel="Parameters (log scale)", title="Model size")
    hours = comp.training_seconds.astype(float) / 3600
    axes[1].bar(comp.model, hours, color=colors)
    axes[1].set(ylabel="Hours", title="Training wall-clock time")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    save_figure(fig, out_dir, "model_comparison")

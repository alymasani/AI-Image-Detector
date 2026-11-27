"""Figures for the notebook and report.

Every function returns a matplotlib Figure; `save_figure` writes it to results/figures/.
Colors follow one rule: blue is always SD 1.4 (the generator the models trained on) and orange
is always Midjourney (the unseen one), so the two can be told apart in every figure.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import roc_auc_score, roc_curve

from detector.metrics import METRIC_NAMES, summarize

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "results" / "figures"

# Chart chrome and ink (light theme; the figures go into a printed report).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Two categorical slots, validated together for color-blind safety and contrast.
BLUE = "#2a78d6"
ORANGE = "#eb6834"

TEST_SETS = ("test_sd14", "test_midjourney")
TEST_SET_LABELS = {"test_sd14": "SD 1.4 (seen)", "test_midjourney": "Midjourney (unseen)"}
TEST_SET_COLORS = {"test_sd14": BLUE, "test_midjourney": ORANGE}
MODEL_LABELS = {"cnn": "CNN (from scratch)", "resnet18": "ResNet-18 (pretrained)"}
METRIC_LABELS = {"accuracy": "Accuracy", "f1": "F1", "auroc": "AUROC"}

LINE_WIDTH = 1.75
STYLE = {
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 200,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.labelcolor": INK_SECONDARY,
    "axes.labelsize": 9.5,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "xtick.color": BASELINE,
    "ytick.color": BASELINE,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.color": INK,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    "lines.linewidth": LINE_WIDTH,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.labelcolor": INK_SECONDARY,
}


def save_figure(fig: Figure, name: str, directory: str | Path = FIGURES_DIR) -> Path:
    """Save as PNG under results/figures/ (or `directory`) and return the path."""
    path = Path(directory) / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path


def plot_roc_curves(
    curves: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray]]],
) -> Figure:
    """One ROC panel per model, with the SD 1.4 and Midjourney curves on top of each other.

    `curves` maps model -> test set -> (labels, prob_fake), e.g. from `read_predictions`.
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            1, len(curves), figsize=(4.6 * len(curves), 4.6), squeeze=False, layout="constrained"
        )
        for ax, (model, by_test_set) in zip(axes[0], curves.items(), strict=True):
            ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, label="Chance")
            for test_set in TEST_SETS:
                if test_set not in by_test_set:
                    continue
                labels, prob_fake = by_test_set[test_set]
                fpr, tpr, _ = roc_curve(labels, prob_fake)
                auroc = roc_auc_score(labels, prob_fake)
                ax.plot(
                    fpr,
                    tpr,
                    color=TEST_SET_COLORS[test_set],
                    label=f"{TEST_SET_LABELS[test_set]}, AUROC {auroc:.3f}",
                )
            ax.set(xlim=(0, 1), ylim=(0, 1.01), aspect="equal")
            ax.set_title(MODEL_LABELS.get(model, model), loc="left")
            ax.set_xlabel("False positive rate (real images flagged as fake)")
            ax.set_ylabel("True positive rate (fakes caught)")
            ax.legend(loc="lower right")
    return fig


def plot_generalization_gap(metrics: pd.DataFrame) -> Figure:
    """Accuracy, F1 and AUROC on SD 1.4 vs Midjourney for each model, averaged over seeds.

    `metrics` has one row per run and test set (the results/metrics.csv format). Bars show the
    mean, whiskers one standard deviation across seeds, and the number above each pair is the
    drop from SD 1.4 to Midjourney.
    """
    summary = summarize(metrics).set_index(["model", "test_set"])
    models = list(dict.fromkeys(metrics["model"]))
    missing = [(m, t) for m in models for t in TEST_SETS if (m, t) not in summary.index]
    if missing:
        raise ValueError(f"metrics has no rows for {missing}")

    bar_width = 0.28
    gap = 0.02  # a sliver of surface between the two bars of a pair
    x = np.arange(len(models))
    offsets = {"test_sd14": -(bar_width + gap) / 2, "test_midjourney": (bar_width + gap) / 2}

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            1, len(METRIC_NAMES), figsize=(10.5, 3.8), sharey=True, layout="constrained"
        )
        for ax, metric in zip(axes, METRIC_NAMES, strict=True):
            ax.grid(axis="x", visible=False)
            ax.axhline(0.5, color=MUTED, linewidth=0.8, zorder=1)
            for test_set in TEST_SETS:
                rows = summary.loc[[(m, test_set) for m in models]]
                ax.bar(
                    x + offsets[test_set],
                    rows[f"{metric}_mean"],
                    width=bar_width,
                    color=TEST_SET_COLORS[test_set],
                    yerr=rows[f"{metric}_std"],
                    error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 3},
                    label=TEST_SET_LABELS[test_set],
                    zorder=2,
                )
            for i, model in enumerate(models):
                seen = summary.loc[(model, "test_sd14")]
                unseen = summary.loc[(model, "test_midjourney")]
                drop = seen[f"{metric}_mean"] - unseen[f"{metric}_mean"]
                top = max(
                    seen[f"{metric}_mean"] + seen[f"{metric}_std"],
                    unseen[f"{metric}_mean"] + unseen[f"{metric}_std"],
                )
                ax.annotate(
                    f"\N{MINUS SIGN}{drop:.2f}" if drop >= 0 else f"+{-drop:.2f}",
                    (x[i], top),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=INK_SECONDARY,
                )
            ax.set_title(METRIC_LABELS[metric], loc="left")
            ax.set_xticks(x, [MODEL_LABELS.get(m, m) for m in models])
            ax.set_ylim(0, 1.12)
            ax.set_yticks(np.linspace(0, 1, 6))
        if len(models) > 1:
            # Put the label in the gap between the first two models' bars, where no bar covers it.
            axes[0].text(0.5, 0.51, "chance", color=MUTED, fontsize=8, va="bottom", ha="center")
        else:
            # With only one model there is no gap between bars (and the label used to end up
            # floating between two panels), so put it just past the right end of the line.
            axes[-1].text(
                1.01,
                0.5,
                "chance",
                transform=axes[-1].get_yaxis_transform(),
                color=MUTED,
                fontsize=8,
                va="center",
                ha="left",
            )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper right", ncols=2)
        fig.supxlabel(
            "Bars: mean over seeds, whiskers: 1 std. "
            "Number above each pair: drop from SD 1.4 to Midjourney.",
            x=0,
            ha="left",
            fontsize=8.5,
            color=INK_SECONDARY,
        )
    return fig


def plot_training_curves(histories: Mapping[str, pd.DataFrame]) -> Figure:
    """Loss and validation AUROC per epoch, one row per run.

    Each history has columns epoch, train_loss, val_loss, val_auroc (a run's history.csv).
    The dot marks the epoch with the best validation AUROC, which is the checkpoint kept.
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            len(histories),
            2,
            figsize=(9, 2.9 * len(histories)),
            squeeze=False,
            layout="constrained",
        )
        for (loss_ax, auroc_ax), (run, history) in zip(axes, histories.items(), strict=True):
            loss_ax.plot(history["epoch"], history["train_loss"], color=BLUE, label="Train")
            loss_ax.plot(history["epoch"], history["val_loss"], color=ORANGE, label="Validation")
            loss_ax.set_title(f"{run}: loss", loc="left")
            loss_ax.set_xlabel("Epoch")
            loss_ax.set_ylabel("Cross-entropy loss")
            loss_ax.set_ylim(bottom=0)
            loss_ax.legend(loc="upper right")

            best = history.loc[history["val_auroc"].idxmax()]
            auroc_ax.plot(history["epoch"], history["val_auroc"], color=BLUE)
            auroc_ax.plot(
                best["epoch"],
                best["val_auroc"],
                marker="o",
                markersize=8,
                color=BLUE,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
                linestyle="none",
            )
            # Above the dot, where the curve can't be (it is the maximum). Near the right edge
            # the label hangs to the left so it stays inside the panel.
            near_end = best["epoch"] > history["epoch"].min() + 0.6 * np.ptp(history["epoch"])
            auroc_ax.annotate(
                f"best: epoch {int(best['epoch'])}, {best['val_auroc']:.3f}",
                (best["epoch"], best["val_auroc"]),
                xytext=(0, 9),
                textcoords="offset points",
                ha="right" if near_end else "center",
                va="bottom",
                fontsize=9,
                color=INK_SECONDARY,
            )
            auroc_ax.margins(y=0.2)
            auroc_ax.set_title(f"{run}: validation AUROC", loc="left")
            auroc_ax.set_xlabel("Epoch")
            auroc_ax.set_ylabel("AUROC")
            for ax in (loss_ax, auroc_ax):
                ax.xaxis.get_major_locator().set_params(integer=True)
    return fig

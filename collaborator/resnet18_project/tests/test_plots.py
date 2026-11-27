import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Rectangle

from detector.plots import (
    BLUE,
    ORANGE,
    plot_generalization_gap,
    plot_roc_curves,
    plot_training_curves,
    save_figure,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def fake_predictions(separation: float, n: int = 400, seed: int = 0):
    """Labels and scores where fakes score higher by roughly `separation`."""
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1], n // 2)
    scores = np.clip(0.5 + (labels - 0.5) * separation + rng.normal(0, 0.2, n), 0, 1)
    return labels, scores


@pytest.fixture
def metrics() -> pd.DataFrame:
    rows = []
    for model, seen, unseen in [("cnn", 0.9, 0.6), ("resnet18", 0.97, 0.75)]:
        for seed in range(3):
            for test_set, base in [("test_sd14", seen), ("test_midjourney", unseen)]:
                value = base + 0.01 * (seed - 1)
                rows.append(
                    {
                        "run": f"{model}_seed{seed}",
                        "model": model,
                        "seed": seed,
                        "test_set": test_set,
                        "accuracy": value,
                        "f1": value - 0.02,
                        "auroc": min(value + 0.02, 1),
                    }
                )
    return pd.DataFrame(rows)


def history(epochs: int = 8) -> pd.DataFrame:
    epoch = np.arange(1, epochs + 1)
    return pd.DataFrame(
        {
            "epoch": epoch,
            "train_loss": 0.7 / epoch,
            "val_loss": 0.6 / epoch + 0.02 * epoch,
            "val_auroc": [0.80, 0.88, 0.92, 0.95, 0.94, 0.93, 0.93, 0.92][:epochs],
        }
    )


def test_roc_curves_one_panel_per_model():
    curves = {
        "cnn": {"test_sd14": fake_predictions(0.6), "test_midjourney": fake_predictions(0.2)},
        "resnet18": {"test_sd14": fake_predictions(0.8), "test_midjourney": fake_predictions(0.4)},
    }
    fig = plot_roc_curves(curves)
    assert len(fig.axes) == 2
    for ax in fig.axes:
        legend = [t.get_text() for t in ax.get_legend().get_texts()]
        assert legend[0] == "Chance"
        assert legend[1].startswith("SD 1.4 (seen), AUROC")
        assert legend[2].startswith("Midjourney (unseen), AUROC")
        assert [line.get_color() for line in ax.get_lines()[1:]] == [BLUE, ORANGE]


def test_generalization_gap_bars_and_drop_labels(metrics):
    fig = plot_generalization_gap(metrics)
    assert len(fig.axes) == 3
    accuracy_ax = fig.axes[0]
    bars = [p for p in accuracy_ax.patches if isinstance(p, Rectangle)]
    heights = sorted(round(b.get_height(), 3) for b in bars)
    assert heights == [0.6, 0.75, 0.9, 0.97]  # means over the 3 seeds
    drops = [t.get_text() for t in accuracy_ax.texts if t.get_text().startswith("\N{MINUS SIGN}")]
    assert drops == ["\N{MINUS SIGN}0.30", "\N{MINUS SIGN}0.22"]


@pytest.mark.parametrize("models", [["cnn", "resnet18"], ["resnet18"]], ids=["two", "one"])
def test_generalization_gap_chance_label_sits_clear_of_the_bars(metrics, models):
    fig = plot_generalization_gap(metrics[metrics["model"].isin(models)])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [t for ax in fig.axes for t in ax.texts if t.get_text() == "chance"]
    assert len(labels) == 1
    label = labels[0].get_window_extent(renderer)
    for ax in fig.axes:
        for bar in ax.patches:
            assert not label.overlaps(bar.get_window_extent(renderer))
    if len(models) == 1:
        # Nothing to sit between, so it goes past the end of the last panel's line.
        assert label.x0 >= fig.axes[-1].get_window_extent(renderer).x1


def test_generalization_gap_needs_both_test_sets(metrics):
    with pytest.raises(ValueError, match="test_midjourney"):
        plot_generalization_gap(metrics[metrics["test_set"] == "test_sd14"])


def test_training_curves_mark_the_best_epoch():
    fig = plot_training_curves({"cnn seed 0": history(), "resnet18 seed 0": history(5)})
    assert len(fig.axes) == 4
    auroc_ax = fig.axes[1]
    assert "best: epoch 4, 0.950" in [t.get_text() for t in auroc_ax.texts]


def test_save_figure_writes_png(tmp_path, metrics):
    path = save_figure(plot_generalization_gap(metrics), "gap", tmp_path)
    assert path == tmp_path / "gap.png"
    assert path.read_bytes().startswith(b"\x89PNG")

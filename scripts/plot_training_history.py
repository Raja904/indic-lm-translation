import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline


def _setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#0f172a",
            "axes.facecolor": "#111827",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#e5e7eb",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "text.color": "#f8fafc",
            "axes.titleweight": "bold",
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "legend.frameon": True,
            "legend.facecolor": "#0b1120",
            "legend.edgecolor": "#334155",
            "grid.color": "#334155",
            "grid.alpha": 0.35,
        }
    )


def _save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {path}")


def _annotate_extreme(ax, x, y, label, color, mode="min"):
    if y.empty:
        return
    idx = y.idxmin() if mode == "min" else y.idxmax()
    xv = x.loc[idx]
    yv = y.loc[idx]
    ax.scatter([xv], [yv], s=70, color=color, edgecolor="white", linewidth=1.2, zorder=5)
    ax.annotate(
        f"{label}: {yv:.2f}",
        xy=(xv, yv),
        xytext=(10, 12 if mode == "min" else -24),
        textcoords="offset points",
        fontsize=10,
        weight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
    )


def _smooth_xy(x, y, points_per_segment=25):
    x_vals = np.asarray(x, dtype=float)
    y_vals = np.asarray(y, dtype=float)

    if len(x_vals) < 3:
        return x_vals, y_vals

    dense_x = np.linspace(x_vals.min(), x_vals.max(), num=(len(x_vals) - 1) * points_per_segment + 1)
    dense_y = np.interp(dense_x, x_vals, y_vals)
    return dense_x, dense_y


def _cubic_spline_xy(x, y, points_per_segment=40):
    x_vals = np.asarray(x, dtype=float)
    y_vals = np.asarray(y, dtype=float)

    if len(x_vals) < 4 or len(np.unique(x_vals)) < 4:
        return _smooth_xy(x_vals, y_vals, points_per_segment)

    dense_x = np.linspace(x_vals.min(), x_vals.max(), num=(len(x_vals) - 1) * points_per_segment + 1)
    spline = make_interp_spline(x_vals, y_vals, k=3)
    dense_y = spline(dense_x)
    return dense_x, dense_y


def _plot_curve_pair(ax, epochs, series_a, series_b, label_a, label_b, color_a, color_b, smooth_fn):
    ax.plot(*smooth_fn(epochs, series_a), color=color_a, lw=3, label=label_a)
    ax.plot(*smooth_fn(epochs, series_b), color=color_b, lw=3, label=label_b)
    ax.scatter(epochs, series_a, color=color_a, s=30, zorder=4)
    ax.scatter(epochs, series_b, color=color_b, s=30, zorder=4)


def plot_training_history(csv_path, output_dir):
    _setup_style()

    df = pd.read_csv(csv_path)
    expected = {"epoch", "train_loss", "val_loss", "bleu", "chrf"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    df = df.sort_values("epoch").reset_index(drop=True)

    epochs = df["epoch"]
    train_loss = df["train_loss"]
    val_loss = df["val_loss"]
    bleu = df["bleu"]
    chrf = df["chrf"]

    colors = {
        "train_loss": "#38bdf8",
        "val_loss": "#f97316",
        "bleu": "#22c55e",
        "chrf": "#a78bfa",
        "accent": "#facc15",
    }

    # Combined dashboard
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    fig.suptitle("Training History Overview", fontsize=18, weight="bold", color="white")

    ax = axes[0]
    _plot_curve_pair(
        ax,
        epochs,
        train_loss,
        val_loss,
        "Train Loss",
        "Val Loss",
        colors["train_loss"],
        colors["val_loss"],
        _smooth_xy,
    )
    ax.fill_between(epochs, train_loss, alpha=0.10, color=colors["train_loss"])
    ax.fill_between(epochs, val_loss, alpha=0.08, color=colors["val_loss"])
    ax.set_title("Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", linewidth=0.7)
    _annotate_extreme(ax, epochs, train_loss, "Best train", colors["train_loss"], mode="min")
    _annotate_extreme(ax, epochs, val_loss, "Best val", colors["val_loss"], mode="min")

    ax = axes[1]
    _plot_curve_pair(
        ax,
        epochs,
        bleu,
        chrf,
        "BLEU",
        "CHRF",
        colors["bleu"],
        colors["chrf"],
        _smooth_xy,
    )
    ax.fill_between(epochs, bleu, alpha=0.10, color=colors["bleu"])
    ax.fill_between(epochs, chrf, alpha=0.10, color=colors["chrf"])
    ax.set_title("Score Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", linewidth=0.7)
    _annotate_extreme(ax, epochs, bleu, "Best BLEU", colors["bleu"], mode="max")
    _annotate_extreme(ax, epochs, chrf, "Best CHRF", colors["chrf"], mode="max")

    _save_fig(fig, os.path.join(output_dir, "training_history_dashboard.png"))

    # Spline dashboard
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    fig.suptitle("Training History Overview - Cubic Spline", fontsize=18, weight="bold", color="white")

    ax = axes[0]
    _plot_curve_pair(
        ax,
        epochs,
        train_loss,
        val_loss,
        "Train Loss",
        "Val Loss",
        colors["train_loss"],
        colors["val_loss"],
        _cubic_spline_xy,
    )
    ax.fill_between(epochs, train_loss, alpha=0.10, color=colors["train_loss"])
    ax.fill_between(epochs, val_loss, alpha=0.08, color=colors["val_loss"])
    ax.set_title("Loss Curves - Spline")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", linewidth=0.7)
    _annotate_extreme(ax, epochs, train_loss, "Best train", colors["train_loss"], mode="min")
    _annotate_extreme(ax, epochs, val_loss, "Best val", colors["val_loss"], mode="min")

    ax = axes[1]
    _plot_curve_pair(
        ax,
        epochs,
        bleu,
        chrf,
        "BLEU",
        "CHRF",
        colors["bleu"],
        colors["chrf"],
        _cubic_spline_xy,
    )
    ax.fill_between(epochs, bleu, alpha=0.10, color=colors["bleu"])
    ax.fill_between(epochs, chrf, alpha=0.10, color=colors["chrf"])
    ax.set_title("Score Curves - Spline")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", linewidth=0.7)
    _annotate_extreme(ax, epochs, bleu, "Best BLEU", colors["bleu"], mode="max")
    _annotate_extreme(ax, epochs, chrf, "Best CHRF", colors["chrf"], mode="max")

    _save_fig(fig, os.path.join(output_dir, "training_history_dashboard_spline.png"))

    # Loss-only chart
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(*_smooth_xy(epochs, train_loss), color=colors["train_loss"], lw=3, label="Train Loss")
    ax.plot(*_smooth_xy(epochs, val_loss), color=colors["val_loss"], lw=3, label="Val Loss")
    ax.scatter(epochs, train_loss, color=colors["train_loss"], s=30, zorder=4)
    ax.scatter(epochs, val_loss, color=colors["val_loss"], s=30, zorder=4)
    ax.set_title("Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", linewidth=0.7)
    _annotate_extreme(ax, epochs, train_loss, "Best train", colors["train_loss"], mode="min")
    _annotate_extreme(ax, epochs, val_loss, "Best val", colors["val_loss"], mode="min")
    _save_fig(fig, os.path.join(output_dir, "loss_curves.png"))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(*_cubic_spline_xy(epochs, train_loss), color=colors["train_loss"], lw=3, label="Train Loss")
    ax.plot(*_cubic_spline_xy(epochs, val_loss), color=colors["val_loss"], lw=3, label="Val Loss")
    ax.scatter(epochs, train_loss, color=colors["train_loss"], s=30, zorder=4)
    ax.scatter(epochs, val_loss, color=colors["val_loss"], s=30, zorder=4)
    ax.set_title("Loss Curve - Cubic Spline")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", linewidth=0.7)
    _annotate_extreme(ax, epochs, train_loss, "Best train", colors["train_loss"], mode="min")
    _annotate_extreme(ax, epochs, val_loss, "Best val", colors["val_loss"], mode="min")
    _save_fig(fig, os.path.join(output_dir, "loss_curves_spline.png"))

    # Score-only chart
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(*_smooth_xy(epochs, bleu), color=colors["bleu"], lw=3, label="BLEU")
    ax.plot(*_smooth_xy(epochs, chrf), color=colors["chrf"], lw=3, label="CHRF")
    ax.scatter(epochs, bleu, color=colors["bleu"], s=30, zorder=4)
    ax.scatter(epochs, chrf, color=colors["chrf"], s=30, zorder=4)
    ax.set_title("Score Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", linewidth=0.7)
    _annotate_extreme(ax, epochs, bleu, "Best BLEU", colors["bleu"], mode="max")
    _annotate_extreme(ax, epochs, chrf, "Best CHRF", colors["chrf"], mode="max")
    _save_fig(fig, os.path.join(output_dir, "score_curves.png"))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(*_cubic_spline_xy(epochs, bleu), color=colors["bleu"], lw=3, label="BLEU")
    ax.plot(*_cubic_spline_xy(epochs, chrf), color=colors["chrf"], lw=3, label="CHRF")
    ax.scatter(epochs, bleu, color=colors["bleu"], s=30, zorder=4)
    ax.scatter(epochs, chrf, color=colors["chrf"], s=30, zorder=4)
    ax.set_title("Score Curves - Cubic Spline")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", linewidth=0.7)
    _annotate_extreme(ax, epochs, bleu, "Best BLEU", colors["bleu"], mode="max")
    _annotate_extreme(ax, epochs, chrf, "Best CHRF", colors["chrf"], mode="max")
    _save_fig(fig, os.path.join(output_dir, "score_curves_spline.png"))


def main():
    parser = argparse.ArgumentParser(description="Plot training history from a CSV file.")
    parser.add_argument(
        "--csv",
        default="lstm_random_exp_a_history.csv",
        help="Path to the history CSV file.",
    )
    parser.add_argument(
        "--outdir",
        default="assets/plots",
        help="Directory where plots will be saved.",
    )
    args = parser.parse_args()
    plot_training_history(args.csv, args.outdir)


if __name__ == "__main__":
    main()

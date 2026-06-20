from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ecg_compression.data import nonzero_lengths


def _save_current(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_metric_curves(metrics: pd.DataFrame, figures_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    for metric, ylabel, filename in [
        ("prd", "PRD [%]", "cr_vs_prd.png"),
        ("quality_score", "QS = CR / PRD", "cr_vs_qs.png"),
        ("rmse", "RMSE", "cr_vs_rmse.png"),
        ("snr_db", "SNR [dB]", "cr_vs_snr.png"),
    ]:
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=metrics,
            x="compression_ratio",
            y=metric,
            hue="method",
            marker="o",
        )
        plt.xscale("log")
        plt.xlabel("Compression ratio (187 / K)")
        plt.ylabel(ylabel)
        _save_current(figures_dir / filename)


def plot_padding_histogram(x: np.ndarray, figures_dir: Path) -> None:
    lengths = nonzero_lengths(x)
    plt.figure(figsize=(8, 5))
    sns.histplot(lengths, bins=40)
    plt.xlabel("Estimated non-zero heartbeat length")
    plt.ylabel("Count")
    _save_current(figures_dir / "padding_lengths.png")


def plot_reconstructions(
    x: np.ndarray,
    x_hat: np.ndarray,
    y: np.ndarray,
    bottleneck_dim: int,
    figures_dir: Path,
) -> None:
    classes = []
    for label in sorted(np.unique(y)):
        index = np.flatnonzero(y == label)
        if len(index):
            classes.append(int(index[0]))
    selected = classes[:6] if classes else list(range(min(6, len(x))))

    fig, axes = plt.subplots(len(selected), 1, figsize=(9, 2.1 * len(selected)), sharex=True)
    if len(selected) == 1:
        axes = [axes]
    for axis, idx in zip(axes, selected):
        axis.plot(x[idx], label="Original", linewidth=1.7)
        axis.plot(x_hat[idx], label="Reconstruction", linewidth=1.2)
        axis.set_ylabel(f"class {int(y[idx])}")
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Sample index")
    fig.suptitle(f"MLP reconstruction examples, K={bottleneck_dim}")
    _save_current(figures_dir / f"reconstructions_k{bottleneck_dim}.png")


def plot_prd_by_class(
    sample_prd: np.ndarray,
    y: np.ndarray,
    class_names: dict[int, str],
    bottleneck_dim: int,
    figures_dir: Path,
) -> None:
    labels = [class_names.get(int(label), str(int(label))) for label in y]
    frame = pd.DataFrame({"PRD": sample_prd, "class": labels})
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=frame, x="class", y="PRD")
    plt.xlabel("MIT-BIH class")
    plt.ylabel("Per-sample PRD [%]")
    _save_current(figures_dir / f"prd_by_class_k{bottleneck_dim}.png")

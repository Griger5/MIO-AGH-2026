from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from ecg_compression.data import CLASS_NAMES, SIGNAL_LENGTH, load_splits, make_loader
from ecg_compression.metrics import metric_summary, per_sample_prd
from ecg_compression.model import MLPAutoencoder
from ecg_compression.plots import (
    plot_metric_curves,
    plot_padding_histogram,
    plot_prd_by_class,
    plot_reconstructions,
)


def reconstruct_with_model(checkpoint_path: Path, x: np.ndarray, batch_size: int) -> tuple[np.ndarray, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    bottleneck_dim = int(checkpoint["bottleneck_dim"])
    hidden_dims = tuple(int(dim) for dim in checkpoint.get("hidden_dims", [128, 64]))
    model = MLPAutoencoder(SIGNAL_LENGTH, bottleneck_dim, hidden_dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    reconstructions = []
    loader = make_loader(x, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_x, _ in loader:
            reconstruction, _ = model(batch_x)
            reconstructions.append(reconstruction.numpy())
    return np.vstack(reconstructions), bottleneck_dim


def evaluate_pca(x_train: np.ndarray, x_test: np.ndarray, bottleneck_dim: int) -> np.ndarray:
    pca = PCA(n_components=bottleneck_dim, random_state=42)
    encoded = pca.fit_transform(x_train)
    del encoded
    return pca.inverse_transform(pca.transform(x_test)).astype(np.float32)


def evaluate_models(
    data_dir: Path,
    results_dir: Path,
    bottlenecks: list[int] | None,
    batch_size: int,
    limit: int | None,
) -> pd.DataFrame:
    splits = load_splits(data_dir, limit=limit)
    model_dir = results_dir / "models"
    checkpoint_paths = sorted(model_dir.glob("mlp_autoencoder_k*.pt"))
    if bottlenecks:
        wanted = {int(value) for value in bottlenecks}
        checkpoint_paths = [
            path for path in checkpoint_paths if int(path.stem.split("_k")[-1]) in wanted
        ]
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No checkpoints found in {model_dir}. Run ecg-train before ecg-evaluate."
        )

    rows: list[dict[str, float | str]] = []
    selected_reconstructions: dict[int, np.ndarray] = {}
    selected_prd: dict[int, np.ndarray] = {}

    for checkpoint_path in checkpoint_paths:
        x_hat, bottleneck_dim = reconstruct_with_model(checkpoint_path, splits.x_test, batch_size)
        row = metric_summary(splits.x_test, x_hat, bottleneck_dim)
        row["method"] = "MLP autoencoder"
        rows.append(row)
        selected_reconstructions[bottleneck_dim] = x_hat
        selected_prd[bottleneck_dim] = per_sample_prd(splits.x_test, x_hat)

        x_hat_pca = evaluate_pca(splits.x_train, splits.x_test, bottleneck_dim)
        pca_row = metric_summary(splits.x_test, x_hat_pca, bottleneck_dim)
        pca_row["method"] = "PCA"
        rows.append(pca_row)

    metrics = pd.DataFrame(rows).sort_values(["method", "bottleneck_dim"])
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_curves(metrics, figures_dir)
    plot_padding_histogram(splits.x_train, figures_dir)

    if selected_reconstructions:
        representative_k = sorted(selected_reconstructions)[len(selected_reconstructions) // 2]
        plot_reconstructions(
            splits.x_test,
            selected_reconstructions[representative_k],
            splits.y_test,
            representative_k,
            figures_dir,
        )
        plot_prd_by_class(
            selected_prd[representative_k],
            splits.y_test,
            CLASS_NAMES,
            representative_k,
            figures_dir,
        )

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved figures to {figures_dir}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained MLP ECG autoencoders.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--bottlenecks", type=int, nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke runs.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    evaluate_models(args.data_dir, args.results_dir, args.bottlenecks, args.batch_size, args.limit)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def select_best_checkpoint(
    results_dir: Path,
    metric: str = "prd",
    method: str = "MLP autoencoder",
) -> Path:
    metrics_path = results_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing {metrics_path}. Run 'make evaluate' after training models first."
        )

    metrics = pd.read_csv(metrics_path)
    required_columns = {"method", "bottleneck_dim", metric}
    missing_columns = required_columns - set(metrics.columns)
    if missing_columns:
        raise ValueError(f"{metrics_path} is missing columns: {sorted(missing_columns)}")

    candidates = metrics[metrics["method"] == method].copy()
    if candidates.empty:
        raise ValueError(f"No rows for method '{method}' in {metrics_path}.")

    best_row = candidates.sort_values(metric, ascending=True).iloc[0]
    bottleneck_dim = int(best_row["bottleneck_dim"])
    checkpoint_path = results_dir / "models" / f"mlp_autoencoder_k{bottleneck_dim}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best metrics row points to K={bottleneck_dim}, but checkpoint is missing: "
            f"{checkpoint_path}"
        )
    return checkpoint_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print the best trained MLP checkpoint path.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--metric", default="prd")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(select_best_checkpoint(args.results_dir, args.metric))


if __name__ == "__main__":
    main()

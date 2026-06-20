from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ecg_compression.data import SIGNAL_LENGTH, load_mitbih_csv
from ecg_compression.metrics import metric_summary, per_sample_prd
from ecg_compression.model import MLPAutoencoder
from ecg_compression.plots import plot_reconstructions


def load_model(checkpoint_path: Path) -> tuple[MLPAutoencoder, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    bottleneck_dim = int(checkpoint["bottleneck_dim"])
    hidden_dims = tuple(int(dim) for dim in checkpoint.get("hidden_dims", [128, 64]))
    model = MLPAutoencoder(SIGNAL_LENGTH, bottleneck_dim, hidden_dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, bottleneck_dim


def infer_samples(
    checkpoint_path: Path,
    input_csv: Path,
    output_dir: Path,
    sample_indices: list[int] | None,
    num_samples: int,
) -> dict[str, object]:
    x, y = load_mitbih_csv(input_csv)
    if sample_indices:
        indices = np.asarray(sample_indices, dtype=np.int64)
    else:
        indices = np.arange(min(num_samples, len(x)), dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= len(x)):
        raise IndexError(f"Sample indices must be in range [0, {len(x) - 1}].")

    selected_x = x[indices]
    selected_y = y[indices]
    model, bottleneck_dim = load_model(checkpoint_path)

    with torch.no_grad():
        batch = torch.from_numpy(selected_x.astype(np.float32))
        reconstruction, latent = model(batch)

    x_hat = reconstruction.numpy()
    latent_values = latent.numpy()
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(x_hat).to_csv(output_dir / "reconstructions.csv", index=False, header=False)
    pd.DataFrame(latent_values).to_csv(output_dir / "latent.csv", index=False, header=False)

    per_sample = per_sample_prd(selected_x, x_hat)
    sample_metrics = pd.DataFrame(
        {
            "source_index": indices,
            "class": selected_y,
            "prd": per_sample,
        }
    )
    sample_metrics.to_csv(output_dir / "sample_metrics.csv", index=False)

    summary = metric_summary(selected_x, x_hat, bottleneck_dim)
    summary_payload: dict[str, object] = {
        **summary,
        "checkpoint": str(checkpoint_path),
        "input_csv": str(input_csv),
        "sample_indices": indices.tolist(),
        "num_samples": int(len(indices)),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    plot_reconstructions(selected_x, x_hat, selected_y, bottleneck_dim, output_dir)
    return summary_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference with a trained ECG autoencoder.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, default=Path("data/raw/mitbih_test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/inference"))
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--num-samples", type=int, default=6)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = infer_samples(
        checkpoint_path=args.checkpoint,
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        sample_indices=args.indices,
        num_samples=args.num_samples,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

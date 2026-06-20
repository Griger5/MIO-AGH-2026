from pathlib import Path

import numpy as np
import pandas as pd

from ecg_compression.data import make_synthetic_beats
from ecg_compression.infer import infer_samples
from ecg_compression.train import train_one_model


def _write_kaggle_like_csv(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    frame = pd.DataFrame(np.column_stack([x, y.astype(np.float32)]))
    frame.to_csv(path, index=False, header=False)


def test_infer_samples_smoke(tmp_path: Path):
    x, y = make_synthetic_beats(n_samples=64)
    csv_path = tmp_path / "mitbih_test.csv"
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "inference"
    _write_kaggle_like_csv(csv_path, x, y)

    train_one_model(
        x_train=x[:48],
        x_val=x[48:],
        bottleneck_dim=4,
        results_dir=results_dir,
        epochs=1,
        batch_size=16,
        learning_rate=1e-3,
        patience=2,
        seed=42,
    )

    summary = infer_samples(
        checkpoint_path=results_dir / "models" / "mlp_autoencoder_k4.pt",
        input_csv=csv_path,
        output_dir=output_dir,
        sample_indices=[0, 2, 4],
        num_samples=3,
    )

    assert summary["num_samples"] == 3
    assert (output_dir / "reconstructions.csv").exists()
    assert (output_dir / "latent.csv").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "sample_metrics.csv").exists()

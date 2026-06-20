from pathlib import Path

import numpy as np
import pandas as pd

from ecg_compression.data import make_synthetic_beats
from ecg_compression.evaluate import evaluate_models
from ecg_compression.train import train_one_model


def _write_kaggle_like_csv(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    frame = pd.DataFrame(np.column_stack([x, y.astype(np.float32)]))
    frame.to_csv(path, index=False, header=False)


def test_evaluate_models_smoke(tmp_path: Path):
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    data_dir.mkdir()

    x, y = make_synthetic_beats(n_samples=80)
    _write_kaggle_like_csv(data_dir / "mitbih_train.csv", x[:60], y[:60])
    _write_kaggle_like_csv(data_dir / "mitbih_test.csv", x[60:], y[60:])

    train_one_model(
        x_train=x[:45],
        x_val=x[45:60],
        bottleneck_dim=4,
        results_dir=results_dir,
        epochs=1,
        batch_size=16,
        learning_rate=1e-3,
        patience=2,
        seed=42,
    )

    metrics = evaluate_models(
        data_dir=data_dir,
        results_dir=results_dir,
        bottlenecks=[4],
        batch_size=16,
        limit=None,
    )

    assert set(metrics["method"]) == {"MLP autoencoder", "PCA"}
    assert (results_dir / "metrics.csv").exists()
    assert (results_dir / "figures" / "cr_vs_prd.png").exists()

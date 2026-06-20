from pathlib import Path

import pandas as pd

from ecg_compression.best_model import select_best_checkpoint


def test_select_best_checkpoint(tmp_path: Path):
    results_dir = tmp_path / "results"
    model_dir = results_dir / "models"
    model_dir.mkdir(parents=True)
    checkpoint = model_dir / "mlp_autoencoder_k16.pt"
    checkpoint.write_bytes(b"checkpoint")
    pd.DataFrame(
        [
            {"method": "MLP autoencoder", "bottleneck_dim": 8, "prd": 20.0},
            {"method": "MLP autoencoder", "bottleneck_dim": 16, "prd": 10.0},
            {"method": "PCA", "bottleneck_dim": 8, "prd": 5.0},
        ]
    ).to_csv(results_dir / "metrics.csv", index=False)

    assert select_best_checkpoint(results_dir) == checkpoint

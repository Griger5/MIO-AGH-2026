from pathlib import Path

from ecg_compression.data import make_synthetic_beats
from ecg_compression.train import train_one_model


def test_train_one_model_smoke(tmp_path: Path):
    x, _ = make_synthetic_beats(n_samples=48)
    x_train = x[:36]
    x_val = x[36:]

    summary = train_one_model(
        x_train=x_train,
        x_val=x_val,
        bottleneck_dim=4,
        results_dir=tmp_path,
        epochs=1,
        batch_size=16,
        learning_rate=1e-3,
        patience=2,
        seed=42,
    )

    assert summary["epochs_ran"] == 1
    assert (tmp_path / "models" / "mlp_autoencoder_k4.pt").exists()

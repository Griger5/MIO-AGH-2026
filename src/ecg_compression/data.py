from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch


SIGNAL_LENGTH = 187
LABEL_COLUMN = 187
CLASS_NAMES = {
    0: "N",
    1: "S",
    2: "V",
    3: "F",
    4: "Q",
}


@dataclass(frozen=True)
class ECGSplits:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def load_mitbih_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a Kaggle MIT-BIH CSV with 187 signal columns and one label column."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {csv_path}. Download Kaggle dataset "
            "'shayanfazeli/heartbeat' and place mitbih_train.csv and "
            "mitbih_test.csv in data/raw/."
        )

    frame = pd.read_csv(csv_path, header=None)
    if frame.shape[1] != SIGNAL_LENGTH + 1:
        raise ValueError(
            f"Expected {SIGNAL_LENGTH + 1} columns in {csv_path}, got {frame.shape[1]}."
        )

    x = frame.iloc[:, :SIGNAL_LENGTH].to_numpy(dtype=np.float32)
    y = frame.iloc[:, LABEL_COLUMN].to_numpy(dtype=np.int64)
    return x, y


def load_splits(
    data_dir: str | Path,
    val_size: float = 0.15,
    seed: int = 42,
    limit: int | None = None,
) -> ECGSplits:
    data_path = Path(data_dir)
    x_train_full, y_train_full = load_mitbih_csv(data_path / "mitbih_train.csv")
    x_test, y_test = load_mitbih_csv(data_path / "mitbih_test.csv")

    if limit is not None:
        x_train_full = x_train_full[:limit]
        y_train_full = y_train_full[:limit]
        x_test = x_test[: max(1, limit // 4)]
        y_test = y_test[: max(1, limit // 4)]

    stratify = y_train_full if len(np.unique(y_train_full)) > 1 else None
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=val_size,
        random_state=seed,
        stratify=stratify,
    )
    return ECGSplits(x_train, x_val, x_test, y_train, y_val, y_test)


def make_loader(
    x: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))
    dataset = TensorDataset(tensor, tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def nonzero_lengths(x: np.ndarray) -> np.ndarray:
    """Estimate useful heartbeat lengths before trailing zero padding starts."""
    is_nonzero = np.abs(x) > 1e-8
    reversed_argmax = np.argmax(is_nonzero[:, ::-1], axis=1)
    has_signal = is_nonzero.any(axis=1)
    lengths = SIGNAL_LENGTH - reversed_argmax
    lengths[~has_signal] = 0
    return lengths.astype(np.int64)


def make_synthetic_beats(
    n_samples: int = 256,
    signal_length: int = SIGNAL_LENGTH,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate small ECG-like data for tests and smoke runs."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, signal_length, dtype=np.float32)
    x = np.zeros((n_samples, signal_length), dtype=np.float32)
    y = rng.integers(0, 5, size=n_samples, dtype=np.int64)

    for idx in range(n_samples):
        shift = rng.normal(0.0, 0.015)
        width = rng.uniform(0.012, 0.024)
        amp = rng.uniform(0.75, 1.0)
        baseline = 0.08 * np.sin(2 * np.pi * (t + rng.random()))
        p_wave = 0.15 * np.exp(-((t - 0.28 - shift) ** 2) / 0.002)
        qrs = amp * np.exp(-((t - 0.48 - shift) ** 2) / (2 * width**2))
        t_wave = 0.3 * np.exp(-((t - 0.68 - shift) ** 2) / 0.01)
        noise = rng.normal(0.0, 0.015, size=signal_length)
        beat = baseline + p_wave + qrs + t_wave + noise
        beat = beat - beat.min()
        beat = beat / max(float(beat.max()), 1e-8)
        x[idx] = beat.astype(np.float32)

    return x, y

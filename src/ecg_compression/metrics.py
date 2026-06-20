from __future__ import annotations

import numpy as np


EPS = 1e-12


def rmse(x: np.ndarray, x_hat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x - x_hat) ** 2)))


def mae(x: np.ndarray, x_hat: np.ndarray) -> float:
    return float(np.mean(np.abs(x - x_hat)))


def prd(x: np.ndarray, x_hat: np.ndarray) -> float:
    numerator = np.sum((x - x_hat) ** 2)
    denominator = np.sum(x**2) + EPS
    return float(100.0 * np.sqrt(numerator / denominator))


def prdn(x: np.ndarray, x_hat: np.ndarray) -> float:
    centered = x - np.mean(x, axis=1, keepdims=True)
    numerator = np.sum((x - x_hat) ** 2)
    denominator = np.sum(centered**2) + EPS
    return float(100.0 * np.sqrt(numerator / denominator))


def snr_db(x: np.ndarray, x_hat: np.ndarray) -> float:
    signal = np.sum(x**2) + EPS
    noise = np.sum((x - x_hat) ** 2) + EPS
    return float(10.0 * np.log10(signal / noise))


def metric_summary(x: np.ndarray, x_hat: np.ndarray, bottleneck_dim: int) -> dict[str, float]:
    compression_ratio = x.shape[1] / bottleneck_dim
    prd_value = prd(x, x_hat)
    return {
        "bottleneck_dim": float(bottleneck_dim),
        "compression_ratio": float(compression_ratio),
        "prd": prd_value,
        "prdn": prdn(x, x_hat),
        "snr_db": snr_db(x, x_hat),
        "rmse": rmse(x, x_hat),
        "mae": mae(x, x_hat),
        "quality_score": float(compression_ratio / max(prd_value, EPS)),
    }


def per_sample_prd(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    numerator = np.sum((x - x_hat) ** 2, axis=1)
    denominator = np.sum(x**2, axis=1) + EPS
    return 100.0 * np.sqrt(numerator / denominator)

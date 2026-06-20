import numpy as np

from ecg_compression.metrics import metric_summary, prd, rmse


def test_metrics_are_zero_for_identical_signals():
    x = np.ones((3, 187), dtype=np.float32)

    assert rmse(x, x) == 0.0
    assert prd(x, x) == 0.0


def test_metric_summary_contains_compression_values():
    x = np.ones((2, 187), dtype=np.float32)
    x_hat = x * 0.9

    summary = metric_summary(x, x_hat, bottleneck_dim=17)

    assert summary["compression_ratio"] == 11.0
    assert summary["prd"] > 0.0
    assert summary["quality_score"] > 0.0

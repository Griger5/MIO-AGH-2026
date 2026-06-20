import numpy as np

from ecg_compression.data import SIGNAL_LENGTH, make_synthetic_beats, nonzero_lengths


def test_synthetic_beats_shape_and_range():
    x, y = make_synthetic_beats(n_samples=12)

    assert x.shape == (12, SIGNAL_LENGTH)
    assert y.shape == (12,)
    assert np.all(x >= 0.0)
    assert np.all(x <= 1.0)


def test_nonzero_lengths_detects_trailing_padding():
    x = np.zeros((2, SIGNAL_LENGTH), dtype=np.float32)
    x[0, :10] = 1.0
    x[1, :25] = 0.5

    assert nonzero_lengths(x).tolist() == [10, 25]

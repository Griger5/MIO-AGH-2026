import torch

from ecg_compression.model import MLPAutoencoder


def test_autoencoder_shapes():
    model = MLPAutoencoder(input_dim=187, bottleneck_dim=8)
    x = torch.rand(4, 187)

    reconstruction, latent = model(x)

    assert reconstruction.shape == (4, 187)
    assert latent.shape == (4, 8)
    assert torch.all(reconstruction >= 0.0)
    assert torch.all(reconstruction <= 1.0)


def test_autoencoder_supports_custom_hidden_dims():
    model = MLPAutoencoder(input_dim=187, bottleneck_dim=8, hidden_dims=(256, 128, 64))
    x = torch.rand(4, 187)

    reconstruction, latent = model(x)

    assert reconstruction.shape == (4, 187)
    assert latent.shape == (4, 8)

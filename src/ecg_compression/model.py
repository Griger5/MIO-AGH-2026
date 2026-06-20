from __future__ import annotations

import torch
from torch import nn


class MLPAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 187,
        bottleneck_dim: int = 16,
        hidden_dims: tuple[int, ...] = (128, 64),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dims = hidden_dims
        self.encoder = self._build_encoder(input_dim, bottleneck_dim, hidden_dims)
        self.decoder = self._build_decoder(input_dim, bottleneck_dim, hidden_dims)

    @staticmethod
    def _build_encoder(
        input_dim: int,
        bottleneck_dim: int,
        hidden_dims: tuple[int, ...],
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(previous_dim, hidden_dim), nn.ReLU()])
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, bottleneck_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_decoder(
        input_dim: int,
        bottleneck_dim: int,
        hidden_dims: tuple[int, ...],
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        previous_dim = bottleneck_dim
        for hidden_dim in reversed(hidden_dims):
            layers.extend([nn.Linear(previous_dim, hidden_dim), nn.ReLU()])
            previous_dim = hidden_dim
        layers.extend([nn.Linear(previous_dim, input_dim), nn.Sigmoid()])
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

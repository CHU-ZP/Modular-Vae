"""Replaceable encoder backbones that output mu and logvar."""

from __future__ import annotations

from torch import nn

from vae.models.transformer_blocks import PatchTransformerEncoder


class MLPEncoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 512, image_shape: tuple[int, int, int] = (1, 28, 28)):
        super().__init__()
        channels, height, width = image_shape
        input_dim = channels * height * width
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class CNNEncoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.features(x)
        return self.mu(h), self.logvar(h)


class TransformerEncoder(PatchTransformerEncoder):
    """Patch-based Transformer encoder for MNIST."""

    pass

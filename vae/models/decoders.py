"""Replaceable decoder backbones that map z back to an image."""

from __future__ import annotations

from torch import nn

from vae.models.transformer_blocks import PatchTransformerDecoder


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 512, image_shape: tuple[int, int, int] = (1, 28, 28)):
        super().__init__()
        self.image_shape = image_shape
        channels, height, width = image_shape
        output_dim = channels * height * width
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.net(z)
        return x.view(z.shape[0], *self.image_shape)


class CNNDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 128 * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.net(z).view(z.shape[0], 128, 7, 7)
        return self.deconv(h)


class TransformerDecoder(PatchTransformerDecoder):
    """Patch-based Transformer decoder for MNIST."""

    pass

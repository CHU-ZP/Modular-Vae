"""The dataset-agnostic VAE wrapper."""

from __future__ import annotations

from torch import nn

from vae.distributions import DiagonalGaussian


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x) -> DiagonalGaussian:
        mu, logvar = self.encoder(x)
        return DiagonalGaussian(mu, logvar)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        posterior = self.encode(x)
        z = posterior.rsample()
        x_recon = self.decode(z)
        return {
            "x_recon": x_recon,
            "z": z,
            "mu": posterior.mu,
            "logvar": posterior.logvar,
            "posterior": posterior,
        }

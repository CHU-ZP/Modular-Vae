"""Small probability distribution helpers used by the VAE."""

from __future__ import annotations

import math

import torch


class DiagonalGaussian:
    """Diagonal Gaussian q(z | x) with differentiable sampling."""

    def __init__(self, mu: torch.Tensor, logvar: torch.Tensor):
        if mu.shape != logvar.shape:
            raise ValueError(f"mu and logvar must share shape, got {mu.shape} and {logvar.shape}")
        self.mu = mu
        self.logvar = logvar

    @property
    def std(self) -> torch.Tensor:
        return torch.exp(0.5 * self.logvar)

    @property
    def var(self) -> torch.Tensor:
        return torch.exp(self.logvar)

    def rsample(self) -> torch.Tensor:
        """Sample z = mu + sigma * eps so gradients flow through mu and sigma."""

        eps = torch.randn_like(self.mu)
        return self.mu + self.std * eps

    def sample(self) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample()

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Return log q(z | x), summed across latent dimensions."""

        log_two_pi = math.log(2.0 * math.pi)
        log_prob = -0.5 * (log_two_pi + self.logvar + (z - self.mu).pow(2) / self.var)
        return log_prob.sum(dim=-1)

    def kl_to_standard_normal(self) -> torch.Tensor:
        """Closed-form KL(q(z | x) || N(0, I)), summed across latent dimensions."""

        return 0.5 * (self.var + self.mu.pow(2) - 1.0 - self.logvar).sum(dim=-1)

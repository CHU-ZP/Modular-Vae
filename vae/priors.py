"""Prior distributions p(z)."""

from __future__ import annotations

import math

import torch
from torch import nn

from vae.flows import RealNVPFlow


class BasePrior(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    def sample(self, num_samples: int, device: torch.device | str | None = None) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class StandardNormalPrior(BasePrior):
    """Standard Gaussian prior p(z) = N(0, I)."""

    def sample(self, num_samples: int, device: torch.device | str | None = None) -> torch.Tensor:
        return torch.randn(num_samples, self.latent_dim, device=device)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        log_two_pi = math.log(2.0 * math.pi)
        return -0.5 * (z.pow(2) + log_two_pi).sum(dim=-1)


class FlowPrior(BasePrior):
    """Flow prior z = f_psi(u), u ~ N(0, I)."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__(latent_dim)
        self.base = StandardNormalPrior(latent_dim)
        self.flow = RealNVPFlow(latent_dim, hidden_dim=hidden_dim, num_layers=num_layers)

    def sample(self, num_samples: int, device: torch.device | str | None = None) -> torch.Tensor:
        u = self.base.sample(num_samples, device=device)
        z, _ = self.flow(u)
        return z

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        u, log_det_inv = self.flow.inverse(z)
        return self.base.log_prob(u) + log_det_inv

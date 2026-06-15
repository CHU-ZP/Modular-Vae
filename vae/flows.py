"""Simple RealNVP-style affine coupling flow for the prior."""

from __future__ import annotations

import torch
from torch import nn


class AffineCouplingLayer(nn.Module):
    """Affine coupling layer with a fixed binary mask."""

    def __init__(self, latent_dim: int, hidden_dim: int, mask: torch.Tensor, scale_limit: float = 2.0):
        super().__init__()
        if mask.shape != (latent_dim,):
            raise ValueError(f"mask must have shape ({latent_dim},), got {mask.shape}")
        self.register_buffer("mask", mask.float())
        self.scale_limit = scale_limit
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

    def _shift_and_log_scale(self, x_masked: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shift, log_scale = self.net(x_masked).chunk(2, dim=-1)
        inv_mask = 1.0 - self.mask
        shift = shift * inv_mask
        log_scale = torch.tanh(log_scale) * self.scale_limit * inv_mask
        return shift, log_scale

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        shift, log_scale = self._shift_and_log_scale(x_masked)
        y = x_masked + (1.0 - self.mask) * (x * torch.exp(log_scale) + shift)
        log_det = log_scale.sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_masked = y * self.mask
        shift, log_scale = self._shift_and_log_scale(y_masked)
        x = y_masked + (1.0 - self.mask) * ((y - shift) * torch.exp(-log_scale))
        log_det_inv = -log_scale.sum(dim=-1)
        return x, log_det_inv


class RealNVPFlow(nn.Module):
    """Stack of affine coupling layers.

    forward: u -> z
    inverse: z -> u
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()
        if latent_dim < 2:
            raise ValueError("RealNVPFlow needs latent_dim >= 2")
        base_mask = (torch.arange(latent_dim) % 2).float()
        layers = []
        for layer_idx in range(num_layers):
            mask = base_mask if layer_idx % 2 == 0 else 1.0 - base_mask
            layers.append(AffineCouplingLayer(latent_dim, hidden_dim, mask))
        self.layers = nn.ModuleList(layers)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = u
        total_log_det = torch.zeros(u.shape[0], device=u.device, dtype=u.dtype)
        for layer in self.layers:
            z, log_det = layer(z)
            total_log_det = total_log_det + log_det
        return z, total_log_det

    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = z
        total_log_det_inv = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in reversed(self.layers):
            u, log_det_inv = layer.inverse(u)
            total_log_det_inv = total_log_det_inv + log_det_inv
        return u, total_log_det_inv

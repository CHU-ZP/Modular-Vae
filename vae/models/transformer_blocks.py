"""Tiny patch-based Transformer backbones for MNIST VAE experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _make_transformer_encoder(d_model: int, num_layers: int, num_heads: int, mlp_ratio: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=int(d_model * mlp_ratio),
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=False,
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int = 28, patch_size: int = 7, channels: int = 1, d_model: int = 128):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.transpose(1, 2)
        return self.proj(patches)


class PatchTransformerEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        patch_size: int = 7,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        image_size: int = 28,
        channels: int = 1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, channels, d_model)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.blocks = _make_transformer_encoder(d_model, num_layers, num_heads, mlp_ratio)
        self.mu = nn.Linear(d_model, latent_dim)
        self.logvar = nn.Linear(d_model, latent_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        tokens = self.blocks(tokens)
        pooled = tokens[:, 0]
        return self.mu(pooled), self.logvar(pooled)


class PatchTransformerDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        patch_size: int = 7,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        image_size: int = 28,
        channels: int = 1,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size * patch_size
        self.from_z = nn.Linear(latent_dim, self.num_patches * d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        self.blocks = _make_transformer_encoder(d_model, num_layers, num_heads, mlp_ratio)
        self.to_patch = nn.Linear(d_model, self.patch_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        tokens = self.from_z(z).view(z.shape[0], self.num_patches, -1)
        tokens = tokens + self.pos_embed
        tokens = self.blocks(tokens)
        patches = self.to_patch(tokens).transpose(1, 2)
        image = F.fold(
            patches,
            output_size=(self.image_size, self.image_size),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        return torch.sigmoid(image)

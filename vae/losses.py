"""ELBO-style objectives for the modular VAE."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _per_sample_reconstruction_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    reconstruction: str = "bce",
) -> torch.Tensor:
    dims = tuple(range(1, x.ndim))
    if reconstruction == "bce":
        return F.binary_cross_entropy(x_recon, x, reduction="none").sum(dim=dims)
    if reconstruction == "mse":
        return F.mse_loss(x_recon, x, reduction="none").sum(dim=dims)
    raise ValueError(f"unknown reconstruction loss: {reconstruction}")


def vae_loss(
    outputs: dict[str, torch.Tensor],
    x: torch.Tensor,
    prior=None,
    beta: float = 1.0,
    kl_mode: str = "analytic",
    reconstruction: str = "bce",
) -> dict[str, torch.Tensor]:
    """Compute reconstruction + beta * KL.

    ``analytic`` uses KL(q(z|x) || N(0, I)).
    ``monte_carlo`` estimates KL(q(z|x) || p(z)) as log q(z|x) - log p(z).
    """

    posterior = outputs["posterior"]
    z = outputs["z"]
    recon_loss = _per_sample_reconstruction_loss(outputs["x_recon"], x, reconstruction).mean()

    log_q = posterior.log_prob(z)
    log_p = prior.log_prob(z) if prior is not None else None

    if kl_mode == "analytic":
        kl_loss = posterior.kl_to_standard_normal().mean()
        if log_p is None:
            log_p_mean = torch.zeros((), device=x.device, dtype=x.dtype)
        else:
            log_p_mean = log_p.mean()
    elif kl_mode == "monte_carlo":
        if log_p is None:
            raise ValueError("monte_carlo KL requires a prior with log_prob(z)")
        kl_loss = (log_q - log_p).mean()
        log_p_mean = log_p.mean()
    else:
        raise ValueError(f"unknown kl_mode: {kl_mode}")

    loss = recon_loss + beta * kl_loss
    return {
        "loss": loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
        "log_q": log_q.mean(),
        "log_p": log_p_mean,
    }

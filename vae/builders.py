"""Factory helpers that turn YAML config into runnable components."""

from __future__ import annotations

from vae.losses import vae_loss
from vae.models.base_vae import VAE
from vae.models.decoders import CNNDecoder, MLPDecoder, TransformerDecoder
from vae.models.encoders import CNNEncoder, MLPEncoder, TransformerEncoder
from vae.priors import FlowPrior, StandardNormalPrior


def _model_cfg(config: dict) -> dict:
    return config.get("model", config)


def build_encoder(config: dict):
    cfg = _model_cfg(config)
    encoder_type = cfg["encoder"].lower()
    latent_dim = int(cfg["latent_dim"])
    if encoder_type == "mlp":
        return MLPEncoder(latent_dim=latent_dim, hidden_dim=int(cfg.get("hidden_dim", 512)))
    if encoder_type == "cnn":
        return CNNEncoder(latent_dim=latent_dim, hidden_dim=int(cfg.get("hidden_dim", 512)))
    if encoder_type == "transformer":
        return TransformerEncoder(
            latent_dim=latent_dim,
            patch_size=int(cfg.get("patch_size", 7)),
            d_model=int(cfg.get("d_model", 128)),
            num_layers=int(cfg.get("num_layers", 2)),
            num_heads=int(cfg.get("num_heads", 4)),
            mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
        )
    raise ValueError(f"unknown encoder type: {encoder_type}")


def build_decoder(config: dict):
    cfg = _model_cfg(config)
    decoder_type = cfg["decoder"].lower()
    latent_dim = int(cfg["latent_dim"])
    if decoder_type == "mlp":
        return MLPDecoder(latent_dim=latent_dim, hidden_dim=int(cfg.get("hidden_dim", 512)))
    if decoder_type == "cnn":
        return CNNDecoder(latent_dim=latent_dim, hidden_dim=int(cfg.get("hidden_dim", 512)))
    if decoder_type == "transformer":
        return TransformerDecoder(
            latent_dim=latent_dim,
            patch_size=int(cfg.get("patch_size", 7)),
            d_model=int(cfg.get("d_model", 128)),
            num_layers=int(cfg.get("num_layers", 2)),
            num_heads=int(cfg.get("num_heads", 4)),
            mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
        )
    raise ValueError(f"unknown decoder type: {decoder_type}")


def build_prior(config: dict):
    model_cfg = _model_cfg(config)
    prior_cfg = config.get("prior", {})
    latent_dim = int(model_cfg["latent_dim"])
    prior_type = prior_cfg.get("type", "standard_normal").lower()
    if prior_type == "standard_normal":
        return StandardNormalPrior(latent_dim=latent_dim)
    if prior_type == "flow":
        return FlowPrior(
            latent_dim=latent_dim,
            num_layers=int(prior_cfg.get("num_flow_layers", 4)),
            hidden_dim=int(prior_cfg.get("flow_hidden_dim", 128)),
        )
    raise ValueError(f"unknown prior type: {prior_type}")


def build_model(config: dict) -> VAE:
    return VAE(encoder=build_encoder(config), decoder=build_decoder(config))


def build_loss_config(config: dict) -> dict:
    cfg = config.get("loss", {})
    return {
        "beta": float(cfg.get("beta", 1.0)),
        "kl_mode": cfg.get("kl_mode", "analytic"),
        "reconstruction": cfg.get("reconstruction", "bce"),
    }


__all__ = [
    "build_encoder",
    "build_decoder",
    "build_prior",
    "build_model",
    "build_loss_config",
    "vae_loss",
]

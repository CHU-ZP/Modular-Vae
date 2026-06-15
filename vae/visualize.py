"""Visualization utilities and command-line visualization script."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image

from vae.builders import build_loss_config, build_model, build_prior
from vae.data.mnist import get_mnist_dataloaders
from vae.losses import vae_loss


def resolve_device(requested: str | None) -> torch.device:
    if requested is None or requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def load_checkpoint(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = build_model(config).to(device)
    prior = build_prior(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    prior.load_state_dict(checkpoint["prior_state_dict"])
    model.eval()
    prior.eval()
    return checkpoint, config, model, prior


def plot_training_curves(history: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not history or not history.get("train"):
        return

    epochs = [entry["epoch"] for entry in history["train"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, key, title in zip(axes, ["loss", "recon_loss", "kl_loss"], ["Loss", "Reconstruction", "KL"]):
        ax.plot(epochs, [entry[key] for entry in history["train"]], label="train")
        if history.get("val"):
            ax.plot(epochs, [entry[key] for entry in history["val"]], label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def save_reconstructions(model, data_loader, device: torch.device, output_path: str | Path, max_images: int = 8) -> None:
    x, _ = next(iter(data_loader))
    x = x[:max_images].to(device)
    recon = model(x)["x_recon"]
    grid = torch.cat([x.cpu(), recon.cpu()], dim=0)
    save_image(grid, output_path, nrow=max_images)


@torch.no_grad()
def save_prior_samples(model, prior, device: torch.device, output_path: str | Path, num_samples: int = 64) -> None:
    z = prior.sample(num_samples, device=device)
    samples = model.decode(z).cpu()
    save_image(samples, output_path, nrow=8)


@torch.no_grad()
def save_latent_interpolation(model, data_loader, device: torch.device, output_path: str | Path, steps: int = 12) -> None:
    x, _ = next(iter(data_loader))
    if x.shape[0] < 2:
        return
    x = x[:2].to(device)
    posterior = model.encode(x)
    z0, z1 = posterior.mu[0], posterior.mu[1]
    weights = torch.linspace(0.0, 1.0, steps, device=device).unsqueeze(1)
    z = (1.0 - weights) * z0.unsqueeze(0) + weights * z1.unsqueeze(0)
    images = model.decode(z).cpu()
    save_image(images, output_path, nrow=steps)


@torch.no_grad()
def save_latent_space_2d(
    model,
    data_loader,
    device: torch.device,
    output_path: str | Path,
    max_batches: int = 40,
) -> None:
    mus = []
    labels = []
    for batch_idx, (x, y) in enumerate(data_loader):
        if batch_idx >= max_batches:
            break
        posterior = model.encode(x.to(device))
        mus.append(posterior.mu.cpu())
        labels.append(y)
    mu = torch.cat(mus).numpy()
    y = torch.cat(labels).numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(mu[:, 0], mu[:, 1], c=y, s=6, cmap="tab10", alpha=0.75)
    ax.set_xlabel("mu[0]")
    ax.set_ylabel("mu[1]")
    ax.set_title("Latent means")
    fig.colorbar(scatter, ax=ax, ticks=range(10))
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def evaluate_loader(model, prior, data_loader, device: torch.device, loss_config: dict) -> dict[str, float]:
    totals = {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0, "log_q": 0.0, "log_p": 0.0}
    total_count = 0
    for x, _ in data_loader:
        x = x.to(device)
        outputs = model(x)
        losses = vae_loss(outputs, x, prior=prior, **loss_config)
        batch_size = x.shape[0]
        total_count += batch_size
        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * batch_size
    return {key: value / total_count for key, value in totals.items()}


def build_data_from_config(config: dict, device: torch.device):
    data_cfg = config.get("data", {})
    if data_cfg.get("name", "mnist").lower() != "mnist":
        raise ValueError("only MNIST is implemented in this demo")
    return get_mnist_dataloaders(
        data_dir=data_cfg.get("data_dir", "data"),
        batch_size=int(data_cfg.get("batch_size", 128)),
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )


def generate_visualizations(checkpoint_path: str | Path, output_dir: str | Path | None = None, device_name: str | None = None) -> None:
    device = resolve_device(device_name)
    checkpoint, config, model, prior = load_checkpoint(checkpoint_path, device)
    if output_dir is None:
        output_dir = Path(checkpoint_path).parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, test_loader = build_data_from_config(config, device)

    save_reconstructions(model, test_loader, device, output_dir / "reconstructions.png")
    save_prior_samples(model, prior, device, output_dir / "samples_from_prior.png")
    save_latent_interpolation(model, test_loader, device, output_dir / "latent_interpolation.png")
    plot_training_curves(checkpoint.get("history", {}), output_dir / "training_curves.png")

    if int(config["model"]["latent_dim"]) == 2:
        save_latent_space_2d(model, test_loader, device, output_dir / "latent_space_2d.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VAE visualizations from a checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    parser.add_argument("--output-dir", default=None, help="Directory for PNG outputs")
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto")
    args = parser.parse_args()
    generate_visualizations(args.checkpoint, args.output_dir, args.device)


if __name__ == "__main__":
    main()

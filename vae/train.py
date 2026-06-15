"""Train a modular VAE on MNIST."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from vae.builders import build_loss_config, build_model, build_prior
from vae.data.mnist import get_mnist_dataloaders
from vae.losses import vae_loss
from vae.visualize import plot_training_curves, save_reconstructions


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(requested: str) -> torch.device:
    if requested in {"auto", None}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def build_dataloaders(config: dict, device: torch.device):
    data_cfg = config.get("data", {})
    if data_cfg.get("name", "mnist").lower() != "mnist":
        raise ValueError("only MNIST is implemented in this demo")
    return get_mnist_dataloaders(
        data_dir=data_cfg.get("data_dir", "data"),
        batch_size=int(data_cfg.get("batch_size", 128)),
        num_workers=int(data_cfg.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )


def run_epoch(model, prior, data_loader, optimizer, device: torch.device, loss_config: dict, train: bool) -> dict[str, float]:
    model.train(train)
    prior.train(train)
    totals = {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0, "log_q": 0.0, "log_p": 0.0}
    total_count = 0
    context = torch.enable_grad() if train else torch.no_grad()
    progress = tqdm(data_loader, leave=False, desc="train" if train else "val")
    with context:
        for x, _ in progress:
            x = x.to(device)
            outputs = model(x)
            losses = vae_loss(outputs, x, prior=prior, **loss_config)
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                optimizer.step()

            batch_size = x.shape[0]
            total_count += batch_size
            for key in totals:
                totals[key] += float(losses[key].detach().cpu()) * batch_size
            progress.set_postfix(loss=f"{float(losses['loss']):.2f}", kl=f"{float(losses['kl_loss']):.2f}")
    return {key: value / total_count for key, value in totals.items()}


def save_checkpoint(path: Path, config: dict, model, prior, optimizer, epoch: int, history: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config,
            "model_state_dict": model.state_dict(),
            "prior_state_dict": prior.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
        },
        path,
    )


def train(config_path: str | Path) -> Path:
    config = load_config(config_path)
    train_cfg = config.get("training", {})
    experiment_name = config.get("experiment_name", Path(config_path).stem)
    output_dir = Path("outputs") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(train_cfg.get("device", "auto"))
    train_loader, val_loader = build_dataloaders(config, device)
    model = build_model(config).to(device)
    prior = build_prior(config).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(prior.parameters()),
        lr=float(train_cfg.get("lr", 1e-3)),
    )
    loss_config = build_loss_config(config)

    history = {"train": [], "val": []}
    epochs = int(train_cfg.get("epochs", 20))
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, prior, train_loader, optimizer, device, loss_config, train=True)
        val_metrics = run_epoch(model, prior, val_loader, optimizer, device, loss_config, train=False)
        train_metrics["epoch"] = epoch
        val_metrics["epoch"] = epoch
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        print(
            f"epoch {epoch:03d}/{epochs:03d} "
            f"train loss={train_metrics['loss']:.2f} val loss={val_metrics['loss']:.2f} "
            f"val recon={val_metrics['recon_loss']:.2f} val kl={val_metrics['kl_loss']:.2f}"
        )
        save_checkpoint(output_dir / "checkpoint.pt", config, model, prior, optimizer, epoch, history)
        plot_training_curves(history, output_dir / "training_curves.png")

    save_reconstructions(model, val_loader, device, output_dir / "reconstructions.png")
    return output_dir / "checkpoint.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a modular MNIST VAE.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    checkpoint_path = train(args.config)
    print(f"saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()

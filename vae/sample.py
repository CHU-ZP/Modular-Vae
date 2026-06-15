"""Sample images from a trained VAE prior."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from vae.visualize import load_checkpoint, resolve_device


@torch.no_grad()
def sample(checkpoint_path: str | Path, output_path: str | Path | None = None, num_samples: int = 64, device_name: str | None = None) -> Path:
    device = resolve_device(device_name)
    _, config, model, prior = load_checkpoint(checkpoint_path, device)
    if output_path is None:
        output_path = Path(checkpoint_path).parent / "samples_from_prior.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    z = prior.sample(num_samples, device=device)
    images = model.decode(z).cpu()
    save_image(images, output_path, nrow=8)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a trained VAE prior.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto")
    args = parser.parse_args()
    output_path = sample(args.checkpoint, args.output, args.num_samples, args.device)
    print(f"saved samples to {output_path}")


if __name__ == "__main__":
    main()

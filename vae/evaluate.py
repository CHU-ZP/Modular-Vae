"""Evaluate a checkpoint on MNIST."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from vae.builders import build_loss_config
from vae.visualize import build_data_from_config, evaluate_loader, load_checkpoint, resolve_device


def evaluate(checkpoint_path: str | Path, device_name: str | None = None) -> dict[str, float]:
    device = resolve_device(device_name)
    _, config, model, prior = load_checkpoint(checkpoint_path, device)
    _, test_loader = build_data_from_config(config, device)
    metrics = evaluate_loader(model, prior, test_loader, device, build_loss_config(config))
    output_path = Path(checkpoint_path).parent / "eval_metrics.yaml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metrics, f, sort_keys=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained VAE checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto")
    args = parser.parse_args()
    metrics = evaluate(args.checkpoint, args.device)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

"""MNIST dataloaders."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    data_dir = Path(data_dir)
    train_dataset = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform, download=True)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, test_loader

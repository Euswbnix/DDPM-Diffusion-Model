"""CIFAR-10 training data pipeline: [-1, 1] tensors, horizontal flips."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def ensure_cifar10(data_dir: str = "data"):
    """Download + integrity-check pass. The only off-clock data work (PROTOCOL)."""
    datasets.CIFAR10(data_dir, train=True, download=True)


def cifar10_loader(
    batch_size: int,
    num_workers: int = 8,
    data_dir: str = "data",
    flip: bool = True,
    device_is_cuda: bool = False,
):
    tfs = []
    if flip:
        tfs.append(transforms.RandomHorizontalFlip())
    tfs += [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]  # -> [-1, 1]
    ds = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=transforms.Compose(tfs)
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=device_is_cuda,
        persistent_workers=num_workers > 0,
    )


def infinite(loader: DataLoader):
    while True:
        for batch, _ in loader:
            yield batch

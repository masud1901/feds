"""Shared data loading and partitioning utilities for FEDS experiments.

Implements the loaders used in the v10/v11 Kaggle notebooks:

- Label-pair non-IID partition for MNIST / CIFAR-10 (DSFL paper setup)
- Dirichlet(α) partition for CIFAR-100
- Optional GPU tensor caching via ``GPUTensorLoader`` for image datasets.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset, idx: np.ndarray) -> None:
        self.base = base
        self.idx = np.asarray(idx)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        return self.base[int(self.idx[i])]


def label_pair_partition(
    dataset: Dataset, num_clients: int, num_classes: int, rng: np.random.Generator
) -> List[np.ndarray]:
    """DSFL-style label-pair non-IID partition."""
    if hasattr(dataset, "targets"):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, "labels"):
        targets = np.array(dataset.labels)
    else:
        targets = np.array([dataset[i][1] for i in range(len(dataset))])

    parts: List[np.ndarray] = []
    for cid in range(num_clients):
        la = cid % num_classes
        lb = (cid + 1) % num_classes
        idx = np.where((targets == la) | (targets == lb))[0].copy()
        rng.shuffle(idx)
        parts.append(idx)
    return parts


def dirichlet_partition(
    dataset: Dataset, num_clients: int, alpha: float, seed: int
) -> List[np.ndarray]:
    """Dirichlet non-IID partition over labels (used for CIFAR-100)."""
    if hasattr(dataset, "targets"):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, "labels"):
        targets = np.array(dataset.labels)
    else:
        targets = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = int(targets.max()) + 1
    rng = np.random.default_rng(seed)

    class_indices = [np.where(targets == c)[0].copy() for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(class_indices[c])

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        props = rng.dirichlet([alpha] * num_clients)
        counts = (props * len(class_indices[c])).astype(int)
        counts[-1] = len(class_indices[c]) - counts[:-1].sum()

        start = 0
        for cid, count in enumerate(counts):
            count = max(int(count), 0)
            client_indices[cid].extend(
                class_indices[c][start : start + count].tolist()
            )
            start += count

    return [np.array(idx, dtype=np.int64) for idx in client_indices]


class GPUTensorLoader:
    """Simple GPU tensor mini-batch loader used in the notebooks."""

    def __init__(self, X_gpu: torch.Tensor, Y_gpu: torch.Tensor, batch_size: int, seed: int = 0) -> None:
        self.X = X_gpu
        self.Y = Y_gpu
        self.bs = batch_size
        self.gen = torch.Generator(device=X_gpu.device)
        self.gen.manual_seed(seed)
        self.n = X_gpu.shape[0]

    def __iter__(self):
        idx = torch.randperm(self.n, device=self.X.device, generator=self.gen)
        for s in range(0, self.n - self.bs + 1, self.bs):
            b = idx[s : s + self.bs]
            yield self.X[b], self.Y[b]

    def __len__(self) -> int:
        return self.n // self.bs


def _load_base_dataset(dataset_name: str):
    if dataset_name == "MNIST":
        tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        train_ds = MNIST("/tmp/data", train=True, download=True, transform=tf)
        test_ds = MNIST("/tmp/data", train=False, download=True, transform=tf)
        num_classes = 10
    elif dataset_name == "CIFAR10":
        tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * 3, (0.5,) * 3),
            ]
        )
        train_ds = CIFAR10("/tmp/data", train=True, download=True, transform=tf)
        test_ds = CIFAR10("/tmp/data", train=False, download=True, transform=tf)
        num_classes = 10
    elif dataset_name == "CIFAR100":
        tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * 3, (0.5,) * 3),
            ]
        )
        train_ds = CIFAR100("/tmp/data", train=True, download=True, transform=tf)
        test_ds = CIFAR100("/tmp/data", train=False, download=True, transform=tf)
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}")
    return train_ds, test_ds, num_classes


def build_loaders(
    dataset_name: str,
    num_clients: int,
    seed: int,
    batch_size: int,
    partition_dir: str = "partitions",
    device: str = "cpu",
    dirichlet_alpha: float = 0.5,
) -> Tuple[List[DataLoader], DataLoader]:
    """Build client train loaders and a global test loader.

    - MNIST / CIFAR-10: label-pair partition.
    - CIFAR-100: Dirichlet(α) partition.

    For image datasets, the full tensors are preloaded to ``device`` once and
    then sliced per client, matching the CUDA-optimised notebook behaviour.
    """
    os.makedirs(partition_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    train_ds, test_ds, num_classes = _load_base_dataset(dataset_name)

    # Partition indices (no on-disk pickle cache to avoid security issues).
    if dataset_name in {"MNIST", "CIFAR10"}:
        parts = label_pair_partition(train_ds, num_clients, num_classes, rng)
    elif dataset_name == "CIFAR100":
        parts = dirichlet_partition(train_ds, num_clients, dirichlet_alpha, seed)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}")

    # Preload full train/test to device once.
    loader_tmp = DataLoader(
        train_ds, batch_size=4096, shuffle=False, num_workers=2, pin_memory=(device == "cuda")
    )
    xs, ys = [], []
    for xb, yb in loader_tmp:
        xs.append(xb)
        ys.append(yb)
    all_X = torch.cat(xs).to(device)
    all_Y = torch.cat(ys).to(device)

    train_loaders: List[DataLoader] = []
    for cid, idx in enumerate(parts):
        idx_arr = np.asarray(idx, dtype=np.int64)
        train_loaders.append(
            GPUTensorLoader(all_X[idx_arr], all_Y[idx_arr], batch_size, seed=seed + cid)
        )

    te_tmp = DataLoader(
        test_ds, batch_size=4096, shuffle=False, num_workers=2, pin_memory=(device == "cuda")
    )
    te_xs, te_ys = [], []
    for xb, yb in te_tmp:
        te_xs.append(xb)
        te_ys.append(yb)
    te_X = torch.cat(te_xs).to(device)
    te_Y = torch.cat(te_ys).to(device)
    test_loader = GPUTensorLoader(te_X, te_Y, batch_size * 2, seed=0)

    return train_loaders, test_loader


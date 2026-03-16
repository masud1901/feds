# -*- coding: utf-8 -*-
"""Shared model definitions used by FedAvg / DSFL / FEDS.

Historically this module only exposed ``MNISTNet`` for artifact creation.
The FEDS v10/v11 experiments use a unified architecture across MNIST and
CIFAR-style datasets so that all baselines share the same capacity.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    """Legacy CNN for MNIST (kept for backwards compatibility)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class UnifiedNet(nn.Module):
    """Unified CNN for MNIST / CIFAR10 / CIFAR100.

    This mirrors the architecture used in the Kaggle notebooks:

      - Three Conv-BN-ReLU blocks with MaxPool/AdaptiveAvgPool
      - Fixed 512-dim representation, followed by two FC layers

    The only dataset-specific pieces are the input channel count and the
    number of output classes.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(2),  # -> [B, 128, 2, 2] => 512 features
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def get_model(dataset: str) -> nn.Module:
    """Return the canonical model for a given dataset.

    - MNIST      -> UnifiedNet(1, 10)
    - CIFAR10    -> UnifiedNet(3, 10)
    - CIFAR100   -> UnifiedNet(3, 100)

    Speech / other datasets are intentionally not wired into the unified
    FEDS experiments and should continue using their existing clients.
    """

    name = dataset.upper()
    if name == "MNIST":
        return UnifiedNet(in_channels=1, num_classes=10)
    if name == "CIFAR10":
        return UnifiedNet(in_channels=3, num_classes=10)
    if name == "CIFAR100":
        return UnifiedNet(in_channels=3, num_classes=100)
    raise ValueError(f"Unsupported dataset for unified FEDS experiments: {dataset!r}")


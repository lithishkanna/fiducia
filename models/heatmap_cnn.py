"""Heatmap CNN: 2D Convolutional Neural Network on the heatmap tensor (Model C).

Processes the 4-channel tensor (bid liquidity, ask liquidity, buy trades, sell trades)
spanning price distance relative to mid (P) and historical time steps (T).
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class HeatmapCNN(nn.Module):
    """2D CNN on the (4, P, T) heatmap tensor."""

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 3,
        num_filters: Optional[list] = None,
        fc_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if num_filters is None:
            num_filters = [32, 64, 128]

        f1 = num_filters[0] if len(num_filters) > 0 else 32
        f2 = num_filters[1] if len(num_filters) > 1 else 64
        f3 = num_filters[2] if len(num_filters) > 2 else 128

        # Conv Block 1
        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(f1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(f2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 3
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(f3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classifier
        self.flatten_dim = f3 * 4 * 4
        self.fc1 = nn.Linear(self.flatten_dim, fc_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

        x: Tensor of shape (batch, 4, P, T).

        Returns:
            Logits of shape (batch, num_classes).
        """
        # Block 1
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        out = self.pool1(out)

        # Block 2
        out = F.leaky_relu(self.bn2(self.conv2(out)), 0.01)
        out = self.pool2(out)

        # Block 3
        out = F.leaky_relu(self.bn3(self.conv3(out)), 0.01)
        out = self.adaptive_pool(out)

        # Flatten & FC
        out = out.reshape(out.size(0), -1)
        out = F.leaky_relu(self.fc1(out), 0.01)
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits

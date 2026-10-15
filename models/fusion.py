"""Dual-branch fusion model: spatial CNN + tabular MLP (Model D).

Spatial branch processes heatmap tensor with configurable price-band pooling ablations:
- 'band_gru' (default): adaptive pooling over price axis to B bands + temporal GRU
- 'gap_time': global average pooling over time axis, keeping price bands
- 'full_gap': global average pooling across both price and time axes

Tabular branch projects engineered order-flow and wall trust features.
Both branches are fused via concatenation and a joint classification head.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SpatialBranch(nn.Module):
    """Spatial 2D-CNN branch with price-band pooling variants."""

    def __init__(
        self,
        in_channels: int = 4,
        spatial_dim: int = 64,
        num_bands: int = 8,
        pooling_variant: str = "band_gru",
        conv_filters: Optional[list] = None,
    ) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim
        self.num_bands = num_bands
        self.pooling_variant = pooling_variant.lower()

        if conv_filters is None:
            conv_filters = [32, 64]

        f1, f2 = conv_filters[0], conv_filters[1]

        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(f1)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(f2)

        if self.pooling_variant == "band_gru":
            # Pool price axis to num_bands, retain time axis intact
            self.band_pool = nn.AdaptiveAvgPool2d((num_bands, None))
            gru_in_dim = f2 * num_bands
            self.temporal_gru = nn.GRU(
                input_size=gru_in_dim,
                hidden_size=spatial_dim,
                batch_first=True,
            )
        elif self.pooling_variant == "gap_time":
            # Pool time axis completely, keep price bands
            self.pool = nn.AdaptiveAvgPool2d((num_bands, 1))
            self.proj = nn.Linear(f2 * num_bands, spatial_dim)
        elif self.pooling_variant == "full_gap":
            # Global average pool over both axes
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.proj = nn.Linear(f2, spatial_dim)
        else:
            raise ValueError(f"Unknown pooling_variant '{pooling_variant}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

        x: Tensor of shape (batch, 4, P, T).

        Returns:
            Spatial embedding of shape (batch, spatial_dim).
        """
        # Convolutions
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        out = F.leaky_relu(self.bn2(self.conv2(out)), 0.01)  # (batch, f2, P, T)

        if self.pooling_variant == "band_gru":
            out = self.band_pool(out)  # (batch, f2, num_bands, T)
            batch_size, channels, bands, time_steps = out.shape
            out = out.permute(0, 3, 1, 2)  # (batch, T, f2, bands)
            out = out.reshape(batch_size, time_steps, channels * bands)
            gru_out, _ = self.temporal_gru(out)
            embed = gru_out[:, -1, :]  # (batch, spatial_dim)
            return embed

        elif self.pooling_variant == "gap_time":
            out = self.pool(out)  # (batch, f2, num_bands, 1)
            out = out.reshape(out.size(0), -1)
            embed = F.leaky_relu(self.proj(out), 0.01)
            return embed

        elif self.pooling_variant == "full_gap":
            out = self.pool(out)  # (batch, f2, 1, 1)
            out = out.reshape(out.size(0), -1)
            embed = F.leaky_relu(self.proj(out), 0.01)
            return embed

        raise RuntimeError("Invalid pooling state")


class TabularBranch(nn.Module):
    """MLP branch for engineered order-flow and wall trust features."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If batch size is 1 during inference, bypass batch norm
        h = self.fc1(x)
        if h.size(0) > 1:
            h = self.bn1(h)
        h = F.leaky_relu(h, 0.01)
        h = self.dropout(h)
        out = F.leaky_relu(self.fc2(h), 0.01)
        return out


class DualBranchFusion(nn.Module):
    """Model D: Dual-Branch Fusion integrating Spatial Tensor and Tabular Order-Flow."""

    def __init__(
        self,
        tabular_in_dim: int,
        num_classes: int = 3,
        spatial_dim: int = 64,
        tabular_dim: int = 64,
        fused_dim: int = 128,
        num_bands: int = 8,
        pooling_variant: str = "band_gru",
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.spatial_branch = SpatialBranch(
            in_channels=4,
            spatial_dim=spatial_dim,
            num_bands=num_bands,
            pooling_variant=pooling_variant,
        )
        self.tabular_branch = TabularBranch(
            in_dim=tabular_in_dim,
            out_dim=tabular_dim,
            hidden_dim=tabular_dim,
            dropout=dropout,
        )

        # Joint head
        joint_in_dim = spatial_dim + tabular_dim
        self.dropout = nn.Dropout(dropout)
        self.fc_joint = nn.Linear(joint_in_dim, fused_dim)
        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(
        self,
        x_tensor: torch.Tensor,
        x_tabular: torch.Tensor,
    ) -> torch.Tensor:
        """Args:

        x_tensor: Heatmap tensor (batch, 4, P, T)
        x_tabular: Tabular features (batch, tabular_in_dim)

        Returns:
            Logits of shape (batch, num_classes)
        """
        spatial_embed = self.spatial_branch(x_tensor)
        tabular_embed = self.tabular_branch(x_tabular)

        fused = torch.cat([spatial_embed, tabular_embed], dim=-1)
        fused = self.dropout(fused)
        fused = F.leaky_relu(self.fc_joint(fused), 0.01)
        logits = self.classifier(fused)
        return logits

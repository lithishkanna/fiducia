"""PyTorch Dataset classes for order book representations and fusion models.

Builds tensors in-memory / memory-mapped without persisting images to disk:
- HeatmapDataset: (4, P, T) tensor + label
- DeepLOBDataset: (1, 4 * levels, T) matrix + label
- FusionDataset: (4, P, T) tensor + tabular features + label
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from features.tensor import TensorNormalizer
from models.baselines import encode_labels

logger = logging.getLogger(__name__)


class HeatmapDataset(Dataset):
    """PyTorch Dataset yielding (4, P, T) heatmap tensors and 3-class labels."""

    def __init__(
        self,
        tensors: np.ndarray,
        labels: np.ndarray,
        normalizer: Optional[TensorNormalizer] = None,
    ) -> None:
        """Args:

        tensors: Array of shape (N, 4, P, T).
        labels: Array of labels in [-1, 0, 1] or [0, 1, 2].
        normalizer: Optional fitted TensorNormalizer.
        """
        if len(tensors) != len(labels):
            raise ValueError(f"Mismatch: {len(tensors)} tensors vs {len(labels)} labels.")

        self.labels = (
            encode_labels(labels)
            if set(np.unique(labels)).issubset({-1, 0, 1})
            else labels.astype(int)
        )

        if normalizer is not None:
            self.tensors = normalizer.transform(tensors).astype(np.float32)
        else:
            self.tensors = tensors.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.tensors[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class DeepLOBDataset(Dataset):
    """PyTorch Dataset yielding top-N order book matrices (1, 4*levels, T) and labels."""

    def __init__(
        self,
        lob_data: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Args:

        lob_data: Array of shape (N, 1, 4*levels, T) or (N, 4*levels, T).
        labels: Array of labels in [-1, 0, 1] or [0, 1, 2].
        """
        if len(lob_data) != len(labels):
            raise ValueError(f"Mismatch: {len(lob_data)} LOB samples vs {len(labels)} labels.")

        if lob_data.ndim == 3:
            # Add channel dimension: (N, 1, 4*levels, T)
            lob_data = lob_data[:, np.newaxis, :, :]

        self.lob_data = lob_data.astype(np.float32)
        self.labels = (
            encode_labels(labels)
            if set(np.unique(labels)).issubset({-1, 0, 1})
            else labels.astype(int)
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.lob_data[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class FusionDataset(Dataset):
    """PyTorch Dataset yielding (tensor, tabular, label) for dual-branch fusion."""

    def __init__(
        self,
        tensors: np.ndarray,
        tabular_features: np.ndarray,
        labels: np.ndarray,
        normalizer: Optional[TensorNormalizer] = None,
    ) -> None:
        """Args:

        tensors: Array of shape (N, 4, P, T).
        tabular_features: Array of shape (N, D_tabular).
        labels: Array of labels in [-1, 0, 1] or [0, 1, 2].
        normalizer: Optional fitted TensorNormalizer.
        """
        if not (len(tensors) == len(tabular_features) == len(labels)):
            raise ValueError(
                f"Length mismatch: {len(tensors)} tensors, "
                f"{len(tabular_features)} tabular, {len(labels)} labels."
            )

        if normalizer is not None:
            self.tensors = normalizer.transform(tensors).astype(np.float32)
        else:
            self.tensors = tensors.astype(np.float32)

        self.tabular = tabular_features.astype(np.float32)
        self.labels = (
            encode_labels(labels)
            if set(np.unique(labels)).issubset({-1, 0, 1})
            else labels.astype(int)
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_tensor = torch.from_numpy(self.tensors[idx])
        x_tab = torch.from_numpy(self.tabular[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x_tensor, x_tab, y

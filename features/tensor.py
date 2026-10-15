"""Heatmap tensor builder (C x P x T) and leakage-free normalization.

Constructs 3D tensors directly in memory without image persistence:
- Channels C = 4: [bid_size, ask_size, executed_buy_vol, executed_sell_vol]
- Price axis P: relative price distance to mid in ticks [-P//2, P//2)
- Time axis T: past T time steps up to reference time t

Normalization parameters (mean, std) are computed exclusively on training data.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class TensorNormalizer:
    """Channel-wise normalizer fitted exclusively on training set tensors."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.mean: Optional[np.ndarray] = None  # shape (C, 1, 1) or (C,)
        self.std: Optional[np.ndarray] = None  # shape (C, 1, 1) or (C,)
        self.is_fitted: bool = False

    def fit(self, tensors: np.ndarray) -> "TensorNormalizer":
        """Compute per-channel mean and std from training tensors.

        Args:
            tensors: array of shape (N, C, P, T) or (C, P, T)
        """
        if tensors.ndim == 3:
            tensors = tensors[np.newaxis, ...]  # (1, C, P, T)

        # Compute mean and std over (N, P, T) axes per channel
        # Result shape: (1, C, 1, 1)
        self.mean = np.mean(tensors, axis=(0, 2, 3), keepdims=True)
        self.std = np.std(tensors, axis=(0, 2, 3), keepdims=True)
        # Avoid division by zero
        self.std = np.where(self.std < self.eps, 1.0, self.std)
        self.is_fitted = True
        return self

    def transform(self, tensor: np.ndarray) -> np.ndarray:
        """Standardize tensor using training statistics: (x - mean) / std.

        Args:
            tensor: shape (C, P, T) or (N, C, P, T)
        """
        if not self.is_fitted:
            raise ValueError("TensorNormalizer must be fitted before transforming.")

        if tensor.ndim == 3:
            mean = self.mean.squeeze(0)  # (C, 1, 1)
            std = self.std.squeeze(0)  # (C, 1, 1)
            return (tensor - mean) / (std + self.eps)
        elif tensor.ndim == 4:
            return (tensor - self.mean) / (self.std + self.eps)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {tensor.shape}")

    def fit_transform(self, tensors: np.ndarray) -> np.ndarray:
        """Fit on training tensors and return transformed output."""
        self.fit(tensors)
        return self.transform(tensors)

    def save(self, file_path: Union[str, Path]) -> None:
        """Save normalizer parameters to JSON."""
        data = {
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "eps": self.eps,
            "is_fitted": self.is_fitted,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "TensorNormalizer":
        """Load fitted normalizer from JSON."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        norm = cls(eps=data.get("eps", 1e-8))
        if data.get("mean") is not None:
            norm.mean = np.array(data["mean"])
            norm.std = np.array(data["std"])
            norm.is_fitted = data.get("is_fitted", True)
        return norm


class HeatmapTensorBuilder:
    """Builds (C x P x T) heatmap tensors from depth levels and trades."""

    def __init__(
        self,
        price_bins: int = 100,
        price_tick_size: float = 0.10,
        time_steps: int = 60,
        log_transform: bool = True,
    ) -> None:
        self.price_bins = price_bins
        self.price_tick_size = price_tick_size
        self.time_steps = time_steps
        self.log_transform = log_transform
        self.half_bins = price_bins // 2

    def price_to_bin(self, price: float, ref_mid: float) -> Optional[int]:
        """Map price relative to ref_mid to a discrete bin index [0, P-1]."""
        if ref_mid <= 0:
            return None
        # Tick offset from mid-price
        ticks = int(np.round((price - ref_mid) / self.price_tick_size))
        bin_idx = self.half_bins + ticks
        if 0 <= bin_idx < self.price_bins:
            return bin_idx
        return None

    def build_tensor(
        self,
        grid_snapshots: List[Dict[str, Any]],
        ref_mid: Optional[float] = None,
    ) -> np.ndarray:
        """Build (4, P, T) tensor from a window of T grid step snapshots.

        Args:
            grid_snapshots: List of T dicts, each with:
                - "mid_price": float
                - "bids": [(price, qty), ...]
                - "asks": [(price, qty), ...]
                - "buy_trades": [(price, qty), ...] optional
                - "sell_trades": [(price, qty), ...] optional
            ref_mid: Mid-price to center the price axis around.
                     Defaults to the mid_price of the latest snapshot (index T-1).

        Returns:
            np.ndarray of shape (4, price_bins, time_steps)
        """
        T = min(len(grid_snapshots), self.time_steps)
        P = self.price_bins
        tensor = np.zeros((4, P, self.time_steps), dtype=np.float32)

        if not grid_snapshots:
            return tensor

        if ref_mid is None or ref_mid <= 0:
            ref_mid = grid_snapshots[-1].get("mid_price", 0.0)

        # Populate trailing T positions (right-aligned in time)
        offset = self.time_steps - T
        for t_idx, snap in enumerate(grid_snapshots[-T:]):
            time_col = offset + t_idx

            # Channel 0: Bids
            for price, qty in snap.get("bids", []):
                b = self.price_to_bin(price, ref_mid)
                if b is not None:
                    tensor[0, b, time_col] += qty

            # Channel 1: Asks
            for price, qty in snap.get("asks", []):
                b = self.price_to_bin(price, ref_mid)
                if b is not None:
                    tensor[1, b, time_col] += qty

            # Channel 2: Executed Buy Volume
            for price, qty in snap.get("buy_trades", []):
                b = self.price_to_bin(price, ref_mid)
                if b is not None:
                    tensor[2, b, time_col] += qty

            # Channel 3: Executed Sell Volume
            for price, qty in snap.get("sell_trades", []):
                b = self.price_to_bin(price, ref_mid)
                if b is not None:
                    tensor[3, b, time_col] += qty

        if self.log_transform:
            tensor = np.log1p(tensor)

        return tensor


def build_heatmap_tensor(
    bid_matrix: np.ndarray,
    ask_matrix: np.ndarray,
    buy_vol_matrix: np.ndarray,
    sell_vol_matrix: np.ndarray,
    log_transform: bool = True,
) -> np.ndarray:
    """Build in-memory heatmap tensor of shape (4, P, T) from 2D matrices."""
    if log_transform:
        bid_matrix = np.log1p(bid_matrix)
        ask_matrix = np.log1p(ask_matrix)
        buy_vol_matrix = np.log1p(buy_vol_matrix)
        sell_vol_matrix = np.log1p(sell_vol_matrix)
    return np.stack([bid_matrix, ask_matrix, buy_vol_matrix, sell_vol_matrix], axis=0).astype(
        np.float32
    )

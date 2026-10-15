"""Walk-forward time-ordered splits with purge gaps and untouched test holdout.

Guarantees:
- Strictly time-ordered: train precedes validation, validation precedes test.
- Purge gap: an interval >= max label horizon between train and validation
  (and between train/val and test) to prevent label overlap leakage.
- Final test period is completely untouched.
"""

import logging
from dataclasses import dataclass
from typing import Generator, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FoldSplit:
    """Metadata and indices for a single walk-forward cross-validation fold."""

    fold_idx: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_start_ts: int
    train_end_ts: int
    val_start_ts: int
    val_end_ts: int
    purge_gap_ms: int

    @property
    def train_size(self) -> int:
        return len(self.train_indices)

    @property
    def val_size(self) -> int:
        return len(self.val_indices)


def create_train_test_split(
    timestamps_ms: np.ndarray,
    test_ratio: float = 0.20,
    purge_gap_sec: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Split time-series into train/validation pool and untouched holdout test set.

    Args:
        timestamps_ms: Array of sorted timestamps in milliseconds.
        test_ratio: Fraction of timeline allocated to holdout test set.
        purge_gap_sec: Purge gap in seconds separating train pool from test set.

    Returns:
        (train_val_indices, test_indices, purge_gap_ms)
    """
    n = len(timestamps_ms)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int), 0

    purge_gap_ms = int(purge_gap_sec * 1000)
    total_duration = timestamps_ms[-1] - timestamps_ms[0]

    test_duration = total_duration * test_ratio
    test_start_ts = timestamps_ms[-1] - test_duration
    train_end_ts = test_start_ts - purge_gap_ms

    train_val_mask = timestamps_ms <= train_end_ts
    test_mask = timestamps_ms >= test_start_ts

    train_val_indices = np.where(train_val_mask)[0]
    test_indices = np.where(test_mask)[0]

    logger.info(
        "Train-test split: %d train/val samples (ends %d), "
        "%d test samples (starts %d), %d ms purge gap",
        len(train_val_indices),
        train_end_ts,
        len(test_indices),
        test_start_ts,
        purge_gap_ms,
    )
    return train_val_indices, test_indices, purge_gap_ms


def walk_forward_splits(
    timestamps_ms: np.ndarray,
    n_splits: int = 5,
    purge_gap_sec: float = 30.0,
    min_train_ratio: float = 0.30,
) -> Generator[FoldSplit, None, None]:
    """Generate expanding-window walk-forward splits with purge gaps.

    For each fold:
      Train: [t_start, t_train_end]
      Purge: (t_train_end, t_val_start)
      Val:   [t_val_start, t_val_end]
    """
    n = len(timestamps_ms)
    if n < n_splits * 2:
        raise ValueError(f"Insufficient samples ({n}) for {n_splits} splits.")

    purge_gap_ms = int(purge_gap_sec * 1000)
    total_duration = timestamps_ms[-1] - timestamps_ms[0]

    if total_duration <= purge_gap_ms * n_splits:
        raise ValueError("Total timeline duration is too short for requested purge gaps.")

    # Fraction available for validation across all folds
    val_timeline_ratio = 1.0 - min_train_ratio
    step_ratio = val_timeline_ratio / n_splits

    for fold in range(n_splits):
        train_ratio = min_train_ratio + (fold * step_ratio * 0.5)
        val_start_ratio = train_ratio
        val_end_ratio = min(1.0, val_start_ratio + step_ratio)

        t_min = timestamps_ms[0]
        train_end_target = t_min + (total_duration * train_ratio)
        val_start_target = train_end_target + purge_gap_ms
        val_end_target = t_min + (total_duration * val_end_ratio)

        train_mask = timestamps_ms <= train_end_target
        val_mask = (timestamps_ms >= val_start_target) & (timestamps_ms <= val_end_target)

        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]

        if len(train_indices) == 0 or len(val_indices) == 0:
            continue

        yield FoldSplit(
            fold_idx=fold,
            train_indices=train_indices,
            val_indices=val_indices,
            train_start_ts=int(timestamps_ms[train_indices[0]]),
            train_end_ts=int(timestamps_ms[train_indices[-1]]),
            val_start_ts=int(timestamps_ms[val_indices[0]]),
            val_end_ts=int(timestamps_ms[val_indices[-1]]),
            purge_gap_ms=purge_gap_ms,
        )

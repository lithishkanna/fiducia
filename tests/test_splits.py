"""Tests for eval/splits.py — walk-forward splits with purge gaps and holdout test set."""

import numpy as np
import pytest

from eval.splits import create_train_test_split, walk_forward_splits


@pytest.fixture
def sample_timestamps():
    """10,000 steps at 500ms intervals = 5,000 seconds total timeline."""
    return np.array([1_000_000 + i * 500 for i in range(10_000)], dtype=np.int64)


class TestSplits:
    """Test time-ordered splitting and purge gap integrity."""

    def test_train_test_split_holdout_and_purge_gap(self, sample_timestamps):
        purge_gap_sec = 30.0
        test_ratio = 0.20

        train_pool, test_set, purge_ms = create_train_test_split(
            sample_timestamps,
            test_ratio=test_ratio,
            purge_gap_sec=purge_gap_sec,
        )

        assert len(train_pool) > 0
        assert len(test_set) > 0
        assert purge_ms == 30_000

        # Max train timestamp must precede min test timestamp by AT LEAST purge_ms
        last_train_ts = sample_timestamps[train_pool[-1]]
        first_test_ts = sample_timestamps[test_set[0]]
        assert first_test_ts - last_train_ts >= purge_ms

        # No index overlap
        assert len(set(train_pool).intersection(set(test_set))) == 0

    def test_walk_forward_splits_purge_gap_and_temporal_order(self, sample_timestamps):
        n_splits = 4
        purge_gap_sec = 30.0
        purge_ms = int(purge_gap_sec * 1000)

        folds = list(
            walk_forward_splits(
                sample_timestamps,
                n_splits=n_splits,
                purge_gap_sec=purge_gap_sec,
                min_train_ratio=0.30,
            )
        )

        assert len(folds) == n_splits

        prev_train_size = 0
        for f in folds:
            assert f.train_size > 0
            assert f.val_size > 0

            # Expanding window: train size increases
            assert f.train_size >= prev_train_size
            prev_train_size = f.train_size

            # Purge gap must be strictly respected between train and validation
            gap = f.val_start_ts - f.train_end_ts
            assert gap >= purge_ms

            # No overlap between train and validation indices
            train_set = set(f.train_indices)
            val_set = set(f.val_indices)
            assert len(train_set.intersection(val_set)) == 0

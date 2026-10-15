"""Tests for features/tensor.py — heatmap tensor builder and leakage-free normalization."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from features.tensor import HeatmapTensorBuilder, TensorNormalizer, build_heatmap_tensor


class TestHeatmapTensor:
    """Test 3D heatmap tensor construction."""

    def test_tensor_shape_and_channels(self):
        builder = HeatmapTensorBuilder(
            price_bins=100,
            price_tick_size=0.10,
            time_steps=60,
            log_transform=True,
        )

        snapshots = []
        for i in range(60):
            snapshots.append(
                {
                    "mid_price": 50000.0,
                    "bids": [(49999.0, 1.5), (49998.0, 3.0)],
                    "asks": [(50001.0, 2.0), (50002.0, 4.0)],
                    "buy_trades": [(50001.0, 0.5)],
                    "sell_trades": [(49999.0, 0.8)],
                }
            )

        tensor = builder.build_tensor(snapshots, ref_mid=50000.0)

        # C = 4, P = 100, T = 60
        assert tensor.shape == (4, 100, 60)
        # Log-transformed non-negative values
        assert np.all(tensor >= 0.0)
        # Bids in channel 0
        assert tensor[0].sum() > 0
        # Asks in channel 1
        assert tensor[1].sum() > 0
        # Buy trades in channel 2
        assert tensor[2].sum() > 0
        # Sell trades in channel 3
        assert tensor[3].sum() > 0

    def test_direct_build_heatmap_tensor(self):
        bids = np.ones((50, 20), dtype=np.float32)
        asks = np.ones((50, 20), dtype=np.float32)
        buys = np.zeros((50, 20), dtype=np.float32)
        sells = np.zeros((50, 20), dtype=np.float32)

        tensor = build_heatmap_tensor(bids, asks, buys, sells, log_transform=True)
        assert tensor.shape == (4, 50, 20)
        assert pytest.approx(tensor[0, 0, 0], abs=1e-4) == np.log1p(1.0)


class TestTensorNormalizer:
    """Test train-only normalization without test set leakage."""

    def test_fit_and_transform(self):
        np.random.seed(42)
        # Synthetic train set of 10 tensors: (10, 4, 50, 30)
        train_tensors = np.random.exponential(scale=2.0, size=(10, 4, 50, 30)).astype(np.float32)

        normalizer = TensorNormalizer()
        transformed_train = normalizer.fit_transform(train_tensors)

        assert normalizer.is_fitted
        # Check channel mean ~ 0 and std ~ 1 over train
        for c in range(4):
            assert pytest.approx(transformed_train[:, c, :, :].mean(), abs=1e-2) == 0.0
            assert pytest.approx(transformed_train[:, c, :, :].std(), abs=1e-2) == 1.0

        # Transform single test tensor
        test_tensor = np.random.exponential(scale=2.0, size=(4, 50, 30)).astype(np.float32)
        transformed_test = normalizer.transform(test_tensor)
        assert transformed_test.shape == (4, 50, 30)

    def test_save_and_load(self):
        train_tensors = np.random.randn(5, 4, 20, 10).astype(np.float32)
        norm = TensorNormalizer().fit(train_tensors)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "norm.json"
            norm.save(file_path)

            loaded = TensorNormalizer.load(file_path)
            assert loaded.is_fitted
            np.testing.assert_allclose(norm.mean, loaded.mean)
            np.testing.assert_allclose(norm.std, loaded.std)

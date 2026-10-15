"""Tests for Phase 6: Deep Learning architectures and PyTorch Datasets."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.datasets import DeepLOBDataset, FusionDataset, HeatmapDataset
from models.deeplob import DeepLOB
from models.fusion import DualBranchFusion
from models.heatmap_cnn import HeatmapCNN


class TestDatasets:
    """Test PyTorch dataset wrapping and batching."""

    def test_heatmap_dataset(self):
        tensors = np.random.randn(10, 4, 30, 20).astype(np.float32)
        labels = np.array([-1, 0, 1, 0, -1, 1, 0, 1, -1, 0])

        ds = HeatmapDataset(tensors, labels)
        assert len(ds) == 10
        x, y = ds[0]
        assert x.shape == (4, 30, 20)
        assert y.item() == 0  # -1 is encoded to 0

        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (4, 4, 30, 20)
        assert batch_y.shape == (4,)

    def test_deeplob_dataset(self):
        lob_data = np.random.randn(8, 40, 30).astype(np.float32)
        labels = np.zeros(8, dtype=int)

        ds = DeepLOBDataset(lob_data, labels)
        assert len(ds) == 8
        x, y = ds[0]
        assert x.shape == (1, 40, 30)

    def test_fusion_dataset(self):
        tensors = np.random.randn(6, 4, 20, 15).astype(np.float32)
        tabular = np.random.randn(6, 12).astype(np.float32)
        labels = np.array([1, 1, -1, -1, 0, 0])

        ds = FusionDataset(tensors, tabular, labels)
        assert len(ds) == 6
        x_tens, x_tab, y = ds[0]
        assert x_tens.shape == (4, 20, 15)
        assert x_tab.shape == (12,)


class TestOverfitSanityChecks:
    """MANDATORY ACCEPTANCE CHECKS: Each model must overfit a tiny subset."""

    def test_deeplob_overfit_tiny_subset(self):
        torch.manual_seed(42)
        model = DeepLOB(num_classes=3, num_levels=10, conv_filters=16, lstm_hidden=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # 4 samples
        x = torch.randn(4, 1, 40, 20)
        y = torch.tensor([0, 1, 2, 1], dtype=torch.long)

        initial_loss = None
        for step in range(50):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        # Loss must drop substantially
        assert final_loss < initial_loss * 0.4
        # Accuracy on tiny subset
        preds = torch.argmax(model(x), dim=1)
        assert torch.equal(preds, y)

    def test_heatmap_cnn_overfit_tiny_subset(self):
        torch.manual_seed(42)
        model = HeatmapCNN(in_channels=4, num_classes=3, num_filters=[16, 32, 64], fc_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(4, 4, 32, 20)
        y = torch.tensor([2, 0, 1, 0], dtype=torch.long)

        initial_loss = None
        for step in range(30):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss * 0.4
        preds = torch.argmax(model(x), dim=1)
        assert torch.equal(preds, y)

    def test_fusion_model_pooling_variants_and_overfit(self):
        torch.manual_seed(42)
        variants = ["band_gru", "gap_time", "full_gap"]

        x_tensor = torch.randn(4, 4, 32, 20)
        x_tab = torch.randn(4, 15)
        y = torch.tensor([1, 2, 0, 1], dtype=torch.long)
        criterion = nn.CrossEntropyLoss()

        for variant in variants:
            model = DualBranchFusion(
                tabular_in_dim=15,
                num_classes=3,
                spatial_dim=32,
                tabular_dim=32,
                fused_dim=64,
                num_bands=4,
                pooling_variant=variant,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            initial_loss = None
            for step in range(30):
                optimizer.zero_grad()
                logits = model(x_tensor, x_tab)
                loss = criterion(logits, y)
                if initial_loss is None:
                    initial_loss = loss.item()
                loss.backward()
                optimizer.step()

            final_loss = loss.item()
            assert final_loss < initial_loss * 0.5
            preds = torch.argmax(model(x_tensor, x_tab), dim=1)
            assert torch.equal(preds, y)

"""Tests for models/baselines.py — Model A tabular baselines and evaluation metrics."""

import numpy as np
import pytest

from models.baselines import (
    TabularBaselineModel,
    decode_labels,
    encode_labels,
    evaluate_classifier,
)


@pytest.fixture
def synthetic_data():
    """Synthetic tabular feature matrix and triple-barrier target."""
    np.random.seed(42)
    n_samples = 300
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    # Generate labels in [-1, 0, 1]
    raw_score = X[:, 0] * 1.5 - X[:, 1] * 1.0 + np.random.randn(n_samples) * 0.5
    y = np.zeros(n_samples, dtype=int)
    y[raw_score > 0.8] = 1
    y[raw_score < -0.8] = -1

    return X, y


class TestEncoding:
    """Test label encoding / decoding roundtrip."""

    def test_roundtrip(self):
        y = np.array([-1, 0, 1, 0, -1, 1])
        encoded = encode_labels(y)
        decoded = decode_labels(encoded)
        np.testing.assert_array_equal(y, decoded)
        assert set(np.unique(encoded)).issubset({0, 1, 2})


class TestLogisticRegressionBaseline:
    """Test Model A Logistic Regression baseline."""

    def test_fit_predict_proba(self, synthetic_data):
        X, y = synthetic_data
        model = TabularBaselineModel(model_type="logistic", seed=42)
        model.fit(X, y)

        assert model.is_fitted
        preds = model.predict(X)
        probs = model.predict_proba(X)

        assert len(preds) == len(y)
        assert set(np.unique(preds)).issubset({-1, 0, 1})
        assert probs.shape == (len(y), 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

        metrics = evaluate_classifier(y, preds, probs)
        assert "balanced_accuracy" in metrics
        assert "macro_f1" in metrics
        assert metrics["balanced_accuracy"] > 0.33  # Better than random guess


class TestXGBoostBaseline:
    """Test Model A XGBoost baseline."""

    def test_xgboost_fit_predict(self, synthetic_data):
        X, y = synthetic_data
        model = TabularBaselineModel(
            model_type="xgboost",
            params={"n_estimators": 50, "max_depth": 3},
            seed=42,
        )
        model.fit(X, y, feature_names=[f"feat_{i}" for i in range(10)])

        assert model.is_fitted
        preds = model.predict(X)
        probs = model.predict_proba(X)

        assert len(preds) == len(y)
        assert probs.shape == (len(y), 3)

        importances = model.get_feature_importances()
        assert len(importances) == 10
        # Feature 0 was strongest in synthetic generator
        assert max(importances, key=importances.get) in ("feat_0", "feat_1")

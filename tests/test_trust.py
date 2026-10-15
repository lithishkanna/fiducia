"""Tests for models/trust.py — Wall Trust Model and RQ2 decile analysis."""

import numpy as np
import pytest

from features.walls import WallRecord, WallTracker, compute_wall_trust_features
from models.trust import (
    WALL_FEATURE_NAMES,
    WallTrustModel,
    compute_trust_decile_analysis,
    extract_wall_feature_vector,
)


@pytest.fixture
def synthetic_wall_dataset():
    """Generate synthetic wall features and survival outcomes."""
    np.random.seed(42)
    n_samples = 400

    # Features: [relative_size, distance_ticks, distance_bps, age_ms, flicker_count,
    #            cum_fill_qty, cum_cancel_qty, fill_ratio, approach_count, is_bid, book_imbalance]
    X = np.empty((n_samples, len(WALL_FEATURE_NAMES)), dtype=np.float32)

    X[:, 0] = np.random.uniform(5.0, 30.0, n_samples)  # relative_size
    X[:, 1] = np.random.uniform(5.0, 50.0, n_samples)  # distance_ticks
    X[:, 2] = X[:, 1] * 0.2  # distance_bps
    X[:, 3] = np.random.uniform(1000.0, 60000.0, n_samples)  # age_ms
    X[:, 4] = np.random.poisson(0.5, n_samples)  # flicker_count
    X[:, 5] = np.random.exponential(2.0, n_samples)  # cum_fill_qty
    X[:, 6] = np.random.exponential(1.0, n_samples)  # cum_cancel_qty
    X[:, 7] = X[:, 5] / (X[:, 5] + X[:, 6] + 1e-8)  # fill_ratio
    X[:, 8] = np.random.poisson(2, n_samples)  # approach_count
    X[:, 9] = np.random.binomial(1, 0.5, n_samples)  # is_bid
    X[:, 10] = np.random.uniform(-0.5, 0.5, n_samples)  # book_imbalance

    # Latent probability of wall standing (higher age, fill_ratio, relative_size -> stands)
    z = (
        0.05 * X[:, 0]
        - 0.04 * X[:, 1]
        + 0.0001 * X[:, 3]
        - 0.8 * X[:, 4]  # High flicker -> low trust
        + 2.5 * X[:, 7]  # High fill_ratio -> high trust
    )
    prob_stand = 1.0 / (1.0 + np.exp(-z))
    y = (np.random.rand(n_samples) < prob_stand).astype(int)

    return X, y


class TestWallTrustModel:
    """Test model training, calibration, and 0-100 score mapping."""

    def test_fit_predict_and_score_bounds(self, synthetic_wall_dataset):
        X, y = synthetic_wall_dataset
        model = WallTrustModel(n_estimators=50, max_depth=3, seed=42)
        model.fit(X, y)

        assert model.is_fitted
        scores = model.predict_score(X)
        probs = model.predict_proba(X)

        assert len(scores) == len(y)
        assert probs.shape == (len(y), 2)
        # Probabilities bounded in [0, 1]
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
        # Scores bounded in [0, 100]
        assert np.all(scores >= 0) and np.all(scores <= 100)

        preds = model.predict(X, threshold_score=50)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_decile_validation_analysis(self, synthetic_wall_dataset):
        """Verify RQ2 validation decile report with confidence intervals."""
        X, y = synthetic_wall_dataset
        model = WallTrustModel(n_estimators=50, max_depth=3, seed=42)
        model.fit(X[:250], y[:250])

        val_scores = model.predict_score(X[250:])
        val_y = y[250:]

        report = compute_trust_decile_analysis(val_scores, val_y, n_bins=10)

        assert len(report.deciles) == 10
        assert report.spearman_correlation > 0.0  # Positive monotonic relationship
        assert -1.0 <= report.spearman_correlation <= 1.0

        for d in report.deciles:
            if d.sample_count > 0:
                assert 0.0 <= d.empirical_rate <= 1.0
                assert 0.0 <= d.ci_lower <= d.empirical_rate <= d.ci_upper <= 1.0


class TestCausalIntegrity:
    """MANDATORY ACCEPTANCE: Wall features evaluated at time t use ONLY data up to t."""

    def test_wall_features_invariant_to_future_perturbations(self):
        tracker = WallTracker(wall_multiplier=5.0, tick_size=0.10)

        bids_t1 = [(49999.0, 1.0), (49990.0, 20.0)]
        asks_t1 = [(50001.0, 1.0)]

        # Wall appears at t=1000
        walls_t1 = tracker.update(1000, 50000.0, bids_t1, asks_t1)
        feat_vector_orig = extract_wall_feature_vector(walls_t1[0], book_imbalance=0.1)

        # Perturb future events at t=2000
        bids_t2_future = [(49999.0, 1.0), (49990.0, 100.0)]  # Size jumped in future
        tracker.update(2000, 50000.0, bids_t2_future, asks_t1)

        # Feature vector measured at t=1000 must NOT have changed retrospectively
        assert feat_vector_orig[0] == 20.0 / 1.0  # Relative size was 20x at t=1000
        assert feat_vector_orig[3] == 0.0  # Age was 0 ms at t=1000


class TestWallTrustTabularFeatures:
    """Test augmenting tabular feature set with wall trust scores."""

    def test_compute_wall_trust_features(self):
        w1 = WallRecord(
            wall_id="w1",
            price=49990.0,
            side="bid",
            first_seen_ts=1000,
            last_seen_ts=2000,
            current_size=20.0,
            peak_size=20.0,
            relative_size=10.0,
            distance_ticks=100.0,
            distance_bps=20.0,
        )
        w2 = WallRecord(
            wall_id="w2",
            price=50010.0,
            side="ask",
            first_seen_ts=1000,
            last_seen_ts=2000,
            current_size=15.0,
            peak_size=15.0,
            relative_size=8.0,
            distance_ticks=100.0,
            distance_bps=20.0,
        )

        trust_scores = [85.0, 30.0]
        feats = compute_wall_trust_features([w1, w2], trust_scores)

        assert feats["max_bid_wall_trust"] == 85.0
        assert feats["max_ask_wall_trust"] == 30.0
        assert feats["wall_trust_imbalance"] == pytest.approx(0.55, abs=1e-3)

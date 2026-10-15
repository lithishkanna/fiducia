"""Wall trust model: calibrated gradient-boosted classifier producing a 0-100 trust score.

Predicts whether a large liquidity wall is trustworthy (remains standing / traded into)
versus fleeting (cancelled / pulled when price approaches).

Outputs:
- Calibrated probability P(trustworthy)
- Trust score mapped to integer range [0, 100]
- Decile outcome rate analysis answering RQ2 with confidence intervals.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

from features.walls import WallRecord

logger = logging.getLogger(__name__)

WALL_FEATURE_NAMES = [
    "relative_size",
    "distance_ticks",
    "distance_bps",
    "age_ms",
    "flicker_count",
    "cum_fill_qty",
    "cum_cancel_qty",
    "fill_ratio",
    "approach_count",
    "is_bid",
    "book_imbalance",
]


def extract_wall_feature_vector(
    wall: WallRecord,
    book_imbalance: float = 0.0,
) -> np.ndarray:
    """Extract strictly historical feature vector for a single wall record."""
    return np.array(
        [
            float(wall.relative_size),
            float(wall.distance_ticks),
            float(wall.distance_bps),
            float(wall.age_ms),
            float(wall.flicker_count),
            float(wall.cum_fill_qty),
            float(wall.cum_cancel_qty),
            float(wall.fill_ratio),
            float(wall.approach_count),
            1.0 if wall.side == "bid" else 0.0,
            float(book_imbalance),
        ],
        dtype=np.float32,
    )


def label_wall_outcome(
    min_dist_seen_ticks: float,
    remaining_size_ratio: float,
    fill_ratio: float,
    proximity_ticks: float = 10.0,
) -> int:
    """Determine ground-truth wall outcome label.

    Returns:
        1 (Trustworthy): wall stood (>=50% size) or had real fills (>=20%) when reached.
        0 (Low-Trust): wall pulled / cancelled before or upon approach.
    """
    # Did price come within proximity window?
    approached = min_dist_seen_ticks <= proximity_ticks

    if approached:
        if remaining_size_ratio >= 0.50 or fill_ratio >= 0.20:
            return 1
        return 0
    else:
        # If price never came within proximity, classified by whether it persisted
        return 1 if remaining_size_ratio >= 0.50 else 0


@dataclass
class DecileBinResult:
    """Statistics for a single trust score decile bin."""

    decile: int
    score_range: Tuple[float, float]
    sample_count: int
    standing_count: int
    empirical_rate: float
    ci_lower: float
    ci_upper: float


@dataclass
class TrustDecileReport:
    """Complete validation experiment report answering RQ2."""

    deciles: List[DecileBinResult]
    spearman_correlation: float
    spearman_pvalue: float
    monotonic: bool
    summary: str


def compute_trust_decile_analysis(
    trust_scores: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
    confidence_level: float = 0.95,
) -> TrustDecileReport:
    """Compute empirical outcome rates across trust deciles with Wilson score CIs.

    Answers RQ2: do low-trust walls get cancelled before being reached more often
    than high-trust walls?
    """
    n = len(trust_scores)
    if n == 0:
        return TrustDecileReport(
            deciles=[],
            spearman_correlation=0.0,
            spearman_pvalue=1.0,
            monotonic=False,
            summary="No samples provided for decile analysis.",
        )

    z = stats.norm.ppf(1.0 - (1.0 - confidence_level) / 2.0)
    bin_edges = np.linspace(0.0, 100.0, n_bins + 1)
    decile_results: List[DecileBinResult] = []

    empirical_rates = []

    for d in range(n_bins):
        low, high = bin_edges[d], bin_edges[d + 1]
        upper_cond = trust_scores <= high if d == n_bins - 1 else trust_scores < high
        mask = (trust_scores >= low) & upper_cond

        count = int(np.sum(mask))
        standing = int(np.sum(y_true[mask])) if count > 0 else 0
        rate = standing / count if count > 0 else 0.0
        empirical_rates.append(rate)

        # Wilson score interval for binomial proportion
        if count > 0:
            p_hat = rate
            denom = 1.0 + (z**2) / count
            center = (p_hat + (z**2) / (2 * count)) / denom
            margin = z * np.sqrt((p_hat * (1.0 - p_hat) + (z**2) / (4 * count)) / count) / denom
            ci_low = max(0.0, center - margin)
            ci_high = min(1.0, center + margin)
        else:
            ci_low, ci_high = 0.0, 0.0

        decile_results.append(
            DecileBinResult(
                decile=d + 1,
                score_range=(low, high),
                sample_count=count,
                standing_count=standing,
                empirical_rate=rate,
                ci_lower=ci_low,
                ci_upper=ci_high,
            )
        )

    # Compute rank correlation
    active_rates = [r.empirical_rate for r in decile_results if r.sample_count > 0]
    active_ranks = list(range(len(active_rates)))

    if len(active_rates) > 2:
        corr_res = stats.spearmanr(active_ranks, active_rates)
        spearman_corr = (
            float(corr_res.statistic) if hasattr(corr_res, "statistic") else float(corr_res[0])
        )
        spearman_pval = (
            float(corr_res.pvalue) if hasattr(corr_res, "pvalue") else float(corr_res[1])
        )
    else:
        spearman_corr, spearman_pval = 0.0, 1.0

    is_monotonic = spearman_corr > 0.80 and spearman_pval < 0.05

    summary = f"RQ2 Support: Spearman rho={spearman_corr:.3f} (p={spearman_pval:.4e}). " + (
        "High trust scores reliably exhibit higher survival/fill rates."
        if spearman_corr > 0.5
        else "Weak or non-monotonic relationship between trust score and persistence."
    )

    return TrustDecileReport(
        deciles=decile_results,
        spearman_correlation=spearman_corr,
        spearman_pvalue=spearman_pval,
        monotonic=is_monotonic,
        summary=summary,
    )


class WallTrustModel:
    """Calibrated model predicting liquidity wall trust score (0 to 100)."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        calibration_method: str = "sigmoid",
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.calibration_method = calibration_method
        self.seed = seed

        # Base estimator
        base_clf = HistGradientBoostingClassifier(
            max_iter=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.seed,
        )

        # Calibrated classifier wrapping base
        self.model = CalibratedClassifierCV(
            estimator=base_clf,
            method=self.calibration_method,
            cv=3,
        )
        self.is_fitted = False
        self.feature_names = WALL_FEATURE_NAMES

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WallTrustModel":
        """Fit calibrated classifier on wall features X and outcomes y in {0, 1}."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities [P(pull), P(stand)]."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        return self.model.predict_proba(X)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated trust scores in [0, 100]."""
        probs = self.predict_proba(X)
        standing_prob = probs[:, 1]
        scores = np.clip(np.round(standing_prob * 100.0), 0.0, 100.0)
        return scores.astype(np.int32)

    def predict(self, X: np.ndarray, threshold_score: int = 50) -> np.ndarray:
        """Binary prediction: 1 if score >= threshold_score else 0."""
        scores = self.predict_score(X)
        return (scores >= threshold_score).astype(np.int32)

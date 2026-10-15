"""Probability calibration via Temperature Scaling, reliability diagrams, and ECE.

Optimizes temperature T on validation set logits exclusively.
Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Metrics and bin statistics from reliability analysis."""

    ece: float
    mce: float
    nll: float
    brier: float
    bin_confs: np.ndarray
    bin_accs: np.ndarray
    bin_counts: np.ndarray


def compute_calibration_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> CalibrationResult:
    """Calculate ECE, MCE, NLL, Brier score, and reliability diagram bins.

    Args:
        probs: Array of predicted probabilities of shape (N, K).
        y_true: True class indices in [0, K-1].
        n_bins: Number of equal-width confidence bins in [0, 1].
    """
    n, num_classes = probs.shape
    y_true = np.asarray(y_true, dtype=int)

    # Top-1 predicted class and confidence
    pred_classes = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    corrects = (pred_classes == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs = np.zeros(n_bins, dtype=np.float64)
    bin_accs = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    ece = 0.0
    mce = 0.0

    for m in range(n_bins):
        low, high = bin_boundaries[m], bin_boundaries[m + 1]
        mask = (confs >= low) & (confs <= high if m == n_bins - 1 else confs < high)
        count = int(np.sum(mask))
        bin_counts[m] = count

        if count > 0:
            avg_conf = float(np.mean(confs[mask]))
            avg_acc = float(np.mean(corrects[mask]))
            bin_confs[m] = avg_conf
            bin_accs[m] = avg_acc

            diff = abs(avg_acc - avg_conf)
            ece += (count / n) * diff
            mce = max(mce, diff)

    # Negative log-likelihood (NLL)
    eps = 1e-12
    p_true = np.clip(probs[np.arange(n), y_true], eps, 1.0)
    nll = float(-np.mean(np.log(p_true)))

    # Multiclass Brier score
    y_one_hot = np.zeros_like(probs)
    y_one_hot[np.arange(n), y_true] = 1.0
    brier = float(np.mean(np.sum((probs - y_one_hot) ** 2, axis=1)))

    return CalibrationResult(
        ece=float(ece),
        mce=float(mce),
        nll=nll,
        brier=brier,
        bin_confs=bin_confs,
        bin_accs=bin_accs,
        bin_counts=bin_counts,
    )


class TemperatureScaler:
    """Post-processing calibration fitting scalar temperature T on validation logits."""

    def __init__(self, eps: float = 1e-4) -> None:
        self.eps = eps
        self.temperature: float = 1.0
        self.is_fitted: bool = False

    def fit(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
    ) -> "TemperatureScaler":
        """Optimize temperature T > 0 on validation logits by minimizing NLL.

        Args:
            logits: Uncalibrated logit matrix (N, K).
            y_true: Ground truth class labels (N,) in [0, K-1].
        """
        n, num_classes = logits.shape
        y_true = np.asarray(y_true, dtype=int)

        def nll_objective(t: float) -> float:
            if t <= self.eps:
                return float("inf")
            scaled_logits = logits / t
            # log(softmax(scaled_logits)) = scaled_logits - logsumexp(scaled_logits)
            lse = logsumexp(scaled_logits, axis=1, keepdims=True)
            log_probs = scaled_logits - lse
            chosen_log_probs = log_probs[np.arange(n), y_true]
            return float(-np.mean(chosen_log_probs))

        res = minimize_scalar(nll_objective, bounds=(0.05, 10.0), method="bounded")
        self.temperature = float(res.x)
        self.is_fitted = True
        logger.info("Optimized temperature T = %.4f (NLL: %.4f)", self.temperature, res.fun)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply fitted temperature to scale logits and compute calibrated probabilities."""
        if not self.is_fitted:
            raise ValueError("TemperatureScaler must be fitted before transforming.")
        scaled_logits = logits / max(self.eps, self.temperature)
        return softmax(scaled_logits, axis=1)

    def fit_transform(self, logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit temperature and return calibrated probabilities."""
        self.fit(logits, y_true)
        return self.transform(logits)

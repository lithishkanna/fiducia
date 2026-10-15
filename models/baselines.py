"""Baseline models on tabular features: Logistic Regression and XGBoost (Model A).

Supports 3-class triple-barrier labels:
- -1: DOWN
-  0: NEUTRAL
- +1: UP

Internally mapped to [0, 1, 2] for model training, and mapped back for inference.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

LABEL_TO_IDX = {-1: 0, 0: 1, 1: 2}
IDX_TO_LABEL = {0: -1, 1: 0, 2: 1}


def encode_labels(y: np.ndarray) -> np.ndarray:
    """Map [-1, 0, 1] labels to [0, 1, 2]."""
    return np.array([LABEL_TO_IDX.get(int(val), 1) for val in y], dtype=int)


def decode_labels(y_idx: np.ndarray) -> np.ndarray:
    """Map [0, 1, 2] class indices back to [-1, 0, 1]."""
    return np.array([IDX_TO_LABEL.get(int(idx), 0) for idx in y_idx], dtype=int)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute comprehensive classification evaluation metrics."""
    # Ensure consistent encoding
    if set(np.unique(y_true)).issubset({-1, 0, 1}):
        y_true_enc = encode_labels(y_true)
        y_pred_enc = encode_labels(y_pred)
    else:
        y_true_enc = y_true
        y_pred_enc = y_pred

    acc = float(accuracy_score(y_true_enc, y_pred_enc))
    bal_acc = float(balanced_accuracy_score(y_true_enc, y_pred_enc))
    macro_f1 = float(f1_score(y_true_enc, y_pred_enc, average="macro", zero_division=0))

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
    }

    # Per-class F1
    per_class_f1 = f1_score(y_true_enc, y_pred_enc, average=None, labels=[0, 1, 2], zero_division=0)
    metrics["f1_down"] = float(per_class_f1[0])
    metrics["f1_neutral"] = float(per_class_f1[1])
    metrics["f1_up"] = float(per_class_f1[2])

    if y_prob is not None:
        try:
            # Ensure prob covers all 3 classes
            if y_prob.shape[1] == 3:
                metrics["log_loss"] = float(log_loss(y_true_enc, y_prob, labels=[0, 1, 2]))
                # Multiclass Brier score: mean of (p_ic - y_ic)^2
                y_one_hot = np.zeros_like(y_prob)
                for i, c in enumerate(y_true_enc):
                    y_one_hot[i, c] = 1.0
                brier = float(np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1)))
                metrics["brier_score"] = brier
        except Exception as e:
            logger.debug("Prob metric calculation note: %s", e)

    return metrics


class TabularBaselineModel:
    """Unified wrapper for Model A baselines (Logistic Regression or XGBoost)."""

    def __init__(
        self,
        model_type: str = "xgboost",
        params: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        self.model_type = model_type.lower()
        self.params = params or {}
        self.seed = seed
        self.model = self._build_model()
        self.is_fitted = False
        self.feature_names_: List[str] = []

    def _build_model(self):
        if self.model_type == "logistic":
            c_val = self.params.get("C", 1.0)
            max_iter = self.params.get("max_iter", 1000)
            class_weight = self.params.get("class_weight", "balanced")
            lr = LogisticRegression(
                C=c_val,
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=self.seed,
            )
            return Pipeline([("scaler", StandardScaler()), ("clf", lr)])

        elif self.model_type == "xgboost":
            if not HAS_XGBOOST:
                raise ImportError(
                    "xgboost is not installed. Please install xgboost or use model_type='logistic'."
                )
            n_est = self.params.get("n_estimators", 200)
            depth = self.params.get("max_depth", 5)
            lr = self.params.get("learning_rate", 0.05)
            subsample = self.params.get("subsample", 0.8)
            colsample = self.params.get("colsample_bytree", 0.8)

            return xgb.XGBClassifier(
                n_estimators=n_est,
                max_depth=depth,
                learning_rate=lr,
                subsample=subsample,
                colsample_bytree=colsample,
                objective="multi:softprob",
                num_class=3,
                random_state=self.seed,
                n_jobs=-1,
                eval_metric="mlogloss",
            )
        else:
            raise ValueError(
                f"Unknown model_type '{self.model_type}'. Choose 'logistic' or 'xgboost'."
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "TabularBaselineModel":
        """Fit model on tabular features X and targets y in [-1, 0, 1]."""
        y_enc = encode_labels(y)
        self.model.fit(X, y_enc)
        self.is_fitted = True
        self.feature_names_ = feature_names or [f"f_{i}" for i in range(X.shape[1])]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels in [-1, 0, 1]."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        y_enc_pred = self.model.predict(X)
        return decode_labels(y_enc_pred)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities: columns correspond to [DOWN (-1), NEUTRAL (0), UP (+1)]."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        return self.model.predict_proba(X)

    def get_feature_importances(self) -> Dict[str, float]:
        """Return feature importance or normalized absolute coefficients."""
        if not self.is_fitted or not self.feature_names_:
            return {}

        if self.model_type == "logistic":
            lr_step = self.model.named_steps["clf"]
            coefs = np.mean(np.abs(lr_step.coef_), axis=0)
            return dict(zip(self.feature_names_, coefs.tolist()))
        elif self.model_type == "xgboost":
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names_, importances.tolist()))
        return {}

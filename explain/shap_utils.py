"""SHAP-based feature attribution for tabular models and deep tabular branches.

Provides feature attribution for:
- Model A: Tabular baselines (XGBoost TreeExplainer and Logistic Regression Explainer)
- Model D: Tabular branch of the Dual-Branch Fusion architecture

All attributions are strictly interpreted as model diagnostics (local and global
feature sensitivities), not causal explanations of market phenomena.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import shap

logger = logging.getLogger(__name__)


@dataclass
class ShapExplanation:
    """Standardized container for SHAP attribution results."""

    values: np.ndarray  # Shape (N, n_features, n_classes) or (N, n_features)
    base_values: Optional[np.ndarray]  # Shape (n_classes,) or (N, n_classes) or scalar
    feature_names: List[str]
    data: Optional[np.ndarray] = None


def compute_tabular_shap(
    model: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    background_samples: int = 100,
) -> ShapExplanation:
    """Compute SHAP feature attributions for tabular models (Model A).

    Supports TabularBaselineModel (XGBoost / Logistic Regression) and raw estimators.

    Args:
        model: Trained TabularBaselineModel, XGBClassifier, or Pipeline.
        X: Feature matrix of shape (N, n_features).
        feature_names: Optional list of feature names.
        background_samples: Number of background samples for kernel/permutation explainers.

    Returns:
        ShapExplanation with standardized attribution arrays.
    """
    X_arr = np.asarray(X, dtype=np.float32)
    n_samples, n_features = X_arr.shape

    if feature_names is None:
        if hasattr(model, "feature_names_") and model.feature_names_:
            names = list(model.feature_names_)
        else:
            names = [f"feature_{i}" for i in range(n_features)]
    else:
        names = list(feature_names)

    # Determine underlying estimator
    underlying = getattr(model, "model", model)
    model_type = getattr(model, "model_type", None)

    # 1. XGBoost fast path via TreeExplainer
    is_xgb = False
    try:
        import xgboost as xgb

        if isinstance(underlying, (xgb.XGBClassifier, xgb.Booster)) or model_type == "xgboost":
            is_xgb = True
    except ImportError:
        pass

    if is_xgb:
        clf = underlying
        explainer = shap.TreeExplainer(clf)
        raw_values = explainer.shap_values(X_arr)
        base_vals = getattr(explainer, "expected_value", None)

        if isinstance(raw_values, list):
            # List of (N, n_features) arrays per class -> stack to (N, n_features, n_classes)
            values_arr = np.stack(raw_values, axis=-1)
        elif isinstance(raw_values, np.ndarray):
            values_arr = raw_values
            if values_arr.ndim == 2:
                values_arr = values_arr[:, :, np.newaxis]
        else:
            raise ValueError(f"Unexpected SHAP values type: {type(raw_values)}")

        return ShapExplanation(
            values=values_arr,
            base_values=np.asarray(base_vals) if base_vals is not None else None,
            feature_names=names,
            data=X_arr,
        )

    # 2. General / Logistic Pipeline path via standard shap.Explainer
    bg_size = min(background_samples, len(X_arr))
    bg_indices = np.linspace(0, len(X_arr) - 1, bg_size, dtype=int)
    background = X_arr[bg_indices]

    if hasattr(model, "predict_proba"):
        predict_fn = model.predict_proba
    elif hasattr(underlying, "predict_proba"):
        predict_fn = underlying.predict_proba
    else:
        predict_fn = underlying

    explainer = shap.Explainer(predict_fn, background)
    explanation = explainer(X_arr)

    values_arr = explanation.values
    base_vals = explanation.base_values

    if values_arr.ndim == 2:
        values_arr = values_arr[:, :, np.newaxis]

    return ShapExplanation(
        values=values_arr,
        base_values=np.asarray(base_vals) if base_vals is not None else None,
        feature_names=names,
        data=X_arr,
    )


def compute_tabular_branch_shap(
    model: Any,
    X_tabular: np.ndarray,
    feature_names: Optional[List[str]] = None,
    background_samples: int = 50,
) -> ShapExplanation:
    """Compute SHAP feature attributions for a deep neural network tabular branch (Model D).

    Args:
        model: PyTorch model with tabular_branch or callable taking tabular tensor.
        X_tabular: Tabular feature matrix of shape (N, n_features).
        feature_names: Optional feature names.
        background_samples: Reference samples used to estimate base rates.

    Returns:
        ShapExplanation container.
    """
    import torch

    X_arr = np.asarray(X_tabular, dtype=np.float32)
    n_samples, n_features = X_arr.shape

    if feature_names is None:
        names = [f"tabular_{i}" for i in range(n_features)]
    else:
        names = list(feature_names)

    # Define prediction wrapper
    def predict_wrapper(x_np: np.ndarray) -> np.ndarray:
        tensor_in = torch.tensor(x_np, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            if hasattr(model, "tabular_branch"):
                # Forward through tabular branch
                out = model.tabular_branch(tensor_in)
                if hasattr(model, "classifier") and hasattr(model, "fc_joint"):
                    # If full fusion model, simulate zero/mean spatial input
                    sp_dim = model.spatial_branch.spatial_dim
                    dummy_spatial = torch.zeros((tensor_in.size(0), sp_dim), dtype=torch.float32)
                    fused = torch.cat([dummy_spatial, out], dim=-1)
                    fused = torch.nn.functional.leaky_relu(model.fc_joint(fused), 0.01)
                    logits = model.classifier(fused)
                    probs = torch.softmax(logits, dim=-1)
                    return probs.cpu().numpy()
                else:
                    return out.cpu().numpy()
            elif callable(model):
                out = model(tensor_in)
                if isinstance(out, torch.Tensor):
                    return out.cpu().numpy()
                return np.asarray(out)
            else:
                raise ValueError("Model does not support forward evaluation for SHAP.")

    bg_size = min(background_samples, len(X_arr))
    bg_indices = np.linspace(0, len(X_arr) - 1, bg_size, dtype=int)
    background = X_arr[bg_indices]

    explainer = shap.Explainer(predict_wrapper, background)
    explanation = explainer(X_arr)

    values_arr = explanation.values
    if values_arr.ndim == 2:
        values_arr = values_arr[:, :, np.newaxis]

    return ShapExplanation(
        values=values_arr,
        base_values=np.asarray(explanation.base_values)
        if explanation.base_values is not None
        else None,
        feature_names=names,
        data=X_arr,
    )


def get_top_features_for_sample(
    shap_explanation: ShapExplanation,
    sample_idx: int,
    target_class: int = 2,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Extract top contributing features for a single sample and target class.

    Args:
        shap_explanation: Computed ShapExplanation.
        sample_idx: Index of sample in the explanation.
        target_class: Class index (0: DOWN, 1: NEUTRAL, 2: UP).
        top_k: Number of highest-magnitude features to return.

    Returns:
        List of dicts containing feature, value, shap_value, and impact direction.
    """
    if sample_idx >= len(shap_explanation.values):
        raise IndexError(f"sample_idx {sample_idx} out of range ({len(shap_explanation.values)})")

    # Shape: (n_features, n_classes) or (n_features,)
    sample_shap = shap_explanation.values[sample_idx]
    if sample_shap.ndim == 2:
        class_shap = sample_shap[:, target_class]
    else:
        class_shap = sample_shap

    sample_data = (
        shap_explanation.data[sample_idx]
        if shap_explanation.data is not None
        else np.zeros(len(class_shap))
    )

    # Sort by absolute magnitude descending
    ranked_indices = np.argsort(np.abs(class_shap))[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        s_val = float(class_shap[idx])
        d_val = float(sample_data[idx])
        name = shap_explanation.feature_names[idx]
        results.append(
            {
                "feature": name,
                "value": d_val,
                "shap_value": s_val,
                "impact": "positive" if s_val > 0 else "negative",
            }
        )

    return results


def compute_feature_importance_summary(
    shap_explanation: ShapExplanation,
    target_class: Optional[int] = None,
) -> Dict[str, float]:
    """Compute mean absolute SHAP value across all samples for each feature.

    Args:
        shap_explanation: Computed ShapExplanation.
        target_class: Specific class index, or None to average across all classes.

    Returns:
        Dictionary mapping feature name to mean absolute SHAP attribution, sorted descending.
    """
    vals = shap_explanation.values  # (N, n_features, n_classes) or (N, n_features)

    if vals.ndim == 3:
        if target_class is not None:
            mean_abs = np.mean(np.abs(vals[:, :, target_class]), axis=0)
        else:
            mean_abs = np.mean(np.mean(np.abs(vals), axis=-1), axis=0)
    else:
        mean_abs = np.mean(np.abs(vals), axis=0)

    importance_dict = {
        name: float(score) for name, score in zip(shap_explanation.feature_names, mean_abs)
    }

    # Sort descending
    return dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

"""Explainability: SHAP, Grad-CAM, and plain-language explanation templates."""

__version__ = "0.1.0"

from explain.gradcam import GradCAM, identify_salient_price_bands
from explain.shap_utils import (
    ShapExplanation,
    compute_feature_importance_summary,
    compute_tabular_branch_shap,
    compute_tabular_shap,
    get_top_features_for_sample,
)
from explain.templates import (
    STANDARD_DISCLAIMER,
    DecisionDiagnostic,
    build_decision_diagnostic,
    format_order_flow_explanation,
    format_wall_explanation,
)

__all__ = [
    "GradCAM",
    "identify_salient_price_bands",
    "ShapExplanation",
    "compute_tabular_shap",
    "compute_tabular_branch_shap",
    "get_top_features_for_sample",
    "compute_feature_importance_summary",
    "DecisionDiagnostic",
    "build_decision_diagnostic",
    "format_wall_explanation",
    "format_order_flow_explanation",
    "STANDARD_DISCLAIMER",
]

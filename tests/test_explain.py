"""Tests for Phase 8: Explainability (SHAP, Grad-CAM, and diagnostic templates)."""

import numpy as np
import pytest
import torch

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
from models.baselines import TabularBaselineModel
from models.fusion import DualBranchFusion, TabularBranch
from models.heatmap_cnn import HeatmapCNN


@pytest.fixture
def synthetic_tabular_data():
    np.random.seed(42)
    n_samples, n_features = 40, 6
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.choice([-1, 0, 1], size=n_samples)
    feature_names = [f"feat_{i}" for i in range(n_features)]
    return X, y, feature_names


def test_shap_tabular_xgboost(synthetic_tabular_data):
    X, y, feature_names = synthetic_tabular_data
    model = TabularBaselineModel(model_type="xgboost", params={"n_estimators": 5, "max_depth": 2})
    model.fit(X, y, feature_names=feature_names)

    explanation = compute_tabular_shap(model, X[:15], feature_names=feature_names)

    assert isinstance(explanation, ShapExplanation)
    assert explanation.values.shape == (15, 6, 3)
    assert explanation.feature_names == feature_names

    # Check sample attribution extraction
    top_feats = get_top_features_for_sample(explanation, sample_idx=0, target_class=2, top_k=3)
    assert len(top_feats) == 3
    for f in top_feats:
        assert "feature" in f
        assert "shap_value" in f
        assert "impact" in f

    # Check feature importance summary
    summary = compute_feature_importance_summary(explanation)
    assert len(summary) == 6
    scores = list(summary.values())
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_shap_tabular_logistic(synthetic_tabular_data):
    X, y, feature_names = synthetic_tabular_data
    model = TabularBaselineModel(model_type="logistic", params={"max_iter": 50})
    model.fit(X, y, feature_names=feature_names)

    explanation = compute_tabular_shap(
        model,
        X[:10],
        feature_names=feature_names,
        background_samples=10,
    )

    assert isinstance(explanation, ShapExplanation)
    assert explanation.values.shape[0] == 10
    assert explanation.values.shape[1] == 6


def test_shap_tabular_branch():
    torch.manual_seed(42)
    np.random.seed(42)
    branch = TabularBranch(in_dim=5, out_dim=3)
    X_tab = np.random.randn(20, 5).astype(np.float32)
    feature_names = [f"tab_{i}" for i in range(5)]

    explanation = compute_tabular_branch_shap(
        branch,
        X_tab[:10],
        feature_names=feature_names,
        background_samples=10,
    )

    assert isinstance(explanation, ShapExplanation)
    assert explanation.values.shape[0] == 10
    assert explanation.values.shape[1] == 5


def test_gradcam_heatmap_cnn():
    torch.manual_seed(42)
    model = HeatmapCNN(in_channels=4, num_classes=3)
    P, T = 40, 20
    single_tensor = torch.randn(4, P, T)
    batch_tensor = torch.randn(2, 4, P, T)

    with GradCAM(model) as gradcam:
        # Single sample CAM
        cam_single = gradcam.generate_cam(single_tensor, target_class=2)
        assert cam_single.shape == (P, T)
        assert 0.0 <= cam_single.min() <= cam_single.max() <= 1.0

        # Batch CAM
        cam_batch = gradcam.generate_cam(batch_tensor, target_class=0)
        assert cam_batch.shape == (2, P, T)
        assert 0.0 <= cam_batch.min() <= cam_batch.max() <= 1.0

        # Salient price bands
        bands = identify_salient_price_bands(cam_single, num_bands=4)
        assert len(bands) == 4
        for b in bands:
            assert "price_offset_ticks" in b
            assert "mean_saliency" in b
            assert b["side"] in ["bid", "ask", "mid"]


def test_gradcam_dual_branch_fusion():
    torch.manual_seed(42)
    model = DualBranchFusion(tabular_in_dim=6, num_classes=3)
    P, T = 30, 15
    x_tensor = torch.randn(2, 4, P, T)
    x_tabular = torch.randn(2, 6)

    with GradCAM(model, target_layer=model.spatial_branch.conv2) as gradcam:
        cam_fusion = gradcam.generate_cam((x_tensor, x_tabular), target_class=2)
        assert cam_fusion.shape == (2, P, T)
        assert 0.0 <= cam_fusion.min() <= cam_fusion.max() <= 1.0


def test_templates_formatting_and_safety():
    forbidden_terms = [
        "spoofing",
        "spoof",
        "manipulation",
        "manipulator",
        "guaranteed profit",
        "sure thing",
        "rigged",
    ]

    # 1. Low trust wall
    low_trust_expl = format_wall_explanation(
        price=65400.0,
        side="ask",
        trust_score=22.0,
        features={"age_ms": 1200, "flicker_count": 3, "fill_ratio": 0.02},
    )
    assert "Ask wall at 65400.00" in low_trust_expl
    assert "22/100" in low_trust_expl
    assert "unusual cancellation behavior" in low_trust_expl

    # 2. High trust wall
    high_trust_expl = format_wall_explanation(
        price=64800.0,
        side="bid",
        trust_score=88.0,
        features={"age_ms": 45000, "fill_ratio": 0.75},
    )
    assert "Bid wall at 64800.00" in high_trust_expl
    assert "88/100" in high_trust_expl
    assert "typical depth persistence" in high_trust_expl

    # 3. Order flow explanation
    of_expl = format_order_flow_explanation(ofi=1.2, depth_imbalance=0.35, trade_flow=5.5)
    assert "upward order-flow pressure" in of_expl

    # 4. Decision diagnostic for trade
    diagnostic_trade = build_decision_diagnostic(
        action=1,
        probabilities=np.array([0.15, 0.20, 0.65]),
        expected_value_bps=3.85,
        confidence_threshold=0.55,
        top_shap_features=[{"feature": "ofi_mean", "shap_value": 0.18}],
        wall_explanations=[high_trust_expl],
    )
    assert isinstance(diagnostic_trade, DecisionDiagnostic)
    assert diagnostic_trade.action_label == "TRADE LONG"
    assert diagnostic_trade.confidence == 0.65
    assert "TRADE LONG" in diagnostic_trade.summary_text
    assert diagnostic_trade.disclaimer == STANDARD_DISCLAIMER

    # 5. Decision diagnostic for abstain
    diagnostic_abstain = build_decision_diagnostic(
        action=0,
        probabilities=np.array([0.34, 0.33, 0.33]),
        expected_value_bps=-1.2,
        confidence_threshold=0.55,
    )
    assert diagnostic_abstain.action_label == "NO TRADE (ABSTAIN)"
    assert "abstain" in diagnostic_abstain.summary_text.lower()

    # Safety check across all generated texts
    all_texts = [
        low_trust_expl,
        high_trust_expl,
        of_expl,
        diagnostic_trade.summary_text,
        diagnostic_trade.disclaimer,
        diagnostic_abstain.summary_text,
    ]
    for text in all_texts:
        for term in forbidden_terms:
            assert term not in text.lower(), f"Forbidden term '{term}' found in text: {text}"

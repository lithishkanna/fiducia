"""Plain-language explanation templates for decision-support diagnostics.

All text generated here is framed as diagnostics of model behavior,
not causal explanations of market movements or market manipulation claims.
Uses 'unusual cancellation behavior' or 'fleeting depth' instead of 'spoofing' or 'manipulation'.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

STANDARD_DISCLAIMER = (
    "Note: Diagnostics reflect internal model feature sensitivities and probability estimates. "
    "They do not constitute financial advice, trade recommendations, or representations of "
    "exchange order integrity."
)


@dataclass
class DecisionDiagnostic:
    """Comprehensive diagnostic container for a model trade / abstain decision."""

    action: int  # -1 (SHORT), 0 (ABSTAIN / NO TRADE), 1 (LONG)
    action_label: str  # 'TRADE LONG', 'TRADE SHORT', or 'NO TRADE (ABSTAIN)'
    expected_value_bps: float
    confidence: float
    prob_down: float
    prob_neutral: float
    prob_up: float
    top_contributing_factors: List[Dict[str, Any]] = field(default_factory=list)
    wall_explanations: List[str] = field(default_factory=list)
    summary_text: str = ""
    disclaimer: str = STANDARD_DISCLAIMER


def format_wall_explanation(
    price: float,
    side: str,
    trust_score: float,
    features: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a plain-language diagnostic explanation for a wall's trust score.

    Args:
        price: Price level of the liquidity wall.
        side: 'bid' or 'ask'.
        trust_score: Trust score in [0, 100].
        features: Optional dictionary of wall characteristics (age_ms, fill_ratio, etc.).

    Returns:
        Safe, diagnostic text explaining the wall rating.
    """
    side_str = side.capitalize()
    score_clamped = max(0.0, min(100.0, float(trust_score)))

    if score_clamped < 35.0:
        diagnostic = (
            f"{side_str} wall at {price:.2f} has trust score {score_clamped:.0f}/100. "
            f"Model diagnostics indicate unusual cancellation behavior and low persistence "
            f"upon approach; associated {'support' if side.lower() == 'bid' else 'resistance'} "
            f"is discounted."
        )
    elif score_clamped > 65.0:
        diagnostic = (
            f"{side_str} wall at {price:.2f} has trust score {score_clamped:.0f}/100. "
            f"Model diagnostics indicate typical depth persistence and solid fill participation; "
            f"liquidity is evaluated as authentic resting depth."
        )
    else:
        diagnostic = (
            f"{side_str} wall at {price:.2f} has trust score {score_clamped:.0f}/100. "
            f"Model diagnostics indicate moderate persistence with balanced fill-to-cancel rates."
        )

    if features:
        extra_details = []
        if "age_ms" in features:
            age_sec = features["age_ms"] / 1000.0
            extra_details.append(f"age: {age_sec:.1f}s")
        if "flicker_count" in features and features["flicker_count"] > 1:
            extra_details.append(f"flicker events: {features['flicker_count']}")
        if "fill_ratio" in features:
            extra_details.append(f"fill ratio: {features['fill_ratio']:.1%}")

        if extra_details:
            diagnostic += f" ({', '.join(extra_details)})"

    return diagnostic


def format_order_flow_explanation(
    ofi: float,
    depth_imbalance: float,
    trade_flow: Optional[float] = None,
) -> str:
    """Format near-mid order flow diagnostics.

    Args:
        ofi: Multi-level Order Flow Imbalance.
        depth_imbalance: Top-N depth imbalance in [-1, 1].
        trade_flow: Optional signed trade volume.

    Returns:
        Order flow summary sentence.
    """
    pressure = "neutral"
    if ofi > 0.5 and depth_imbalance > 0.1:
        pressure = "upward order-flow pressure with predominant bid accumulation"
    elif ofi < -0.5 and depth_imbalance < -0.1:
        pressure = "downward order-flow pressure with predominant ask accumulation"
    elif ofi > 0:
        pressure = "mild net buying order flow"
    elif ofi < 0:
        pressure = "mild net selling order flow"

    text = (
        f"Order flow diagnostics reflect {pressure} "
        f"(OFI: {ofi:+.2f}, Depth Imbalance: {depth_imbalance:+.2f})"
    )
    if trade_flow is not None:
        text += f", aggressive trade volume: {trade_flow:+.2f} BTC"
    text += "."
    return text


def build_decision_diagnostic(
    action: int,
    probabilities: np.ndarray,
    expected_value_bps: float,
    confidence_threshold: float = 0.5,
    top_shap_features: Optional[List[Dict[str, Any]]] = None,
    wall_explanations: Optional[List[str]] = None,
    abstention_reason: Optional[str] = None,
) -> DecisionDiagnostic:
    """Build a comprehensive, human-readable DecisionDiagnostic.

    Args:
        action: Output action: -1 (SHORT), 0 (ABSTAIN), 1 (LONG).
        probabilities: Array or list of 3 probabilities [p_down, p_neutral, p_up].
        expected_value_bps: Net expected value in basis points.
        confidence_threshold: Hurdle used for abstention.
        top_shap_features: List of top contributing SHAP features.
        wall_explanations: Optional list of wall diagnostics.
        abstention_reason: Specific reason if model abstained.

    Returns:
        Structured DecisionDiagnostic instance.
    """
    probs = np.asarray(probabilities, dtype=float).ravel()
    p_down = float(probs[0])
    p_neutral = float(probs[1]) if len(probs) > 1 else 0.0
    p_up = float(probs[2]) if len(probs) > 2 else 0.0

    top_features = top_shap_features or []
    walls = wall_explanations or []

    if action == 1:
        label = "TRADE LONG"
        conf = p_up
        summary = (
            f"Decision: {label}. Calibrated confidence {conf:.1%} exceeds threshold "
            f"({confidence_threshold:.1%}) with positive expected value of "
            f"+{expected_value_bps:.2f} bps after transaction costs."
        )
    elif action == -1:
        label = "TRADE SHORT"
        conf = p_down
        summary = (
            f"Decision: {label}. Calibrated confidence {conf:.1%} exceeds threshold "
            f"({confidence_threshold:.1%}) with positive expected value of "
            f"+{expected_value_bps:.2f} bps after transaction costs."
        )
    else:
        label = "NO TRADE (ABSTAIN)"
        conf = max(p_down, p_neutral, p_up)
        reason = (
            abstention_reason
            or "expected value below transaction cost hurdle or confidence insufficient"
        )
        summary = (
            f"Decision: {label}. The model opted to abstain because {reason} "
            f"(EV: {expected_value_bps:+.2f} bps, max confidence: {conf:.1%})."
        )

    # Append top feature drivers
    if top_features:
        driver_strings = [
            f"{f['feature']} "
            f"({'+' if f.get('shap_value', 0) > 0 else ''}{f.get('shap_value', 0):.3f})"
            for f in top_features[:3]
        ]
        summary += f" Key model sensitivities: {', '.join(driver_strings)}."

    # Append wall notes
    if walls:
        summary += " " + " ".join(walls[:2])

    return DecisionDiagnostic(
        action=action,
        action_label=label,
        expected_value_bps=float(expected_value_bps),
        confidence=float(conf),
        prob_down=p_down,
        prob_neutral=p_neutral,
        prob_up=p_up,
        top_contributing_factors=top_features,
        wall_explanations=walls,
        summary_text=summary,
        disclaimer=STANDARD_DISCLAIMER,
    )

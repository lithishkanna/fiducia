"""Abstention policy via Expected-Value (EV) thresholding and risk-coverage curves.

Decision rule:
  EV_long  = P(up) * G_up - P(down) * L_down - costs
  EV_short = P(down) * G_down - P(up) * L_up - costs

Action is taken (+1 or -1) ONLY if EV >= min_ev AND confidence >= min_conf.
Otherwise, the model abstains (0, NO TRADE).

Thresholds are selected on validation splits only.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from eval.backtest import EventDrivenBacktester, FeeScenario
from models.baselines import decode_labels

logger = logging.getLogger(__name__)


@dataclass
class RiskCoverageResult:
    """Risk-coverage curve diagnostics."""

    coverages: np.ndarray
    selective_accuracies: np.ndarray
    selective_returns: np.ndarray
    confidence_thresholds: np.ndarray


def compute_expected_value(
    probs: np.ndarray,
    cost: float,
    gain: float = 0.0010,
    loss: float = 0.0010,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Expected Values for LONG and SHORT directions.

    Args:
        probs: Array of shape (N, 3) where columns are [DOWN, NEUTRAL, UP].
        cost: Roundtrip fee + slippage in decimal (e.g. 12 bps = 0.0012).
        gain: Target gain on barrier hit (decimal).
        loss: Expected loss on adverse barrier hit (decimal).

    Returns:
        (ev_long, ev_short) arrays of shape (N,).
    """
    p_down = probs[:, 0]
    p_up = probs[:, 2]

    ev_long = (p_up * gain) - (p_down * loss) - cost
    ev_short = (p_down * gain) - (p_up * loss) - cost
    return ev_long, ev_short


def apply_abstention_rule(
    probs: np.ndarray,
    ev_threshold: float,
    conf_threshold: float,
    cost: float,
    gain: float = 0.0010,
    loss: float = 0.0010,
) -> np.ndarray:
    """Generate trade signals in [-1, 0, 1] using the EV & confidence abstention rule.

    Returns:
        Array of integer signals in [-1, 0, 1].
    """
    n = len(probs)
    signals = np.zeros(n, dtype=int)

    ev_long, ev_short = compute_expected_value(probs, cost=cost, gain=gain, loss=loss)
    p_down = probs[:, 0]
    p_up = probs[:, 2]

    long_cond = (ev_long >= ev_threshold) & (p_up >= conf_threshold) & (ev_long > ev_short)
    short_cond = (ev_short >= ev_threshold) & (p_down >= conf_threshold) & (ev_short > ev_long)

    signals[long_cond] = 1
    signals[short_cond] = -1
    return signals


def tune_abstention_thresholds(
    val_probs: np.ndarray,
    val_mid_prices: np.ndarray,
    cost: float,
    conf_grid: Optional[List[float]] = None,
    ev_grid: Optional[List[float]] = None,
    gain: float = 0.0010,
    loss: float = 0.0010,
    min_trade_count: int = 5,
) -> Tuple[float, float, float]:
    """Select optimal (ev_threshold, conf_threshold) strictly on validation data.

    Returns:
        (best_ev_thresh, best_conf_thresh, best_pnl)
    """
    if conf_grid is None:
        conf_grid = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    if ev_grid is None:
        ev_grid = [0.0, 0.0001, 0.0002, 0.0004, 0.0006]

    best_pnl = -float("inf")
    best_ev = 0.0
    best_conf = 0.50

    # Simplified fast evaluation over validation returns
    n = len(val_probs)
    dummy_ts = np.arange(n, dtype=np.int64) * 500
    tester = EventDrivenBacktester()

    for c_th in conf_grid:
        for ev_th in ev_grid:
            sigs = apply_abstention_rule(
                val_probs,
                ev_threshold=ev_th,
                conf_threshold=c_th,
                cost=cost,
                gain=gain,
                loss=loss,
            )
            trade_mask = sigs != 0
            if np.sum(trade_mask) < min_trade_count:
                continue

            res = tester.run(dummy_ts, val_mid_prices, sigs)
            if res.net_pnl > best_pnl:
                best_pnl = res.net_pnl
                best_ev = ev_th
                best_conf = c_th

    logger.info(
        "Tuned validation abstention: ev_thresh=%.5f, conf_thresh=%.2f (PnL=%.2f)",
        best_ev,
        best_conf,
        best_pnl,
    )
    return best_ev, best_conf, best_pnl


def compute_risk_coverage_curve(
    probs: np.ndarray,
    y_true: np.ndarray,
    returns: np.ndarray,
    n_thresholds: int = 40,
) -> RiskCoverageResult:
    """Compute selective risk-coverage curves varying confidence threshold.

    Answers how selective accuracy and returns behave as the model abstains more.
    """
    y_true = np.asarray(y_true, dtype=int)
    max_confs = np.max(probs, axis=1)
    pred_classes = np.argmax(probs, axis=1)
    # Map predictions back to [-1, 0, 1] if needed
    decoded_preds = decode_labels(pred_classes)

    thresholds = np.linspace(0.34, 0.95, n_thresholds)
    coverages = []
    accuracies = []
    selective_returns = []

    n = len(probs)

    for th in thresholds:
        active_mask = max_confs >= th
        count = int(np.sum(active_mask))
        cov = count / n if n > 0 else 0.0
        coverages.append(cov)

        if count > 0:
            active_correct = np.sum(pred_classes[active_mask] == y_true[active_mask])
            acc = active_correct / count
            # Return realized when taking action
            ret = float(np.mean(returns[active_mask] * decoded_preds[active_mask]))
            accuracies.append(acc)
            selective_returns.append(ret)
        else:
            accuracies.append(1.0)
            selective_returns.append(0.0)

    return RiskCoverageResult(
        coverages=np.array(coverages),
        selective_accuracies=np.array(accuracies),
        selective_returns=np.array(selective_returns),
        confidence_thresholds=thresholds,
    )


def compare_with_and_without_abstention(
    timestamps_ms: np.ndarray,
    mid_prices: np.ndarray,
    probs: np.ndarray,
    fee_scenario: Optional[FeeScenario] = None,
    ev_thresh: float = 0.0001,
    conf_thresh: float = 0.50,
) -> Dict[str, Any]:
    """Compare backtest performance WITH abstention vs WITHOUT abstention (RQ3).

    Returns:
        Dict comparing Trade counts, Net PnL, Sharpe ratio, and Max Drawdown.
    """
    scenario = fee_scenario or FeeScenario()
    cost_dec = scenario.total_roundtrip_cost_bps * 1e-4
    tester = EventDrivenBacktester(fee_scenario=scenario)

    # 1. WITHOUT abstention: raw argmax signals
    raw_preds = decode_labels(np.argmax(probs, axis=1))
    res_no_abstain = tester.run(timestamps_ms, mid_prices, raw_preds)

    # 2. WITH abstention: EV and confidence rule
    abstain_sigs = apply_abstention_rule(
        probs,
        ev_threshold=ev_thresh,
        conf_threshold=conf_thresh,
        cost=cost_dec,
    )
    res_with_abstain = tester.run(timestamps_ms, mid_prices, abstain_sigs)

    report = {
        "scenario": scenario.name,
        "without_abstention": {
            "total_trades": res_no_abstain.total_trades,
            "net_pnl": res_no_abstain.net_pnl,
            "win_rate": res_no_abstain.win_rate,
            "sharpe_ratio": res_no_abstain.sharpe_ratio,
            "max_drawdown_pct": res_no_abstain.max_drawdown_pct,
        },
        "with_abstention": {
            "total_trades": res_with_abstain.total_trades,
            "net_pnl": res_with_abstain.net_pnl,
            "win_rate": res_with_abstain.win_rate,
            "sharpe_ratio": res_with_abstain.sharpe_ratio,
            "max_drawdown_pct": res_with_abstain.max_drawdown_pct,
            "abstention_rate": float(np.mean(abstain_sigs == 0)),
        },
        "rq3_pnl_improvement": res_with_abstain.net_pnl - res_no_abstain.net_pnl,
    }
    return report

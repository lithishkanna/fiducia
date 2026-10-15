"""Event-driven cost-aware backtester with fee scenarios, slippage, and gap handling.

Features:
- Maker and taker fee scenarios.
- Slippage / execution latency penalty per leg.
- Gap handling: absolutely NO trading during gaps; existing positions closed.
- Position limits (default 1.0 BTC unit).
- Detailed per-trade logging and portfolio equity curve.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeeScenario:
    """Transaction cost parameters."""

    name: str = "taker_taker"
    entry_fee_bps: float = 5.0  # 5.0 bps = 0.05%
    exit_fee_bps: float = 5.0
    slippage_bps: float = 1.0  # 1.0 bps execution penalty per leg

    @property
    def total_roundtrip_cost_bps(self) -> float:
        return self.entry_fee_bps + self.exit_fee_bps + (2.0 * self.slippage_bps)


@dataclass
class TradeRecord:
    """Detailed log for a single closed trade."""

    trade_id: int
    entry_ts: int
    exit_ts: int
    side: int  # +1 for LONG, -1 for SHORT
    size: float
    entry_price_raw: float
    entry_price_exec: float
    exit_price_raw: float
    exit_price_exec: float
    gross_pnl: float
    fee_cost: float
    slippage_cost: float
    net_pnl: float
    net_return: float
    holding_time_ms: int
    exit_reason: str


@dataclass
class BacktestResult:
    """Summary metrics and trade logs from a backtest run."""

    scenario: str
    initial_capital: float
    final_capital: float
    net_pnl: float
    gross_pnl: float
    total_fees: float
    total_slippage: float
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))


class EventDrivenBacktester:
    """Event-driven simulator evaluating trading signals with execution costs."""

    def __init__(
        self,
        fee_scenario: Optional[FeeScenario] = None,
        position_limit: float = 1.0,
        initial_capital: float = 10000.0,
    ) -> None:
        self.fee_scenario = fee_scenario or FeeScenario()
        self.position_limit = position_limit
        self.initial_capital = initial_capital

    def run(
        self,
        timestamps_ms: np.ndarray,
        mid_prices: np.ndarray,
        signals: np.ndarray,
        gap_flags: Optional[np.ndarray] = None,
    ) -> BacktestResult:
        """Execute backtest over time series.

        Args:
            timestamps_ms: Array of timestamps in milliseconds.
            mid_prices: Array of mid prices.
            signals: Array of target signals in [-1, 0, 1].
            gap_flags: Array of booleans indicating stream gap intervals.

        Returns:
            BacktestResult with trade-by-trade analytics and equity curve.
        """
        n = len(mid_prices)
        if gap_flags is None:
            gap_flags = np.zeros(n, dtype=bool)

        current_position = 0.0  # +1.0 for LONG, -1.0 for SHORT, 0.0 for FLAT
        entry_price_raw = 0.0
        entry_price_exec = 0.0
        entry_ts = 0
        trade_id = 0

        trades: List[TradeRecord] = []
        equity = self.initial_capital
        equity_curve = np.zeros(n, dtype=np.float64)

        slip_factor = self.fee_scenario.slippage_bps * 1e-4
        entry_fee_factor = self.fee_scenario.entry_fee_bps * 1e-4
        exit_fee_factor = self.fee_scenario.exit_fee_bps * 1e-4

        for i in range(n):
            ts = int(timestamps_ms[i])
            price = float(mid_prices[i])
            sig = int(signals[i])
            in_gap = bool(gap_flags[i])

            # 1. Gap Handling: Force close any active position, reject new trades
            if in_gap:
                if current_position != 0.0:
                    trade_id += 1
                    exit_raw = price
                    # Adverse slippage on exit
                    exit_exec = (
                        exit_raw * (1.0 - slip_factor)
                        if current_position > 0
                        else exit_raw * (1.0 + slip_factor)
                    )

                    gross_pnl = current_position * (exit_raw - entry_price_raw)
                    fee = (
                        (entry_price_raw * entry_fee_factor) + (exit_raw * exit_fee_factor)
                    ) * abs(current_position)
                    slip = (
                        abs(entry_price_exec - entry_price_raw) + abs(exit_exec - exit_raw)
                    ) * abs(current_position)
                    net_pnl = gross_pnl - fee - slip

                    trades.append(
                        TradeRecord(
                            trade_id=trade_id,
                            entry_ts=entry_ts,
                            exit_ts=ts,
                            side=int(np.sign(current_position)),
                            size=abs(current_position),
                            entry_price_raw=entry_price_raw,
                            entry_price_exec=entry_price_exec,
                            exit_price_raw=exit_raw,
                            exit_price_exec=exit_exec,
                            gross_pnl=gross_pnl,
                            fee_cost=fee,
                            slippage_cost=slip,
                            net_pnl=net_pnl,
                            net_return=net_pnl / (entry_price_raw * abs(current_position)),
                            holding_time_ms=ts - entry_ts,
                            exit_reason="gap",
                        )
                    )
                    equity += net_pnl
                    current_position = 0.0

                equity_curve[i] = equity
                continue

            # 2. Position Transitions
            target_pos = float(sig) * self.position_limit  # +1.0, -1.0, or 0.0

            if target_pos != current_position:
                # Close existing position if any
                if current_position != 0.0:
                    trade_id += 1
                    exit_raw = price
                    exit_exec = (
                        exit_raw * (1.0 - slip_factor)
                        if current_position > 0
                        else exit_raw * (1.0 + slip_factor)
                    )

                    gross_pnl = current_position * (exit_raw - entry_price_raw)
                    fee = (
                        (entry_price_raw * entry_fee_factor) + (exit_raw * exit_fee_factor)
                    ) * abs(current_position)
                    slip = (
                        abs(entry_price_exec - entry_price_raw) + abs(exit_exec - exit_raw)
                    ) * abs(current_position)
                    net_pnl = gross_pnl - fee - slip

                    trades.append(
                        TradeRecord(
                            trade_id=trade_id,
                            entry_ts=entry_ts,
                            exit_ts=ts,
                            side=int(np.sign(current_position)),
                            size=abs(current_position),
                            entry_price_raw=entry_price_raw,
                            entry_price_exec=entry_price_exec,
                            exit_price_raw=exit_raw,
                            exit_price_exec=exit_exec,
                            gross_pnl=gross_pnl,
                            fee_cost=fee,
                            slippage_cost=slip,
                            net_pnl=net_pnl,
                            net_return=net_pnl / (entry_price_raw * abs(current_position)),
                            holding_time_ms=ts - entry_ts,
                            exit_reason="signal" if target_pos == 0.0 else "reverse",
                        )
                    )
                    equity += net_pnl
                    current_position = 0.0

                # Open new position if target is non-zero
                if target_pos != 0.0:
                    current_position = target_pos
                    entry_ts = ts
                    entry_price_raw = price
                    entry_price_exec = (
                        entry_price_raw * (1.0 + slip_factor)
                        if current_position > 0
                        else entry_price_raw * (1.0 - slip_factor)
                    )

            # Mark to market equity
            unrealized = (
                current_position * (price - entry_price_raw) if current_position != 0.0 else 0.0
            )
            equity_curve[i] = equity + unrealized

        # Close position at end of simulation
        if current_position != 0.0:
            trade_id += 1
            exit_raw = float(mid_prices[-1])
            exit_exec = (
                exit_raw * (1.0 - slip_factor)
                if current_position > 0
                else exit_raw * (1.0 + slip_factor)
            )
            gross_pnl = current_position * (exit_raw - entry_price_raw)
            fee = ((entry_price_raw * entry_fee_factor) + (exit_raw * exit_fee_factor)) * abs(
                current_position
            )
            slip = (abs(entry_price_exec - entry_price_raw) + abs(exit_exec - exit_raw)) * abs(
                current_position
            )
            net_pnl = gross_pnl - fee - slip

            trades.append(
                TradeRecord(
                    trade_id=trade_id,
                    entry_ts=entry_ts,
                    exit_ts=int(timestamps_ms[-1]),
                    side=int(np.sign(current_position)),
                    size=abs(current_position),
                    entry_price_raw=entry_price_raw,
                    entry_price_exec=entry_price_exec,
                    exit_price_raw=exit_raw,
                    exit_price_exec=exit_exec,
                    gross_pnl=gross_pnl,
                    fee_cost=fee,
                    slippage_cost=slip,
                    net_pnl=net_pnl,
                    net_return=net_pnl / (entry_price_raw * abs(current_position)),
                    holding_time_ms=int(timestamps_ms[-1]) - entry_ts,
                    exit_reason="end_of_data",
                )
            )
            equity += net_pnl
            equity_curve[-1] = equity

        # Calculate summary statistics
        tot_trades = len(trades)
        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl < 0]
        win_rate = len(wins) / tot_trades if tot_trades > 0 else 0.0

        gross_win_sum = sum(t.gross_pnl for t in wins)
        gross_loss_sum = abs(sum(t.gross_pnl for t in losses))
        profit_factor = (
            gross_win_sum / gross_loss_sum
            if gross_loss_sum > 0
            else (10.0 if gross_win_sum > 0 else 0.0)
        )

        total_fees = sum(t.fee_cost for t in trades)
        total_slippage = sum(t.slippage_cost for t in trades)
        tot_gross_pnl = sum(t.gross_pnl for t in trades)
        tot_net_pnl = sum(t.net_pnl for t in trades)

        # Sharpe ratio on periodic returns
        returns = np.diff(equity_curve) / np.maximum(1e-8, equity_curve[:-1])
        if len(returns) > 1 and np.std(returns) > 1e-9:
            # Assuming ~500ms steps, annualize: sqrt(2 * 3600 * 24 * 365) ~ 7927
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(7927.0))
        else:
            sharpe = 0.0

        # Max drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / np.maximum(1e-8, running_max)
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        return BacktestResult(
            scenario=self.fee_scenario.name,
            initial_capital=self.initial_capital,
            final_capital=equity,
            net_pnl=tot_net_pnl,
            gross_pnl=tot_gross_pnl,
            total_fees=total_fees,
            total_slippage=total_slippage,
            total_trades=tot_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd * 100.0,
            trades=trades,
            equity_curve=equity_curve,
        )

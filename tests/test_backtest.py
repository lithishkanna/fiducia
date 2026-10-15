"""Tests for eval/backtest.py — event-driven cost-aware backtester and sanity checks."""

import numpy as np
import pytest

from eval.backtest import EventDrivenBacktester, FeeScenario


@pytest.fixture
def flat_market():
    """Flat market price series for sanity testing cost mechanics."""
    n = 1000
    timestamps = np.array([1000 + i * 500 for i in range(n)], dtype=np.int64)
    prices = np.full(n, 50000.0)
    return timestamps, prices


class TestBacktesterSanityChecks:
    """MANDATORY ACCEPTANCE CHECKS: verify backtester behavior on edge cases."""

    def test_always_neutral_produces_zero_trades(self, flat_market):
        """Sanity check: Strategy predicting NEUTRAL (0) must execute 0 trades."""
        timestamps, prices = flat_market
        signals = np.zeros(len(prices), dtype=int)  # Always NEUTRAL (0)

        tester = EventDrivenBacktester()
        res = tester.run(timestamps, prices, signals)

        assert res.total_trades == 0
        assert len(res.trades) == 0
        assert res.net_pnl == 0.0
        assert res.gross_pnl == 0.0
        assert res.total_fees == 0.0
        assert res.total_slippage == 0.0
        assert res.final_capital == res.initial_capital

    def test_random_strategy_loses_cost_of_trades(self, flat_market):
        """Sanity check: Random trades on flat price lose roughly the cost of trades."""
        timestamps, prices = flat_market
        np.random.seed(42)

        # Flip between -1, 0, 1 randomly
        signals = np.random.choice([-1, 0, 1], size=len(prices), p=[0.3, 0.4, 0.3])

        fee_scenario = FeeScenario(
            entry_fee_bps=5.0,
            exit_fee_bps=5.0,
            slippage_bps=1.0,
        )
        tester = EventDrivenBacktester(fee_scenario=fee_scenario, position_limit=1.0)
        res = tester.run(timestamps, prices, signals)

        assert res.total_trades > 20
        # Since price is completely flat, gross PnL is exactly 0
        assert pytest.approx(res.gross_pnl, abs=1e-5) == 0.0
        # Total fees and slippage must be positive
        assert res.total_fees > 0.0
        assert res.total_slippage > 0.0
        # Net PnL must be strictly negative, exactly equal to -(fees + slippage)
        assert res.net_pnl < 0.0
        assert pytest.approx(res.net_pnl, abs=1e-5) == -(res.total_fees + res.total_slippage)

    def test_no_trading_during_gaps_and_position_closure(self):
        """Verify that positions are closed when entering a gap and no new trades open."""
        n = 100
        timestamps = np.array([1000 + i * 500 for i in range(n)], dtype=np.int64)
        prices = np.full(n, 50000.0)

        # Signal says LONG for first 50 steps
        signals = np.ones(n, dtype=int)

        # Gap interval at steps 20 to 40
        gap_flags = np.zeros(n, dtype=bool)
        gap_flags[20:40] = True

        tester = EventDrivenBacktester()
        res = tester.run(timestamps, prices, signals, gap_flags=gap_flags)

        # Check trades: trade 1 was closed at step 20 due to gap
        gap_closes = [t for t in res.trades if t.exit_reason == "gap"]
        assert len(gap_closes) >= 1
        assert gap_closes[0].exit_ts == timestamps[20]

        # Ensure no new trades were entered during the gap [20:40]
        for t in res.trades:
            assert not (timestamps[20] < t.entry_ts < timestamps[40])

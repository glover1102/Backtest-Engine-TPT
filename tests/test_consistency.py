"""
Tests for backtest_engine.consistency

Covers:
* Net P/L calculation from per-day P/L.
* Highest profit day identification.
* Consistency percentage: highest_day / net_pl.
* PASS when consistency_pct < 50%.
* FAIL when consistency_pct ≥ 50%, updated_profit_goal = net_pl × 2.
* Minimum trading days check (5 days).
* Days-to-target calculation.
* Edge cases: all losing days, zero net PnL.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from backtest_engine.consistency import (
    ConsistencyResult,
    compute_consistency,
    profit_target_reached_after_days,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_trades(day_pnls: dict[date, float]) -> pd.DataFrame:
    """Create a minimal trades DataFrame with one trade per day."""
    rows = []
    for d, pnl in sorted(day_pnls.items()):
        rows.append(
            {
                "trade_number": 1,
                "symbol": "TEST",
                "entry_time": pd.Timestamp(f"{d} 09:00", tz="US/Eastern"),
                "exit_time": pd.Timestamp(f"{d} 10:00", tz="US/Eastern"),
                "tpt_trading_day": d,
                "net_pnl": pnl,
                "effective_pnl": pnl,
                "size_qty": 1.0,
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Basic calculations
# ─────────────────────────────────────────────────────────────────────────────

class TestBasicCalculations:
    def test_net_pl_sum(self):
        trades = _make_trades(
            {
                date(2026, 4, 6): 1_000.0,
                date(2026, 4, 7): 2_000.0,
                date(2026, 4, 8): -500.0,
            }
        )
        result = compute_consistency(trades)
        assert abs(result.net_pl - 2_500.0) < 0.01

    def test_highest_profit_day(self):
        trades = _make_trades(
            {
                date(2026, 4, 6): 1_000.0,
                date(2026, 4, 7): 5_000.0,
                date(2026, 4, 8): 500.0,
            }
        )
        result = compute_consistency(trades)
        assert abs(result.highest_profit_day - 5_000.0) < 0.01
        assert result.highest_profit_day_date == date(2026, 4, 7)

    def test_trading_day_count(self):
        trades = _make_trades(
            {
                date(2026, 4, 6): 100.0,
                date(2026, 4, 7): 200.0,
                date(2026, 4, 8): 300.0,
            }
        )
        result = compute_consistency(trades)
        assert result.n_trading_days == 3


# ─────────────────────────────────────────────────────────────────────────────
# Consistency PASS (< 50%)
# ─────────────────────────────────────────────────────────────────────────────

class TestConsistencyPass:
    def test_passes_when_under_50_pct(self):
        """3 equal-profit days → each day = 33.3% → passes."""
        trades = _make_trades(
            {
                date(2026, 4, 6): 3_000.0,
                date(2026, 4, 7): 3_000.0,
                date(2026, 4, 8): 3_000.0,
                date(2026, 4, 9): 3_000.0,
                date(2026, 4, 10): 3_000.0,
            }
        )
        result = compute_consistency(trades)
        assert result.passes_consistency
        assert abs(result.consistency_pct - 1 / 5) < 0.001
        assert result.updated_profit_goal is None

    def test_passes_at_49_pct(self):
        """Best day is 49% of net PnL → just passes."""
        trades = _make_trades(
            {
                date(2026, 4, 6): 4_900.0,
                date(2026, 4, 7): 1_100.0,   # 4900 / (4900+1100+4000) = 49%
                date(2026, 4, 8): 4_000.0,
            }
        )
        result = compute_consistency(trades)
        assert result.passes_consistency


# ─────────────────────────────────────────────────────────────────────────────
# Consistency FAIL (≥ 50%)
# ─────────────────────────────────────────────────────────────────────────────

class TestConsistencyFail:
    def test_fails_at_exactly_50_pct(self):
        """Exactly 50% → FAIL."""
        trades = _make_trades(
            {
                date(2026, 4, 6): 5_000.0,
                date(2026, 4, 7): 5_000.0,
            }
        )
        result = compute_consistency(trades)
        assert not result.passes_consistency  # 5000/10000 = 50% → fail

    def test_fails_over_50_pct(self):
        trades = _make_trades(
            {
                date(2026, 4, 6): 8_000.0,
                date(2026, 4, 7): 2_000.0,
            }
        )
        result = compute_consistency(trades)
        assert not result.passes_consistency
        assert abs(result.consistency_pct - 0.80) < 0.001

    def test_updated_profit_goal_is_net_pl_times_2(self):
        """updated_profit_goal must equal net_pl × 2."""
        trades = _make_trades(
            {
                date(2026, 4, 6): 9_000.0,   # 90% → fails consistency
                date(2026, 4, 7): 1_000.0,
            }
        )
        result = compute_consistency(trades)
        assert result.updated_profit_goal is not None
        assert abs(result.updated_profit_goal - result.net_pl * 2) < 0.01

    def test_large_single_day_fails(self):
        """One massive day and many tiny days — should fail."""
        trades = _make_trades(
            {
                date(2026, 4, 6): 9_000.0,
                date(2026, 4, 7): 100.0,
                date(2026, 4, 8): 100.0,
                date(2026, 4, 9): 100.0,
                date(2026, 4, 10): 100.0,
            }
        )
        result = compute_consistency(trades)
        assert not result.passes_consistency
        assert result.updated_profit_goal is not None


# ─────────────────────────────────────────────────────────────────────────────
# Minimum trading days
# ─────────────────────────────────────────────────────────────────────────────

class TestMinTradingDays:
    def test_passes_with_5_days(self):
        trades = _make_trades(
            {d: 100.0 for d in [
                date(2026, 4, 6), date(2026, 4, 7), date(2026, 4, 8),
                date(2026, 4, 9), date(2026, 4, 10),
            ]}
        )
        result = compute_consistency(trades, min_trading_days=5)
        assert result.passes_min_days

    def test_fails_with_4_days(self):
        trades = _make_trades(
            {d: 100.0 for d in [
                date(2026, 4, 6), date(2026, 4, 7), date(2026, 4, 8),
                date(2026, 4, 9),
            ]}
        )
        result = compute_consistency(trades, min_trading_days=5)
        assert not result.passes_min_days

    def test_custom_min_days(self):
        trades = _make_trades({date(2026, 4, 6): 100.0, date(2026, 4, 7): 200.0})
        result = compute_consistency(trades, min_trading_days=2)
        assert result.passes_min_days


# ─────────────────────────────────────────────────────────────────────────────
# Days to target
# ─────────────────────────────────────────────────────────────────────────────

class TestDaysToTarget:
    def test_target_reached_day_5(self):
        day_pnls = {
            date(2026, 4, 6): 1_000.0,
            date(2026, 4, 7): 2_000.0,
            date(2026, 4, 8): 1_000.0,
            date(2026, 4, 9): 1_000.0,
            date(2026, 4, 10): 4_100.0,   # cumulative = 9,100 ≥ 9,000
        }
        result = profit_target_reached_after_days(day_pnls, profit_target=9_000.0)
        assert result == 5

    def test_target_never_reached(self):
        day_pnls = {
            date(2026, 4, 6): 500.0,
            date(2026, 4, 7): 500.0,
        }
        result = profit_target_reached_after_days(day_pnls, profit_target=9_000.0)
        assert result is None

    def test_target_reached_day_1(self):
        day_pnls = {date(2026, 4, 6): 10_000.0}
        result = profit_target_reached_after_days(day_pnls, profit_target=9_000.0)
        assert result == 1


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_all_losing_days(self):
        """All negative P/L days → consistency passes (no profit to cap)."""
        trades = _make_trades(
            {
                date(2026, 4, 6): -1_000.0,
                date(2026, 4, 7): -500.0,
            }
        )
        result = compute_consistency(trades)
        assert result.passes_consistency
        assert result.updated_profit_goal is None

    def test_zero_net_pnl(self):
        trades = _make_trades(
            {
                date(2026, 4, 6): 1_000.0,
                date(2026, 4, 7): -1_000.0,
            }
        )
        result = compute_consistency(trades)
        # net_pl = 0 → consistency_pct = 0 → passes
        assert result.passes_consistency

    def test_empty_trades(self):
        result = compute_consistency(pd.DataFrame())
        assert result.net_pl == 0.0
        assert result.n_trading_days == 0
        assert not result.passes_min_days

    def test_consistency_pct_calculation(self):
        """Verify the exact formula: highest_profit_day / net_pl."""
        trades = _make_trades(
            {
                date(2026, 4, 6): 3_000.0,
                date(2026, 4, 7): 6_000.0,
                date(2026, 4, 8): 1_000.0,
            }
        )
        result = compute_consistency(trades)
        expected_pct = 6_000.0 / 10_000.0
        assert abs(result.consistency_pct - expected_pct) < 0.0001

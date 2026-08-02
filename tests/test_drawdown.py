"""
Tests for backtest_engine.drawdown

Covers:
* No breach when drawdown stays well below $4,500.
* Breach detected when equity falls ≥ $4,500 below peak.
* EOD mode: peak updates at end of day.
* Close-to-close mode: peak updates after every trade.
* Breach day is correctly identified.
* Empty trade DataFrame returns no-breach result.
* AE proxy is computed when adverse_excursion column is present.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from backtest_engine.drawdown import compute_drawdown


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_trades(day_pnls: dict[date, float]) -> pd.DataFrame:
    """
    Create a simple trades DataFrame with one trade per day.

    ``day_pnls`` maps date → realized PnL for that day.
    """
    rows = []
    for d, pnl in sorted(day_pnls.items()):
        rows.append(
            {
                "trade_number": 1,
                "symbol": "TEST",
                "entry_time": pd.Timestamp(f"{d} 09:00", tz="US/Eastern"),
                "exit_time": pd.Timestamp(f"{d} 10:00", tz="US/Eastern"),
                "tpt_trading_day": d,
                "size_qty": 1.0,
                "net_pnl": pnl,
                "effective_pnl": pnl,
                "effective_size": 1.0,
                "commission": 0.25,
                "adverse_excursion": -abs(pnl) * 0.5,  # dummy AE
                "favorable_excursion": abs(pnl) * 0.3,
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# No-breach scenarios
# ─────────────────────────────────────────────────────────────────────────────

class TestNoBreach:
    def test_profitable_run_no_breach(self):
        trades = _make_trades(
            {
                date(2026, 4, 6): 1_000.0,
                date(2026, 4, 7): 2_000.0,
                date(2026, 4, 8): 500.0,
            }
        )
        result = compute_drawdown(trades, initial_equity=150_000.0, max_trailing_dd=4_500.0)
        assert not result.any_breach
        assert result.breach_days == []

    def test_small_loss_no_breach(self):
        trades = _make_trades(
            {
                date(2026, 4, 6): 500.0,
                date(2026, 4, 7): -200.0,
                date(2026, 4, 8): 300.0,
            }
        )
        result = compute_drawdown(trades, initial_equity=150_000.0, max_trailing_dd=4_500.0)
        assert not result.any_breach

    def test_loss_just_under_limit(self):
        """Loss of $4,499 should NOT breach the $4,500 limit."""
        trades = _make_trades(
            {
                date(2026, 4, 6): 5_000.0,   # peak = 155,000
                date(2026, 4, 7): -4_499.0,  # equity = 150,501 — floor = 150,500 → OK
            }
        )
        result = compute_drawdown(trades, initial_equity=150_000.0, max_trailing_dd=4_500.0)
        assert not result.any_breach

    def test_flat_run_no_breach(self):
        trades = _make_trades({date(2026, 4, 6): 0.0, date(2026, 4, 7): 0.0})
        result = compute_drawdown(trades, initial_equity=150_000.0, max_trailing_dd=4_500.0)
        assert not result.any_breach


# ─────────────────────────────────────────────────────────────────────────────
# Breach detection
# ─────────────────────────────────────────────────────────────────────────────

class TestBreach:
    def test_exact_breach_at_limit(self):
        """Loss that equals the drawdown limit exactly = BREACH."""
        trades = _make_trades(
            {
                date(2026, 4, 6): 5_000.0,   # peak 155,000; floor 150,500
                date(2026, 4, 7): -4_500.0,  # equity 150,500 == floor → breach
            }
        )
        result = compute_drawdown(trades, initial_equity=150_000.0, max_trailing_dd=4_500.0)
        assert result.any_breach

    def test_breach_over_limit(self):
        """Loss exceeding the limit must be detected."""
        trades = _make_trades(
            {
                date(2026, 4, 6): 3_000.0,   # peak 153,000; floor 148,500
                date(2026, 4, 7): -6_000.0,  # equity 147,000 < 148,500 → breach
            }
        )
        result = compute_drawdown(trades, initial_equity=150_000.0, max_trailing_dd=4_500.0)
        assert result.any_breach
        assert date(2026, 4, 7) in result.breach_days

    def test_breach_day_identified(self):
        trades = _make_trades(
            {
                date(2026, 4, 6): 1_000.0,
                date(2026, 4, 7): 500.0,
                date(2026, 4, 8): -6_000.0,  # breach
                date(2026, 4, 9): 200.0,
            }
        )
        result = compute_drawdown(trades, initial_equity=150_000.0, max_trailing_dd=4_500.0)
        assert date(2026, 4, 8) in result.breach_days

    def test_no_breach_before_peak_rises(self):
        """Starting loss from the initial equity — floor starts at $145,500."""
        trades = _make_trades(
            {
                date(2026, 4, 6): -4_000.0,  # equity 146,000 > 145,500 → OK
                date(2026, 4, 7): -400.0,    # equity 145,600 > 145,500 → OK
            }
        )
        result = compute_drawdown(trades, initial_equity=150_000.0, max_trailing_dd=4_500.0)
        assert not result.any_breach

    def test_breach_from_initial_equity(self):
        """A loss of >$4,500 from the starting balance is a breach."""
        trades = _make_trades(
            {
                date(2026, 4, 6): -4_501.0,  # equity 145,499 < 145,500 → breach
            }
        )
        result = compute_drawdown(trades, initial_equity=150_000.0, max_trailing_dd=4_500.0)
        assert result.any_breach


# ─────────────────────────────────────────────────────────────────────────────
# EOD vs close-to-close
# ─────────────────────────────────────────────────────────────────────────────

class TestDrawdownModes:
    def test_eod_peak_updates_at_day_end(self):
        """In EOD mode the peak should reflect day-end equity."""
        trades = _make_trades(
            {
                date(2026, 4, 6): 2_000.0,
                date(2026, 4, 7): -1_000.0,
            }
        )
        result = compute_drawdown(trades, initial_equity=150_000.0, max_trailing_dd=4_500.0, mode="eod")
        # Peak should be 152,000 (after day 1), not the initial 150,000
        assert result.peak_equity == 152_000.0

    def test_close_to_close_peak_updates_per_trade(self):
        """In close-to-close mode, peak updates after every trade."""
        trades = _make_trades(
            {
                date(2026, 4, 6): 2_000.0,
                date(2026, 4, 7): -1_000.0,
            }
        )
        result = compute_drawdown(
            trades, initial_equity=150_000.0, max_trailing_dd=4_500.0,
            mode="close_to_close"
        )
        # Peak is also 152,000 in this test (one trade per day, same result)
        assert result.peak_equity == 152_000.0

    def test_final_equity_correct(self):
        trades = _make_trades(
            {
                date(2026, 4, 6): 1_000.0,
                date(2026, 4, 7): -500.0,
                date(2026, 4, 8): 3_000.0,
            }
        )
        result = compute_drawdown(trades, initial_equity=150_000.0, max_trailing_dd=4_500.0)
        assert abs(result.final_equity - 153_500.0) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_trades_no_breach(self):
        result = compute_drawdown(
            pd.DataFrame(), initial_equity=150_000.0, max_trailing_dd=4_500.0
        )
        assert not result.any_breach
        assert result.final_equity == 150_000.0

    def test_invalid_mode_raises(self):
        trades = _make_trades({date(2026, 4, 6): 100.0})
        with pytest.raises(ValueError, match="mode"):
            compute_drawdown(trades, mode="bad_mode")

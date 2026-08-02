"""
Tests for backtest_engine.session

Covers:
* assign_tpt_trading_day: trades before 18:00 belong to the same calendar day.
* assign_tpt_trading_day: trades at/after 18:00 belong to the NEXT calendar day.
* filter_trades (drop mode): weekend trades are removed.
* filter_trades (drop mode): session-crossing trades (past 4:55 PM) are removed.
* filter_trades (drop mode): normal weekday trades are retained.
* filter_trades (flatten mode): boundary trades are retained but flagged.
* Friday→Sunday weekend detection.
* tpt_trading_day column is added after filtering.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from backtest_engine.parser import parse_csv
from backtest_engine.session import (
    assign_tpt_trading_day,
    filter_trades,
)


# ─────────────────────────────────────────────────────────────────────────────
# assign_tpt_trading_day
# ─────────────────────────────────────────────────────────────────────────────

def _et(dt_str: str) -> pd.Timestamp:
    return pd.Timestamp(dt_str, tz="US/Eastern")


class TestAssignTptTradingDay:
    def test_morning_trade_same_day(self):
        ts = _et("2026-04-07 09:30")
        assert assign_tpt_trading_day(ts) == date(2026, 4, 7)

    def test_afternoon_trade_same_day(self):
        ts = _et("2026-04-07 14:00")
        assert assign_tpt_trading_day(ts) == date(2026, 4, 7)

    def test_just_before_6pm_same_day(self):
        ts = _et("2026-04-07 17:59")
        assert assign_tpt_trading_day(ts) == date(2026, 4, 7)

    def test_at_6pm_next_day(self):
        """A trade opened at exactly 18:00 belongs to the next calendar day."""
        ts = _et("2026-04-07 18:00")
        assert assign_tpt_trading_day(ts) == date(2026, 4, 8)

    def test_after_6pm_next_day(self):
        ts = _et("2026-04-07 21:30")
        assert assign_tpt_trading_day(ts) == date(2026, 4, 8)

    def test_midnight_trade_next_day(self):
        """A midnight trade (after 18:00) belongs to the NEXT calendar day."""
        ts = _et("2026-04-07 23:59")
        assert assign_tpt_trading_day(ts) == date(2026, 4, 8)

    def test_early_morning_same_day(self):
        """A 2:00 AM trade (before 18:00) belongs to today."""
        ts = _et("2026-04-08 02:00")
        assert assign_tpt_trading_day(ts) == date(2026, 4, 8)


# ─────────────────────────────────────────────────────────────────────────────
# Weekend / session filter
# ─────────────────────────────────────────────────────────────────────────────

class TestFilterTrades:
    def test_normal_trades_retained(self, parsed_mgc1):
        """Trades 1 and 2 (normal weekday) must survive the drop filter."""
        filtered, stats = filter_trades(parsed_mgc1, mode="drop")
        retained_trades = set(filtered["trade_number"].tolist())
        assert 1 in retained_trades
        assert 2 in retained_trades

    def test_weekend_trade_dropped(self, parsed_mgc1):
        """Trade 3 (entry Fri 16:00, exit Sun 18:30) must be removed."""
        filtered, stats = filter_trades(parsed_mgc1, mode="drop")
        assert 3 not in filtered["trade_number"].tolist()
        assert stats["dropped"] >= 1

    def test_session_crossing_trade_dropped(self, parsed_mgc1):
        """Trade 4 (entry Tue 16:30, exit Tue 17:30 — past 16:55) must be removed."""
        filtered, stats = filter_trades(parsed_mgc1, mode="drop")
        assert 4 not in filtered["trade_number"].tolist()

    def test_drop_count(self, parsed_mgc1):
        """Exactly 2 legs should be dropped (trades 3 and 4)."""
        _, stats = filter_trades(parsed_mgc1, mode="drop")
        assert stats["dropped"] == 2

    def test_flatten_retains_violations(self, parsed_mgc1):
        """In flatten mode, violated trades are retained (not dropped)."""
        filtered, stats = filter_trades(parsed_mgc1, mode="flatten")
        # All 4 legs should still be present
        assert len(filtered) == 4
        assert stats["flattened"] == 2
        assert stats["dropped"] == 0

    def test_tpt_trading_day_column_added(self, parsed_mgc1):
        """Filter must add the tpt_trading_day column."""
        filtered, _ = filter_trades(parsed_mgc1, mode="drop")
        assert "tpt_trading_day" in filtered.columns

    def test_tpt_trading_day_correct_for_morning_trade(self, parsed_mgc1):
        """Trade 1 (Mon Apr 6, 09:00) → tpt_trading_day = 2026-04-06."""
        filtered, _ = filter_trades(parsed_mgc1, mode="drop")
        t1 = filtered[filtered["trade_number"] == 1].iloc[0]
        assert t1["tpt_trading_day"] == date(2026, 4, 6)

    def test_tpt_trading_day_correct_for_tue_trade(self, parsed_mgc1):
        """Trade 2 (Tue Apr 7, 09:00) → tpt_trading_day = 2026-04-07."""
        filtered, _ = filter_trades(parsed_mgc1, mode="drop")
        t2 = filtered[filtered["trade_number"] == 2].iloc[0]
        assert t2["tpt_trading_day"] == date(2026, 4, 7)

    def test_invalid_mode_raises(self, parsed_mgc1):
        with pytest.raises(ValueError, match="session_mode"):
            filter_trades(parsed_mgc1, mode="invalid")


# ─────────────────────────────────────────────────────────────────────────────
# Weekend boundary edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestWeekendBoundary:
    def _make_trade(self, entry_str: str, exit_str: str) -> pd.DataFrame:
        """Helper: create a single-trade DataFrame for boundary testing."""
        return pd.DataFrame(
            [
                {
                    "trade_number": 1,
                    "symbol": "TEST",
                    "entry_time": pd.Timestamp(entry_str, tz="US/Eastern"),
                    "exit_time": pd.Timestamp(exit_str, tz="US/Eastern"),
                    "entry_price": 1000.0,
                    "exit_price": 1001.0,
                    "size_qty": 1.0,
                    "net_pnl": 10.0,
                    "commission": 0.25,
                    "adverse_excursion": -5.0,
                    "favorable_excursion": 12.0,
                    "cumulative_pnl": 10.0,
                    "leg_type": "Exit Long",
                }
            ]
        )

    def test_friday_before_cutoff_not_weekend(self):
        """Entry Fri 09:00, exit Fri 14:00 — entirely before Fri 17:00 → not weekend."""
        df = self._make_trade("2026-04-10 09:00", "2026-04-10 14:00")
        filtered, stats = filter_trades(df, mode="drop")
        assert len(filtered) == 1
        assert stats["dropped"] == 0

    def test_friday_entry_before_cutoff_exit_after_is_weekend(self):
        """Entry Fri 16:00, exit Sat 10:00 — crosses Fri 17:00 → weekend trade."""
        df = self._make_trade("2026-04-10 16:00", "2026-04-12 10:00")
        filtered, stats = filter_trades(df, mode="drop")
        assert len(filtered) == 0
        assert stats["dropped"] == 1

    def test_sunday_after_1800_not_weekend(self):
        """Entry Sun 18:01, exit Mon 09:00 — after weekend end → not weekend."""
        df = self._make_trade("2026-04-12 18:01", "2026-04-13 09:00")
        filtered, stats = filter_trades(df, mode="drop")
        # Entry is after weekend end → normal session trade
        # BUT it may cross a daily boundary (exits next day). Let's check:
        # Entry Sun 18:01 → tpt_trading_day = Mon Apr 13
        # Session close for this trade: Mon 16:55 → exit Mon 09:00 is before 16:55 → OK
        assert len(filtered) == 1
        assert stats["dropped"] == 0

    def test_within_weekend_block_dropped(self):
        """Entry Sat 12:00, exit Sun 12:00 — entirely within weekend block → dropped."""
        df = self._make_trade("2026-04-11 12:00", "2026-04-12 12:00")
        # Both entry and exit overlap Fri 17:00 → Sun 18:00 (entry is Saturday)
        filtered, stats = filter_trades(df, mode="drop")
        assert len(filtered) == 0

"""
Tests for backtest_engine.parser

Covers:
* CSV parsing returns only Exit rows (no Entry rows in output)
* Entry/exit pairing by trade number
* TP1 / TP2 partial-exit legs both appear, paired with the same entry
* No double-counting (sum of net_pnl == last cumulative value reported)
* Missing column raises ValueError
* Entry without matching exit produces a warning, not a crash
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from backtest_engine.parser import parse_csv


# ─────────────────────────────────────────────────────────────────────────────
# Basic parsing
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_returns_only_exits(parsed_mgc1):
    """Only Exit rows should appear in the result."""
    assert all("Exit" in lt for lt in parsed_mgc1["leg_type"])


def test_parse_has_correct_columns(parsed_mgc1):
    expected = {
        "trade_number", "symbol", "entry_time", "exit_time",
        "entry_price", "exit_price", "size_qty", "net_pnl",
        "commission", "adverse_excursion", "favorable_excursion",
        "cumulative_pnl", "leg_type",
    }
    assert expected.issubset(set(parsed_mgc1.columns))


def test_parse_symbol_assigned(parsed_mgc1):
    assert (parsed_mgc1["symbol"] == "MGC1").all()


def test_parse_trade_count(parsed_mgc1):
    """Four Exit rows in simple_mgc1_csv → 4 legs."""
    assert len(parsed_mgc1) == 4


def test_parse_entry_time_populated(parsed_mgc1):
    """entry_time must be set for every leg (not NaT)."""
    assert not parsed_mgc1["entry_time"].isna().any()


def test_parse_exit_time_populated(parsed_mgc1):
    assert not parsed_mgc1["exit_time"].isna().any()


def test_parse_pnl_numeric(parsed_mgc1):
    assert parsed_mgc1["net_pnl"].dtype in (float, "float64")


def test_parse_sorted_by_exit_time(parsed_mgc1):
    times = parsed_mgc1["exit_time"].tolist()
    assert times == sorted(times)


# ─────────────────────────────────────────────────────────────────────────────
# Entry / exit pairing
# ─────────────────────────────────────────────────────────────────────────────

def test_entry_time_matches_entry_row(parsed_mgc1):
    """
    Trade 1: entry at 2026-04-06 09:00.
    Trade 2: entry at 2026-04-07 09:00.
    """
    t1 = parsed_mgc1[parsed_mgc1["trade_number"] == 1].iloc[0]
    assert t1["entry_time"].hour == 9
    assert t1["entry_time"].day == 6

    t2 = parsed_mgc1[parsed_mgc1["trade_number"] == 2].iloc[0]
    assert t2["entry_time"].day == 7


def test_correct_pnl_per_trade(parsed_mgc1):
    t1 = parsed_mgc1[parsed_mgc1["trade_number"] == 1].iloc[0]
    assert abs(t1["net_pnl"] - 9.75) < 0.01

    t2 = parsed_mgc1[parsed_mgc1["trade_number"] == 2].iloc[0]
    assert abs(t2["net_pnl"] - 9.75) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# TP1 / TP2 partial exits (same trade number, multiple legs)
# ─────────────────────────────────────────────────────────────────────────────

def test_tp_legs_both_present(tp_csv):
    df = parse_csv(tp_csv, symbol="TEST")
    assert len(df) == 2


def test_tp_legs_same_entry_time(tp_csv):
    df = parse_csv(tp_csv, symbol="TEST")
    # Both legs share trade number 10 and the same entry row
    assert df["trade_number"].nunique() == 1
    entry_times = df["entry_time"].unique()
    assert len(entry_times) == 1


def test_tp_legs_no_double_counting(tp_csv):
    """Sum of per-leg PnL should equal the last cumulative_pnl value."""
    df = parse_csv(tp_csv, symbol="TEST")
    pnl_sum = df["net_pnl"].sum()
    last_cumulative = df["cumulative_pnl"].iloc[-1]
    assert abs(pnl_sum - last_cumulative) < 1.0


def test_m2k1_tp_legs(parsed_m2k1):
    """Trade 5 has TP1 (size 3) and TP2 (size 7) legs."""
    t5 = parsed_m2k1[parsed_m2k1["trade_number"] == 5]
    assert len(t5) == 2
    sizes = sorted(t5["size_qty"].tolist())
    assert sizes == [3.0, 7.0]


# ─────────────────────────────────────────────────────────────────────────────
# Validation / error handling
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_column_raises():
    bad_csv = io.StringIO(
        "Trade number,Type,Date and time\n"
        "1,Entry Long,2026-04-06 09:00\n"
    )
    with pytest.raises(ValueError, match="missing required columns"):
        parse_csv(bad_csv, symbol="X")


def test_orphan_exit_skipped(caplog):
    """An Exit row with no preceding Entry should produce a warning and be skipped."""
    csv_text = (
        "Trade number,Type,Date and time,Signal,Price USD,Size (qty),Size (value),"
        "Net PnL USD,Return %,Commission USD,Favorable excursion USD,"
        "Favorable excursion %,Adverse excursion USD,Adverse excursion %,"
        "Cumulative PnL USD,Cumulative PnL %,Duration (bars)\n"
        "99,Exit Long,2026-04-06 10:00,Sell,2010.0,1,2010.0,"
        "9.75,0.49,0.25,12.0,0.6,-5.0,-0.25,9.75,0.49,4\n"
    )
    import logging
    with caplog.at_level(logging.WARNING, logger="backtest_engine.parser"):
        df = parse_csv(io.StringIO(csv_text), symbol="X")
    assert df.empty
    assert "no matching Entry" in caplog.text


def test_empty_csv_returns_empty_df():
    csv_text = (
        "Trade number,Type,Date and time,Signal,Price USD,Size (qty),Size (value),"
        "Net PnL USD,Return %,Commission USD,Favorable excursion USD,"
        "Favorable excursion %,Adverse excursion USD,Adverse excursion %,"
        "Cumulative PnL USD,Cumulative PnL %,Duration (bars)\n"
    )
    df = parse_csv(io.StringIO(csv_text), symbol="MGC1")
    assert df.empty

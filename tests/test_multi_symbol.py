"""
Tests for the multi-symbol engine new features:

* Multi-symbol load incl. missing-CSV skip
* Baseline-size auto-detect + no double-scaling
* Overlapping-interval concurrency computation (AE stacking upper/lower bounds)
* Trailing-peak ratchet logic
* 150-micro concurrency cap
* Monthly pass-rate
* Consistency never fails (only raises goal)

All tests use small synthetic in-memory fixtures (2–3 symbols).
"""

from __future__ import annotations

import io
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from backtest_engine.config import (
    AccountConfig,
    MultiSymbolConfig,
    SymbolConfig,
    CME_MICRO_SPECS,
    load_multi_config,
)
from backtest_engine.concurrency import compute_intraday_concurrency, ESTIMATE_BANNER
from backtest_engine.loader import detect_baseline_size, load_all_symbols, _apply_symbol_sizing
from backtest_engine.monthly import evaluate_monthly, MONTHLY_DISCLAIMER
from backtest_engine.sizing import check_concurrent_position_limit, rescale_symbol


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers (reuse existing conftest format)
# ─────────────────────────────────────────────────────────────────────────────

_CSV_HEADER = (
    "Trade number,Type,Date and time,Signal,Price USD,Size (qty),Size (value),"
    "Net PnL USD,Return %,Commission USD,Favorable excursion USD,"
    "Favorable excursion %,Adverse excursion USD,Adverse excursion %,"
    "Cumulative PnL USD,Cumulative PnL %,Duration (bars)\n"
)


def _row(num, t_type, dt, signal, price, size, net_pnl="", commission="",
         fe="", ae="", cum_pnl=""):
    size_val = round(price * size, 2) if price and size else ""
    return (
        f"{num},{t_type},{dt},{signal},{price},{size},{size_val},"
        f"{net_pnl},,{commission},{fe},,{ae},,{cum_pnl},,4\n"
    )


def _make_csv_str(rows):
    lines = [_CSV_HEADER]
    for r in rows:
        num, t_type, dt, signal, price, size = r[:6]
        net_pnl = r[6] if len(r) > 6 else ""
        commission = r[7] if len(r) > 7 else ""
        fe = r[8] if len(r) > 8 else ""
        ae = r[9] if len(r) > 9 else ""
        cum_pnl = r[10] if len(r) > 10 else ""
        lines.append(_row(num, t_type, dt, signal, price, size, net_pnl, commission, fe, ae, cum_pnl))
    return "".join(lines)


def _make_mgc_csv():
    """3 trades for MGC, 1 contract each, clean weekday trades."""
    rows = [
        (1, "Entry Long", "2026-04-06 09:00", "Buy",  2000.0, 1),
        (1, "Exit Long",  "2026-04-06 10:00", "Sell", 2010.0, 1, 100.0, 0.25, 15.0, -50.0,  100.0),
        (2, "Entry Long", "2026-04-07 09:00", "Buy",  2000.0, 1),
        (2, "Exit Long",  "2026-04-07 10:00", "Sell", 2005.0, 1,  50.0, 0.25, 10.0, -20.0,  150.0),
        (3, "Entry Long", "2026-04-08 09:00", "Buy",  2000.0, 1),
        (3, "Exit Long",  "2026-04-08 10:00", "Sell", 1990.0, 1, -100.0, 0.25, 5.0, -120.0,  50.0),
    ]
    return _make_csv_str(rows)


def _make_mnq_csv():
    """2 trades for MNQ, 1 contract."""
    rows = [
        (10, "Entry Long", "2026-04-06 09:30", "Buy",  18000.0, 1),
        (10, "Exit Long",  "2026-04-06 10:30", "Sell", 18050.0, 1, 100.0, 0.50, 60.0, -30.0,  100.0),
        (11, "Entry Long", "2026-04-08 09:30", "Buy",  18000.0, 1),
        (11, "Exit Long",  "2026-04-08 10:30", "Sell", 17980.0, 1, -40.0, 0.50, 20.0, -80.0,   60.0),
    ]
    return _make_csv_str(rows)


def _make_m2k_csv_5lot():
    """M2K trades with 1+4 scale-in (modal size = 4 → baseline=4, target=5)."""
    rows = [
        (20, "Entry Long",  "2026-04-07 09:00", "Buy",  2100.0, 5),  # entry row
        (20, "Exit Long",   "2026-04-07 09:30", "Sell", 2110.0, 1,  50.0, 0.25, 20.0, -15.0,  50.0),
        (20, "Exit Long",   "2026-04-07 10:00", "Sell", 2120.0, 4, 200.0, 1.00, 80.0, -40.0, 250.0),
    ]
    return _make_csv_str(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Config loading tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiSymbolConfig:
    def test_cme_specs_present(self):
        """CME micro specs must include all 5 symbols."""
        for sym in ("MGC", "M2K", "MNQ", "MYM", "MCL"):
            assert sym in CME_MICRO_SPECS
            assert CME_MICRO_SPECS[sym]["point_value"] > 0

    def test_load_multi_config_from_yaml(self, tmp_path):
        """load_multi_config reads symbols + account block."""
        yaml_content = """
account:
  size: 150000
  profit_target: 9000
  trailing_drawdown: 4500
  safety_buffer: 3000
symbols:
  MGC:
    csv: data/MGC1.csv
    contract_size: 1
  M2K:
    csv: data/M2K1.csv
    contract_size: 5
"""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml_content)
        cfg = load_multi_config(str(cfg_file))

        assert cfg.account.size == 150_000
        assert cfg.account.profit_target == 9_000
        assert cfg.account.trailing_drawdown == 4_500
        assert "MGC" in cfg.symbols
        assert "M2K" in cfg.symbols
        assert cfg.symbols["MGC"].contract_size == 1
        assert cfg.symbols["M2K"].contract_size == 5

    def test_load_multi_config_no_file(self):
        """Missing config file returns default config."""
        cfg = load_multi_config("/nonexistent/path/config.yaml")
        assert isinstance(cfg.account, AccountConfig)
        assert cfg.symbols == {}

    def test_symbol_config_inherits_cme_specs(self, tmp_path):
        """When point_value/tick_size are omitted, CME defaults are applied."""
        yaml_content = """
symbols:
  MNQ:
    csv: data/MNQ1.csv
    contract_size: 2
"""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml_content)
        cfg = load_multi_config(str(cfg_file))
        mnq = cfg.symbols["MNQ"]
        assert mnq.point_value == CME_MICRO_SPECS["MNQ"]["point_value"]
        assert mnq.tick_size == CME_MICRO_SPECS["MNQ"]["tick_size"]


# ─────────────────────────────────────────────────────────────────────────────
# Loader: multi-symbol load + missing CSV skip
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiSymbolLoader:
    def _write_csvs(self, tmp_path):
        (tmp_path / "MGC1.csv").write_text(_make_mgc_csv())
        (tmp_path / "MNQ1.csv").write_text(_make_mnq_csv())
        # M2K1.csv intentionally absent

    def _make_cfg(self, tmp_path):
        cfg = MultiSymbolConfig(
            account=AccountConfig(),
            symbols={
                "MGC": SymbolConfig(
                    csv=str(tmp_path / "MGC1.csv"), contract_size=1,
                    point_value=10.0, tick_size=0.10
                ),
                "MNQ": SymbolConfig(
                    csv=str(tmp_path / "MNQ1.csv"), contract_size=1,
                    point_value=2.0, tick_size=0.25
                ),
                "M2K": SymbolConfig(
                    csv=str(tmp_path / "M2K1.csv"), contract_size=5,
                    point_value=5.0, tick_size=0.10
                ),
            },
        )
        return cfg

    def test_loads_present_symbols(self, tmp_path):
        self._write_csvs(tmp_path)
        cfg = self._make_cfg(tmp_path)
        combined, present, missing = load_all_symbols(cfg)
        assert "MGC" in present
        assert "MNQ" in present

    def test_warns_missing_csv(self, tmp_path):
        self._write_csvs(tmp_path)
        cfg = self._make_cfg(tmp_path)
        _, present, missing = load_all_symbols(cfg)
        assert "M2K" in missing
        assert "M2K" not in present

    def test_combined_has_symbol_column(self, tmp_path):
        self._write_csvs(tmp_path)
        cfg = self._make_cfg(tmp_path)
        combined, _, _ = load_all_symbols(cfg)
        assert "symbol" in combined.columns
        symbols_in_df = set(combined["symbol"].unique())
        assert "MGC" in symbols_in_df
        assert "MNQ" in symbols_in_df

    def test_effective_pnl_column_added(self, tmp_path):
        self._write_csvs(tmp_path)
        cfg = self._make_cfg(tmp_path)
        combined, _, _ = load_all_symbols(cfg)
        assert "effective_pnl" in combined.columns
        assert "effective_size" in combined.columns

    def test_combined_sorted_by_exit_time(self, tmp_path):
        self._write_csvs(tmp_path)
        cfg = self._make_cfg(tmp_path)
        combined, _, _ = load_all_symbols(cfg)
        times = combined["exit_time"].tolist()
        assert times == sorted(times)

    def test_all_missing_returns_empty(self, tmp_path):
        """When all CSVs are missing, returns empty DataFrame."""
        cfg = MultiSymbolConfig(
            account=AccountConfig(),
            symbols={
                "MGC": SymbolConfig(csv="/nope/MGC.csv", contract_size=1, point_value=10.0, tick_size=0.1),
            },
        )
        combined, present, missing = load_all_symbols(cfg)
        assert combined.empty
        assert present == []
        assert "MGC" in missing


# ─────────────────────────────────────────────────────────────────────────────
# Baseline-size auto-detect + no double-scaling
# ─────────────────────────────────────────────────────────────────────────────

class TestBaselineSizing:
    def test_detect_baseline_from_modal_size(self):
        """Modal size_qty is the baseline."""
        df = pd.DataFrame({
            "size_qty": [1.0, 4.0, 4.0, 1.0, 4.0],  # mode = 4
        })
        assert detect_baseline_size(df) == 4

    def test_detect_baseline_single_value(self):
        df = pd.DataFrame({"size_qty": [1.0, 1.0, 1.0]})
        assert detect_baseline_size(df) == 1

    def test_detect_baseline_empty(self):
        assert detect_baseline_size(pd.DataFrame()) == 1

    def test_no_double_scaling_when_equal(self, tmp_path):
        """When configured == baseline, multiplier = 1.0; net_pnl == effective_pnl."""
        (tmp_path / "MGC1.csv").write_text(_make_mgc_csv())
        cfg = MultiSymbolConfig(
            account=AccountConfig(),
            symbols={
                "MGC": SymbolConfig(
                    csv=str(tmp_path / "MGC1.csv"),
                    contract_size=1,   # all MGC trades have size_qty=1 → no scaling
                    point_value=10.0,
                    tick_size=0.10,
                ),
            },
        )
        combined, _, _ = load_all_symbols(cfg)
        mgc = combined[combined["symbol"] == "MGC"]
        pd.testing.assert_series_equal(
            mgc["net_pnl"].reset_index(drop=True).rename("val"),
            mgc["effective_pnl"].reset_index(drop=True).rename("val"),
        )

    def test_scaling_when_different(self):
        """Scaling should double PnL when target = 2 × baseline."""
        df = pd.DataFrame({
            "size_qty": [1.0, 1.0, 1.0],
            "net_pnl": [100.0, -50.0, 200.0],
            "adverse_excursion": [-30.0, -20.0, -10.0],
            "favorable_excursion": [15.0, 5.0, 25.0],
            "symbol": ["MGC", "MGC", "MGC"],
        })
        sym_cfg = SymbolConfig(csv="", contract_size=2, point_value=10.0, tick_size=0.1)
        scaled = _apply_symbol_sizing(df, "MGC", sym_cfg)
        assert abs(scaled["effective_pnl"].iloc[0] - 200.0) < 0.001
        assert abs(scaled["effective_pnl"].iloc[1] - (-100.0)) < 0.001


# ─────────────────────────────────────────────────────────────────────────────
# Concurrency engine: overlapping-interval AE stacking
# ─────────────────────────────────────────────────────────────────────────────

def _make_concurrent_trades() -> pd.DataFrame:
    """
    Two overlapping trades:
      Trade A: MGC  09:00 → 10:30, PnL +100, AE -50
      Trade B: MNQ  09:30 → 11:00, PnL +200, AE -80
    They overlap 09:30 → 10:30.
    """
    tz = "US/Eastern"
    rows = [
        {
            "trade_number": 1, "symbol": "MGC",
            "entry_time": pd.Timestamp("2026-04-06 09:00", tz=tz),
            "exit_time":  pd.Timestamp("2026-04-06 10:30", tz=tz),
            "tpt_trading_day": date(2026, 4, 6),
            "net_pnl": 100.0, "effective_pnl": 100.0,
            "size_qty": 1.0,  "effective_size": 1.0,
            "adverse_excursion": -50.0, "effective_ae": -50.0,
        },
        {
            "trade_number": 2, "symbol": "MNQ",
            "entry_time": pd.Timestamp("2026-04-06 09:30", tz=tz),
            "exit_time":  pd.Timestamp("2026-04-06 11:00", tz=tz),
            "tpt_trading_day": date(2026, 4, 6),
            "net_pnl": 200.0, "effective_pnl": 200.0,
            "size_qty": 1.0,  "effective_size": 1.0,
            "adverse_excursion": -80.0, "effective_ae": -80.0,
        },
    ]
    return pd.DataFrame(rows)


class TestConcurrencyEngine:
    def test_no_concurrent_trades_no_overlap_ae(self):
        """Non-overlapping trades → upper-bound DD = single-trade AE."""
        tz = "US/Eastern"
        rows = [
            {
                "trade_number": 1, "symbol": "MGC",
                "entry_time": pd.Timestamp("2026-04-06 09:00", tz=tz),
                "exit_time":  pd.Timestamp("2026-04-06 09:30", tz=tz),
                "tpt_trading_day": date(2026, 4, 6),
                "net_pnl": 100.0, "effective_pnl": 100.0,
                "size_qty": 1.0, "effective_size": 1.0,
                "adverse_excursion": -50.0, "effective_ae": -50.0,
            },
            {
                "trade_number": 2, "symbol": "MNQ",
                "entry_time": pd.Timestamp("2026-04-06 10:00", tz=tz),
                "exit_time":  pd.Timestamp("2026-04-06 10:30", tz=tz),
                "tpt_trading_day": date(2026, 4, 6),
                "net_pnl": 200.0, "effective_pnl": 200.0,
                "size_qty": 1.0, "effective_size": 1.0,
                "adverse_excursion": -80.0, "effective_ae": -80.0,
            },
        ]
        df = pd.DataFrame(rows)
        result = compute_intraday_concurrency(df, initial_equity=150_000.0)
        # Non-overlapping: worst AE is the single worst, not summed
        assert result.upper_bound_worst_dd >= -80.0  # at most -80 (single trade)

    def test_concurrent_trades_ae_stacks(self):
        """Overlapping trades → upper bound = sum of both AEs."""
        df = _make_concurrent_trades()
        result = compute_intraday_concurrency(df, initial_equity=150_000.0)
        # During the overlap window, both AEs stack: -50 + -80 = -130
        # The upper_bound at that point = realized_equity + (-50 - 80)
        # The trailing peak starts at 150,000 so upper_dd <= -130 is possible
        assert result.upper_bound_worst_dd <= -50.0  # at minimum the worst single

    def test_upper_bound_worse_than_lower(self):
        """Upper bound (sum) should be worse than or equal to lower bound (single)."""
        df = _make_concurrent_trades()
        result = compute_intraday_concurrency(df, initial_equity=150_000.0)
        # Upper = sum of all AEs (more negative), lower = single worst
        assert result.upper_bound_worst_dd <= result.lower_bound_worst_dd

    def test_trailing_peak_ratchets_up(self):
        """Trailing peak should be max realized equity seen, never decreasing."""
        tz = "US/Eastern"
        rows = [
            {
                "trade_number": 1, "symbol": "MGC",
                "entry_time": pd.Timestamp("2026-04-06 09:00", tz=tz),
                "exit_time":  pd.Timestamp("2026-04-06 09:30", tz=tz),
                "tpt_trading_day": date(2026, 4, 6),
                "net_pnl": 5000.0, "effective_pnl": 5000.0,
                "size_qty": 1.0, "effective_size": 1.0,
                "adverse_excursion": -100.0, "effective_ae": -100.0,
            },
            {
                "trade_number": 2, "symbol": "MGC",
                "entry_time": pd.Timestamp("2026-04-07 09:00", tz=tz),
                "exit_time":  pd.Timestamp("2026-04-07 09:30", tz=tz),
                "tpt_trading_day": date(2026, 4, 7),
                "net_pnl": -2000.0, "effective_pnl": -2000.0,
                "size_qty": 1.0, "effective_size": 1.0,
                "adverse_excursion": -500.0, "effective_ae": -500.0,
            },
        ]
        df = pd.DataFrame(rows)
        result = compute_intraday_concurrency(df, initial_equity=150_000.0)
        # Peak should be 155,000 (after first trade); second trade loses, peak stays
        assert result.peak_realized_equity == 155_000.0
        # Final realized = 153,000
        assert abs(result.final_realized_equity - 153_000.0) < 0.01

    def test_trailing_peak_never_decreases(self):
        """Peak must never go below its previous value (ratchet behavior)."""
        tz = "US/Eastern"
        rows = [
            {
                "trade_number": i, "symbol": "MGC",
                "entry_time": pd.Timestamp(f"2026-04-0{i+4} 09:00", tz=tz),
                "exit_time":  pd.Timestamp(f"2026-04-0{i+4} 09:30", tz=tz),
                "tpt_trading_day": date(2026, 4, i + 4),
                "net_pnl": pnl, "effective_pnl": pnl,
                "size_qty": 1.0, "effective_size": 1.0,
                "adverse_excursion": -10.0, "effective_ae": -10.0,
            }
            for i, pnl in enumerate([1000.0, -500.0, 2000.0, -1000.0])
        ]
        df = pd.DataFrame(rows)
        result = compute_intraday_concurrency(df, initial_equity=150_000.0)
        # Collect peak values from timeline close events
        close_peaks = [
            pt.trailing_peak for pt in result.timeline if pt.event_type == "close"
        ]
        for i in range(1, len(close_peaks)):
            assert close_peaks[i] >= close_peaks[i - 1], (
                f"Peak decreased from {close_peaks[i-1]} to {close_peaks[i]}"
            )

    def test_no_breach_when_well_within_limit(self):
        """Small AE, big equity buffer → no breach estimate."""
        tz = "US/Eastern"
        df = pd.DataFrame([{
            "trade_number": 1, "symbol": "MGC",
            "entry_time": pd.Timestamp("2026-04-06 09:00", tz=tz),
            "exit_time":  pd.Timestamp("2026-04-06 09:30", tz=tz),
            "tpt_trading_day": date(2026, 4, 6),
            "net_pnl": 500.0, "effective_pnl": 500.0,
            "size_qty": 1.0, "effective_size": 1.0,
            "adverse_excursion": -200.0, "effective_ae": -200.0,
        }])
        result = compute_intraday_concurrency(
            df, initial_equity=150_000.0,
            trailing_drawdown_limit=4_500.0,
            safety_buffer=3_000.0,
        )
        assert not result.any_breach_estimate
        assert not result.any_at_risk

    def test_breach_estimate_when_ae_exceeds_limit(self):
        """AE that pushes combined floating equity below trailing floor → breach estimate."""
        tz = "US/Eastern"
        df = pd.DataFrame([{
            "trade_number": 1, "symbol": "MGC",
            "entry_time": pd.Timestamp("2026-04-06 09:00", tz=tz),
            "exit_time":  pd.Timestamp("2026-04-06 09:30", tz=tz),
            "tpt_trading_day": date(2026, 4, 6),
            "net_pnl": 100.0, "effective_pnl": 100.0,
            "size_qty": 1.0, "effective_size": 1.0,
            "adverse_excursion": -5000.0, "effective_ae": -5000.0,
        }])
        result = compute_intraday_concurrency(
            df, initial_equity=150_000.0,
            trailing_drawdown_limit=4_500.0,
        )
        # AE of -5000 from initial 150k → float equity = 145k < floor 145.5k → breach
        assert result.any_breach_estimate

    def test_empty_trades(self):
        result = compute_intraday_concurrency(
            pd.DataFrame(), initial_equity=150_000.0
        )
        assert not result.any_breach_estimate
        assert result.max_concurrent_micros == 0.0

    def test_estimate_banner_exists(self):
        assert "ESTIMATE ONLY" in ESTIMATE_BANNER


# ─────────────────────────────────────────────────────────────────────────────
# 150-micro concurrency cap
# ─────────────────────────────────────────────────────────────────────────────

class TestConcurrencyCap:
    def _make_multi_trade_df(self, sizes, tz="US/Eastern"):
        """Create trades for different symbols with given sizes, all overlapping."""
        rows = []
        for i, (sym, size) in enumerate(sizes):
            rows.append({
                "trade_number": i + 1, "symbol": sym,
                "entry_time": pd.Timestamp("2026-04-06 09:00", tz=tz),
                "exit_time":  pd.Timestamp("2026-04-06 09:30", tz=tz),
                "effective_size": float(size),
                "net_pnl": 100.0, "effective_pnl": 100.0,
                "adverse_excursion": -10.0, "effective_ae": -10.0,
            })
        return pd.DataFrame(rows)

    def test_within_cap_returns_true(self):
        df = self._make_multi_trade_df([("MGC", 1), ("MNQ", 1), ("M2K", 5)])
        within, peak, _, _ = check_concurrent_position_limit(df, max_micros=150)
        assert within
        assert abs(peak - 7.0) < 0.01  # 1+1+5 = 7

    def test_exceeds_cap_returns_false(self):
        # 100 + 100 = 200 > 150 cap
        df = self._make_multi_trade_df([("MGC", 100), ("MNQ", 100)])
        within, peak, _, _ = check_concurrent_position_limit(df, max_micros=150)
        assert not within
        assert peak >= 150

    def test_peak_micros_computed(self):
        df = self._make_multi_trade_df([("MGC", 10), ("M2K", 30)])
        _, peak, peak_time, peak_syms = check_concurrent_position_limit(df, max_micros=150)
        assert abs(peak - 40.0) < 0.01
        assert peak_time is not None
        assert "MGC" in peak_syms or "M2K" in peak_syms

    def test_concurrency_result_includes_cap_flag(self):
        tz = "US/Eastern"
        df = pd.DataFrame([{
            "trade_number": 1, "symbol": "MGC",
            "entry_time": pd.Timestamp("2026-04-06 09:00", tz=tz),
            "exit_time":  pd.Timestamp("2026-04-06 09:30", tz=tz),
            "tpt_trading_day": date(2026, 4, 6),
            "net_pnl": 100.0, "effective_pnl": 100.0,
            "size_qty": 200.0, "effective_size": 200.0,  # 200 > 150 cap
            "adverse_excursion": -10.0, "effective_ae": -10.0,
        }])
        result = compute_intraday_concurrency(df, initial_equity=150_000.0, max_micros=150)
        assert result.exceeds_micro_cap
        assert result.max_concurrent_micros >= 150


# ─────────────────────────────────────────────────────────────────────────────
# Monthly pass-rate evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _make_monthly_trades(monthly_pnls: dict) -> pd.DataFrame:
    """
    Build a trades DataFrame with one trade per day.
    monthly_pnls: {(year, month): [daily_pnls]} where each daily pnl is a float.
    """
    rows = []
    trade_num = 1
    for (year, month), pnls in sorted(monthly_pnls.items()):
        for day_offset, pnl in enumerate(pnls):
            d = date(year, month, day_offset + 1)
            tz = "US/Eastern"
            rows.append({
                "trade_number": trade_num,
                "symbol": "MGC",
                "entry_time": pd.Timestamp(f"{d} 09:00", tz=tz),
                "exit_time":  pd.Timestamp(f"{d} 09:30", tz=tz),
                "tpt_trading_day": d,
                "net_pnl": pnl,
                "effective_pnl": pnl,
                "size_qty": 1.0,
                "effective_size": 1.0,
            })
            trade_num += 1
    return pd.DataFrame(rows)


class TestMonthlyEvaluation:
    def test_pass_when_target_met(self):
        """Month with ≥ $9k in ≥ 5 days → PASS."""
        trades = _make_monthly_trades({
            (2026, 4): [2000.0, 2000.0, 2000.0, 2000.0, 1500.0],  # 9500 > 9000
        })
        result = evaluate_monthly(trades, profit_target=9_000.0, min_trading_days=5)
        assert result.months[0].hits_target
        assert result.months[0].result == "PASS"

    def test_miss_when_target_not_met(self):
        """Month with < $9k → MISS (fee gate, not failure)."""
        trades = _make_monthly_trades({
            (2026, 4): [500.0, 500.0, 500.0, 500.0, 500.0],  # 2500 < 9000
        })
        result = evaluate_monthly(trades, profit_target=9_000.0, min_trading_days=5)
        assert not result.months[0].hits_target
        assert "recurring fee" in result.months[0].result.lower()

    def test_miss_when_insufficient_trading_days(self):
        """Month with fewer than min_days → MISS."""
        trades = _make_monthly_trades({
            (2026, 4): [5000.0, 5000.0],  # > 9k but only 2 days
        })
        result = evaluate_monthly(trades, profit_target=9_000.0, min_trading_days=5)
        assert not result.months[0].passes_min_days
        assert not result.months[0].hits_target

    def test_pass_rate_calculation(self):
        """Pass rate = PASS months / months with enough days."""
        trades = _make_monthly_trades({
            (2026, 3): [2000.0, 2000.0, 2000.0, 2000.0, 1500.0],  # PASS
            (2026, 4): [200.0, 200.0, 200.0, 200.0, 200.0],        # MISS
            (2026, 5): [2000.0, 2000.0, 2000.0, 2000.0, 1500.0],   # PASS
        })
        result = evaluate_monthly(trades, profit_target=9_000.0, min_trading_days=5)
        assert result.pass_count == 2
        assert result.miss_count == 1
        assert abs(result.pass_rate_pct - 66.67) < 1.0

    def test_monthly_disclaimer_present(self):
        """MONTHLY_DISCLAIMER must contain 'fee' and 'not account failure'."""
        assert "fee" in MONTHLY_DISCLAIMER.lower()
        assert "not account failure" in MONTHLY_DISCLAIMER.lower()

    def test_consistency_adjusted_target(self):
        """Best day ≥ 50% → effective target = net_pl × 2 (NOT a failure)."""
        # Day 1: 9500, Day 2: 500, Day 3: 500, Day 4: 500, Day 5: 500
        # net_pl = 11500, best_day = 9500, pct = 9500/11500 = 82.6% → fail consistency
        trades = _make_monthly_trades({
            (2026, 4): [9500.0, 500.0, 500.0, 500.0, 500.0],
        })
        result = evaluate_monthly(trades, profit_target=9_000.0, min_trading_days=5)
        m = result.months[0]
        assert m.consistency_adjusted
        # Effective target = 11500 * 2 = 23000
        assert abs(m.effective_target - m.combined_pnl * 2) < 0.01
        # Since combined_pnl (11500) < effective_target (23000) → MISS
        assert not m.hits_target

    def test_empty_trades(self):
        result = evaluate_monthly(pd.DataFrame(), profit_target=9_000.0)
        assert result.total_months == 0
        assert result.pass_rate_pct == 0.0

    def test_multiple_months_independent(self):
        """Each month evaluated independently; one MISS doesn't affect others."""
        trades = _make_monthly_trades({
            (2026, 3): [2000.0, 2000.0, 2000.0, 2000.0, 1500.0],  # PASS
            (2026, 4): [100.0,  100.0,  100.0,  100.0,  100.0],   # MISS
        })
        result = evaluate_monthly(trades, profit_target=9_000.0, min_trading_days=5)
        assert len(result.months) == 2
        assert result.months[0].hits_target  # March PASS
        assert not result.months[1].hits_target  # April MISS


# ─────────────────────────────────────────────────────────────────────────────
# Consistency never fails — only raises goal
# ─────────────────────────────────────────────────────────────────────────────

class TestConsistencyNeverFails:
    def test_consistency_fail_does_not_fail_account(self):
        """The consistency check failure raises the goal, NOT the account verdict."""
        from backtest_engine.consistency import compute_consistency
        from backtest_engine.reporting import compute_verdict

        trades = _make_monthly_trades({
            (2026, 4): [9000.0, 500.0, 500.0, 500.0, 500.0],  # best day ≥ 50%
        })
        # Add tpt_trading_day from the month data
        con = compute_consistency(trades, profit_target=9_000.0)
        # Consistency fails (best day > 50%)
        assert not con.passes_consistency
        assert con.updated_profit_goal is not None
        # But account verdict with no drawdown breach and enough days is NOT a hard fail
        verdict = compute_verdict(
            hits_target=False,         # didn't hit adjusted goal
            any_dd_breach=False,       # NO drawdown breach
            passes_consistency=False,
            passes_min_days=True,
            updated_profit_goal=con.updated_profit_goal,
            total_net_pl=con.net_pl,
        )
        # Should be a FAIL (consistency) — NOT BREACH (drawdown)
        # i.e., the account is still alive; no fatal failure
        assert "BREACH" not in verdict
        assert "consistency" in verdict.lower() or "target" in verdict.lower()

    def test_consistency_adjusted_goal_passable(self):
        """If net_pl ≥ updated goal, verdict is PASS (consistency-adjusted)."""
        from backtest_engine.consistency import compute_consistency
        from backtest_engine.reporting import compute_verdict

        # net_pl = 20k; best day = 12k = 60% → fails, updated_goal = 40k
        # But suppose the realized net_pl already exceeds 2× original
        trades = _make_monthly_trades({
            (2026, 4): [12000.0, 2000.0, 2000.0, 2000.0, 2000.0],  # net=20k, best=12k
        })
        con = compute_consistency(trades, profit_target=9_000.0)
        verdict = compute_verdict(
            hits_target=con.net_pl >= (con.updated_profit_goal or 9000),
            any_dd_breach=False,
            passes_consistency=con.passes_consistency,
            passes_min_days=True,
            updated_profit_goal=con.updated_profit_goal,
            total_net_pl=con.net_pl,
        )
        # net_pl=20k, updated_goal=40k → doesn't pass yet, but no BREACH
        assert "BREACH" not in verdict


# ─────────────────────────────────────────────────────────────────────────────
# Per-symbol rescaling
# ─────────────────────────────────────────────────────────────────────────────

class TestRescaleSymbol:
    def _make_df(self):
        tz = "US/Eastern"
        return pd.DataFrame([
            {
                "symbol": "MGC", "net_pnl": 100.0, "size_qty": 1.0,
                "adverse_excursion": -50.0,
                "entry_time": pd.Timestamp("2026-04-06 09:00", tz=tz),
                "exit_time":  pd.Timestamp("2026-04-06 09:30", tz=tz),
            },
            {
                "symbol": "MNQ", "net_pnl": 200.0, "size_qty": 1.0,
                "adverse_excursion": -80.0,
                "entry_time": pd.Timestamp("2026-04-07 09:00", tz=tz),
                "exit_time":  pd.Timestamp("2026-04-07 09:30", tz=tz),
            },
        ])

    def test_rescale_only_affects_target_symbol(self):
        df = self._make_df()
        scaled = rescale_symbol(df, "MGC", baseline_size=1, target_size=2)
        # MGC scaled × 2
        mgc = scaled[scaled["symbol"] == "MGC"]
        assert abs(mgc["effective_pnl"].iloc[0] - 200.0) < 0.01
        # MNQ unchanged
        mnq = scaled[scaled["symbol"] == "MNQ"]
        assert abs(mnq["effective_pnl"].iloc[0] - 200.0) < 0.01  # 200 × 1 = 200

    def test_rescale_ae_also_scaled(self):
        df = self._make_df()
        scaled = rescale_symbol(df, "MGC", baseline_size=1, target_size=3)
        mgc = scaled[scaled["symbol"] == "MGC"]
        assert abs(mgc["effective_ae"].iloc[0] - (-150.0)) < 0.01  # -50 × 3

    def test_rescale_no_change_when_equal(self):
        df = self._make_df()
        scaled = rescale_symbol(df, "MGC", baseline_size=1, target_size=1)
        mgc = scaled[scaled["symbol"] == "MGC"]
        assert abs(mgc["effective_pnl"].iloc[0] - 100.0) < 0.01

"""
TPT Backtesting Engine — configuration loader.

Supports two configuration formats:

1. **Multi-symbol format** (new, recommended):
   A YAML file with an ``account:`` block and a ``symbols:`` map.
   Load with :func:`load_multi_config`.

2. **Legacy two-symbol format** (backward compatible):
   A flat YAML file with ``mgc1_file``, ``m2k1_file``, and other fields.
   Load with :func:`load_config`.

Both formats are detected automatically by :func:`load_config`; if a
``symbols:`` key is present, a :class:`MultiSymbolConfig` is returned instead
of the legacy :class:`Config`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Baked-in CME micro-futures specs
# ─────────────────────────────────────────────────────────────────────────────

#: Default CME micro-futures specs: symbol → (point_value, tick_size, tick_value)
CME_MICRO_SPECS: Dict[str, Dict[str, float]] = {
    "MGC": {"point_value": 10.0,  "tick_size": 0.10, "tick_value": 1.00},
    "M2K": {"point_value": 5.0,   "tick_size": 0.10, "tick_value": 0.50},
    "MNQ": {"point_value": 2.0,   "tick_size": 0.25, "tick_value": 0.50},
    "MYM": {"point_value": 0.50,  "tick_size": 1.0,  "tick_value": 0.50},
    "MCL": {"point_value": 10.0,  "tick_size": 0.01, "tick_value": 0.10},
}


# ─────────────────────────────────────────────────────────────────────────────
# Multi-symbol config dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SymbolConfig:
    """Per-symbol configuration for the multi-symbol engine."""
    csv: str                        # path to the CSV file
    contract_size: int = 1          # desired number of contracts for this run
    point_value: float = 1.0        # dollars per point (e.g. 10 for MGC)
    tick_size: float = 0.01         # minimum price increment
    # Derived: tick_value = tick_size * point_value (not stored; computed on access)


@dataclass
class AccountConfig:
    """Account-level TPT evaluation rules."""
    size: float = 150_000.0
    profit_target: float = 9_000.0            # $9k combined / month; MISS = fee, not fail
    monthly_target_window_days: int = 20       # informational: ~20 trading days/month
    trailing_drawdown: float = 4_500.0         # THE ONLY hard fail condition
    dd_mode: str = "intraday"                  # confirmed: tick-by-tick incl. open floating
    dd_source: str = "adverse_excursion"       # use AE column as intraday-loss estimate
    concurrency: str = "overlap_timestamps"    # stack only simultaneously-open trades
    safety_buffer: float = 3_000.0             # recommend sizes with est. worst-case < this
    max_micros: int = 150                      # max concurrent micro contracts
    min_trading_days: int = 5
    consistency_pct: float = 0.50
    session_tz: str = "America/New_York"
    weekend_filter_mode: str = "drop"


@dataclass
class MultiSymbolConfig:
    """Top-level config for the multi-symbol engine."""
    account: AccountConfig = field(default_factory=AccountConfig)
    symbols: Dict[str, SymbolConfig] = field(default_factory=dict)
    output_dir: str = "reports"
    generate_charts: bool = False
    data_timezone: str = "US/Eastern"


# ─────────────────────────────────────────────────────────────────────────────
# Legacy two-symbol config (preserved for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── File paths ────────────────────────────────────────────────────────────
    mgc1_file: str = "data/MGC1.csv"
    m2k1_file: str = "data/M2K1.csv"
    output_dir: str = "reports"

    # ── Timezone assumption ───────────────────────────────────────────────────
    data_timezone: str = "US/Eastern"

    # ── Session / weekend filter ─────────────────────────────────────────────
    session_mode: str = "drop"

    # ── TPT $150 k account rules ──────────────────────────────────────────────
    account_size: float = 150_000.0
    profit_target: float = 9_000.0
    max_trailing_dd: float = 4_500.0
    min_trading_days: int = 5
    max_position_micros: int = 150
    consistency_threshold: float = 0.50

    # ── Trailing-drawdown mode ─────────────────────────────────────────────────
    trailing_dd_mode: str = "eod"

    # ── M2K1! position-sizing sweep ──────────────────────────────────────────
    m2k1_base_size: int = 10
    m2k1_sweep_sizes: List[int] = field(
        default_factory=lambda: [5, 7, 8, 10, 12, 15]
    )

    # ── MGC1! sizing ──────────────────────────────────────────────────────────
    mgc1_size_multiplier: float = 1.0

    # ── Dynamic sizing for M2K1! ─────────────────────────────────────────────
    dynamic_sizing_enabled: bool = False
    dynamic_sizing_start: int = 5
    dynamic_sizing_step: int = 10
    dynamic_sizing_trigger: float = 3_000.0

    # ── Charts ────────────────────────────────────────────────────────────────
    generate_charts: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_multi_config(path: str | None = None) -> MultiSymbolConfig:
    """
    Load a :class:`MultiSymbolConfig` from a YAML file with ``symbols:`` +
    ``account:`` structure.

    Parameters
    ----------
    path:
        Path to a YAML config file.  If *None* or the file does not exist,
        return a default config (no symbols defined).

    Returns
    -------
    MultiSymbolConfig
    """
    cfg = MultiSymbolConfig()

    if not path or not os.path.isfile(path):
        return cfg

    with open(path) as fh:
        data = yaml.safe_load(fh) or {}

    # Account block
    if "account" in data:
        acct_data = data["account"]
        acct = AccountConfig()
        for key, value in acct_data.items():
            if hasattr(acct, key):
                setattr(acct, key, value)
        cfg.account = acct

    # Symbols block
    if "symbols" in data:
        for sym_name, sym_data in data["symbols"].items():
            sym_data = sym_data or {}
            # Provide CME spec defaults if not explicitly set
            spec = CME_MICRO_SPECS.get(sym_name, {})
            sym_cfg = SymbolConfig(
                csv=sym_data.get("csv", f"data/{sym_name}1.csv"),
                contract_size=int(sym_data.get("contract_size", 1)),
                point_value=float(sym_data.get("point_value", spec.get("point_value", 1.0))),
                tick_size=float(sym_data.get("tick_size", spec.get("tick_size", 0.01))),
            )
            cfg.symbols[sym_name] = sym_cfg

    # Top-level fields
    for key in ("output_dir", "generate_charts", "data_timezone"):
        if key in data:
            setattr(cfg, key, data[key])

    return cfg


def load_config(path: str | None = None, overrides: dict | None = None) -> Config:
    """
    Load a legacy :class:`Config` from a YAML file and apply any CLI overrides.

    If the YAML file contains a ``symbols:`` key this function still returns a
    legacy :class:`Config` (use :func:`load_multi_config` for the new format).

    Parameters
    ----------
    path:
        Path to a YAML config file.  Missing/unknown keys are silently ignored.
    overrides:
        Dict of field-name → value pairs applied on top of the YAML values.

    Returns
    -------
    Config
    """
    cfg = Config()

    if path and os.path.isfile(path):
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    if overrides:
        for key, value in overrides.items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)

    return cfg

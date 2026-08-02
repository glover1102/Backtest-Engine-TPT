"""
TPT Backtesting Engine — configuration loader.

All TPT rule values and engine parameters are centralised here.
Command-line flags and YAML config files both map onto this dataclass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import yaml


@dataclass
class Config:
    # ── File paths ────────────────────────────────────────────────────────────
    mgc1_file: str = "data/MGC1.csv"
    m2k1_file: str = "data/M2K1.csv"
    output_dir: str = "reports"

    # ── Timezone assumption ───────────────────────────────────────────────────
    # The TradingView CSV timestamps are assumed to already be in this timezone.
    # Set to 'UTC' and adjust if your export is in UTC.
    data_timezone: str = "US/Eastern"

    # ── Session / weekend filter ─────────────────────────────────────────────
    # "drop"    — remove any trade whose open interval crosses a boundary.
    # "flatten" — approximate forced exit at the 4:55 PM ET boundary using the
    #             trade's recorded PnL (documented approximation; no OHLC).
    session_mode: str = "drop"

    # ── TPT $150 k account rules (do NOT change unless TPT changes its rules) ──
    account_size: float = 150_000.0
    profit_target: float = 9_000.0        # $9,000 profit target
    max_trailing_dd: float = 4_500.0      # $4,500 trailing drawdown limit
    min_trading_days: int = 5             # minimum qualifying trading days
    max_position_micros: int = 150        # max open micros at any time
    consistency_threshold: float = 0.50  # 50 % single-day cap

    # ── Trailing-drawdown mode ─────────────────────────────────────────────────
    # "eod"            — peak updates at end of each TPT trading day (default).
    # "close_to_close" — peak updates after every individual closed trade.
    trailing_dd_mode: str = "eod"

    # ── M2K1! position-sizing sweep ──────────────────────────────────────────
    # Original total position size encoded in the CSV (3 + 7 scale-in = 10).
    m2k1_base_size: int = 10
    # Effective total sizes to sweep; PnL scales linearly (documented assumption).
    m2k1_sweep_sizes: List[int] = field(
        default_factory=lambda: [5, 7, 8, 10, 12, 15]
    )

    # ── MGC1! sizing (typically 1–2 lots; keep as-is unless changing strategy) ─
    mgc1_size_multiplier: float = 1.0

    # ── Dynamic sizing for M2K1! ─────────────────────────────────────────────
    # Start at a lower size and step up once the account equity is up by the
    # trigger amount — protects the trailing drawdown early in the evaluation.
    dynamic_sizing_enabled: bool = False
    dynamic_sizing_start: int = 5         # initial M2K1! size
    dynamic_sizing_step: int = 10         # target M2K1! size after trigger
    dynamic_sizing_trigger: float = 3_000.0  # equity profit (+$3k) to trigger step-up

    # ── Charts ────────────────────────────────────────────────────────────────
    generate_charts: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str | None = None, overrides: dict | None = None) -> Config:
    """
    Load a ``Config`` from a YAML file (optional) and apply any CLI overrides.

    Parameters
    ----------
    path:
        Path to a YAML config file.  Missing/unknown keys are silently ignored.
    overrides:
        Dict of field-name → value pairs applied on top of the YAML values.

    Returns
    -------
    Config
        Populated configuration object.
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

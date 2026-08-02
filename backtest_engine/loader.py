"""
TPT Backtesting Engine — multi-symbol CSV loader.

Loads an arbitrary list of symbol CSVs (as defined in the ``symbols:`` map of
the YAML config), tags each trade with its ``symbol``, and returns a single
combined DataFrame.

Missing CSV files produce a warning and are skipped — they are not errors.
Symbols present in ``data/`` but not listed in the config are ignored.

Baseline size auto-detection
----------------------------
Each symbol's CSV may encode a multi-lot scale-in strategy.  The "natural"
(baseline) contract size is inferred from the **modal** ``Size (qty)`` value
across all exit rows.  The scaling multiplier is then:

    multiplier = configured_size / baseline_size

If ``configured_size == baseline_size`` the multiplier is 1.0 (no scaling,
no double-counting).

All scaled PnL is stored in the ``effective_pnl`` column; the original
``net_pnl`` is preserved unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .config import MultiSymbolConfig, SymbolConfig
from .parser import parse_csv

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_all_symbols(
    cfg: MultiSymbolConfig,
    data_timezone: str = "US/Eastern",
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Load CSVs for all symbols defined in *cfg.symbols*, skip missing files,
    and return a combined trade DataFrame.

    Parameters
    ----------
    cfg:
        Multi-symbol configuration.
    data_timezone:
        Timezone the CSV timestamps are assumed to be in.

    Returns
    -------
    combined_trades, present_symbols, missing_symbols
        * ``combined_trades``  — concatenated DataFrame of all loaded symbols,
          with ``effective_pnl`` and ``effective_size`` columns added.
        * ``present_symbols``  — symbols that were successfully loaded.
        * ``missing_symbols``  — symbols whose CSV was not found (warned, not
          failed).
    """
    frames: List[pd.DataFrame] = []
    present: List[str] = []
    missing: List[str] = []

    for sym_name, sym_cfg in cfg.symbols.items():
        csv_path = Path(sym_cfg.csv)
        if not csv_path.exists():
            logger.warning(
                "[%s] CSV not found at '%s' — skipping symbol.  "
                "Drop the file in place to include it automatically.",
                sym_name,
                csv_path,
            )
            missing.append(sym_name)
            continue

        try:
            raw = parse_csv(csv_path, symbol=sym_name, data_timezone=data_timezone)
        except Exception as exc:
            logger.warning("[%s] Failed to parse '%s': %s — skipping.", sym_name, csv_path, exc)
            missing.append(sym_name)
            continue

        if raw.empty:
            logger.warning("[%s] CSV parsed but contains no trade legs — skipping.", sym_name)
            missing.append(sym_name)
            continue

        # Auto-detect baseline size and apply per-symbol scaling
        raw = _apply_symbol_sizing(raw, sym_name, sym_cfg)

        frames.append(raw)
        present.append(sym_name)
        logger.info("[%s] Loaded %d trade legs.", sym_name, len(raw))

    if not frames:
        empty = pd.DataFrame(
            columns=[
                "trade_number", "symbol", "entry_time", "exit_time",
                "entry_price", "exit_price", "size_qty", "net_pnl",
                "commission", "adverse_excursion", "favorable_excursion",
                "cumulative_pnl", "leg_type", "effective_pnl", "effective_size",
                "baseline_size",
            ]
        )
        return empty, present, missing

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("exit_time").reset_index(drop=True)

    logger.info(
        "Loaded %d total trade legs across %d symbol(s): %s. "
        "Missing: %s.",
        len(combined),
        len(present),
        present,
        missing if missing else "none",
    )
    return combined, present, missing


def detect_baseline_size(trades: pd.DataFrame) -> int:
    """
    Return the modal ``Size (qty)`` across exit rows as the baseline contract
    size.

    Uses the mode of the ``size_qty`` column.  Falls back to 1 if the column
    is absent or all-NaN.
    """
    if trades.empty or "size_qty" not in trades.columns:
        return 1
    qty = trades["size_qty"].dropna()
    if qty.empty:
        return 1
    mode_val = qty.mode()
    if mode_val.empty:
        return max(1, int(round(qty.median())))
    return max(1, int(round(float(mode_val.iloc[0]))))


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_symbol_sizing(
    trades: pd.DataFrame,
    sym_name: str,
    sym_cfg: SymbolConfig,
) -> pd.DataFrame:
    """
    Add ``effective_pnl``, ``effective_size``, and ``baseline_size`` columns.

    Scaling rule (first-order linear approximation):
        multiplier  = configured_size / baseline_size
        effective_pnl  = net_pnl  × multiplier
        effective_size = size_qty × multiplier

    When configured_size == baseline_size the multiplier is exactly 1.0
    (no double-scaling).

    The linear scaling approximation assumes:
    * Slippage and commission scale linearly with contract count.
    * No partial-fill or liquidity constraints at the target size.
    * The same entry/exit price would have been achieved at the larger size.
    """
    df = trades.copy()
    baseline = detect_baseline_size(df)
    configured = sym_cfg.contract_size
    multiplier = configured / baseline if baseline > 0 else 1.0

    df["baseline_size"] = baseline
    df["effective_size"] = df["size_qty"] * multiplier
    df["effective_pnl"] = df["net_pnl"] * multiplier

    # Scale adverse/favorable excursion by the same multiplier
    if "adverse_excursion" in df.columns:
        df["effective_ae"] = df["adverse_excursion"] * multiplier
    if "favorable_excursion" in df.columns:
        df["effective_fe"] = df["favorable_excursion"] * multiplier

    logger.debug(
        "[%s] Baseline size=%d, configured=%d, multiplier=%.4f.",
        sym_name,
        baseline,
        configured,
        multiplier,
    )
    return df

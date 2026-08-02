"""
TPT Backtesting Engine — position sizing.

Multi-symbol per-symbol sizing
===============================
Each symbol's trade log may encode a multi-lot scale-in strategy.  The
baseline size is auto-detected from the modal ``Size (qty)`` across exit rows
(see :func:`~backtest_engine.loader.detect_baseline_size`).

The scaling multiplier for each symbol is:

    multiplier = configured_size / baseline_size

If ``configured_size == baseline_size`` the multiplier is 1.0 (no scaling,
no double-counting).

``effective_pnl`` and ``effective_size`` columns are added.  The original
``net_pnl`` and ``size_qty`` columns are preserved unchanged.

Linear scaling approximation
-----------------------------
* Slippage and commission scale linearly with contract count.
* No partial-fill or liquidity constraints.
* The same entry/exit price would have been achieved at the larger size.

These assumptions hold well for micro futures in normal market conditions but
may overstate returns during low-liquidity periods.

Concurrency-cap check
=====================
:func:`check_concurrent_position_limit` evaluates the maximum *simultaneous*
open contract count (sum of effective_size across all overlapping open
intervals).  This is more accurate than checking per-trade size because TPT's
150-micro cap applies to ALL symbols combined at any given instant.

Legacy two-symbol sizing functions are preserved for backward compatibility.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Static sizing
# ─────────────────────────────────────────────────────────────────────────────

def apply_static_sizing(
    trades: pd.DataFrame,
    *,
    symbol: str,
    base_size: int,
    target_size: int,
    mgc1_multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    Apply a static size multiplier to trades for *symbol*.

    For ``symbol == "M2K1"``:  ``multiplier = target_size / base_size``.
    For ``symbol == "MGC1"``:  ``multiplier = mgc1_multiplier``.

    Adds two new columns:
    * ``effective_pnl``  — scaled net PnL.
    * ``effective_size`` — scaled position size.

    Parameters
    ----------
    trades:
        Combined DataFrame (both symbols) after session filtering.
    symbol:
        ``"M2K1"`` or ``"MGC1"`` — the symbol to scale.
    base_size:
        Original total position size in the M2K1! log (default 10).
    target_size:
        Desired total position size for the sweep step.
    mgc1_multiplier:
        Multiplier for MGC1! trades (default 1.0 = no change).

    Returns
    -------
    pd.DataFrame
        Copy of *trades* with ``effective_pnl`` and ``effective_size`` added.
    """
    df = trades.copy()
    if "effective_pnl" not in df.columns:
        df["effective_pnl"] = df["net_pnl"].copy()
        df["effective_size"] = df["size_qty"].copy()

    m2k1_mult = target_size / base_size
    logger.debug(
        "Static sizing: M2K1! %d → %d lots (×%.4f), MGC1! ×%.4f.",
        base_size,
        target_size,
        m2k1_mult,
        mgc1_multiplier,
    )

    df.loc[df["symbol"] == "M2K1", "effective_pnl"] = (
        df.loc[df["symbol"] == "M2K1", "net_pnl"] * m2k1_mult
    )
    df.loc[df["symbol"] == "M2K1", "effective_size"] = (
        df.loc[df["symbol"] == "M2K1", "size_qty"] * m2k1_mult
    )
    df.loc[df["symbol"] == "MGC1", "effective_pnl"] = (
        df.loc[df["symbol"] == "MGC1", "net_pnl"] * mgc1_multiplier
    )
    df.loc[df["symbol"] == "MGC1", "effective_size"] = (
        df.loc[df["symbol"] == "MGC1", "size_qty"] * mgc1_multiplier
    )

    return df


def apply_dynamic_sizing(
    trades: pd.DataFrame,
    *,
    base_size: int,
    start_size: int,
    step_size: int,
    trigger_profit: float,
    initial_equity: float,
    mgc1_multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    Apply dynamic sizing to M2K1! trades.

    Starts at ``start_size`` lots and switches to ``step_size`` lots once
    the running account equity exceeds ``initial_equity + trigger_profit``.
    The multiplier is applied to PnL linearly.

    Parameters
    ----------
    trades:
        Time-ordered combined DataFrame (both symbols, already session-filtered).
    base_size:
        Original total position size in the M2K1! log.
    start_size:
        M2K1! size used at the start of the evaluation.
    step_size:
        M2K1! size after the trigger threshold is reached.
    trigger_profit:
        Equity rise above initial_equity that activates the step-up.
    initial_equity:
        Starting account balance (e.g. $150,000).
    mgc1_multiplier:
        Multiplier for MGC1! trades.

    Returns
    -------
    pd.DataFrame
        Copy of *trades* with ``effective_pnl``, ``effective_size``, and
        ``dynamic_phase`` (``"start"`` or ``"step"``) columns.
    """
    df = trades.copy().sort_values("exit_time").reset_index(drop=True)
    df["effective_pnl"] = 0.0
    df["effective_size"] = df["size_qty"].copy()
    df["dynamic_phase"] = "start"

    equity = initial_equity
    trigger_level = initial_equity + trigger_profit
    stepped_up = False
    step_triggered_at: Optional[int] = None

    for idx, row in df.iterrows():
        if row["symbol"] == "MGC1":
            mult = mgc1_multiplier
        else:
            if not stepped_up and equity >= trigger_level:
                stepped_up = True
                step_triggered_at = idx
                logger.info(
                    "Dynamic sizing: stepped up M2K1! from %d to %d lots "
                    "at trade index %d (equity $%.2f).",
                    start_size, step_size, idx, equity,
                )
            phase_size = step_size if stepped_up else start_size
            mult = phase_size / base_size
            df.at[idx, "dynamic_phase"] = "step" if stepped_up else "start"

        df.at[idx, "effective_pnl"] = row["net_pnl"] * mult
        df.at[idx, "effective_size"] = row["size_qty"] * mult
        equity += df.at[idx, "effective_pnl"]

    if step_triggered_at is not None:
        logger.info(
            "Dynamic sizing summary: %d trade(s) at %d lots, %d at %d lots.",
            step_triggered_at,
            start_size,
            len(df) - step_triggered_at,
            step_size,
        )
    else:
        logger.info(
            "Dynamic sizing: trigger never reached; all %d trades ran at %d lots.",
            len(df),
            start_size,
        )

    return df


def check_position_limit(
    trades: pd.DataFrame,
    max_micros: int = 150,
) -> bool:
    """
    Check that no individual trade leg exceeds ``max_micros`` contracts.

    Returns True if all trades are within the limit, False otherwise.
    Also logs any violations.
    """
    if "effective_size" not in trades.columns:
        check_col = "size_qty"
    else:
        check_col = "effective_size"

    violations = trades[trades[check_col] > max_micros]
    if not violations.empty:
        logger.warning(
            "Position-limit breach: %d trade leg(s) exceed %d micros.\n%s",
            len(violations),
            max_micros,
            violations[["symbol", "exit_time", check_col]].to_string(),
        )
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Multi-symbol concurrent position-cap check
# ─────────────────────────────────────────────────────────────────────────────

def check_concurrent_position_limit(
    trades: pd.DataFrame,
    max_micros: int = 150,
) -> Tuple[bool, float, Optional[pd.Timestamp], List[str]]:
    """
    Check the CONCURRENT micro-contract count across all symbols.

    Unlike :func:`check_position_limit` (which checks per-trade-leg size),
    this function computes the maximum *simultaneous* open contract count by
    summing ``effective_size`` across all trades whose ``[entry_time, exit_time]``
    intervals overlap at any point in time.

    Parameters
    ----------
    trades:
        Combined (all symbols), session-filtered, sized DataFrame with
        ``entry_time``, ``exit_time``, ``effective_size``, and ``symbol``.
    max_micros:
        Concurrent micro-contract cap (150 for TPT $150k).

    Returns
    -------
    within_limit, peak_concurrent, peak_time, peak_symbols
        * ``within_limit``     — True if peak concurrent ≤ max_micros.
        * ``peak_concurrent``  — highest simultaneous contract count observed.
        * ``peak_time``        — timestamp when the peak occurred.
        * ``peak_symbols``     — symbols that were open at the peak.
    """
    if trades.empty:
        return True, 0.0, None, []

    size_col = "effective_size" if "effective_size" in trades.columns else "size_qty"

    # Build (timestamp, +size, symbol) for open events
    # and (timestamp, -size, symbol) for close events.
    events: List[Tuple[pd.Timestamp, float, str]] = []
    for _, row in trades.iterrows():
        sz = float(row[size_col])
        sym = str(row["symbol"])
        events.append((row["entry_time"], +sz, sym))
        events.append((row["exit_time"],  -sz, sym))

    # Sort: process closes before opens at the same timestamp
    events.sort(key=lambda e: (e[0], e[1]))  # closes are negative → before opens

    current = 0.0
    peak = 0.0
    peak_time: Optional[pd.Timestamp] = None
    peak_symbols: List[str] = []
    open_symbols: Dict[str, int] = {}  # sym → count of open legs

    for ts, delta, sym in events:
        current += delta
        if delta > 0:
            open_symbols[sym] = open_symbols.get(sym, 0) + 1
        else:
            cnt = open_symbols.get(sym, 0) - 1
            if cnt <= 0:
                open_symbols.pop(sym, None)
            else:
                open_symbols[sym] = cnt

        if current > peak:
            peak = current
            peak_time = ts
            peak_symbols = list(open_symbols.keys())

    within_limit = peak <= max_micros
    if not within_limit:
        logger.warning(
            "Concurrent position-limit breach: peak %.0f micros > %d cap "
            "(at %s, symbols: %s).",
            peak,
            max_micros,
            peak_time,
            peak_symbols,
        )
    else:
        logger.info(
            "Concurrent position check OK: peak %.0f micros ≤ %d cap.",
            peak,
            max_micros,
        )

    return within_limit, peak, peak_time, peak_symbols


def rescale_symbol(
    trades: pd.DataFrame,
    symbol: str,
    baseline_size: int,
    target_size: int,
) -> pd.DataFrame:
    """
    Rescale a single symbol's trades to *target_size* from *baseline_size*.

    Adds/updates ``effective_pnl``, ``effective_size``, and (if present)
    ``effective_ae`` / ``effective_fe`` columns.

    Parameters
    ----------
    trades:
        Combined multi-symbol DataFrame that already has ``effective_pnl`` /
        ``effective_size`` columns (e.g. from :func:`~backtest_engine.loader.load_all_symbols`).
    symbol:
        The symbol name to rescale (e.g. ``"M2K"``).
    baseline_size:
        The currently assumed baseline (usually the auto-detected modal size).
    target_size:
        The desired target size.

    Returns
    -------
    pd.DataFrame
        Copy with updated effective columns for *symbol*; other symbols
        are unchanged.
    """
    df = trades.copy()
    if "effective_pnl" not in df.columns:
        df["effective_pnl"] = df["net_pnl"].copy()
    if "effective_size" not in df.columns:
        df["effective_size"] = df["size_qty"].copy()

    mult = target_size / baseline_size if baseline_size > 0 else 1.0
    mask = df["symbol"] == symbol
    df.loc[mask, "effective_pnl"] = df.loc[mask, "net_pnl"] * mult
    df.loc[mask, "effective_size"] = df.loc[mask, "size_qty"] * mult
    if "adverse_excursion" in df.columns:
        df.loc[mask, "effective_ae"] = df.loc[mask, "adverse_excursion"] * mult
    if "favorable_excursion" in df.columns:
        df.loc[mask, "effective_fe"] = df.loc[mask, "favorable_excursion"] * mult

    logger.debug(
        "Rescaled %s: baseline=%d, target=%d, multiplier=%.4f.",
        symbol, baseline_size, target_size, mult,
    )
    return df


"""
TPT Backtesting Engine — position sizing.

M2K1! sizing sweep
==================
The M2K1! trade log was produced with a scale-in pattern of 3 + 7 = 10 lots
total (base_size = 10).  Because PnL is proportional to contract count, we can
approximate any target size by a linear multiplier:

    multiplier  = target_size / base_size
    scaled_pnl  = original_pnl  × multiplier
    scaled_size = original_size × multiplier

**This is a first-order approximation.**  It assumes:
* Slippage and commission scale linearly with contract count.
* No partial-fill or liquidity constraints.
* The same entry/exit price would have been achieved at the larger size.

These assumptions hold well for micro futures in normal market conditions but
may overstate returns during low-liquidity periods.

MGC1! stays at its native sizing (multiplier = 1.0 by default).

Dynamic sizing (optional)
=========================
Start M2K1! at ``start_size`` lots and step up to ``step_size`` lots once the
combined account equity has risen above ``initial_equity + trigger_profit``.
This protects the trailing drawdown early in the evaluation.
"""

from __future__ import annotations

import logging
from typing import Optional

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

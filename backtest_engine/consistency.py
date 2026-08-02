"""
TPT Backtesting Engine — profit consistency rule.

TPT Profit Consistency Rule (for the $150,000 Evaluation)
==========================================================
No single trading day's profit may be **≥ 50 %** of total net P/L.

Formula:

    consistency_pct = highest_profit_day / net_pl

If ``consistency_pct < 0.50`` → the account **qualifies** (consistency check
passes).

If ``consistency_pct ≥ 0.50`` → the account does **not fail outright**.
Instead the evaluation target is automatically raised:

    updated_profit_goal = net_pl × 2

The trader must keep trading until:
* net_pl reaches the updated goal, AND
* consistency_pct < 0.50.

Minimum trading days
--------------------
The evaluation also requires at least **5 trading days** with at least one
trade each.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyResult:
    """Result of the profit consistency check for one sizing scenario."""
    net_pl: float
    highest_profit_day: float
    highest_profit_day_date: Optional[date]
    consistency_pct: float
    passes_consistency: bool           # highest_day < 50 % of net_pl
    updated_profit_goal: Optional[float]  # net_pl × 2 if consistency fails
    n_trading_days: int
    passes_min_days: bool
    per_day_pnl: Dict[date, float]     # date → daily net PnL


def compute_consistency(
    trades: pd.DataFrame,
    *,
    profit_target: float = 9_000.0,
    consistency_threshold: float = 0.50,
    min_trading_days: int = 5,
) -> ConsistencyResult:
    """
    Compute the TPT profit-consistency check for the given trade set.

    Parameters
    ----------
    trades:
        Session-filtered, sized DataFrame with ``tpt_trading_day`` and
        ``effective_pnl`` columns.
    profit_target:
        Primary profit target ($9,000 for TPT $150k).
    consistency_threshold:
        Maximum allowed fraction of net P/L attributable to the single best
        day (default 0.50 = 50 %).
    min_trading_days:
        Minimum number of trading days required (default 5).

    Returns
    -------
    ConsistencyResult
    """
    if trades.empty:
        return ConsistencyResult(
            net_pl=0.0,
            highest_profit_day=0.0,
            highest_profit_day_date=None,
            consistency_pct=0.0,
            passes_consistency=True,
            updated_profit_goal=None,
            n_trading_days=0,
            passes_min_days=False,
            per_day_pnl={},
        )

    pnl_col = "effective_pnl" if "effective_pnl" in trades.columns else "net_pnl"

    per_day: Dict[date, float] = (
        trades.groupby("tpt_trading_day")[pnl_col]
        .sum()
        .to_dict()
    )

    net_pl = sum(per_day.values())
    n_days = len(per_day)

    if not per_day:
        return ConsistencyResult(
            net_pl=0.0,
            highest_profit_day=0.0,
            highest_profit_day_date=None,
            consistency_pct=0.0,
            passes_consistency=True,
            updated_profit_goal=None,
            n_trading_days=0,
            passes_min_days=False,
            per_day_pnl=per_day,
        )

    # Highest *profit* day (only positive days count for the consistency check)
    profit_days = {d: v for d, v in per_day.items() if v > 0}
    if not profit_days:
        return ConsistencyResult(
            net_pl=net_pl,
            highest_profit_day=0.0,
            highest_profit_day_date=None,
            consistency_pct=0.0,
            passes_consistency=True,
            updated_profit_goal=None,
            n_trading_days=n_days,
            passes_min_days=n_days >= min_trading_days,
            per_day_pnl=per_day,
        )

    best_day_date = max(profit_days, key=profit_days.__getitem__)
    best_day_pnl = profit_days[best_day_date]

    # Consistency percentage — only meaningful when net_pl > 0
    if net_pl > 0:
        consistency_pct = best_day_pnl / net_pl
    else:
        # Net P/L is zero or negative — no consistency issue (no profit to cap)
        consistency_pct = 0.0

    passes = consistency_pct < consistency_threshold

    updated_goal: Optional[float] = None
    if not passes and net_pl > 0:
        updated_goal = net_pl * 2
        logger.warning(
            "Consistency check FAILED: highest day $%.2f = %.1f%% of net P/L $%.2f "
            "(≥ %.0f%% threshold).  Updated profit goal = $%.2f.",
            best_day_pnl,
            consistency_pct * 100,
            net_pl,
            consistency_threshold * 100,
            updated_goal,
        )
    else:
        logger.info(
            "Consistency check PASSED: highest day $%.2f = %.1f%% of net P/L $%.2f "
            "(< %.0f%% threshold).",
            best_day_pnl,
            consistency_pct * 100,
            net_pl,
            consistency_threshold * 100,
        )

    return ConsistencyResult(
        net_pl=net_pl,
        highest_profit_day=best_day_pnl,
        highest_profit_day_date=best_day_date,
        consistency_pct=consistency_pct,
        passes_consistency=passes,
        updated_profit_goal=updated_goal,
        n_trading_days=n_days,
        passes_min_days=n_days >= min_trading_days,
        per_day_pnl=per_day,
    )


def profit_target_reached_after_days(
    per_day_pnl: Dict[date, float],
    profit_target: float,
) -> Optional[int]:
    """
    Return the number of trading days after which the cumulative profit first
    reaches *profit_target*, or None if it never does.
    """
    cumulative = 0.0
    for i, (_, pnl) in enumerate(sorted(per_day_pnl.items()), start=1):
        cumulative += pnl
        if cumulative >= profit_target:
            return i
    return None

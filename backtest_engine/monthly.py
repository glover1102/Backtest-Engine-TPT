"""
TPT Backtesting Engine — monthly profit-target evaluation.

TPT Monthly Profit-Target Rule (fee gate, NOT account failure)
==============================================================
* Each calendar month, combined net P/L across all symbols must reach $9,000
  within approximately 20 trading days.
* **MISSING the $9,000 target is NOT a failure.**  It means the trader pays
  another recurring fee to continue the evaluation.  The account is NOT failed.
* The evaluation is ONLY failed by the $4,500 intraday trailing drawdown breach.

Per-month evaluation
---------------------
For each calendar month present in the trade data:
1. Sum combined ``effective_pnl`` across all symbols for all trading days in
   that month.
2. Apply the consistency-adjustment: if the best single day ≥ 50% of net P/L,
   the effective target is ``net_pl × 2`` (not a failure; just a higher bar).
3. PASS  — combined net P/L ≥ effective target AND trading days ≥ min_days.
4. MISS  — combined net P/L < effective target, OR insufficient trading days.
   (A MISS means: pay the recurring fee and continue.  NOT account failure.)

Output
-------
* Per-month table: month, trading_days, combined_pnl, effective_target, result.
* Overall monthly pass-rate % (PASS months / total months with ≥ min_days).
* Explicitly labelled: miss = fee, not account failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Prominent label for all monthly output
MONTHLY_DISCLAIMER = (
    "NOTE: Monthly $9k = recurring-fee gate, NOT account failure.  "
    "MISS = pay another month's fee and continue.  "
    "ONLY a $4,500 intraday trailing drawdown breach FAILS the account."
)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MonthResult:
    """Per-calendar-month evaluation result."""
    month: str                           # "YYYY-MM"
    n_trading_days: int
    combined_pnl: float
    per_symbol_pnl: Dict[str, float]
    highest_day_pnl: float
    consistency_pct: float               # highest_day / net_pl (0 if net_pl ≤ 0)
    consistency_adjusted: bool           # True if effective_target = net_pl × 2
    effective_target: float              # $9,000 or net_pl×2 if consistency fails
    passes_min_days: bool
    hits_target: bool
    result: str                          # "PASS" or "MISS (pay recurring fee)"


@dataclass
class MonthlyEvalResult:
    """Full monthly evaluation result across all months."""
    months: List[MonthResult]
    total_months: int
    months_with_enough_days: int         # months with n_trading_days ≥ min_days
    pass_count: int
    miss_count: int
    pass_rate_pct: float                 # PASS / months_with_enough_days × 100
    disclaimer: str = MONTHLY_DISCLAIMER


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_monthly(
    trades: pd.DataFrame,
    *,
    profit_target: float = 9_000.0,
    consistency_threshold: float = 0.50,
    min_trading_days: int = 5,
) -> MonthlyEvalResult:
    """
    Evaluate the monthly $9,000 profit target for each calendar month.

    Parameters
    ----------
    trades:
        Session-filtered, sized DataFrame with ``tpt_trading_day``,
        ``effective_pnl``, and ``symbol`` columns.
    profit_target:
        Monthly profit target ($9,000 for TPT $150k).
    consistency_threshold:
        Best-day cap fraction (0.50 = 50%).
    min_trading_days:
        Minimum number of trading days to count a month.

    Returns
    -------
    MonthlyEvalResult
    """
    if trades.empty:
        return MonthlyEvalResult(
            months=[],
            total_months=0,
            months_with_enough_days=0,
            pass_count=0,
            miss_count=0,
            pass_rate_pct=0.0,
        )

    pnl_col = "effective_pnl" if "effective_pnl" in trades.columns else "net_pnl"

    df = trades.copy()
    df["month"] = pd.to_datetime(df["tpt_trading_day"]).dt.to_period("M").astype(str)

    month_results: List[MonthResult] = []

    for month, month_df in sorted(df.groupby("month")):
        # Trading days in this month
        trading_days = sorted(month_df["tpt_trading_day"].unique())
        n_days = len(trading_days)

        # Per-day combined PnL
        day_pnls: Dict[date, float] = (
            month_df.groupby("tpt_trading_day")[pnl_col].sum().to_dict()
        )
        combined_pnl = sum(day_pnls.values())

        # Per-symbol PnL
        per_sym: Dict[str, float] = (
            month_df.groupby("symbol")[pnl_col].sum().to_dict()
        )

        # Best profit day
        profit_days = {d: v for d, v in day_pnls.items() if v > 0}
        highest_day_pnl = max(profit_days.values()) if profit_days else 0.0

        # Consistency check
        if combined_pnl > 0 and highest_day_pnl > 0:
            consistency_pct = highest_day_pnl / combined_pnl
        else:
            consistency_pct = 0.0

        consistency_adjusted = consistency_pct >= consistency_threshold and combined_pnl > 0
        if consistency_adjusted:
            effective_target = combined_pnl * 2
        else:
            effective_target = profit_target

        passes_min_days = n_days >= min_trading_days
        hits_target = passes_min_days and combined_pnl >= effective_target

        result_str = "PASS" if hits_target else "MISS (pay recurring fee)"

        month_results.append(
            MonthResult(
                month=str(month),
                n_trading_days=n_days,
                combined_pnl=combined_pnl,
                per_symbol_pnl=per_sym,
                highest_day_pnl=highest_day_pnl,
                consistency_pct=consistency_pct,
                consistency_adjusted=consistency_adjusted,
                effective_target=effective_target,
                passes_min_days=passes_min_days,
                hits_target=hits_target,
                result=result_str,
            )
        )

    months_with_days = [m for m in month_results if m.passes_min_days]
    pass_count = sum(1 for m in months_with_days if m.hits_target)
    miss_count = len(months_with_days) - pass_count
    pass_rate = (pass_count / len(months_with_days) * 100.0) if months_with_days else 0.0

    logger.info(
        "Monthly evaluation: %d months, %d with ≥%d days, %d PASS / %d MISS → %.0f%% pass rate.  "
        "%s",
        len(month_results),
        len(months_with_days),
        min_trading_days,
        pass_count,
        miss_count,
        pass_rate,
        MONTHLY_DISCLAIMER,
    )

    return MonthlyEvalResult(
        months=month_results,
        total_months=len(month_results),
        months_with_enough_days=len(months_with_days),
        pass_count=pass_count,
        miss_count=miss_count,
        pass_rate_pct=pass_rate,
    )

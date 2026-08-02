"""
TPT Backtesting Engine — trailing drawdown engine.

TPT $150,000 Trailing Drawdown Rules
=====================================
* Maximum trailing drawdown: **$4,500**.
* The drawdown *floor* = peak equity − $4,500.
* The floor can only move **up** — it never drops when equity falls.
* If account equity falls to or below the floor on any trade close (or
  end-of-day), that day is flagged as a **BREACH** and the evaluation fails.

Close-to-close limitation
--------------------------
We do NOT have 1-minute OHLC data.  The equity curve is reconstructed from
**closed trade PnL only** (close-to-close).  Intra-trade floating drawdown
(open positions going against you before being closed) is NOT captured.

Optional AE proxy
-----------------
The ``Adverse excursion USD`` column from the CSV provides a per-trade
worst-case intra-trade move.  We use it as a conservative upper-bound proxy:

    daily_ae_proxy_dd = worst_realized_day_equity
                        + sum(daily_ae * sizing_multiplier)

This is labelled ``max_floating_dd_proxy`` in the output and is explicitly
approximate.

Two trailing-drawdown modes
---------------------------
``"eod"``            — the peak is updated at the **end of each TPT trading
                       day** (safer for the trader; mirrors how TPT actually
                       evaluates progress at the close of each session).

``"close_to_close"`` — the peak is updated after **every individual closed
                       trade**.  This is more conservative; any trade that
                       creates a new equity high immediately tightens the floor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DayResult:
    """Drawdown statistics for a single TPT trading day."""
    trading_day: date
    pnl: float                    # combined realised PnL for the day
    equity_start: float           # equity at start of day
    equity_end: float             # equity at end of day
    peak_start: float             # trailing peak at start of day
    peak_end: float               # trailing peak at end of day (after EOD update)
    floor: float                  # drawdown floor in effect at end of day
    drawdown_from_peak: float     # equity_end − peak_end  (≤ 0 means loss)
    max_drawdown_intraday: float  # worst close-to-close equity seen during day
    breach: bool                  # True if drawdown ≥ max_trailing_dd
    # AE proxy fields (approximate, requires adverse_excursion column)
    ae_proxy_worst_equity: Optional[float] = None
    ae_proxy_dd_from_peak: Optional[float] = None
    # Metadata
    n_trades: int = 0
    symbols: List[str] = field(default_factory=list)


@dataclass
class DrawdownResult:
    """Full drawdown analysis result for one sizing scenario."""
    mode: str                        # "eod" or "close_to_close"
    initial_equity: float
    final_equity: float
    peak_equity: float
    max_trailing_dd: float           # config threshold
    max_realized_dd: float           # worst drawdown_from_peak seen
    max_realized_dd_day: Optional[date]
    any_breach: bool
    breach_days: List[date]
    days: List[DayResult]
    # AE proxy (if computed)
    ae_proxy_max_dd: Optional[float] = None
    ae_proxy_breach_days: Optional[List[date]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_drawdown(
    trades: pd.DataFrame,
    *,
    initial_equity: float = 150_000.0,
    max_trailing_dd: float = 4_500.0,
    mode: str = "eod",
    include_ae_proxy: bool = True,
) -> DrawdownResult:
    """
    Compute the trailing-drawdown curve for a time-ordered trade DataFrame.

    Parameters
    ----------
    trades:
        Combined (both symbols), session-filtered, sized DataFrame with
        ``exit_time``, ``tpt_trading_day``, ``effective_pnl``, and optionally
        ``adverse_excursion`` + ``effective_size``.
    initial_equity:
        Starting account balance.
    max_trailing_dd:
        Drawdown limit that triggers a breach ($4,500 for TPT $150k).
    mode:
        ``"eod"`` or ``"close_to_close"`` — when to update the trailing peak.
    include_ae_proxy:
        If True and ``adverse_excursion`` is present, compute the optional
        conservative floating-DD proxy using per-trade adverse excursion.

    Returns
    -------
    DrawdownResult
    """
    if mode not in ("eod", "close_to_close"):
        raise ValueError(f"mode must be 'eod' or 'close_to_close', got '{mode}'.")

    if trades.empty:
        return DrawdownResult(
            mode=mode,
            initial_equity=initial_equity,
            final_equity=initial_equity,
            peak_equity=initial_equity,
            max_trailing_dd=max_trailing_dd,
            max_realized_dd=0.0,
            max_realized_dd_day=None,
            any_breach=False,
            breach_days=[],
            days=[],
        )

    df = trades.sort_values("exit_time").copy()
    pnl_col = "effective_pnl" if "effective_pnl" in df.columns else "net_pnl"

    trading_days_sorted = sorted(df["tpt_trading_day"].unique())

    equity = initial_equity
    peak = initial_equity
    day_results: List[DayResult] = []

    # AE proxy accumulators
    ae_proxy_breach_days: List[date] = []
    ae_proxy_max_dd: Optional[float] = None

    for tday in trading_days_sorted:
        day_trades = df[df["tpt_trading_day"] == tday].sort_values("exit_time")

        eq_start = equity
        peak_start = peak
        intraday_min = equity
        # The floor in effect at the START of the day (before any EOD update).
        # For EOD mode: this is the floor throughout the day.
        # For close_to_close mode: the floor may rise as winning trades close.
        floor_intraday = peak_start - max_trailing_dd

        # --- Intraday equity walk (close-to-close) ---
        intraday_breach = False
        for _, row in day_trades.iterrows():
            leg_pnl = float(row[pnl_col])
            equity += leg_pnl

            intraday_min = min(intraday_min, equity)

            # In close_to_close mode update peak after EVERY trade
            if mode == "close_to_close":
                # Check breach BEFORE updating the peak (peak not yet updated)
                if equity <= floor_intraday:
                    intraday_breach = True
                peak = max(peak, equity)
                floor_intraday = peak - max_trailing_dd
            else:
                # EOD mode: floor is fixed at floor_intraday (based on peak_start)
                if equity <= floor_intraday:
                    intraday_breach = True

        # --- EOD peak update (EOD mode only) ---
        if mode == "eod":
            peak = max(peak, equity)

        floor = peak - max_trailing_dd
        dd_from_peak = equity - peak  # ≤ 0 means drawdown

        # Breach if end-of-day equity is at/below EOD floor, OR if intraday
        # equity touched the floor that was in effect at the time.
        breach = (equity <= floor) or intraday_breach

        day_result = DayResult(
            trading_day=tday,
            pnl=equity - eq_start,
            equity_start=eq_start,
            equity_end=equity,
            peak_start=peak_start,
            peak_end=peak,
            floor=floor,
            drawdown_from_peak=dd_from_peak,
            max_drawdown_intraday=intraday_min - peak,
            breach=breach,
            n_trades=len(day_trades),
            symbols=list(day_trades["symbol"].unique()),
        )

        # --- AE proxy (optional conservative floating-DD estimate) ---
        if include_ae_proxy and "adverse_excursion" in df.columns:
            ae_col = "adverse_excursion"
            # Sum absolute AE values for the day (AE is reported as negative)
            # Scale by the effective_size / size_qty ratio to approximate sizing
            ae_sum = 0.0
            for _, row in day_trades.iterrows():
                ae_val = float(row[ae_col]) if pd.notna(row[ae_col]) else 0.0
                if "effective_size" in row.index and pd.notna(row["effective_size"]) and pd.notna(row["size_qty"]) and row["size_qty"] > 0:
                    ae_scale = float(row["effective_size"]) / float(row["size_qty"])
                else:
                    ae_scale = 1.0
                ae_sum += ae_val * ae_scale  # ae_val is already negative

            # Worst-case: assume all AE happened from the worst intraday equity point
            ae_proxy_worst = intraday_min + ae_sum  # ae_sum < 0, so this is lower
            ae_proxy_dd = ae_proxy_worst - peak
            day_result.ae_proxy_worst_equity = ae_proxy_worst
            day_result.ae_proxy_dd_from_peak = ae_proxy_dd

            if ae_proxy_worst <= floor:
                ae_proxy_breach_days.append(tday)

        day_results.append(day_result)

    # Summarise
    breach_days = [d.trading_day for d in day_results if d.breach]
    all_dd = [d.max_drawdown_intraday for d in day_results]
    worst_idx = int(pd.Series(all_dd).idxmin()) if all_dd else -1
    max_realized_dd = min(all_dd) if all_dd else 0.0
    max_realized_dd_day = day_results[worst_idx].trading_day if worst_idx >= 0 else None

    ae_max = None
    if include_ae_proxy and any(d.ae_proxy_dd_from_peak is not None for d in day_results):
        ae_values = [d.ae_proxy_dd_from_peak for d in day_results if d.ae_proxy_dd_from_peak is not None]
        ae_max = min(ae_values) if ae_values else None

    result = DrawdownResult(
        mode=mode,
        initial_equity=initial_equity,
        final_equity=equity,
        peak_equity=peak,
        max_trailing_dd=max_trailing_dd,
        max_realized_dd=max_realized_dd,
        max_realized_dd_day=max_realized_dd_day,
        any_breach=bool(breach_days),
        breach_days=breach_days,
        days=day_results,
        ae_proxy_max_dd=ae_max,
        ae_proxy_breach_days=ae_proxy_breach_days if ae_proxy_breach_days else None,
    )

    _log_summary(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log_summary(r: DrawdownResult) -> None:
    status = "BREACH" if r.any_breach else "OK"
    logger.info(
        "Drawdown [%s, %s]: final equity $%.2f | peak $%.2f | "
        "worst realized DD $%.2f | breach days: %s.",
        r.mode,
        status,
        r.final_equity,
        r.peak_equity,
        r.max_realized_dd,
        r.breach_days if r.breach_days else "none",
    )

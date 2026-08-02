"""
TPT Backtesting Engine — intraday concurrency-based drawdown estimator.

⚠️  ESTIMATE ONLY — intraday tick-by-tick breach cannot be confirmed without
    1-minute/tick OHLC data.  Size defensively using the UPPER BOUND.

Background
----------
We do NOT have 1-minute or tick OHLC.  The trade logs are close-to-close per
trade, with an ``Adverse excursion USD`` column giving each trade's WORST
intra-trade floating loss — but NOT the timestamp of that low.

Therefore:
* A true tick-by-tick trailing-drawdown breach cannot be definitively confirmed.
* This module computes a conservative **concurrency-based estimate** of the
  worst-case combined floating drawdown.
* Every drawdown verdict is labelled: "ESTIMATE ONLY — intraday tick-by-tick
  breach cannot be confirmed without 1-minute/tick OHLC."

Algorithm
---------
1.  Build a time-ordered event stream from all symbols' trades (after session
    filtering + sizing).  Each trade is represented as an interval
    [entry_time, exit_time] with:
    - ``effective_pnl`` — scaled realized PnL (recorded at exit_time)
    - ``effective_ae``  — scaled adverse excursion (worst intra-trade floating
      loss; negative value; available anytime during the open interval)

2.  Maintain a **trailing peak equity** that:
    - Starts at ``initial_equity``.
    - Ratchets up (increases) whenever a trade closes with a cumulative equity
      above the current peak.
    - NEVER moves down.

3.  At each moment in time, identify the SET of currently open trades (those
    whose [entry_time, exit_time] brackets the current moment).

4.  Compute two bounds on the combined worst-case floating equity at that moment:
    - **Upper bound** (conservative / size defensively against this):
          realized_equity + sum(effective_ae for all open trades)
      Assumes ALL concurrent trades hit their worst simultaneously.
    - **Lower bound** (optimistic):
          realized_equity + min(effective_ae for all open trades)
      Assumes only the single worst trade hits its low; others are flat.

5.  Combined floating drawdown from trailing peak:
          upper_bound_dd = upper_bound_equity - trailing_peak   (≤ 0 = drawdown)
          lower_bound_dd = lower_bound_equity - trailing_peak

6.  Flag **AT RISK** when the upper-bound drawdown reaches or exceeds:
          -safety_buffer  (e.g. −$3,000; i.e. within $1,500 of the $4,500 limit)
    Flag **BREACH ESTIMATE** when it reaches or exceeds:
          -trailing_drawdown_limit  (−$4,500)

7.  Report the **peak concurrent micros** (sum of effective_size across all
    simultaneously open trades) and whether it ever exceeds 150.

Realized-equity waypoints
--------------------------
Because we only have trade-close timestamps (not tick data), we reconstruct
the realized equity curve at close events only.  Between close events, the
realized equity is unchanged (the last close value).

The trailing peak is updated at each close event if the new realized equity
exceeds the current peak.  This is the conservative approach — in reality the
peak might ratchet more aggressively on unrealized gains, but we can only
observe realized ones.

Concurrency-event based approach
---------------------------------
We enumerate every distinct interval: open a trade → close a trade.  At each
trade-open and trade-close event we:
- Add/remove the trade from the "active set".
- Compute the current realized equity (incremented at each close).
- Compute upper/lower AE stacking for the active set.
- Check whether the estimated worst-case equity is below the DD floor.

This gives us O(n log n) complexity and ensures we check every possible
"critical point" in the concurrency timeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ── Prominent disclaimer — printed on every drawdown output ────────────────
ESTIMATE_BANNER = (
    "⚠️  ESTIMATE ONLY — intraday tick-by-tick breach cannot be confirmed "
    "without 1-minute/tick OHLC data.  Size defensively against the UPPER BOUND."
)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConcurrencyPoint:
    """Snapshot at a single concurrency-event point."""
    timestamp: pd.Timestamp
    event_type: str                      # "open" or "close"
    trade_idx: int
    symbol: str
    realized_equity: float               # equity from closed trades so far
    trailing_peak: float                 # ratcheted peak at this moment
    active_trade_indices: List[int]
    active_symbols: List[str]
    n_concurrent_micros: float           # sum of effective_size of open trades
    # AE stacking
    upper_ae_sum: float                  # sum of effective_ae (all open trades)
    lower_ae_min: float                  # min(effective_ae) — single-worst-trade bound
    upper_bound_equity: float            # realized_equity + upper_ae_sum
    lower_bound_equity: float            # realized_equity + lower_ae_min
    upper_dd_from_peak: float            # upper_bound_equity − trailing_peak (≤ 0)
    lower_dd_from_peak: float            # lower_bound_equity − trailing_peak
    at_risk_upper: bool                  # upper DD ≤ −safety_buffer
    breach_estimate_upper: bool          # upper DD ≤ −trailing_drawdown_limit


@dataclass
class ConcurrencyResult:
    """
    Full concurrency-based intraday drawdown estimation result.

    All drawdown figures are ESTIMATES based on adverse-excursion stacking.
    See module-level docstring for methodology and limitations.
    """
    # Config inputs
    initial_equity: float
    trailing_drawdown_limit: float
    safety_buffer: float

    # Realized (close-to-close) summary
    final_realized_equity: float
    peak_realized_equity: float          # peak equity at trade-close waypoints

    # Concurrency summary
    max_concurrent_micros: float
    max_concurrent_micros_time: Optional[pd.Timestamp]
    max_concurrent_micros_symbols: List[str]
    exceeds_micro_cap: bool

    # Intraday floating DD estimates (upper bound — defensive figure)
    upper_bound_worst_dd: float          # most negative upper_dd_from_peak seen
    upper_bound_worst_time: Optional[pd.Timestamp]
    upper_bound_worst_symbols: List[str]  # symbols open at worst point

    # Lower bound (optimistic)
    lower_bound_worst_dd: float

    # Verdict
    any_breach_estimate: bool            # upper bound ≥ trailing_drawdown_limit
    any_at_risk: bool                    # upper bound ≥ safety_buffer
    breach_estimate_points: List[ConcurrencyPoint]
    at_risk_points: List[ConcurrencyPoint]

    # Full timeline (one row per event)
    timeline: List[ConcurrencyPoint] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_intraday_concurrency(
    trades: pd.DataFrame,
    *,
    initial_equity: float = 150_000.0,
    trailing_drawdown_limit: float = 4_500.0,
    safety_buffer: float = 3_000.0,
    max_micros: int = 150,
) -> ConcurrencyResult:
    """
    Compute the concurrency-based intraday trailing-drawdown estimate.

    Parameters
    ----------
    trades:
        Session-filtered, sized DataFrame with ``entry_time``, ``exit_time``,
        ``effective_pnl``, ``effective_size``, ``symbol``, and ``effective_ae``
        (adverse excursion scaled to target size; negative values).
        Must contain ``tpt_trading_day``.
    initial_equity:
        Starting account balance.
    trailing_drawdown_limit:
        The hard-fail DD limit ($4,500 for TPT $150k).
    safety_buffer:
        Warn/flag when estimated worst-case DD exceeds this level ($3,000
        default, giving a $1,500 buffer below the fatal $4,500 limit).
    max_micros:
        Maximum concurrent micro contracts allowed (150 for TPT $150k).

    Returns
    -------
    ConcurrencyResult
        See class docstring for field descriptions.
    """
    if trades.empty:
        return ConcurrencyResult(
            initial_equity=initial_equity,
            trailing_drawdown_limit=trailing_drawdown_limit,
            safety_buffer=safety_buffer,
            final_realized_equity=initial_equity,
            peak_realized_equity=initial_equity,
            max_concurrent_micros=0.0,
            max_concurrent_micros_time=None,
            max_concurrent_micros_symbols=[],
            exceeds_micro_cap=False,
            upper_bound_worst_dd=0.0,
            upper_bound_worst_time=None,
            upper_bound_worst_symbols=[],
            lower_bound_worst_dd=0.0,
            any_breach_estimate=False,
            any_at_risk=False,
            breach_estimate_points=[],
            at_risk_points=[],
            timeline=[],
        )

    df = trades.copy().reset_index(drop=True)

    # Ensure effective_ae exists (fall back to adverse_excursion if needed)
    if "effective_ae" not in df.columns:
        if "adverse_excursion" in df.columns:
            df["effective_ae"] = df["adverse_excursion"]
        else:
            df["effective_ae"] = 0.0

    # Ensure effective_size exists
    if "effective_size" not in df.columns:
        df["effective_size"] = df.get("size_qty", pd.Series(1.0, index=df.index))

    # Build event list: (timestamp, event_type, trade_idx)
    # "open" events happen at entry_time; "close" events at exit_time.
    events: List[Tuple[pd.Timestamp, str, int]] = []
    for idx, row in df.iterrows():
        events.append((row["entry_time"], "open",  int(idx)))
        events.append((row["exit_time"],  "close", int(idx)))

    # Sort: close before open at the same timestamp (trade exits before new ones open)
    events.sort(key=lambda e: (e[0], 0 if e[1] == "close" else 1))

    # State
    realized_equity = initial_equity
    trailing_peak = initial_equity
    active_set: Set[int] = set()
    timeline: List[ConcurrencyPoint] = []

    # Summaries
    max_concurrent_micros = 0.0
    max_concurrent_micros_time: Optional[pd.Timestamp] = None
    max_concurrent_micros_symbols: List[str] = []

    upper_bound_worst_dd = 0.0
    upper_bound_worst_time: Optional[pd.Timestamp] = None
    upper_bound_worst_symbols: List[str] = []

    lower_bound_worst_dd = 0.0

    breach_estimate_points: List[ConcurrencyPoint] = []
    at_risk_points: List[ConcurrencyPoint] = []

    for ts, evt, idx in events:
        row = df.iloc[idx]

        if evt == "close":
            # Update realized equity BEFORE removing from active set
            realized_equity += float(row["effective_pnl"])
            trailing_peak = max(trailing_peak, realized_equity)
            active_set.discard(idx)
        else:  # open
            active_set.add(idx)

        # Snapshot active set
        active_list = sorted(active_set)
        active_symbols = list(df.iloc[active_list]["symbol"].unique()) if active_list else []

        # Concurrent micro count
        concurrent_micros = float(
            df.iloc[active_list]["effective_size"].sum() if active_list else 0.0
        )

        # AE stacking
        if active_list:
            ae_values = df.iloc[active_list]["effective_ae"].fillna(0.0).tolist()
            upper_ae_sum = sum(min(v, 0.0) for v in ae_values)   # sum negatives
            lower_ae_min = min(ae_values) if ae_values else 0.0   # single worst
        else:
            upper_ae_sum = 0.0
            lower_ae_min = 0.0

        upper_equity = realized_equity + upper_ae_sum
        lower_equity = realized_equity + lower_ae_min
        upper_dd = upper_equity - trailing_peak
        lower_dd = lower_equity - trailing_peak

        # Flags
        at_risk = upper_dd <= -safety_buffer
        breach_est = upper_dd <= -trailing_drawdown_limit

        pt = ConcurrencyPoint(
            timestamp=ts,
            event_type=evt,
            trade_idx=idx,
            symbol=str(row["symbol"]),
            realized_equity=realized_equity,
            trailing_peak=trailing_peak,
            active_trade_indices=active_list,
            active_symbols=active_symbols,
            n_concurrent_micros=concurrent_micros,
            upper_ae_sum=upper_ae_sum,
            lower_ae_min=lower_ae_min,
            upper_bound_equity=upper_equity,
            lower_bound_equity=lower_equity,
            upper_dd_from_peak=upper_dd,
            lower_dd_from_peak=lower_dd,
            at_risk_upper=at_risk,
            breach_estimate_upper=breach_est,
        )
        timeline.append(pt)

        # Update concurrency-cap tracking
        if concurrent_micros > max_concurrent_micros:
            max_concurrent_micros = concurrent_micros
            max_concurrent_micros_time = ts
            max_concurrent_micros_symbols = list(active_symbols)

        # Update worst-DD tracking
        if upper_dd < upper_bound_worst_dd:
            upper_bound_worst_dd = upper_dd
            upper_bound_worst_time = ts
            upper_bound_worst_symbols = list(active_symbols)

        if lower_dd < lower_bound_worst_dd:
            lower_bound_worst_dd = lower_dd

        if breach_est:
            breach_estimate_points.append(pt)
        if at_risk:
            at_risk_points.append(pt)

    exceeds_micro_cap = max_concurrent_micros > max_micros

    result = ConcurrencyResult(
        initial_equity=initial_equity,
        trailing_drawdown_limit=trailing_drawdown_limit,
        safety_buffer=safety_buffer,
        final_realized_equity=realized_equity,
        peak_realized_equity=trailing_peak,
        max_concurrent_micros=max_concurrent_micros,
        max_concurrent_micros_time=max_concurrent_micros_time,
        max_concurrent_micros_symbols=max_concurrent_micros_symbols,
        exceeds_micro_cap=exceeds_micro_cap,
        upper_bound_worst_dd=upper_bound_worst_dd,
        upper_bound_worst_time=upper_bound_worst_time,
        upper_bound_worst_symbols=upper_bound_worst_symbols,
        lower_bound_worst_dd=lower_bound_worst_dd,
        any_breach_estimate=bool(breach_estimate_points),
        any_at_risk=bool(at_risk_points),
        breach_estimate_points=breach_estimate_points,
        at_risk_points=at_risk_points,
        timeline=timeline,
    )

    _log_concurrency_summary(result)
    return result


def get_concurrent_trades_at(
    trades: pd.DataFrame,
    timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """
    Return all trades in *trades* that are open at *timestamp*
    (i.e. entry_time <= timestamp < exit_time).
    """
    mask = (trades["entry_time"] <= timestamp) & (trades["exit_time"] > timestamp)
    return trades[mask]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log_concurrency_summary(r: ConcurrencyResult) -> None:
    status = "BREACH EST." if r.any_breach_estimate else (
        "AT RISK" if r.any_at_risk else "OK"
    )
    logger.info(
        "Concurrency DD [%s | %s]: upper-bound worst DD $%.2f | "
        "lower-bound worst DD $%.2f | peak micros %.0f (cap %d) | "
        "at-risk events: %d | breach-est events: %d.",
        status,
        ESTIMATE_BANNER[:30] + "…",
        r.upper_bound_worst_dd,
        r.lower_bound_worst_dd,
        r.max_concurrent_micros,
        r.trailing_drawdown_limit + r.safety_buffer,  # cap = initial equity context
        len(r.at_risk_points),
        len(r.breach_estimate_points),
    )

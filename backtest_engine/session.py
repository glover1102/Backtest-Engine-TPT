"""
TPT Backtesting Engine — session / weekend filter.

TPT $150,000 Evaluation — trading-hours rules
=============================================
* The TPT trading day runs from **6:00 PM ET** to **5:00 PM ET** the next
  calendar day.
* Any position still open at **4:55 PM ET** is force-closed by TPT.
* A trade opened *at or after* 6:00 PM ET belongs to the **next** calendar
  trading day.
* **Weekend block:** positions may NOT be held from **Friday 17:00 ET**
  through **Sunday 18:00 ET**.

Two filter modes
----------------
``drop``    (default) — remove any trade whose open interval [entry, exit]
            overlaps with a boundary (daily 4:55 PM or weekend block).

``flatten`` — approximate forced exit: keep the trade but cap the exit time at
            the boundary and retain the originally recorded PnL as the best
            available estimate (since no 1-minute OHLC is available).
            **This is an approximation** — the true PnL at the forced-exit
            bar is unknown.  The approximation is clearly labelled in the
            output.

The engine computes close-to-close PnL only; intraday floating losses are
NOT captured.  See README for full limitation disclosure.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ── TPT boundary constants ─────────────────────────────────────────────────
_SESSION_OPEN_HOUR = 18     # 6:00 PM ET — session open
_SESSION_CLOSE_HOUR = 17    # 5:00 PM ET — official session end
_FORCE_FLAT_HOUR = 16       # 4:55 PM ET
_FORCE_FLAT_MINUTE = 55
_WEEKEND_START_WEEKDAY = 4  # Friday (0=Mon … 4=Fri)
_WEEKEND_START_HOUR = 17    # 5:00 PM ET Friday
_WEEKEND_END_WEEKDAY = 6    # Sunday
_WEEKEND_END_HOUR = 18      # 6:00 PM ET Sunday


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def assign_tpt_trading_day(entry_time: pd.Timestamp) -> date:
    """
    Return the TPT trading-day date for a trade entered at *entry_time*.

    TPT rule: a trade opened at or after 6:00 PM ET belongs to the **next**
    calendar day's trading session.

    Parameters
    ----------
    entry_time:
        Timezone-aware timestamp (must already be in ET or have tz info).

    Returns
    -------
    date
        The calendar date that *owns* this trade in TPT's books.
    """
    et = _to_et(entry_time)
    if et.hour >= _SESSION_OPEN_HOUR:
        return (et + timedelta(days=1)).date()
    return et.date()


def filter_trades(
    trades: pd.DataFrame,
    mode: str = "drop",
) -> Tuple[pd.DataFrame, dict]:
    """
    Apply TPT session and weekend filters to a trade DataFrame.

    Parameters
    ----------
    trades:
        DataFrame as returned by ``parser.parse_csv``.  Must contain
        ``entry_time`` and ``exit_time`` as tz-aware timestamps.
    mode:
        ``"drop"`` or ``"flatten"``.

    Returns
    -------
    filtered_trades, stats
        * ``filtered_trades`` — cleaned DataFrame with an added
          ``tpt_trading_day`` column (``datetime.date``).
        * ``stats`` — summary dict with ``dropped``, ``flattened``,
          ``pnl_dropped``, ``pnl_flattened`` keys.
    """
    if mode not in ("drop", "flatten"):
        raise ValueError(f"session_mode must be 'drop' or 'flatten', got '{mode}'.")

    df = trades.copy()

    # Classify each trade
    df["_weekend_violation"] = df.apply(
        lambda r: _crosses_weekend(r["entry_time"], r["exit_time"]), axis=1
    )
    df["_session_violation"] = df.apply(
        lambda r: _crosses_daily_boundary(r["entry_time"], r["exit_time"]), axis=1
    )

    any_violation = df["_weekend_violation"] | df["_session_violation"]
    violated = df[any_violation]

    stats = {
        "dropped": 0,
        "flattened": 0,
        "pnl_dropped": 0.0,
        "pnl_flattened": 0.0,
    }

    if mode == "drop":
        stats["dropped"] = int(any_violation.sum())
        stats["pnl_dropped"] = float(violated["net_pnl"].sum())
        df = df[~any_violation].copy()

        if stats["dropped"]:
            logger.info(
                "Session filter [drop]: removed %d trade legs (PnL impact: $%.2f).",
                stats["dropped"],
                stats["pnl_dropped"],
            )
    else:  # flatten
        # Mark violated trades as approximated; retain their PnL as-is.
        # APPROXIMATION: true forced-exit PnL (at 4:55 PM or Fri 17:00) is
        # unknown without 1-minute OHLC.  We retain the recorded PnL.
        stats["flattened"] = int(any_violation.sum())
        stats["pnl_flattened"] = float(violated["net_pnl"].sum())
        if stats["flattened"]:
            df.loc[any_violation, "flattened_approximation"] = True
            logger.info(
                "Session filter [flatten]: approximated %d trade legs "
                "(recorded PnL retained as approximation; PnL sum: $%.2f).  "
                "⚠️  True forced-exit PnL is unknown without 1-min OHLC.",
                stats["flattened"],
                stats["pnl_flattened"],
            )

    # Drop internal classification columns
    df = df.drop(columns=["_weekend_violation", "_session_violation"], errors="ignore")

    # Assign TPT trading day (based on entry time)
    df["tpt_trading_day"] = df["entry_time"].apply(assign_tpt_trading_day)

    # Exclude trades on weekends themselves (Saturday / Sunday) if not already removed
    df = _exclude_weekend_days(df)

    df = df.sort_values("exit_time").reset_index(drop=True)
    return df, stats


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_et(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert a tz-aware timestamp to US/Eastern."""
    if ts.tzinfo is None:
        raise ValueError(f"Timestamp {ts!r} has no timezone info.")
    return ts.tz_convert("US/Eastern")


def _get_session_force_flat(entry_time: pd.Timestamp) -> pd.Timestamp:
    """
    Return the 4:55 PM ET of the TPT session the trade entered in.

    Session logic:
    * Entry before 18:00 ET → session ends at 16:55 ET *same day*.
    * Entry at/after 18:00 ET → session ends at 16:55 ET *next* calendar day.
    """
    et = _to_et(entry_time)
    if et.hour >= _SESSION_OPEN_HOUR:
        target_date = et.date() + timedelta(days=1)
    else:
        target_date = et.date()

    force_flat = et.replace(
        year=target_date.year,
        month=target_date.month,
        day=target_date.day,
        hour=_FORCE_FLAT_HOUR,
        minute=_FORCE_FLAT_MINUTE,
        second=0,
        microsecond=0,
    )
    return force_flat


def _crosses_daily_boundary(
    entry_time: pd.Timestamp, exit_time: pd.Timestamp
) -> bool:
    """
    Return True if the trade was still open past the 4:55 PM ET force-flat.

    A trade violates the daily boundary when its exit is **after** the
    force-flat time of the session it was entered in.
    """
    force_flat = _get_session_force_flat(entry_time)
    exit_et = _to_et(exit_time)
    return exit_et > force_flat


def _crosses_weekend(
    entry_time: pd.Timestamp, exit_time: pd.Timestamp
) -> bool:
    """
    Return True if the trade interval [entry, exit] overlaps with the TPT
    weekend block: **Friday 17:00 ET → Sunday 18:00 ET**.

    We check every weekend that falls within the trade's duration to handle
    (rare) multi-week hold attempts.
    """
    entry_et = _to_et(entry_time)
    exit_et = _to_et(exit_time)

    # Walk backwards from the Sunday on/before the exit to find any overlapping
    # weekend within the trade's open interval.
    cursor = exit_et
    for _ in range(14):  # check up to two weeks back
        # Find the Friday of cursor's week
        days_since_fri = (cursor.weekday() - _WEEKEND_START_WEEKDAY) % 7
        fri_17 = cursor.replace(
            hour=_WEEKEND_START_HOUR, minute=0, second=0, microsecond=0
        ) - timedelta(days=int(days_since_fri))
        sun_18 = fri_17 + timedelta(days=2, hours=1)

        if fri_17 >= exit_et:
            # Weekend is entirely after the exit — no overlap
            cursor -= timedelta(days=7)
            continue

        if sun_18 <= entry_et:
            # Weekend is entirely before the entry — stop searching
            break

        # Overlap: entry < sun_18 AND exit > fri_17
        if entry_et < sun_18 and exit_et > fri_17:
            return True

        cursor -= timedelta(days=7)

    return False


def _exclude_weekend_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove any trades whose ``tpt_trading_day`` falls on a Saturday or Sunday.
    (These should already be gone after the weekend filter, but this is a
    safety net.)
    """
    if df.empty or "tpt_trading_day" not in df.columns:
        return df

    mask = df["tpt_trading_day"].apply(
        lambda d: d.weekday() in (5, 6)  # Saturday=5, Sunday=6
    )
    # mask may be an object-dtype boolean Series; convert to bool explicitly
    mask = mask.astype(bool)
    if mask.any():
        logger.debug(
            "Removed %d trade(s) with tpt_trading_day on Saturday/Sunday.",
            int(mask.sum()),
        )
        df = df[~mask].copy()
    return df

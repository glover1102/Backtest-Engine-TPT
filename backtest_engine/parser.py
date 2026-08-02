"""
TPT Backtesting Engine — CSV parser.

Parses TradingView "Any Strategy Converter" trade-log CSV files.

Expected CSV columns
--------------------
Trade number, Type, Date and time, Signal, Price USD, Size (qty),
Size (value), Net PnL USD, Return %, Commission USD,
Favorable excursion USD, Favorable excursion %,
Adverse excursion USD, Adverse excursion %,
Cumulative PnL USD, Cumulative PnL %, Duration (bars)

Parsing rules
-------------
* Each ``Trade number`` may span multiple rows: one (or more) Entry rows and
  one (or more) Exit rows (TP1 / TP2 partial-exit legs).
* **Only Exit rows** carry realised PnL; Entry rows are used solely to recover
  the entry timestamp and price.
* For each Exit row, the matching Entry row is the most-recent preceding row
  with the same Trade number and a Type that starts with "Entry".
* The running sum of ``Net PnL USD`` across all Exit rows should equal the
  final ``Cumulative PnL USD`` value.  Any mismatch is logged.

Returned DataFrame columns
--------------------------
trade_number, symbol, entry_time, exit_time,
entry_price, exit_price, size_qty (from exit row),
net_pnl, commission, adverse_excursion, favorable_excursion,
cumulative_pnl (as reported), leg_type (e.g. "Exit Long")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import IO, Union

import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Column name constants (raw CSV headers)
# ─────────────────────────────────────────────────────────────────────────────
_COL_TRADE_NUM = "Trade number"
_COL_TYPE = "Type"
_COL_DATETIME = "Date and time"
_COL_SIGNAL = "Signal"
_COL_PRICE = "Price USD"
_COL_SIZE_QTY = "Size (qty)"
_COL_NET_PNL = "Net PnL USD"
_COL_COMMISSION = "Commission USD"
_COL_FAVORABLE = "Favorable excursion USD"
_COL_ADVERSE = "Adverse excursion USD"
_COL_CUM_PNL = "Cumulative PnL USD"

_REQUIRED_COLS = [
    _COL_TRADE_NUM,
    _COL_TYPE,
    _COL_DATETIME,
    _COL_PRICE,
    _COL_SIZE_QTY,
    _COL_NET_PNL,
    _COL_COMMISSION,
    _COL_FAVORABLE,
    _COL_ADVERSE,
    _COL_CUM_PNL,
]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parse_csv(
    source: Union[str, Path, IO],
    symbol: str,
    data_timezone: str = "US/Eastern",
) -> pd.DataFrame:
    """
    Parse a TradingView Any-Strategy-Converter CSV and return a tidy
    DataFrame of **closed trade legs**.

    Parameters
    ----------
    source:
        File path, ``pathlib.Path``, or any file-like object readable by
        ``pandas.read_csv``.
    symbol:
        Identifier string added to the ``symbol`` column (e.g. ``"MGC1"``).
    data_timezone:
        The timezone the CSV timestamps are assumed to be in.
        The returned ``entry_time`` / ``exit_time`` columns are
        timezone-aware ``pd.Timestamp`` objects in this timezone.

    Returns
    -------
    pd.DataFrame
        One row per closed leg with columns documented in the module docstring.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    raw = _read_raw(source)
    _validate_columns(raw)

    # Parse timestamps — naive, then localise
    raw[_COL_DATETIME] = pd.to_datetime(raw[_COL_DATETIME], format="%Y-%m-%d %H:%M")
    raw[_COL_DATETIME] = raw[_COL_DATETIME].dt.tz_localize(data_timezone, ambiguous="infer")

    # Numeric coercion (some fields are blank on Entry rows)
    for col in [_COL_NET_PNL, _COL_COMMISSION, _COL_FAVORABLE, _COL_ADVERSE, _COL_CUM_PNL]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw[_COL_PRICE] = pd.to_numeric(raw[_COL_PRICE], errors="coerce")
    raw[_COL_SIZE_QTY] = pd.to_numeric(raw[_COL_SIZE_QTY], errors="coerce")
    raw[_COL_TRADE_NUM] = pd.to_numeric(raw[_COL_TRADE_NUM], errors="coerce").astype("Int64")

    trades = _pair_entries_exits(raw, symbol)
    _validate_cumulative_pnl(trades, raw)

    logger.info(
        "[%s] Parsed %d closed trade legs from %s.",
        symbol,
        len(trades),
        source if isinstance(source, (str, Path)) else "<stream>",
    )
    return trades


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_raw(source: Union[str, Path, IO]) -> pd.DataFrame:
    """Read the raw CSV, tolerating Windows-style line endings."""
    return pd.read_csv(source, dtype=str, keep_default_na=False)


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")


def _is_exit(type_str: str) -> bool:
    return str(type_str).strip().lower().startswith("exit")


def _is_entry(type_str: str) -> bool:
    return str(type_str).strip().lower().startswith("entry")


def _pair_entries_exits(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Match each Exit row to its most-recent preceding Entry row of the same
    trade number, then build the tidy trade-legs DataFrame.

    This handles:
    * Simple 1-entry / 1-exit trades.
    * Partial-exit (TP1 / TP2) legs: multiple exit rows per trade number; each
      is paired with the **same** entry (the most-recent one before the exit).
    """
    records = []

    # Index all Entry rows by trade_number for fast lookup.
    # We iterate in row order so "most-recent" is the last entry seen so far.
    latest_entry: dict[int, dict] = {}

    for _, row in raw.iterrows():
        t_num = row[_COL_TRADE_NUM]
        t_type = str(row[_COL_TYPE]).strip()

        if _is_entry(t_type):
            latest_entry[t_num] = {
                "entry_time": row[_COL_DATETIME],
                "entry_price": pd.to_numeric(row[_COL_PRICE], errors="coerce"),
                "entry_type": t_type,
            }
        elif _is_exit(t_type):
            entry_info = latest_entry.get(t_num)
            if entry_info is None:
                logger.warning(
                    "[%s] Exit for trade #%s has no matching Entry row — skipping.",
                    symbol,
                    t_num,
                )
                continue

            records.append(
                {
                    "trade_number": t_num,
                    "symbol": symbol,
                    "entry_time": entry_info["entry_time"],
                    "exit_time": row[_COL_DATETIME],
                    "entry_price": entry_info["entry_price"],
                    "exit_price": pd.to_numeric(row[_COL_PRICE], errors="coerce"),
                    "size_qty": pd.to_numeric(row[_COL_SIZE_QTY], errors="coerce"),
                    "net_pnl": pd.to_numeric(row[_COL_NET_PNL], errors="coerce"),
                    "commission": pd.to_numeric(row[_COL_COMMISSION], errors="coerce"),
                    "adverse_excursion": pd.to_numeric(row[_COL_ADVERSE], errors="coerce"),
                    "favorable_excursion": pd.to_numeric(row[_COL_FAVORABLE], errors="coerce"),
                    "cumulative_pnl": pd.to_numeric(row[_COL_CUM_PNL], errors="coerce"),
                    "leg_type": t_type,
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "trade_number", "symbol", "entry_time", "exit_time",
                "entry_price", "exit_price", "size_qty", "net_pnl",
                "commission", "adverse_excursion", "favorable_excursion",
                "cumulative_pnl", "leg_type",
            ]
        )

    df = pd.DataFrame(records)
    df = df.sort_values("exit_time").reset_index(drop=True)
    return df


def _validate_cumulative_pnl(trades: pd.DataFrame, raw: pd.DataFrame) -> None:
    """
    Cross-check the running sum of net_pnl against the Cumulative PnL column.

    The final Cumulative PnL USD value in the raw CSV should equal the sum of
    all Net PnL USD on Exit rows (within a small floating-point tolerance).
    A mismatch indicates double-counting or data issues and is logged as a
    warning.
    """
    if trades.empty:
        return

    computed_total = trades["net_pnl"].sum()

    # Last non-NaN value in the Cumulative PnL column across all rows
    cum_series = pd.to_numeric(raw[_COL_CUM_PNL], errors="coerce").dropna()
    if cum_series.empty:
        return
    reported_total = cum_series.iloc[-1]

    diff = abs(computed_total - reported_total)
    if diff > 1.0:  # allow $1 rounding tolerance
        logger.warning(
            "Cumulative PnL mismatch: computed sum of Net PnL = $%.2f, "
            "CSV reports $%.2f (difference $%.2f).  "
            "Possible double-counting or data anomaly.",
            computed_total,
            reported_total,
            diff,
        )
    else:
        logger.debug(
            "Cumulative PnL validation OK: computed $%.2f ≈ reported $%.2f.",
            computed_total,
            reported_total,
        )

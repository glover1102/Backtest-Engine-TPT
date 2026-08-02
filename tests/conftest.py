"""
Shared pytest fixtures for the TPT backtesting engine tests.

Synthetic CSV data
------------------
We use minimal in-memory CSV strings so tests never depend on real data files.

Key dates used (verified with Python datetime):
  2026-04-06 Monday
  2026-04-07 Tuesday
  2026-04-08 Wednesday
  2026-04-09 Thursday
  2026-04-10 Friday
  2026-04-11 Saturday
  2026-04-12 Sunday
  2026-04-13 Monday

All timestamps are assumed to be US/Eastern.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from backtest_engine.parser import parse_csv


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

_CSV_HEADER = (
    "Trade number,Type,Date and time,Signal,Price USD,Size (qty),Size (value),"
    "Net PnL USD,Return %,Commission USD,Favorable excursion USD,"
    "Favorable excursion %,Adverse excursion USD,Adverse excursion %,"
    "Cumulative PnL USD,Cumulative PnL %,Duration (bars)\n"
)


def _row(
    num: int,
    row_type: str,
    dt: str,
    signal: str,
    price: float,
    size: float,
    net_pnl: float = "",
    commission: float = "",
    fe: float = "",
    ae: float = "",
    cum_pnl: float = "",
) -> str:
    """Build one CSV data row."""
    size_value = round(price * size, 2) if price and size else ""
    return (
        f"{num},{row_type},{dt},{signal},{price},{size},{size_value},"
        f"{net_pnl},{'' if net_pnl == '' else round(net_pnl / (price * size) * 100, 4) if price and size else ''},"
        f"{commission},{fe},{''},  {ae},{''},"
        f"{cum_pnl},'',4\n"
    )


def make_csv(rows: list[tuple]) -> io.StringIO:
    """
    Build a CSV StringIO from a list of row tuples:
    (num, type, dt, signal, price, size, net_pnl, commission, fe, ae, cum_pnl)
    """
    lines = [_CSV_HEADER]
    for r in rows:
        num, row_type, dt, signal, price, size = r[:6]
        net_pnl = r[6] if len(r) > 6 else ""
        commission = r[7] if len(r) > 7 else ""
        fe = r[8] if len(r) > 8 else ""
        ae = r[9] if len(r) > 9 else ""
        cum_pnl = r[10] if len(r) > 10 else ""
        lines.append(_row(num, row_type, dt, signal, price, size, net_pnl, commission, fe, ae, cum_pnl))
    return io.StringIO("".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_mgc1_csv() -> io.StringIO:
    """
    A minimal MGC1 CSV with 4 trade legs:
      1 - Normal weekday trade (Mon Apr 6)
      2 - Normal weekday trade (Tue Apr 7)
      3 - Weekend trade: entry Fri Apr 10 16:00, exit Sun Apr 12 18:30  ← weekend
      4 - Session-crossing: entry Tue Apr 7 16:30, exit Tue Apr 7 17:30 ← crosses 4:55 PM
    """
    rows = [
        # Trade 1: Mon Apr 6 — normal intraday, PnL +9.75
        (1, "Entry Long", "2026-04-06 09:00", "Buy",        2000.0, 1),
        (1, "Exit Long",  "2026-04-06 10:00", "Sell",       2010.0, 1,  9.75, 0.25, 12.0, -5.0,  9.75),
        # Trade 2: Tue Apr 7 — normal intraday, PnL +9.75
        (2, "Entry Long", "2026-04-07 09:00", "Buy",        2000.0, 1),
        (2, "Exit Long",  "2026-04-07 10:00", "Sell",       2010.0, 1,  9.75, 0.25, 12.0, -5.0, 19.50),
        # Trade 3: Weekend hold — entry Fri 16:00, exit Sun 18:30
        (3, "Entry Long", "2026-04-10 16:00", "Buy",        2000.0, 1),
        (3, "Exit Long",  "2026-04-12 18:30", "Sell",       2005.0, 1, -4.75, 0.25,  3.0,-15.0, 14.75),
        # Trade 4: Session violation — entry Tue 16:30, exit Tue 17:30 (past 16:55)
        (4, "Entry Long", "2026-04-07 16:30", "Buy",        2000.0, 1),
        (4, "Exit Long",  "2026-04-07 17:30", "Sell",       2015.0, 1, 14.75, 0.25,  8.0, -3.0, 29.50),
    ]
    return make_csv(rows)


@pytest.fixture
def simple_m2k1_csv() -> io.StringIO:
    """
    A minimal M2K1 CSV with scale-in trades (3+7 lots).
    All trades are normal weekday (no session violations).
    """
    rows = [
        # Trade 5: TP1 leg — 3 lots
        (5, "Entry Long",  "2026-04-06 09:30", "Buy",  2100.0, 10),
        (5, "Exit Long",   "2026-04-06 10:30", "Sell", 2110.0, 3,  28.75, 0.75, 35.0, -10.0,  28.75),
        # Trade 5: TP2 leg — 7 lots (same trade number, same entry)
        (5, "Exit Long",   "2026-04-06 11:00", "Sell", 2115.0, 7,  68.25, 1.75, 80.0, -15.0,  97.00),
        # Trade 6: losing day on Apr 7
        (6, "Entry Short", "2026-04-07 09:30", "Sell Short", 2100.0, 10),
        (6, "Exit Short",  "2026-04-07 10:30", "Buy to Cover", 2110.0, 10, -102.50, 2.50, 5.0, -120.0, -5.50),
    ]
    return make_csv(rows)


@pytest.fixture
def parsed_mgc1(simple_mgc1_csv) -> pd.DataFrame:
    return parse_csv(simple_mgc1_csv, symbol="MGC1")


@pytest.fixture
def parsed_m2k1(simple_m2k1_csv) -> pd.DataFrame:
    return parse_csv(simple_m2k1_csv, symbol="M2K1")


@pytest.fixture
def tp_csv() -> io.StringIO:
    """CSV with TP1 / TP2 partial-exit legs for the same trade number."""
    rows = [
        (10, "Entry Long", "2026-04-08 09:00", "Buy",  3000.0, 10),
        (10, "Exit Long",  "2026-04-08 10:00", "Sell", 3010.0, 3,   28.75, 0.75, 40.0, -8.0,  28.75),
        (10, "Exit Long",  "2026-04-08 11:00", "Sell", 3020.0, 7,   68.25, 1.75, 90.0,-12.0,  97.00),
    ]
    return make_csv(rows)

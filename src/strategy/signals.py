"""
MoM Sniper Backtest Engine - Signal Generation Module

Entry logic: MOM zero-line crossover is the primary trigger.
All other conditions act as hard gates (must all be true).

LONG entry (all must be true):
  1. MOM crosses above 0
  2. Supertrend direction = +1  (close > Supertrend)
  3. EMA_fast > EMA_slow        (golden-cross macro bias)
  4. MACD histogram > 0         (momentum confirmation)
  5. RSI > 50 and RSI < 70      (healthy momentum, not overbought)
  6. OBV slope > 0              (volume confirmation)

SHORT entry (mirror of long):
  1. MOM crosses below 0
  2. Supertrend direction = -1
  3. EMA_fast < EMA_slow
  4. MACD histogram < 0
  5. RSI < 50 and RSI > 30
  6. OBV slope < 0
"""

import numpy as np
import pandas as pd
from typing import Dict


def generate_entry_signals(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Generate MoM Sniper entry signals.

    Adds columns:
        entry_signal  (+1 long, -1 short, 0 flat)
    """
    result = df.copy()

    rsi_oversold   = params.get('rsi_oversold',   30.0)
    rsi_overbought = params.get('rsi_overbought', 70.0)

    # MOM zero-line cross
    mom_cross_up   = (result['mom'] > 0) & (result['mom_prev'].fillna(0) <= 0)
    mom_cross_down = (result['mom'] < 0) & (result['mom_prev'].fillna(0) >= 0)

    # --- Long conditions ---
    long_signal = (
        mom_cross_up &
        (result['st_direction'] == 1) &
        (result['ema_bias']     == 1) &
        (result['macd_histogram'] > 0) &
        (result['rsi'] > 50) &
        (result['rsi'] < rsi_overbought) &
        (result['obv_slope'] > 0)
    )

    # --- Short conditions ---
    short_signal = (
        mom_cross_down &
        (result['st_direction'] == -1) &
        (result['ema_bias']     == -1) &
        (result['macd_histogram'] < 0) &
        (result['rsi'] < 50) &
        (result['rsi'] > rsi_oversold) &
        (result['obv_slope'] < 0)
    )

    result['entry_signal'] = np.where(long_signal,   1,
                              np.where(short_signal, -1, 0))

    return result


def generate_all_signals(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Generate all MoM Sniper entry signals.

    Args:
        df:     DataFrame with all indicators calculated.
        params: Strategy parameters.

    Returns:
        DataFrame enriched with entry_signal column.
    """
    return generate_entry_signals(df, params)

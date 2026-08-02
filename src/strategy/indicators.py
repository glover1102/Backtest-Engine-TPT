"""
MoM Sniper Backtest Engine - Technical Indicators Module

Implements all indicators used by the Momentum Sniper strategy:

1. MOM         (Momentum Oscillator)            – primary trigger
2. Supertrend                                    – directional bias filter
3. EMA Cross   (50 / 200)                       – macro trend filter
4. ATR         (Average True Range)             – volatility / TPT sizing
5. MACD                                          – momentum confirmation
6. RSI                                           – quality filter
7. OBV         (On-Balance Volume)              – volume confirmation
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _true_range(df: pd.DataFrame) -> pd.Series:
    """Return the True Range series."""
    hl  = df['high'] - df['low']
    hpc = (df['high'] - df['close'].shift(1)).abs()
    lpc = (df['low']  - df['close'].shift(1)).abs()
    return pd.concat([hl, hpc, lpc], axis=1).max(axis=1)


# ---------------------------------------------------------------------------
# Individual indicators
# ---------------------------------------------------------------------------

def calculate_mom(df: pd.DataFrame, mom_length: int = 10) -> pd.DataFrame:
    """
    Momentum Oscillator: MOM = Close - Close.shift(n)

    Adds columns:
        mom
        mom_prev   (previous bar's MOM, used for cross detection)
    """
    result = df.copy()
    result['mom']      = df['close'] - df['close'].shift(mom_length)
    result['mom_prev'] = result['mom'].shift(1)
    return result


def calculate_supertrend(df: pd.DataFrame, st_atr_period: int = 10,
                          st_multiplier: float = 3.0) -> pd.DataFrame:
    """
    Supertrend directional bias filter.

    Adds columns:
        st_upper, st_lower  – final upper/lower bands
        supertrend          – current Supertrend line value
        st_direction        – +1 (bullish) or -1 (bearish)
    """
    result = df.copy()

    tr  = _true_range(df)
    atr = tr.ewm(alpha=1.0 / st_atr_period, adjust=False).mean()

    hl2   = (df['high'] + df['low']) / 2.0
    ub_basic = hl2 + st_multiplier * atr
    lb_basic = hl2 - st_multiplier * atr

    final_ub  = ub_basic.copy()
    final_lb  = lb_basic.copy()
    direction = pd.Series(1, index=df.index, dtype=int)

    for i in range(1, len(df)):
        # Final upper band
        if (ub_basic.iloc[i] < final_ub.iloc[i - 1] or
                df['close'].iloc[i - 1] > final_ub.iloc[i - 1]):
            final_ub.iloc[i] = ub_basic.iloc[i]
        else:
            final_ub.iloc[i] = final_ub.iloc[i - 1]

        # Final lower band
        if (lb_basic.iloc[i] > final_lb.iloc[i - 1] or
                df['close'].iloc[i - 1] < final_lb.iloc[i - 1]):
            final_lb.iloc[i] = lb_basic.iloc[i]
        else:
            final_lb.iloc[i] = final_lb.iloc[i - 1]

        # Direction
        if df['close'].iloc[i] > final_ub.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df['close'].iloc[i] < final_lb.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

    result['st_upper']     = final_ub
    result['st_lower']     = final_lb
    result['supertrend']   = np.where(direction == 1, final_lb, final_ub)
    result['st_direction'] = direction

    return result


def calculate_ema(df: pd.DataFrame, ema_fast: int = 50,
                  ema_slow: int = 200) -> pd.DataFrame:
    """
    50 / 200 EMA cross for macro trend bias.

    Adds columns:
        ema_fast, ema_slow
        ema_bias  (+1 golden cross, -1 death cross)
    """
    result = df.copy()
    result['ema_fast'] = df['close'].ewm(span=ema_fast,  adjust=False).mean()
    result['ema_slow'] = df['close'].ewm(span=ema_slow,  adjust=False).mean()
    result['ema_bias'] = np.where(result['ema_fast'] > result['ema_slow'], 1, -1)
    return result


def calculate_atr(df: pd.DataFrame, atr_length: int = 14) -> pd.DataFrame:
    """
    ATR used for stop-loss and TPT level sizing.

    Adds column:
        atr
    """
    result = df.copy()
    tr = _true_range(df)
    result['atr'] = tr.ewm(alpha=1.0 / atr_length, adjust=False).mean()
    return result


def calculate_macd(df: pd.DataFrame, macd_fast: int = 12,
                   macd_slow: int = 26, macd_signal: int = 9) -> pd.DataFrame:
    """
    MACD for momentum confirmation.

    Adds columns:
        macd, macd_signal_line, macd_histogram
    """
    result = df.copy()
    ema_f = df['close'].ewm(span=macd_fast,   adjust=False).mean()
    ema_s = df['close'].ewm(span=macd_slow,   adjust=False).mean()
    result['macd']             = ema_f - ema_s
    result['macd_signal_line'] = result['macd'].ewm(span=macd_signal, adjust=False).mean()
    result['macd_histogram']   = result['macd'] - result['macd_signal_line']
    return result


def calculate_rsi(df: pd.DataFrame, rsi_length: int = 14) -> pd.DataFrame:
    """
    RSI quality filter using Wilder's smoothing.

    Adds column:
        rsi
    """
    result = df.copy()
    delta    = df['close'].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / rsi_length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_length, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    result['rsi'] = (100 - (100 / (1 + rs))).fillna(50)
    return result


def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    On-Balance Volume for volume confirmation.

    Adds columns:
        obv
        obv_slope  (1-bar difference of OBV)
    """
    result    = df.copy()
    direction = np.sign(df['close'].diff()).fillna(0)
    result['obv']       = (direction * df['volume']).cumsum()
    result['obv_slope'] = result['obv'].diff()
    return result


# ---------------------------------------------------------------------------
# Composite entry
# ---------------------------------------------------------------------------

def calculate_all_indicators(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Calculate all MoM Sniper indicators.

    Args:
        df:     DataFrame with OHLCV data.
        params: Strategy parameters dict.

    Returns:
        DataFrame enriched with all indicator columns.
    """
    required = ['open', 'high', 'low', 'close', 'volume']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = df.copy()

    result = calculate_mom(result,
                           mom_length=params.get('mom_length', 10))

    result = calculate_supertrend(result,
                                  st_atr_period=params.get('st_atr_period', 10),
                                  st_multiplier=params.get('st_multiplier', 3.0))

    result = calculate_ema(result,
                           ema_fast=params.get('ema_fast', 50),
                           ema_slow=params.get('ema_slow', 200))

    result = calculate_atr(result,
                           atr_length=params.get('atr_length', 14))

    result = calculate_macd(result,
                             macd_fast=params.get('macd_fast',   12),
                             macd_slow=params.get('macd_slow',   26),
                             macd_signal=params.get('macd_signal', 9))

    result = calculate_rsi(result,
                           rsi_length=params.get('rsi_length', 14))

    result = calculate_obv(result)

    return result

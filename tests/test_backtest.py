"""
Tests for the MoM Sniper Backtest Engine.

Covers:
  - Indicator calculations (MOM, Supertrend, EMA, ATR, MACD, RSI, OBV)
  - Signal generation
  - TPT exit level calculation and partial-exit state machine
  - Full backtest run on synthetic data
  - Data loader helpers
  - Performance metrics
"""

import pytest
import numpy as np
import pandas as pd
import io
import tempfile
import os

from src.strategy.indicators import (
    calculate_mom,
    calculate_supertrend,
    calculate_ema,
    calculate_atr,
    calculate_macd,
    calculate_rsi,
    calculate_obv,
    calculate_all_indicators,
)
from src.strategy.signals import generate_all_signals
from src.strategy.exits import ExitManager, Position
from src.backtest import Backtester
from src.metrics import calculate_metrics
from src.data.loader import DataLoader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with a gentle uptrend."""
    rng = np.random.default_rng(seed)
    dates  = pd.date_range('2020-01-01', periods=n, freq='D')
    close  = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    close  = np.maximum(close, 1.0)
    spread = np.abs(rng.normal(0, 0.5, n))
    high   = close + spread
    low    = close - spread
    open_  = close - rng.normal(0, 0.3, n)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)

    return pd.DataFrame({
        'open':   open_,
        'high':   high,
        'low':    low,
        'close':  close,
        'volume': volume,
    }, index=dates)


DEFAULT_PARAMS = {
    'mom_length':     10,
    'st_atr_period':  10,
    'st_multiplier':  3.0,
    'ema_fast':       50,
    'ema_slow':       200,
    'atr_length':     14,
    'macd_fast':      12,
    'macd_slow':      26,
    'macd_signal':    9,
    'rsi_length':     14,
    'rsi_oversold':   30.0,
    'rsi_overbought': 70.0,
    # TPT
    'sl_multiplier':   1.5,
    'tpt1_multiplier': 1.0,
    'tpt2_multiplier': 2.0,
    'tpt3_multiplier': 3.0,
    'tpt1_size':       0.50,
    'tpt2_size':       0.30,
    'trail_atr_mult':  1.0,
    'use_trailing':    True,
}


# ---------------------------------------------------------------------------
# Indicator tests
# ---------------------------------------------------------------------------

class TestMOM:
    def test_basic_calculation(self):
        df = make_ohlcv(50)
        out = calculate_mom(df, mom_length=10)
        assert 'mom' in out.columns
        assert 'mom_prev' in out.columns

    def test_values_match_manual(self):
        df = make_ohlcv(50)
        out = calculate_mom(df, mom_length=5)
        expected = df['close'] - df['close'].shift(5)
        pd.testing.assert_series_equal(out['mom'], expected, check_names=False)

    def test_no_nan_after_warmup(self):
        df = make_ohlcv(50)
        out = calculate_mom(df, mom_length=5)
        assert out['mom'].iloc[5:].isna().sum() == 0


class TestSupertrend:
    def test_columns_exist(self):
        df = make_ohlcv(100)
        out = calculate_supertrend(df)
        for col in ('st_upper', 'st_lower', 'supertrend', 'st_direction'):
            assert col in out.columns

    def test_direction_values(self):
        df = make_ohlcv(100)
        out = calculate_supertrend(df)
        assert set(out['st_direction'].unique()).issubset({1, -1})

    def test_supertrend_tracks_direction(self):
        df = make_ohlcv(100)
        out = calculate_supertrend(df)
        # When bullish, supertrend == st_lower
        mask_bull = out['st_direction'] == 1
        if mask_bull.any():
            pd.testing.assert_series_equal(
                out.loc[mask_bull, 'supertrend'],
                out.loc[mask_bull, 'st_lower'],
                check_names=False,
            )


class TestEMA:
    def test_columns_exist(self):
        df = make_ohlcv(300)
        out = calculate_ema(df)
        for col in ('ema_fast', 'ema_slow', 'ema_bias'):
            assert col in out.columns

    def test_bias_values(self):
        df = make_ohlcv(300)
        out = calculate_ema(df)
        assert set(out['ema_bias'].unique()).issubset({1, -1})


class TestATR:
    def test_non_negative(self):
        df = make_ohlcv(100)
        out = calculate_atr(df)
        assert (out['atr'].dropna() >= 0).all()


class TestMACD:
    def test_columns_exist(self):
        df = make_ohlcv(100)
        out = calculate_macd(df)
        for col in ('macd', 'macd_signal_line', 'macd_histogram'):
            assert col in out.columns

    def test_histogram_is_diff(self):
        df = make_ohlcv(100)
        out = calculate_macd(df)
        expected = out['macd'] - out['macd_signal_line']
        pd.testing.assert_series_equal(out['macd_histogram'], expected, check_names=False)


class TestRSI:
    def test_range(self):
        df = make_ohlcv(100)
        out = calculate_rsi(df)
        rsi = out['rsi'].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()


class TestOBV:
    def test_columns_exist(self):
        df = make_ohlcv(100)
        out = calculate_obv(df)
        assert 'obv' in out.columns
        assert 'obv_slope' in out.columns

    def test_slope_is_obv_diff(self):
        df = make_ohlcv(100)
        out = calculate_obv(df)
        expected = out['obv'].diff()
        pd.testing.assert_series_equal(out['obv_slope'], expected, check_names=False)


class TestCalculateAllIndicators:
    def test_required_columns_added(self):
        df = make_ohlcv(300)
        out = calculate_all_indicators(df, DEFAULT_PARAMS)
        for col in ('mom', 'st_direction', 'ema_fast', 'atr',
                    'macd_histogram', 'rsi', 'obv'):
            assert col in out.columns

    def test_missing_column_raises(self):
        df = make_ohlcv(300).drop(columns=['volume'])
        with pytest.raises(ValueError, match='Missing required columns'):
            calculate_all_indicators(df, DEFAULT_PARAMS)


# ---------------------------------------------------------------------------
# Signal tests
# ---------------------------------------------------------------------------

class TestSignals:
    def test_entry_signal_values(self):
        df = make_ohlcv(300)
        df_ind = calculate_all_indicators(df, DEFAULT_PARAMS)
        df_sig = generate_all_signals(df_ind, DEFAULT_PARAMS)
        assert set(df_sig['entry_signal'].unique()).issubset({-1, 0, 1})

    def test_no_long_against_trend(self):
        """Long signals must only occur when Supertrend is bullish (+1)."""
        df = make_ohlcv(300)
        df_ind = calculate_all_indicators(df, DEFAULT_PARAMS)
        df_sig = generate_all_signals(df_ind, DEFAULT_PARAMS)
        long_bars = df_sig[df_sig['entry_signal'] == 1]
        assert (long_bars['st_direction'] == 1).all()

    def test_no_short_against_trend(self):
        """Short signals must only occur when Supertrend is bearish (-1)."""
        df = make_ohlcv(300)
        df_ind = calculate_all_indicators(df, DEFAULT_PARAMS)
        df_sig = generate_all_signals(df_ind, DEFAULT_PARAMS)
        short_bars = df_sig[df_sig['entry_signal'] == -1]
        assert (short_bars['st_direction'] == -1).all()


# ---------------------------------------------------------------------------
# TPT exit tests
# ---------------------------------------------------------------------------

class TestExitLevels:
    def test_long_levels_order(self):
        """For a long trade: SL < entry < TPT1 < TPT2 < TPT3."""
        mgr = ExitManager(DEFAULT_PARAMS)
        pos = Position(entry_price=100.0, entry_bar=0, direction=1)
        pos = mgr.calculate_exit_levels(pos, atr=1.0)

        assert pos.stop_loss < pos.entry_price < pos.tpt1 < pos.tpt2 < pos.tpt3

    def test_short_levels_order(self):
        """For a short trade: TPT3 < TPT2 < TPT1 < entry < SL."""
        mgr = ExitManager(DEFAULT_PARAMS)
        pos = Position(entry_price=100.0, entry_bar=0, direction=-1)
        pos = mgr.calculate_exit_levels(pos, atr=1.0)

        assert pos.tpt3 < pos.tpt2 < pos.tpt1 < pos.entry_price < pos.stop_loss

    def test_risk_equals_sl_distance(self):
        mgr = ExitManager(DEFAULT_PARAMS)
        pos = Position(entry_price=100.0, entry_bar=0, direction=1)
        pos = mgr.calculate_exit_levels(pos, atr=2.0)

        assert abs(pos.risk - DEFAULT_PARAMS['sl_multiplier'] * 2.0) < 1e-9

    def test_stop_loss_triggered(self):
        mgr = ExitManager(DEFAULT_PARAMS)
        pos = Position(entry_price=100.0, entry_bar=0, direction=1)
        pos = mgr.calculate_exit_levels(pos, atr=1.0)

        bar = pd.Series({'high': 99.0, 'low': 95.0, 'close': 97.0})
        should_exit, reason, exit_price, _ = mgr.check_exit(pos, bar)

        assert should_exit
        assert reason == 'stop_loss'

    def test_tpt1_triggered_long(self):
        mgr = ExitManager(DEFAULT_PARAMS)
        pos = Position(entry_price=100.0, entry_bar=0, direction=1)
        pos = mgr.calculate_exit_levels(pos, atr=1.0)

        # TPT1 = entry + tpt1_mult * risk = 100 + 1.0 * 1.5 = 101.5
        bar = pd.Series({'high': 102.0, 'low': 100.0, 'close': 101.5})
        should_exit, reason, exit_price, size = mgr.check_exit(pos, bar)

        assert should_exit
        assert reason == 'tpt1'
        assert size == pytest.approx(pos.size * DEFAULT_PARAMS['tpt1_size'])

    def test_tpt1_moves_sl_to_breakeven(self):
        mgr = ExitManager(DEFAULT_PARAMS)
        pos = Position(entry_price=100.0, entry_bar=0, direction=1)
        pos = mgr.calculate_exit_levels(pos, atr=1.0)
        pos = mgr.adjust_stop_after_partial(pos, 'tpt1')

        assert pos.tpt1_hit is True
        assert pos.stop_loss == pytest.approx(pos.entry_price)

    def test_tpt2_moves_sl_to_tpt1(self):
        mgr = ExitManager(DEFAULT_PARAMS)
        pos = Position(entry_price=100.0, entry_bar=0, direction=1)
        pos = mgr.calculate_exit_levels(pos, atr=1.0)
        pos = mgr.adjust_stop_after_partial(pos, 'tpt1')
        pos = mgr.adjust_stop_after_partial(pos, 'tpt2')

        assert pos.tpt2_hit is True
        assert pos.stop_loss == pytest.approx(pos.tpt1)

    def test_partial_sizes_sum_to_one(self):
        tpt1_s = DEFAULT_PARAMS['tpt1_size']
        tpt2_s = DEFAULT_PARAMS['tpt2_size']
        tpt3_s = 1.0 - tpt1_s - tpt2_s
        assert abs(tpt1_s + tpt2_s + tpt3_s - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Backtester integration test
# ---------------------------------------------------------------------------

class TestBacktester:
    def test_returns_expected_types(self):
        df = make_ohlcv(400)
        bt = Backtester(initial_capital=100_000)
        equity, trades = bt.run(df, DEFAULT_PARAMS)

        assert isinstance(equity, pd.Series)
        assert isinstance(trades, pd.DataFrame)

    def test_equity_starts_at_initial_capital(self):
        df = make_ohlcv(400)
        bt = Backtester(initial_capital=100_000)
        equity, _ = bt.run(df, DEFAULT_PARAMS)

        assert equity.iloc[0] == pytest.approx(100_000)

    def test_equity_always_positive(self):
        df = make_ohlcv(400)
        bt = Backtester(initial_capital=100_000)
        equity, _ = bt.run(df, DEFAULT_PARAMS)

        assert (equity > 0).all()

    def test_equity_length_matches_data(self):
        df = make_ohlcv(400)
        bt = Backtester(initial_capital=100_000)
        equity, _ = bt.run(df, DEFAULT_PARAMS)

        assert len(equity) == len(df)

    def test_trade_columns_present(self):
        df = make_ohlcv(400)
        bt = Backtester(initial_capital=100_000)
        _, trades = bt.run(df, DEFAULT_PARAMS)

        if len(trades) > 0:
            for col in ('entry_date', 'exit_date', 'entry_price', 'exit_price',
                        'direction', 'exit_reason', 'pnl', 'pnl_percent'):
                assert col in trades.columns


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_keys_present(self):
        df = make_ohlcv(400)
        bt = Backtester(initial_capital=100_000)
        equity, trades = bt.run(df, DEFAULT_PARAMS)
        metrics = calculate_metrics(equity, trades, 100_000)

        for key in ('Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)',
                    'Win Rate (%)', 'Profit Factor', 'Total Trades'):
            assert key in metrics

    def test_total_return_type(self):
        df = make_ohlcv(400)
        bt = Backtester(initial_capital=100_000)
        equity, trades = bt.run(df, DEFAULT_PARAMS)
        metrics = calculate_metrics(equity, trades, 100_000)

        assert isinstance(metrics['Total Return (%)'], float)

    def test_win_rate_in_range(self):
        df = make_ohlcv(400)
        bt = Backtester(initial_capital=100_000)
        equity, trades = bt.run(df, DEFAULT_PARAMS)
        metrics = calculate_metrics(equity, trades, 100_000)

        wr = metrics['Win Rate (%)']
        assert 0.0 <= wr <= 100.0


# ---------------------------------------------------------------------------
# DataLoader tests
# ---------------------------------------------------------------------------

class TestDataLoader:
    def _make_csv(self, content: str) -> str:
        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        )
        tmp.write(content)
        tmp.close()
        return tmp.name

    def test_load_standard_csv(self):
        dates = pd.date_range('2020-01-01', periods=60, freq='D')
        csv = "date,open,high,low,close,volume\n"
        for i, d in enumerate(dates):
            csv += f"{d.date()},{100+i},{101+i},{99+i},{100+i},{1000}\n"
        path = self._make_csv(csv)
        try:
            loader = DataLoader()
            df = loader.load_csv(path)
            assert 'close' in df.columns
            assert len(df) == 60
        finally:
            os.unlink(path)

    def test_validate_data_requires_min_rows(self):
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        csv = "date,open,high,low,close,volume\n"
        for i, d in enumerate(dates):
            csv += f"{d.date()},100,101,99,100,1000\n"
        path = self._make_csv(csv)
        try:
            loader = DataLoader()
            df = loader.load_csv(path)
            assert loader.validate_data(df) is False
        finally:
            os.unlink(path)

    def test_missing_volume_filled(self):
        dates = pd.date_range('2020-01-01', periods=60, freq='D')
        csv = "date,open,high,low,close\n"
        for i, d in enumerate(dates):
            csv += f"{d.date()},{100+i},{101+i},{99+i},{100+i}\n"
        path = self._make_csv(csv)
        try:
            loader = DataLoader()
            df = loader.load_csv(path)
            assert 'volume' in df.columns
            assert (df['volume'] == 0).all()
        finally:
            os.unlink(path)

    def test_handle_missing_data_drops_nan_rows(self):
        df = make_ohlcv(60)
        df.loc[df.index[5], 'close'] = np.nan
        loader = DataLoader()
        cleaned = loader.handle_missing_data(df)
        # forward fill should eliminate the NaN
        assert cleaned['close'].isna().sum() == 0

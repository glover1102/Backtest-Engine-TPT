"""
MoM Sniper Backtest Engine - Core Backtesting Engine

Orchestrates: data → indicators → signals → simulation → metrics.

The simulation loop supports the three-level TPT partial-exit system:
  - At TPT1: take 50%, move SL to break-even
  - At TPT2: take 30%, move SL to TPT1
  - At TPT3: take remaining 20%
  - Trailing ATR stop tracks the remaining position after TPT1
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

from .strategy.indicators import calculate_all_indicators
from .strategy.signals import generate_all_signals
from .strategy.exits import ExitManager, Position

logger = logging.getLogger(__name__)


class Backtester:
    """Core backtesting engine for the MoM Sniper strategy."""

    def __init__(self, initial_capital: float = 100_000):
        self.initial_capital = initial_capital
        self.equity_curve: pd.Series   = None
        self.trades:        pd.DataFrame = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame,
            params: Dict) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Run a full backtest.

        Args:
            df:     OHLCV DataFrame (must have open, high, low, close, volume).
            params: Strategy parameters dict.

        Returns:
            (equity_curve, trades_df)
        """
        logger.info(f"Starting backtest with {len(df)} bars")

        df_ind = calculate_all_indicators(df.copy(), params)
        df_sig = generate_all_signals(df_ind, params)

        equity_curve, trades_df = self._simulate(df_sig, params)

        self.equity_curve = equity_curve
        self.trades       = trades_df

        logger.info(f"Backtest complete. Total trades: {len(trades_df)}")
        return equity_curve, trades_df

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def _simulate(self, df: pd.DataFrame,
                  params: Dict) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Bar-by-bar simulation supporting three-level TPT partial exits.

        Each trade record captures individual partial and final exits so
        that TPT hit rates can be analysed.
        """
        equity       = [self.initial_capital]
        closed_trades: List[Dict] = []
        position: Position = None
        exit_mgr   = ExitManager(params)

        for i in range(1, len(df)):
            bar          = df.iloc[i]
            current_equity = equity[-1]

            # ---- Open new position (only when flat) ----
            if position is None and bar['entry_signal'] != 0:
                atr_val  = bar.get('atr', bar['close'] * 0.01)
                position = Position(
                    entry_price = bar['close'],
                    entry_bar   = i,
                    direction   = int(bar['entry_signal']),
                    size        = 1.0,
                )
                position = exit_mgr.calculate_exit_levels(position, atr_val)
                logger.debug(
                    f"[{df.index[i]}] ENTER "
                    f"{'LONG' if position.direction == 1 else 'SHORT'} "
                    f"@ {position.entry_price:.4f}  "
                    f"SL={position.stop_loss:.4f}  "
                    f"TPT1={position.tpt1:.4f}  "
                    f"TPT2={position.tpt2:.4f}  "
                    f"TPT3={position.tpt3:.4f}"
                )

            # ---- Manage open position ----
            if position is not None:
                atr_val  = bar.get('atr', bar['close'] * 0.01)

                if exit_mgr.use_trailing:
                    position = exit_mgr.update_trailing_stop(
                        position, bar['close'], atr_val
                    )

                should_exit, reason, exit_price, exit_size = \
                    exit_mgr.check_exit(position, bar)

                if should_exit:
                    pnl = self._calc_pnl(
                        current_equity, position.entry_price,
                        exit_price, position.direction, exit_size
                    )
                    current_equity += pnl

                    closed_trades.append({
                        'entry_date':  df.index[position.entry_bar],
                        'exit_date':   df.index[i],
                        'entry_price': position.entry_price,
                        'exit_price':  exit_price,
                        'direction':   'LONG' if position.direction == 1 else 'SHORT',
                        'exit_reason': reason,
                        'size':        exit_size,
                        'pnl':         pnl,
                        'pnl_percent': self._pnl_pct(
                            position.entry_price, exit_price, position.direction
                        ) * 100,
                        'bars_held':   i - position.entry_bar,
                        'tpt1':        position.tpt1,
                        'tpt2':        position.tpt2,
                        'tpt3':        position.tpt3,
                        'risk':        position.risk,
                    })

                    logger.debug(
                        f"[{df.index[i]}] EXIT {reason} @ {exit_price:.4f}  "
                        f"PnL={pnl:.2f}"
                    )

                    if reason in ('tpt1', 'tpt2'):
                        # Partial exit – adjust levels and keep position open
                        position = exit_mgr.adjust_stop_after_partial(
                            position, reason
                        )
                    else:
                        # Full exit – close position
                        position = None

            equity.append(current_equity)

        # ---- Close any open position at end of data ----
        if position is not None:
            exit_price = df['close'].iloc[-1]
            pnl = self._calc_pnl(
                equity[-1], position.entry_price,
                exit_price, position.direction, position.remaining_size
            )
            equity[-1] += pnl

            closed_trades.append({
                'entry_date':  df.index[position.entry_bar],
                'exit_date':   df.index[-1],
                'entry_price': position.entry_price,
                'exit_price':  exit_price,
                'direction':   'LONG' if position.direction == 1 else 'SHORT',
                'exit_reason': 'end_of_data',
                'size':        position.remaining_size,
                'pnl':         pnl,
                'pnl_percent': self._pnl_pct(
                    position.entry_price, exit_price, position.direction
                ) * 100,
                'bars_held':   len(df) - 1 - position.entry_bar,
                'tpt1':        position.tpt1,
                'tpt2':        position.tpt2,
                'tpt3':        position.tpt3,
                'risk':        position.risk,
            })

        equity_curve = pd.Series(equity, index=df.index[:len(equity)])
        trades_df    = pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame()

        return equity_curve, trades_df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pnl_pct(entry: float, exit_p: float, direction: int) -> float:
        if direction == 1:
            return (exit_p - entry) / entry
        return (entry - exit_p) / entry

    @staticmethod
    def _calc_pnl(current_equity: float, entry: float, exit_p: float,
                  direction: int, size: float) -> float:
        pct = Backtester._pnl_pct(entry, exit_p, direction)
        return current_equity * pct * size

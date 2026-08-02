"""
MoM Sniper Backtest Engine - TPT Exit Management Module

Implements the three-level Take Profit Target (TPT) exit system:

  Risk = sl_multiplier × ATR  (distance from entry to stop)

  TPT1 = entry ± tpt1_multiplier × risk   → close tpt1_size  (default 50%)
  TPT2 = entry ± tpt2_multiplier × risk   → close tpt2_size  (default 30%)
  TPT3 = entry ± tpt3_multiplier × risk   → close remaining  (default 20%)

  Stop-loss adjustments:
    After TPT1 hit → move SL to break-even
    After TPT2 hit → move SL to TPT1 level

  Trailing stop (optional):
    A trailing ATR stop activates after TPT1 is hit and tightens
    to entry ± trail_atr_mult × ATR from the best price seen.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Position data-class
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Represents an open trading position."""
    entry_price:      float
    entry_bar:        int
    direction:        int      # +1 long, -1 short
    size:             float = 1.0

    # Risk-adjusted exit levels
    stop_loss:        float = 0.0
    tpt1:             float = 0.0
    tpt2:             float = 0.0
    tpt3:             float = 0.0
    risk:             float = 0.0   # absolute price distance to initial SL

    # State tracking
    tpt1_hit:         bool  = False
    tpt2_hit:         bool  = False
    remaining_size:   float = 1.0
    best_price:       float = 0.0   # highest price seen for long (lowest for short)

    # Trailing stop (active after TPT1)
    trailing_stop:    float = 0.0


# ---------------------------------------------------------------------------
# Exit manager
# ---------------------------------------------------------------------------

class ExitManager:
    """Manages all exit logic for the MoM Sniper TPT system."""

    def __init__(self, params: Dict):
        self.sl_multiplier    = params.get('sl_multiplier',    1.5)
        self.tpt1_multiplier  = params.get('tpt1_multiplier',  1.0)
        self.tpt2_multiplier  = params.get('tpt2_multiplier',  2.0)
        self.tpt3_multiplier  = params.get('tpt3_multiplier',  3.0)
        self.tpt1_size        = params.get('tpt1_size',        0.50)  # 50 %
        self.tpt2_size        = params.get('tpt2_size',        0.30)  # 30 %
        # remaining 20 % closed at TPT3 or trailing stop
        self.trail_atr_mult   = params.get('trail_atr_mult',   1.0)
        self.use_trailing     = params.get('use_trailing',     True)

    # ------------------------------------------------------------------
    # Level calculation
    # ------------------------------------------------------------------

    def calculate_exit_levels(self, position: Position, atr: float) -> Position:
        """
        Set all exit levels for a freshly opened position.

        Args:
            position: Newly created Position object.
            atr:      ATR value at entry bar.

        Returns:
            Position with all exit levels filled in.
        """
        risk = self.sl_multiplier * atr
        d    = position.direction  # +1 long / -1 short

        position.risk         = risk
        position.stop_loss    = position.entry_price - d * risk
        position.tpt1         = position.entry_price + d * self.tpt1_multiplier * risk
        position.tpt2         = position.entry_price + d * self.tpt2_multiplier * risk
        position.tpt3         = position.entry_price + d * self.tpt3_multiplier * risk

        position.trailing_stop = position.stop_loss   # initialise at SL
        position.best_price    = position.entry_price
        position.remaining_size = position.size

        return position

    # ------------------------------------------------------------------
    # Trailing stop update
    # ------------------------------------------------------------------

    def update_trailing_stop(self, position: Position, current_price: float,
                              current_atr: float) -> Position:
        """
        Update the trailing stop after TPT1 has been hit.

        Args:
            position:      Open position.
            current_price: Current bar's close price.
            current_atr:   ATR at current bar (for ATR-based trailing).

        Returns:
            Position with updated trailing_stop and best_price.
        """
        if not position.tpt1_hit:
            return position

        d = position.direction

        if d == 1:  # long
            if current_price > position.best_price:
                position.best_price = current_price
            new_trail = position.best_price - self.trail_atr_mult * current_atr
            # Never move trailing stop backwards
            position.trailing_stop = max(position.trailing_stop, new_trail)
        else:       # short
            if current_price < position.best_price:
                position.best_price = current_price
            new_trail = position.best_price + self.trail_atr_mult * current_atr
            position.trailing_stop = min(position.trailing_stop, new_trail)

        return position

    # ------------------------------------------------------------------
    # Exit check
    # ------------------------------------------------------------------

    def check_exit(self, position: Position,
                   bar: pd.Series) -> Tuple[bool, Optional[str], float, float]:
        """
        Check whether any exit condition is triggered on the current bar.

        Priority order:
          1. Stop-loss / trailing stop
          2. TPT1 (partial)
          3. TPT2 (partial)
          4. TPT3 (full remaining)
          5. Opposite entry signal (full remaining)

        Args:
            position: Current open position.
            bar:      Current bar (requires 'high', 'low', 'close',
                      optionally 'entry_signal', 'atr').

        Returns:
            (should_exit, exit_reason, exit_price, exit_size)
        """
        high   = bar['high']
        low    = bar['low']
        close  = bar['close']
        d      = position.direction

        if d == 1:  # ------- LONG -------
            # Stop loss / trailing stop (highest priority)
            active_stop = (position.trailing_stop if position.tpt1_hit
                           else position.stop_loss)
            if low <= active_stop:
                return True, 'stop_loss', active_stop, position.remaining_size

            # TPT3 – full exit
            if not position.tpt2_hit and high >= position.tpt3:
                return True, 'tpt3', position.tpt3, position.remaining_size
            if position.tpt2_hit and high >= position.tpt3:
                return True, 'tpt3', position.tpt3, position.remaining_size

            # TPT2 – partial
            if position.tpt1_hit and not position.tpt2_hit and high >= position.tpt2:
                size = position.size * self.tpt2_size
                return True, 'tpt2', position.tpt2, size

            # TPT1 – partial
            if not position.tpt1_hit and high >= position.tpt1:
                size = position.size * self.tpt1_size
                return True, 'tpt1', position.tpt1, size

            # Opposite signal – close remaining
            if bar.get('entry_signal', 0) == -1:
                return True, 'signal_reversal', close, position.remaining_size

        else:       # ------- SHORT -------
            active_stop = (position.trailing_stop if position.tpt1_hit
                           else position.stop_loss)
            if high >= active_stop:
                return True, 'stop_loss', active_stop, position.remaining_size

            if not position.tpt2_hit and low <= position.tpt3:
                return True, 'tpt3', position.tpt3, position.remaining_size
            if position.tpt2_hit and low <= position.tpt3:
                return True, 'tpt3', position.tpt3, position.remaining_size

            if position.tpt1_hit and not position.tpt2_hit and low <= position.tpt2:
                size = position.size * self.tpt2_size
                return True, 'tpt2', position.tpt2, size

            if not position.tpt1_hit and low <= position.tpt1:
                size = position.size * self.tpt1_size
                return True, 'tpt1', position.tpt1, size

            if bar.get('entry_signal', 0) == 1:
                return True, 'signal_reversal', close, position.remaining_size

        return False, None, 0.0, 0.0

    # ------------------------------------------------------------------
    # SL adjustment after partial exits
    # ------------------------------------------------------------------

    def adjust_stop_after_partial(self, position: Position,
                                   exit_reason: str) -> Position:
        """
        Move stop loss after a partial take-profit is hit.

        TPT1 hit → SL moves to break-even
        TPT2 hit → SL moves to TPT1 level
        """
        if exit_reason == 'tpt1':
            position.tpt1_hit       = True
            # Move SL to break-even
            position.stop_loss      = position.entry_price
            position.trailing_stop  = position.entry_price
            remaining = position.size * (1.0 - self.tpt1_size)
            position.remaining_size = round(remaining, 8)

        elif exit_reason == 'tpt2':
            position.tpt2_hit       = True
            # Move SL to TPT1 level
            position.stop_loss      = position.tpt1
            position.trailing_stop  = position.tpt1
            taken = position.size * (self.tpt1_size + self.tpt2_size)
            position.remaining_size = round(position.size - taken, 8)

        return position

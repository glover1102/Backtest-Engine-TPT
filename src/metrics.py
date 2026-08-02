"""
MoM Sniper Backtest Engine - Performance Metrics Module

Calculates and formats all key metrics from a completed backtest.
"""

import numpy as np
import pandas as pd
from typing import Dict


def calculate_metrics(equity_curve: pd.Series, trades_df: pd.DataFrame,
                      initial_capital: float = 100_000) -> Dict[str, float]:
    """
    Compute comprehensive performance metrics.

    Args:
        equity_curve:    Series of portfolio equity values over time.
        trades_df:       DataFrame with one row per (partial) trade exit.
        initial_capital: Starting capital.

    Returns:
        Dictionary with performance metrics.
    """
    metrics: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Return metrics
    # ------------------------------------------------------------------
    final_equity = equity_curve.iloc[-1]
    metrics['Total Return (%)']  = (final_equity - initial_capital) / initial_capital * 100

    returns = equity_curve.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        # Annualise assuming daily bars (252 trading days)
        metrics['Sharpe Ratio'] = (returns.mean() / returns.std()) * np.sqrt(252)
        metrics['Sortino Ratio'] = (
            returns.mean() / returns[returns < 0].std() * np.sqrt(252)
            if (returns < 0).any() else 0.0
        )
    else:
        metrics['Sharpe Ratio']  = 0.0
        metrics['Sortino Ratio'] = 0.0

    # CAGR (assumes daily bars)
    n_years = len(equity_curve) / 252
    if n_years > 0 and final_equity > 0:
        metrics['CAGR (%)'] = ((final_equity / initial_capital) ** (1 / n_years) - 1) * 100
    else:
        metrics['CAGR (%)'] = 0.0

    # ------------------------------------------------------------------
    # Risk metrics
    # ------------------------------------------------------------------
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max * 100
    metrics['Max Drawdown (%)'] = drawdown.min()

    # Calmar ratio
    if metrics['Max Drawdown (%)'] < 0:
        metrics['Calmar Ratio'] = metrics['CAGR (%)'] / abs(metrics['Max Drawdown (%)'])
    else:
        metrics['Calmar Ratio'] = 0.0

    # ------------------------------------------------------------------
    # Trade metrics
    # ------------------------------------------------------------------
    if len(trades_df) > 0 and 'pnl' in trades_df.columns:
        winners = trades_df[trades_df['pnl'] > 0]
        losers  = trades_df[trades_df['pnl'] < 0]

        metrics['Total Trades']    = len(trades_df)
        metrics['Winning Trades']  = len(winners)
        metrics['Losing Trades']   = len(losers)
        metrics['Win Rate (%)']    = len(winners) / len(trades_df) * 100

        gross_profit = winners['pnl'].sum() if len(winners) > 0 else 0.0
        gross_loss   = abs(losers['pnl'].sum()) if len(losers) > 0 else 0.0
        metrics['Profit Factor'] = gross_profit / gross_loss if gross_loss > 0 else 0.0

        metrics['Avg Win ($)']  = winners['pnl'].mean()  if len(winners) > 0 else 0.0
        metrics['Avg Loss ($)'] = losers['pnl'].mean()   if len(losers)  > 0 else 0.0

        if len(winners) > 0 and len(losers) > 0:
            metrics['Avg Win/Loss Ratio'] = abs(metrics['Avg Win ($)'] / metrics['Avg Loss ($)'])
        else:
            metrics['Avg Win/Loss Ratio'] = 0.0

        # TPT hit rates
        if 'exit_reason' in trades_df.columns:
            reasons = trades_df['exit_reason'].value_counts().to_dict()
            n = len(trades_df)
            metrics['TPT1 Hit Rate (%)'] = reasons.get('tpt1', 0) / n * 100
            metrics['TPT2 Hit Rate (%)'] = reasons.get('tpt2', 0) / n * 100
            metrics['TPT3 Hit Rate (%)'] = reasons.get('tpt3', 0) / n * 100
            metrics['Stop Loss Rate (%)'] = reasons.get('stop_loss', 0) / n * 100

        # Average bars held
        if 'bars_held' in trades_df.columns:
            metrics['Avg Bars Held'] = trades_df['bars_held'].mean()

        # Expectancy per trade (in % of equity)
        if 'pnl_percent' in trades_df.columns:
            metrics['Expectancy (%)'] = trades_df['pnl_percent'].mean()

    else:
        for key in ['Total Trades', 'Winning Trades', 'Losing Trades',
                    'Win Rate (%)', 'Profit Factor', 'Avg Win ($)',
                    'Avg Loss ($)', 'Avg Win/Loss Ratio',
                    'TPT1 Hit Rate (%)', 'TPT2 Hit Rate (%)',
                    'TPT3 Hit Rate (%)', 'Stop Loss Rate (%)',
                    'Avg Bars Held', 'Expectancy (%)']:
            metrics[key] = 0.0

    return metrics


def print_metrics(metrics: Dict[str, object]) -> None:
    """Print performance metrics in a formatted table."""
    print()
    print("=" * 62)
    print("  MoM SNIPER BACKTEST — PERFORMANCE METRICS")
    print("=" * 62)

    print("\n  Return Metrics")
    print(f"    Total Return:          {metrics['Total Return (%)']:>10.2f}%")
    print(f"    CAGR:                  {metrics['CAGR (%)']:>10.2f}%")
    print(f"    Sharpe Ratio:          {metrics['Sharpe Ratio']:>10.2f}")
    print(f"    Sortino Ratio:         {metrics['Sortino Ratio']:>10.2f}")

    print("\n  Risk Metrics")
    print(f"    Max Drawdown:          {metrics['Max Drawdown (%)']:>10.2f}%")
    print(f"    Calmar Ratio:          {metrics['Calmar Ratio']:>10.2f}")

    print("\n  Trade Metrics")
    print(f"    Total Trades:          {metrics['Total Trades']:>10.0f}")
    print(f"    Winning Trades:        {metrics['Winning Trades']:>10.0f}")
    print(f"    Losing Trades:         {metrics['Losing Trades']:>10.0f}")
    print(f"    Win Rate:              {metrics['Win Rate (%)']:>10.2f}%")
    print(f"    Profit Factor:         {metrics['Profit Factor']:>10.2f}")
    print(f"    Avg Win:               ${metrics['Avg Win ($)']:>9.2f}")
    print(f"    Avg Loss:              ${metrics['Avg Loss ($)']:>9.2f}")
    print(f"    Avg Win/Loss Ratio:    {metrics['Avg Win/Loss Ratio']:>10.2f}")
    print(f"    Expectancy:            {metrics['Expectancy (%)']:>10.2f}%")

    print("\n  TPT Analysis")
    print(f"    TPT1 Hit Rate:         {metrics['TPT1 Hit Rate (%)']:>10.2f}%")
    print(f"    TPT2 Hit Rate:         {metrics['TPT2 Hit Rate (%)']:>10.2f}%")
    print(f"    TPT3 Hit Rate:         {metrics['TPT3 Hit Rate (%)']:>10.2f}%")
    print(f"    Stop Loss Rate:        {metrics['Stop Loss Rate (%)']:>10.2f}%")

    print()
    print("=" * 62)
    print()

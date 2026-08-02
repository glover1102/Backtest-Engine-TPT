"""
MoM Sniper Backtest Engine — Main Entry Point

Usage:
  python main.py --data data/ohlcv_data.csv
  python main.py --data data/SPY_5y.csv --config config/strategy_params.yaml
  python main.py --data data/SPY_5y.csv --initial-capital 50000
"""

import argparse
import yaml
import logging
from pathlib import Path

from src.data.loader import DataLoader
from src.backtest import Backtester
from src.metrics import calculate_metrics, print_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config/strategy_params.yaml') -> dict:
    """Load and flatten a YAML strategy config file."""
    with open(config_path, 'r') as fh:
        raw = yaml.safe_load(fh)

    params: dict = {}
    for section, values in raw.items():
        if isinstance(values, dict):
            params.update(values)
        else:
            params[section] = values

    return params


def main() -> None:
    parser = argparse.ArgumentParser(
        description='MoM Sniper Backtest Engine with TPT exit system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --data data/SPY_5y.csv
  python main.py --data data/BTC_daily.csv --initial-capital 10000
        """,
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/ohlcv_data.csv',
        help='Path to CSV file with OHLCV data (default: data/ohlcv_data.csv)',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/strategy_params.yaml',
        help='Path to strategy YAML configuration (default: config/strategy_params.yaml)',
    )
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=100_000,
        help='Starting capital for the backtest (default: 100000)',
    )

    args = parser.parse_args()

    try:
        logger.info(f"Loading configuration from {args.config}")
        params = load_config(args.config)

        logger.info(f"Loading data from {args.data}")
        loader = DataLoader()
        df = loader.load_csv(args.data)

        if not loader.validate_data(df):
            logger.error("Data validation failed — aborting.")
            return

        df = loader.handle_missing_data(df)
        logger.info(
            f"Loaded {len(df)} bars  "
            f"({df.index[0]} → {df.index[-1]})"
        )

        logger.info("Running backtest …")
        backtester = Backtester(initial_capital=args.initial_capital)
        equity_curve, trades_df = backtester.run(df, params)

        logger.info("Calculating performance metrics …")
        metrics = calculate_metrics(equity_curve, trades_df, args.initial_capital)
        print_metrics(metrics)

        # ------------------------------------------------------------------
        # Save outputs
        # ------------------------------------------------------------------
        if len(trades_df) > 0:
            trades_path = 'backtest_trades.csv'
            trades_df.to_csv(trades_path, index=False)
            logger.info(f"Trade log saved to {trades_path}")

        equity_path = 'backtest_equity.csv'
        equity_curve.to_csv(equity_path, header=['equity'])
        logger.info(f"Equity curve saved to {equity_path}")

    except FileNotFoundError as exc:
        logger.error(f"File not found: {exc}")
        logger.info("Make sure your data CSV exists at the specified path.")
    except Exception as exc:
        logger.error(f"Backtest failed: {exc}", exc_info=True)


if __name__ == '__main__':
    main()

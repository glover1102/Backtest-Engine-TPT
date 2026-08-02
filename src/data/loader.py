"""
MoM Sniper Backtest Engine - Data Loader Module

Handles loading and validation of OHLCV data from CSV files.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ['open', 'high', 'low', 'close']
OPTIONAL_COLUMNS = ['volume']

# Common date column aliases
DATE_COLUMN_ALIASES = ['date', 'datetime', 'time', 'timestamp', 'Date', 'Datetime', 'Time']

# Common column name aliases (lowercase)
COLUMN_ALIASES = {
    'open':   ['open', 'Open', 'OPEN', 'o'],
    'high':   ['high', 'High', 'HIGH', 'h'],
    'low':    ['low',  'Low',  'LOW',  'l'],
    'close':  ['close', 'Close', 'CLOSE', 'c', 'adj close', 'Adj Close'],
    'volume': ['volume', 'Volume', 'VOLUME', 'vol', 'Vol'],
}


class DataLoader:
    """Loads and pre-processes OHLCV data from CSV files."""

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load OHLCV data from a CSV file.

        The loader normalises column names and attempts to parse a date/time
        column into a DatetimeIndex.

        Args:
            filepath: Path to the CSV file.

        Returns:
            DataFrame with columns open, high, low, close, volume indexed by
            datetime.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If required OHLCV columns cannot be found.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath)
        logger.debug(f"Raw columns: {list(df.columns)}")

        df = self._normalise_columns(df)
        df = self._parse_date_index(df)
        df = df.sort_index()

        # Ensure numeric dtypes
        for col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to canonical lowercase names."""
        rename_map = {}
        for canonical, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in df.columns and canonical not in df.columns:
                    rename_map[alias] = canonical
                    break
        if rename_map:
            df = df.rename(columns=rename_map)

        # Add volume column filled with zeros if absent
        if 'volume' not in df.columns:
            df['volume'] = 0.0
            logger.debug("Volume column not found; defaulting to 0.")

        return df

    def _parse_date_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and parse the datetime index."""
        for alias in DATE_COLUMN_ALIASES:
            if alias in df.columns:
                df[alias] = pd.to_datetime(df[alias], format='mixed', dayfirst=False)
                df = df.set_index(alias)
                df.index.name = 'date'
                return df

        # If the existing index looks like dates, use it directly
        try:
            df.index = pd.to_datetime(df.index, format='mixed', dayfirst=False)
            df.index.name = 'date'
            return df
        except Exception:
            pass

        logger.warning("Could not identify a date column; using default integer index.")
        return df

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that the DataFrame contains the required OHLCV columns and
        has no critical data quality issues.

        Args:
            df: DataFrame to validate.

        Returns:
            True if valid, False otherwise.
        """
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False

        if len(df) < 50:
            logger.error(f"Insufficient data: {len(df)} bars (minimum 50 required)")
            return False

        # Check for negative prices
        for col in REQUIRED_COLUMNS:
            if (df[col] <= 0).any():
                logger.warning(f"Column '{col}' contains non-positive values.")

        # High >= Low sanity check
        invalid_bars = (df['high'] < df['low']).sum()
        if invalid_bars > 0:
            logger.warning(f"{invalid_bars} bars have high < low.")

        return True

    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill or drop missing values in the OHLCV DataFrame.

        Args:
            df: DataFrame with potential NaN values.

        Returns:
            Cleaned DataFrame.
        """
        initial_len = len(df)

        # Forward-fill price data, then drop any remaining NaNs in required cols
        price_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
        df[price_cols] = df[price_cols].ffill()

        # Fill missing volume with 0
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0.0)

        df = df.dropna(subset=price_cols)
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with missing price data.")

        return df

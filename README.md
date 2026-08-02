# Backtest Engine TPT — MoM Sniper

A Python backtesting framework for the **Momentum Sniper** (MoM Sniper) strategy with a three-level **Take Profit Target (TPT)** exit system.

---

## Overview

The **MoM Sniper** strategy uses the raw `MOM` (Momentum) oscillator as its primary entry trigger. A high-conviction "sniper" entry only fires when **all** of the following conditions align simultaneously:

| # | Condition | Indicator |
|---|-----------|-----------|
| 1 | MOM crosses above / below zero | **MOM Oscillator** |
| 2 | Price is above / below Supertrend | **Supertrend** |
| 3 | Golden / death EMA cross | **EMA 50 / 200** |
| 4 | MACD histogram positive / negative | **MACD** |
| 5 | RSI in healthy momentum zone (50–70 / 30–50) | **RSI** |
| 6 | OBV slope rising / falling | **OBV** |

### TPT Exit System

Risk is defined as **SL multiplier × ATR** from entry. Take-profit levels are scaled multiples of that risk:

| Level | Distance | Size Closed | SL Adjustment |
|-------|----------|-------------|---------------|
| **TPT1** | 1× risk | 50 % | Move SL to break-even |
| **TPT2** | 2× risk | 30 % | Move SL to TPT1 level |
| **TPT3** | 3× risk | 20 % (remaining) | — |
| **Stop Loss** | 1.5× ATR | 100 % remaining | — |

After TPT1 is hit an **ATR trailing stop** tracks the remaining position.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your CSV file in the data/ directory (see Data Format below)
# 3. Run the backtest
python main.py --data data/your_data.csv

# 4. View results
#    - Console: formatted performance metrics
#    - backtest_trades.csv: trade-by-trade log
#    - backtest_equity.csv: equity curve
```

---

## Project Structure

```
Backtest-Engine-TPT/
├── src/
│   ├── strategy/
│   │   ├── indicators.py   # MOM, Supertrend, EMA, ATR, MACD, RSI, OBV
│   │   ├── signals.py      # Entry signal generation
│   │   └── exits.py        # Three-level TPT exit manager
│   ├── data/
│   │   └── loader.py       # CSV loading & validation
│   ├── backtest.py         # Core simulation engine
│   └── metrics.py          # Performance metrics
├── config/
│   └── strategy_params.yaml
├── data/                   # Place CSV data files here
├── tests/
│   └── test_backtest.py
├── requirements.txt
└── main.py
```

---

## Data Format

CSV files must contain at minimum: `date`, `open`, `high`, `low`, `close`.  
`volume` is optional (defaults to 0 if absent).

```csv
date,open,high,low,close,volume
2020-01-02,100.00,101.50,99.20,100.80,1500000
2020-01-03,100.80,102.00,100.10,101.20,1200000
...
```

---

## Configuration (`config/strategy_params.yaml`)

All strategy parameters are in one YAML file grouped by function:

```yaml
mom:
  mom_length: 10          # MOM look-back period

supertrend:
  st_atr_period: 10
  st_multiplier: 3.0

ema:
  ema_fast: 50
  ema_slow: 200

tpt:
  sl_multiplier:   1.5    # SL = 1.5 × ATR from entry
  tpt1_multiplier: 1.0    # TPT1 = 1× risk
  tpt2_multiplier: 2.0    # TPT2 = 2× risk
  tpt3_multiplier: 3.0    # TPT3 = 3× risk
  tpt1_size:       0.50   # 50 % closed at TPT1
  tpt2_size:       0.30   # 30 % closed at TPT2
```

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/ohlcv_data.csv` | Path to OHLCV CSV |
| `--config` | `config/strategy_params.yaml` | Strategy configuration |
| `--initial-capital` | `100000` | Starting portfolio capital |

---

## Performance Metrics

```
==============================================================
  MoM SNIPER BACKTEST — PERFORMANCE METRICS
==============================================================

  Return Metrics
    Total Return:            XX.XX%
    CAGR:                    XX.XX%
    Sharpe Ratio:            X.XX
    Sortino Ratio:           X.XX

  Risk Metrics
    Max Drawdown:           -XX.XX%
    Calmar Ratio:            X.XX

  Trade Metrics
    Total Trades:               XX
    Win Rate:                XX.XX%
    Profit Factor:            X.XX

  TPT Analysis
    TPT1 Hit Rate:           XX.XX%
    TPT2 Hit Rate:           XX.XX%
    TPT3 Hit Rate:           XX.XX%
    Stop Loss Rate:          XX.XX%
==============================================================
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Disclaimer

This software is for educational and research purposes only.  
Past performance does not guarantee future results. Always test
thoroughly before using any strategy with real capital.

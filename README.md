# TPT Backtesting Engine

A Python backtesting and analysis engine that evaluates whether running two futures strategies together can **pass a Take Profit Trader (TPT) $150,000 evaluation account** under its real rules.

The engine ingests two TradingView "Any Strategy Converter" trade-log CSV exports (MGC1! Micro Gold and M2K1! Micro Russell), applies TPT session/weekend rules, sweeps M2K1! position sizing, and reports pass/fail against the TPT ruleset.

---

## ⚠️ Critical Limitation: Close-to-Close Only (No Intraday OHLC)

**This engine computes drawdown from closed trade PnL only (close-to-close).**

TPT's trailing drawdown is evaluated on real-time marked-to-market equity, which includes **open-position floating losses**. Since 1-minute OHLC data is not available, the engine cannot detect intraday drawdown breaches that occur *while* a position is open.

As a proxy, the engine uses the `Adverse Excursion USD` field from the CSV to produce a **conservative upper-bound estimate** of floating drawdown (`max_floating_dd_proxy`). This proxy is clearly labelled in all reports as an approximation, not a confirmed breach.

**Conclusion:** a scenario that shows no breach in this engine is *necessary but not sufficient* to confirm the evaluation would pass. You should monitor intraday equity closely in live trading.

---

## TPT $150,000 Account Rules (Implemented)

| Rule | Value |
|------|-------|
| Account size | $150,000 |
| Max position | 15 standard OR 150 micro contracts |
| Profit target | $9,000 |
| Maximum trailing drawdown | $4,500 (trails peak equity) |
| Trading hours | 6:00 PM ET → 5:00 PM ET next day |
| Auto-flatten time | 4:55 PM ET daily |
| Weekend block | Friday 17:00 ET → Sunday 18:00 ET |
| Minimum trading days | 5 (any day with ≥1 trade) |
| Profit consistency | No single day ≥ 50% of net P/L |

### Profit Consistency Rule Detail

If `highest_profit_day / net_pl ≥ 50%`, the account does **not** fail outright. Instead:

```
updated_profit_goal = net_pl × 2
```

The trader must continue trading until the highest day is `< 50%` of the updated net P/L.

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, pandas ≥ 2.0, numpy ≥ 1.24, PyYAML ≥ 6.0, pytz ≥ 2023.3, matplotlib ≥ 3.7 (optional for charts)

---

## Placing Your Data Files

Place your TradingView CSV exports in the `data/` directory:

```
data/
├── MGC1.csv    ← Any_Strategy_Converter export for COMEX:MINI:MGC1!
└── M2K1.csv    ← Strategy export for CME:MINI:M2K1!
```

Default filenames are configurable (see `--mgc1` / `--m2k1` flags or config file).

### Expected CSV Format

The files must be TradingView **"Any Strategy Converter"** exports with this header:

```
Trade number,Type,Date and time,Signal,Price USD,Size (qty),Size (value),
Net PnL USD,Return %,Commission USD,Favorable excursion USD,
Favorable excursion %,Adverse excursion USD,Adverse excursion %,
Cumulative PnL USD,Cumulative PnL %,Duration (bars)
```

- **`Type`** values: `Entry Long`, `Entry Short`, `Exit Long`, `Exit Short`
- Each Trade number may have multiple Exit rows (TP1, TP2 partial-exit legs)
- `Net PnL USD` on Exit rows is the per-leg realised P&L (net of commission)
- `Date and time` is assumed to be **US/Eastern** — change `data_timezone` in config if your export uses UTC or another timezone

---

## Running the Engine

### Quickstart (all defaults)

```bash
python -m backtest_engine.run
```

### With a config file

```bash
cp config.example.yaml config.yaml
python -m backtest_engine.run --config config.yaml
```

### Common CLI overrides

```bash
# Custom file paths
python -m backtest_engine.run --mgc1 data/My_Gold.csv --m2k1 data/My_Russell.csv

# Session filter mode
python -m backtest_engine.run --session-mode drop      # (default) remove boundary trades
python -m backtest_engine.run --session-mode flatten   # approximate forced-exit PnL

# Trailing drawdown mode
python -m backtest_engine.run --dd-mode eod            # (default) peak updates EOD
python -m backtest_engine.run --dd-mode close_to_close # peak updates per trade

# Custom size sweep
python -m backtest_engine.run --sizes 5 7 10 12 15

# Enable dynamic sizing + generate charts
python -m backtest_engine.run --dynamic --charts

# Verbose logging
python -m backtest_engine.run -v
```

### All CLI options

| Flag | Description | Default |
|------|-------------|---------|
| `--config FILE` | YAML config file | none |
| `--mgc1 FILE` | MGC1! CSV path | `data/MGC1.csv` |
| `--m2k1 FILE` | M2K1! CSV path | `data/M2K1.csv` |
| `--session-mode` | `drop` or `flatten` | `drop` |
| `--dd-mode` | `eod` or `close_to_close` | `eod` |
| `--sizes N [N ...]` | M2K1! sizes to sweep | `5 7 8 10 12 15` |
| `--dynamic` | Enable dynamic sizing | disabled |
| `--charts` | Generate PNG charts | disabled |
| `--output-dir DIR` | Reports directory | `reports/` |
| `--timezone TZ` | CSV data timezone | `US/Eastern` |
| `-v / --verbose` | Debug logging | off |

---

## Configuration Reference

See [`config.example.yaml`](config.example.yaml) for the complete annotated example. Key options:

### Timezone Assumption

```yaml
# Timezone of the "Date and time" column in your CSV exports.
# Change to "UTC" if your TradingView export uses UTC.
data_timezone: "US/Eastern"
```

### Session Filter

```yaml
# "drop"    — remove trades crossing 4:55 PM ET or Fri 17:00–Sun 18:00 ET (recommended)
# "flatten" — retain but approximate forced-exit PnL (APPROXIMATION; no OHLC)
session_mode: "drop"
```

### Trailing Drawdown Mode

```yaml
# "eod"            — peak updates once at EOD (mirrors typical prop-firm evaluation)
# "close_to_close" — peak updates after every closed trade (more conservative)
trailing_dd_mode: "eod"
```

### M2K1! Sizing Sweep

```yaml
# Original total position size in the M2K1! CSV (3-lot TP1 + 7-lot TP2 = 10)
m2k1_base_size: 10
# PnL scales linearly: effective_pnl = original_pnl * (target / base)
# APPROXIMATION: assumes linear scaling, no liquidity impact from larger size
m2k1_sweep_sizes: [5, 7, 8, 10, 12, 15]
mgc1_size_multiplier: 1.0
```

### Dynamic Sizing

```yaml
dynamic_sizing_enabled: false
dynamic_sizing_start: 5       # M2K1! lots at start of evaluation
dynamic_sizing_step: 10       # M2K1! lots after trigger is reached
dynamic_sizing_trigger: 3000.0  # Trigger when equity is +$3,000 above start
```

---

## Outputs

### Console Report (per scenario)

Each scenario prints month-by-month P&L, drawdown status, consistency check, and a verdict:
`PASS` / `FAIL (target)` / `BREACH (drawdown)` / `FAIL (consistency)`

### CSV Outputs (`reports/`)

| File | Contents |
|------|----------|
| `summary.csv` | All scenarios, one row each |
| `<scenario>_monthly.csv` | Month-by-month PnL |
| `<scenario>_drawdown.csv` | Per-day drawdown curve |
| `<scenario>_daily_pnl.csv` | Per-day PnL + cumulative |

### Charts (optional, `reports/*.png`)

Enable with `--charts`. Equity curve + drawdown underwater plot per scenario.

---

## Approximations & Limitations

| Approximation | Description |
|--------------|-------------|
| **Close-to-close drawdown** | Intraday floating losses unknown without 1-min OHLC. Only realized (closed) drawdown is detected. |
| **AE proxy** | Per-trade `Adverse Excursion USD` summed daily as a conservative upper-bound floating-DD estimate. Not a confirmed breach. |
| **Flatten mode** | Forced-exit PnL approximated as originally-recorded PnL (no OHLC to determine true exit price). |
| **Linear PnL scaling** | M2K1! PnL scaled by `target/base`. Assumes no slippage increase at larger sizes. |

---

## Running Tests

```bash
pytest tests/ -v
```

Tests use **synthetic in-memory CSV fixtures** — no real data files required.

---

## Project Structure

```
backtest_engine/
├── config.py        — Config dataclass + YAML loader
├── parser.py        — CSV parsing, entry/exit pairing, PnL validation
├── session.py       — TPT trading day assignment, session/weekend filter
├── sizing.py        — M2K1! static + dynamic sizing sweep
├── drawdown.py      — Trailing drawdown engine (EOD + close-to-close)
├── consistency.py   — Profit consistency rule + updated profit goal
├── reporting.py     — Console + CSV reports, optional charts
└── run.py           — CLI entry point

tests/               — pytest suite (synthetic fixtures, no real data)
data/                — Place your CSV files here (gitignored)
reports/             — Output directory (gitignored, except .gitkeep)
config.example.yaml  — Annotated configuration example
```

---

## References

- [TPT $150k Evaluation Rules](https://takeprofittraderhelp.zendesk.com/hc/en-us/categories/15135982702621-Test-Rules)

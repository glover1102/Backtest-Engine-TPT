# TPT Backtesting Engine

A Python backtesting and analysis engine that evaluates whether running multiple futures strategies together can **pass a Take Profit Trader (TPT) $150,000 evaluation account** under its confirmed rules.

The engine supports up to 5 CME micro-futures symbols simultaneously: **MGC (Micro Gold), M2K (Micro Russell 2000), MNQ (Micro Nasdaq-100), MYM (Micro Dow), MCL (Micro Crude Oil)**. It ingests TradingView "Any Strategy Converter" trade-log CSV exports, applies TPT session/weekend rules, computes an **intraday concurrency-based drawdown estimate**, and reports survival/fee-gate verdicts.

---

## ⚠️ CRITICAL LIMITATION: No Intraday OHLC Data

> **ESTIMATE ONLY — intraday tick-by-tick breach cannot be confirmed without 1-minute/tick OHLC data. Size defensively against the UPPER BOUND.**

TPT's trailing drawdown is evaluated on **real-time marked-to-market equity**, including **open-position floating losses**, measured tick-by-tick. The trade logs are close-to-close per trade. We only have the `Adverse Excursion USD` column giving each trade's worst intra-trade floating loss — but **not its timestamp**.

Therefore:
- A true tick-by-tick trailing-drawdown breach **cannot be definitively confirmed** from this data.
- The engine computes a **conservative concurrency-based estimate**: it stacks the `Adverse Excursion` values of simultaneously-open trades to bound the worst-case combined floating drawdown.
- Every drawdown verdict is clearly labelled with an "ESTIMATE ONLY" banner.

**To get a definitive intraday drawdown validation:** source 1-minute or tick OHLC data for each symbol and integrate it with the trade log.

---

## TPT $150,000 Confirmed Evaluation Rules

| Rule | Value | Notes |
|------|-------|-------|
| Account size | $150,000 | |
| Max position | 150 micro contracts | Across ALL symbols simultaneously |
| Monthly profit target | $9,000 combined | Within ~20 trading days; **MISS = pay recurring fee, NOT account failure** |
| **Maximum trailing drawdown** | **$4,500** | **THE ONLY HARD FAIL CONDITION** — intraday/real-time, includes open floating losses |
| Trading session | 6:00 PM ET → 5:00 PM ET | Trade opened ≥ 6 PM counts toward next day |
| Auto-flatten | 4:55 PM ET daily | No positions held overnight |
| Weekend block | Friday 17:00 ET → Sunday 18:00 ET | |
| Minimum trading days | 5 per evaluation | |
| Profit consistency | Best day < 50% of net P/L | Failure = goal raised to `net_pl × 2`, **NOT account failure** |

### Verdict Hierarchy

The **ONLY** thing that fails the account is the `$4,500 intraday trailing drawdown breach`:

```
⛔ BREACH (drawdown)       ← ONLY hard fail; the account is dead
⚠️  AT RISK (est. DD ≥ buffer) ← within safety margin; size down
✓  SURVIVED                ← account alive; then check the fee gate:
   Monthly $9k → PASS or MISS (pay recurring fee, NOT failure)
```

### Profit Consistency Rule

If `highest_profit_day / net_pl ≥ 50%`, the account does **not fail**. Instead:
```
updated_profit_goal = net_pl × 2
```
The trader must continue trading until consistency < 50%.  This is a **goal adjustment**, never a fatal condition.

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, pandas ≥ 2.0, numpy ≥ 1.24, PyYAML ≥ 6.0, pytz ≥ 2023.3, matplotlib ≥ 3.7 (optional for charts)

---

## Placing Your Data Files

```
data/
├── MGC1.csv    ← Micro Gold  (required for 3-symbol run)
├── M2K1.csv    ← Micro Russell 2000  (5-lot scale-in file; baseline auto-detected)
├── MNQ1.csv    ← Micro Nasdaq-100  (add when available)
├── MYM1.csv    ← Micro Dow  (add when available; no code change needed)
└── MCL1.csv    ← Micro Crude Oil  (add when available; no code change needed)
```

**Missing CSV files produce a warning and are skipped** — they are not errors. The engine runs with whatever files are present. Dropping a new CSV into `data/` and listing it in `config.yaml` automatically includes it without any code changes.

**M2K baseline:** The default `M2K1.csv` should be the **5-lot** file (1+4 scale-in). The baseline contract size is auto-detected from the modal `Size (qty)` column, and PnL is scaled linearly to the configured `contract_size`.

### Expected CSV Format

TradingView **"Any Strategy Converter"** exports with this header:

```
Trade number,Type,Date and time,Signal,Price USD,Size (qty),Size (value),
Net PnL USD,Return %,Commission USD,Favorable excursion USD,
Favorable excursion %,Adverse excursion USD,Adverse excursion %,
Cumulative PnL USD,Cumulative PnL %,Duration (bars)
```

- **`Type`** values: `Entry Long`, `Entry Short`, `Exit Long`, `Exit Short`
- Each Trade number may have multiple Exit rows (TP1, TP2 partial-exit legs)
- `Net PnL USD` on Exit rows is the per-leg realized P&L (net of commission)
- `Adverse Excursion USD` is the worst intra-trade floating loss (used for DD estimation)
- `Date and time` is assumed to be **US/Eastern** timezone

---

## Running the Engine

### Multi-symbol engine (new, recommended)

```bash
# With config file (recommended)
cp config.example.yaml config.yaml
python -m backtest_engine.run_multi --config config.yaml

# Without config file (scans data/ for default filenames)
python -m backtest_engine.run_multi

# With EOD close-to-close comparison
python -m backtest_engine.run_multi --config config.yaml --eod-comparison

# Generate charts
python -m backtest_engine.run_multi --config config.yaml --charts

# Verbose logging
python -m backtest_engine.run_multi --config config.yaml -v
```

### Legacy two-symbol engine (backward compatible)

```bash
cp config.example.yaml config.yaml
python -m backtest_engine.run --config config.yaml
```

---

## Multi-Symbol Config Schema

See [`config.example.yaml`](config.example.yaml) for the full annotated example.

```yaml
account:
  size: 150000
  profit_target: 9000            # combined/month; MISS = fee, not fail
  trailing_drawdown: 4500        # THE ONLY hard fail
  safety_buffer: 3000            # target: est. worst-case DD stays under this
  max_micros: 150
  min_trading_days: 5
  consistency_pct: 0.50
  weekend_filter_mode: drop

symbols:
  MGC: { csv: data/MGC1.csv, contract_size: 1, point_value: 10,  tick_size: 0.10 }
  M2K: { csv: data/M2K1.csv, contract_size: 5, point_value: 5,   tick_size: 0.10 }
  MNQ: { csv: data/MNQ1.csv, contract_size: 1, point_value: 2,   tick_size: 0.25 }
  MYM: { csv: data/MYM1.csv, contract_size: 1, point_value: 0.50,tick_size: 1.0  }
  MCL: { csv: data/MCL1.csv, contract_size: 1, point_value: 10,  tick_size: 0.01 }
```

### Baked-in CME Micro Specs

| Symbol | Instrument | point_value | tick_size | tick_value |
|--------|-----------|-------------|-----------|-----------|
| MGC | Micro Gold | 10 | 0.10 | 1.00 |
| M2K | Micro Russell 2000 | 5 | 0.10 | 0.50 |
| MNQ | Micro Nasdaq-100 | 2 | 0.25 | 0.50 |
| MYM | Micro Dow | 0.50 | 1.0 | 0.50 |
| MCL | Micro Crude Oil | 10 | 0.01 | 0.10 |

---

## Intraday Concurrency-Based Trailing Drawdown Engine

### How it works

1. **Build event stream**: Every trade's `[entry_time, exit_time]` interval is decomposed into open/close events, sorted by timestamp.

2. **Trailing peak ratchet**: The realized equity (from closed trades) is tracked. A `trailing_peak` variable starts at `initial_equity` and ratchets up (never down) whenever a trade close produces a new equity high.

3. **At each event**, identify the set of currently open trades. Compute two bounds on combined worst-case floating equity:
   - **Upper bound (conservative — size defensively against this)**:
     ```
     upper_equity = realized_equity + sum(adverse_excursion of all open trades)
     ```
     Assumes ALL concurrent trades hit their worst simultaneously.
   - **Lower bound (optimistic)**:
     ```
     lower_equity = realized_equity + min(adverse_excursion of open trades)
     ```
     Assumes only the single worst trade hits its low.

4. **Drawdown from peak**:
   ```
   upper_dd = upper_equity - trailing_peak   (≤ 0 = drawdown)
   lower_dd = lower_equity - trailing_peak
   ```

5. **Flags**:
   - `AT RISK`: `upper_dd ≤ -safety_buffer` (e.g. −$3,000)
   - `BREACH ESTIMATE`: `upper_dd ≤ -trailing_drawdown_limit` (e.g. −$4,500)

### Why concurrency matters

If MGC's worst −$500 AE trade and MNQ's worst −$300 AE trade happen on **different** days, they never stack. The engine only adds AE values for trades whose `[entry_time, exit_time]` intervals overlap in time.

**Concurrency is your friend**: non-concurrent bad trades don't inflate combined drawdown risk.

### 150-micro concurrency cap

The engine sums `effective_size` across all simultaneously-open trades to compute the peak concurrent micro-contract count. This is checked against the 150-micro cap (not just per-trade size).

---

## Monthly $9k Evaluation (Fee Gate, Not Failure)

For each calendar month in the data:
- Combined net P/L is summed across all symbols and trading days.
- If it reaches `$9,000` (or the consistency-adjusted goal) within ≥ 5 trading days → **PASS**.
- If not → **MISS** = pay the recurring fee and continue. **NOT account failure**.

Output: per-month `PASS/MISS` table + overall monthly pass-rate %.

```
NOTE: Monthly $9k = recurring-fee gate, NOT account failure.
MISS = pay another month's fee and continue.
ONLY a $4,500 intraday trailing drawdown breach FAILS the account.
```

---

## Outputs

### Console Report

```
  SYMBOLS LOADED: MGC, M2K, MNQ
  ⚠️  MISSING: MYM, MCL

  PER-SYMBOL CONTRIBUTION (month-by-month)
  Month       MGC       M2K      MNQ    COMBINED
  2026-03   ...

  MONTHLY $9K EVALUATION
  2026-03    21    $6,500   $9,000  MISS (pay recurring fee)
  2026-04    20   $11,200   $9,000  PASS
  Monthly pass-rate: 1/2 → 50%  (MISS = recurring fee, not failure)

  INTRADAY TRAILING DRAWDOWN ESTIMATE
  ⚠️  ESTIMATE ONLY — intraday tick-by-tick breach cannot be confirmed...
  Upper-bound worst DD: -$2,800  ← size defensively against this
  Lower-bound worst DD: -$1,200  ← optimistic
  Fatal DD limit: -$4,500

  OVERALL VERDICT: ✓ SURVIVED — drawdown estimate within safety margin
  Monthly $9k: 1/2 PASS (50% pass-rate) ← MISS = recurring fee, not failure
```

### CSV Outputs (`reports/`)

| File | Contents |
|------|----------|
| `combined_monthly.csv` | Per-month P/L, PASS/MISS, per-symbol breakdown |
| `combined_trades.csv` | All trades with effective PnL/size |
| `concurrency_timeline_worst.csv` | Top 200 worst-DD concurrency events |
| `summary.csv` (legacy) | All scenarios, one row each |

### Charts (optional, `reports/*.png`)

Enable with `--charts`. Equity curve + drawdown underwater plot with upper/lower AE bounds.

---

## Approximations & Limitations

| Approximation | Description |
|--------------|-------------|
| **No intraday OHLC** | Worst-case floating drawdown ESTIMATED from `Adverse Excursion USD`; NOT a confirmed breach. |
| **AE timestamp unknown** | AE lows are not timestamped; concurrency-based stacking uses open intervals as proxy. |
| **Linear PnL scaling** | Per-symbol PnL scaled by `configured_size / baseline_size`. Assumes linear scaling; no liquidity constraints. |
| **Close-to-close peak** | Trailing peak updated at trade-close events only; true real-time peak may be higher. |
| **Flatten mode** | Forced-exit PnL approximated as originally-recorded PnL (no OHLC). |

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
├── config.py        — Config dataclasses (multi-symbol + legacy) + YAML loaders
├── parser.py        — CSV parsing, entry/exit pairing, PnL validation
├── loader.py        — Multi-symbol CSV loader (warn on missing, baseline auto-detect)
├── session.py       — TPT trading day assignment, session/weekend filter
├── sizing.py        — Per-symbol sizing, concurrency cap check
├── concurrency.py   — Intraday concurrency-based trailing-DD engine (AE bounds)
├── monthly.py       — Monthly $9k pass/miss evaluation (fee-gate, not failure)
├── drawdown.py      — EOD/close-to-close trailing drawdown engine (legacy)
├── consistency.py   — Profit consistency rule + updated profit goal
├── reporting.py     — Console + CSV reports (multi-symbol + legacy), charts
├── run_multi.py     — Multi-symbol CLI entry point
└── run.py           — Legacy two-symbol CLI entry point

tests/               — pytest suite (synthetic fixtures, no real data)
data/                — Place your CSV files here (gitignored)
reports/             — Output directory (gitignored, except .gitkeep)
config.example.yaml  — Annotated configuration example (multi-symbol schema)
```

---

## References

- [TPT $150k Evaluation Rules](https://takeprofittraderhelp.zendesk.com/hc/en-us/categories/15135982702621-Test-Rules)


"""
TPT Backtesting Engine — multi-symbol CLI entry point.

Usage
-----
    python -m backtest_engine.run_multi [--config config.yaml] [options]

All options can also be set in the YAML config file; CLI flags override YAML.
Run ``python -m backtest_engine.run_multi --help`` for the full option list.

Config file format
------------------
See ``config.example.yaml`` for the full multi-symbol configuration schema.
The minimum required structure is::

    account:
      size: 150000
      profit_target: 9000
      trailing_drawdown: 4500
      safety_buffer: 3000
      max_micros: 150

    symbols:
      MGC: { csv: data/MGC1.csv, contract_size: 1 }
      M2K: { csv: data/M2K1.csv, contract_size: 5 }
      MNQ: { csv: data/MNQ1.csv, contract_size: 1 }

Missing CSV files produce a warning and are skipped — no code change needed
when new CSVs are added.

Verdict hierarchy
-----------------
The ONLY hard fail is ``AT RISK (est. floating DD ≥ $4,500)``.
Monthly $9k misses = pay the recurring fee, NOT account failure.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional

import pandas as pd

from .config import (
    AccountConfig,
    CME_MICRO_SPECS,
    MultiSymbolConfig,
    SymbolConfig,
    load_multi_config,
)
from .concurrency import compute_intraday_concurrency
from .consistency import compute_consistency
from .drawdown import compute_drawdown
from .loader import load_all_symbols, detect_baseline_size
from .monthly import evaluate_monthly
from .reporting import (
    export_multi_symbol_csv,
    print_multi_symbol_report,
    print_size_recommendation,
)
from .session import filter_trades
from .sizing import check_concurrent_position_limit

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m backtest_engine.run_multi",
        description="TPT $150,000 Multi-Symbol Evaluation Backtesting Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config", metavar="FILE", default="config.yaml",
        help="Path to YAML config file (default: config.yaml).",
    )
    p.add_argument(
        "--data-dir", metavar="DIR", default=None,
        help="Override data/ directory for all symbol CSVs.",
    )
    p.add_argument(
        "--output-dir", metavar="DIR", default=None,
        help="Directory for CSV outputs (default: reports/).",
    )
    p.add_argument(
        "--session-mode", choices=["drop", "flatten"], default=None,
        help="How to handle session/weekend boundary violations (default: drop).",
    )
    p.add_argument(
        "--charts", action="store_true", default=False,
        help="Generate and save matplotlib charts.",
    )
    p.add_argument(
        "--eod-comparison", action="store_true", default=False,
        help="Also run the EOD close-to-close drawdown mode for comparison.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Core engine
# ─────────────────────────────────────────────────────────────────────────────

def run_multi(
    cfg: MultiSymbolConfig,
    *,
    session_mode: str = "drop",
    generate_charts: bool = False,
    eod_comparison: bool = False,
) -> None:
    """
    Execute the full multi-symbol backtest pipeline.

    Parameters
    ----------
    cfg:
        Multi-symbol configuration.
    session_mode:
        How to handle session/weekend boundary violations.
    generate_charts:
        Whether to generate matplotlib charts.
    eod_comparison:
        Whether to also run EOD close-to-close drawdown mode for comparison.
    """
    acct = cfg.account
    data_tz = cfg.data_timezone

    # ── 1. Load all symbol CSVs ─────────────────────────────────────────────
    if not cfg.symbols:
        logger.warning(
            "No symbols defined in config.  Add a 'symbols:' map to your config "
            "file.  See config.example.yaml for the schema."
        )
        _print_no_symbols_help()
        return

    logger.info("Loading %d configured symbol(s)…", len(cfg.symbols))
    combined_raw, present, missing = load_all_symbols(cfg, data_timezone=data_tz)

    if not present:
        print(
            "\nERROR: No symbol CSVs found.  "
            "Place your CSV files in the paths defined in config.yaml.\n"
            "See config.example.yaml for the schema.\n",
            file=sys.stderr,
        )
        return

    # ── 2. Session / weekend filter ─────────────────────────────────────────
    weekend_mode = acct.weekend_filter_mode or session_mode
    logger.info("Applying session filter (mode=%s)…", weekend_mode)
    filtered, session_stats = filter_trades(combined_raw, mode=weekend_mode)
    logger.info(
        "After filter: %d legs remain.  Dropped=%d, Flattened=%d.",
        len(filtered),
        session_stats["dropped"],
        session_stats["flattened"],
    )

    if filtered.empty:
        print(
            "ERROR: No trades remain after session filtering.  "
            "Check your CSV files and session_mode setting.",
            file=sys.stderr,
        )
        return

    # ── 3. Collect per-symbol baseline sizes ───────────────────────────────
    baseline_sizes: Dict[str, int] = {}
    current_sizes: Dict[str, int] = {}
    for sym in present:
        sym_trades = filtered[filtered["symbol"] == sym]
        baseline = detect_baseline_size(sym_trades)
        baseline_sizes[sym] = baseline
        current_sizes[sym] = cfg.symbols[sym].contract_size
        logger.info(
            "[%s] baseline=%d, configured=%d.", sym, baseline, cfg.symbols[sym].contract_size
        )

    # ── 4. Concurrent position-cap check ───────────────────────────────────
    within_cap, peak_micros, peak_time, peak_syms = check_concurrent_position_limit(
        filtered, max_micros=acct.max_micros
    )
    if not within_cap:
        logger.warning(
            "Peak concurrent micros %.0f exceeds cap %d.  "
            "Consider reducing contract sizes.",
            peak_micros,
            acct.max_micros,
        )

    # ── 5. Intraday concurrency-based drawdown estimate ────────────────────
    logger.info("Computing intraday concurrency-based drawdown estimate…")
    concurrency = compute_intraday_concurrency(
        filtered,
        initial_equity=acct.size,
        trailing_drawdown_limit=acct.trailing_drawdown,
        safety_buffer=acct.safety_buffer,
        max_micros=acct.max_micros,
    )

    # ── 6. Monthly profit-target evaluation ────────────────────────────────
    logger.info("Evaluating monthly $%.0f profit targets…", acct.profit_target)
    monthly = evaluate_monthly(
        filtered,
        profit_target=acct.profit_target,
        consistency_threshold=acct.consistency_pct,
        min_trading_days=acct.min_trading_days,
    )

    # ── 7. Overall consistency check ───────────────────────────────────────
    consistency = compute_consistency(
        filtered,
        profit_target=acct.profit_target,
        consistency_threshold=acct.consistency_pct,
        min_trading_days=acct.min_trading_days,
    )

    # ── 8. (Optional) EOD close-to-close drawdown for comparison ──────────
    if eod_comparison:
        logger.info("Running EOD close-to-close drawdown mode for comparison…")
        dd_eod = compute_drawdown(
            filtered,
            initial_equity=acct.size,
            max_trailing_dd=acct.trailing_drawdown,
            mode="eod",
        )
        _print_eod_comparison(dd_eod)

    # ── 9. Console report ─────────────────────────────────────────────────
    print_multi_symbol_report(
        trades=filtered,
        present_symbols=present,
        missing_symbols=missing,
        concurrency=concurrency,
        monthly=monthly,
        consistency=consistency,
        profit_target=acct.profit_target,
        max_micros=acct.max_micros,
    )

    print_size_recommendation(
        present_symbols=present,
        baseline_sizes=baseline_sizes,
        current_sizes=current_sizes,
        concurrency=concurrency,
        monthly=monthly,
        safety_buffer=acct.safety_buffer,
        trailing_dd_limit=acct.trailing_drawdown,
        max_micros=acct.max_micros,
    )

    # ── 10. CSV export ────────────────────────────────────────────────────
    os.makedirs(cfg.output_dir, exist_ok=True)
    export_multi_symbol_csv(
        trades=filtered,
        monthly=monthly,
        concurrency=concurrency,
        output_dir=cfg.output_dir,
    )

    # ── 11. Optional charts ───────────────────────────────────────────────
    if generate_charts or cfg.generate_charts:
        _generate_multi_charts(filtered, concurrency, monthly, cfg.output_dir)


def _print_eod_comparison(dd_eod) -> None:
    """Print a brief EOD close-to-close comparison block."""
    print("\n" + "─" * 60)
    print("  EOD CLOSE-TO-CLOSE COMPARISON (for reference only)")
    print("  NOTE: intraday mode is the governing metric for TPT.")
    print("─" * 60)
    print(f"  Max realized DD (EOD): ${dd_eod.max_realized_dd:,.2f}")
    print(f"  Any breach (EOD): {'YES ⛔' if dd_eod.any_breach else 'NO ✓'}")
    if dd_eod.breach_days:
        for d in dd_eod.breach_days[:5]:
            print(f"    → EOD breach day: {d}")
    print("─" * 60 + "\n")


def _generate_multi_charts(
    trades: pd.DataFrame,
    concurrency,
    monthly,
    output_dir: str,
) -> None:
    """Generate and save equity + drawdown charts for multi-symbol run."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib not available — skipping charts.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Equity curve from concurrency timeline
    if not concurrency.timeline:
        return

    times = [pt.timestamp for pt in concurrency.timeline if pt.event_type == "close"]
    equities = [pt.realized_equity for pt in concurrency.timeline if pt.event_type == "close"]
    upper_dds = [pt.upper_dd_from_peak for pt in concurrency.timeline if pt.event_type == "close"]
    lower_dds = [pt.lower_dd_from_peak for pt in concurrency.timeline if pt.event_type == "close"]

    if not times:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    ax1.plot(times, equities, color="steelblue", label="Realized equity")
    ax1.set_ylabel("Account Equity ($)")
    ax1.set_title("Combined Multi-Symbol Equity Curve")
    ax1.legend(fontsize=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(times, upper_dds, 0, color="salmon", alpha=0.5, label="Upper-bound DD (conservative)")
    ax2.fill_between(times, lower_dds, 0, color="orange", alpha=0.3, label="Lower-bound DD (optimistic)")
    ax2.axhline(-concurrency.trailing_drawdown_limit, color="red", linestyle="--",
                label=f"Fatal DD limit (${concurrency.trailing_drawdown_limit:,.0f})")
    ax2.axhline(-concurrency.safety_buffer, color="orange", linestyle=":",
                label=f"Safety buffer (${concurrency.safety_buffer:,.0f})")
    ax2.set_ylabel("Estimated Drawdown from Peak ($)")
    ax2.set_xlabel("Time")
    ax2.legend(fontsize=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "combined_equity_drawdown.png")
    plt.savefig(chart_path, dpi=150)
    plt.close(fig)
    logger.info("Saved combined chart to %s", chart_path)


def _print_no_symbols_help() -> None:
    print(
        "\n"
        "No symbols are configured.  Create a config.yaml with a 'symbols:' block.\n"
        "Example:\n"
        "\n"
        "  account:\n"
        "    size: 150000\n"
        "    profit_target: 9000\n"
        "    trailing_drawdown: 4500\n"
        "\n"
        "  symbols:\n"
        "    MGC: { csv: data/MGC1.csv, contract_size: 1 }\n"
        "    M2K: { csv: data/M2K1.csv, contract_size: 5 }\n"
        "    MNQ: { csv: data/MNQ1.csv, contract_size: 1 }\n"
        "\n"
        "See config.example.yaml for the complete schema.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Default config builder (for runs with no config file)
# ─────────────────────────────────────────────────────────────────────────────

def _build_default_config(data_dir: str = "data") -> MultiSymbolConfig:
    """
    Build a default MultiSymbolConfig by scanning *data_dir* for CSVs matching
    the expected filenames (MGC1.csv, M2K1.csv, MNQ1.csv, MYM1.csv, MCL1.csv).
    """
    defaults: Dict[str, Dict] = {
        "MGC": {"csv": f"{data_dir}/MGC1.csv", "contract_size": 1},
        "M2K": {"csv": f"{data_dir}/M2K1.csv", "contract_size": 5},
        "MNQ": {"csv": f"{data_dir}/MNQ1.csv", "contract_size": 1},
        "MYM": {"csv": f"{data_dir}/MYM1.csv", "contract_size": 1},
        "MCL": {"csv": f"{data_dir}/MCL1.csv", "contract_size": 1},
    }
    symbols: Dict[str, SymbolConfig] = {}
    for sym, d in defaults.items():
        spec = CME_MICRO_SPECS.get(sym, {})
        symbols[sym] = SymbolConfig(
            csv=d["csv"],
            contract_size=d["contract_size"],
            point_value=spec.get("point_value", 1.0),
            tick_size=spec.get("tick_size", 0.01),
        )

    return MultiSymbolConfig(
        account=AccountConfig(),
        symbols=symbols,
        output_dir="reports",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config_path = args.config
    if os.path.isfile(config_path):
        cfg = load_multi_config(config_path)
        logger.info("Loaded multi-symbol config from %s", config_path)
    else:
        if config_path != "config.yaml":
            logger.warning("Config file '%s' not found — using defaults.", config_path)
        data_dir = args.data_dir or "data"
        cfg = _build_default_config(data_dir)
        logger.info("No config.yaml found — using defaults (scanning %s/ for CSVs).", data_dir)

    # Apply CLI overrides
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.charts:
        cfg.generate_charts = True
    if args.data_dir:
        # Rewrite csv paths to use the specified data_dir
        for sym, sym_cfg in cfg.symbols.items():
            fname = os.path.basename(sym_cfg.csv)
            sym_cfg.csv = os.path.join(args.data_dir, fname)

    session_mode = args.session_mode or cfg.account.weekend_filter_mode or "drop"

    run_multi(
        cfg,
        session_mode=session_mode,
        generate_charts=args.charts or cfg.generate_charts,
        eod_comparison=args.eod_comparison,
    )


if __name__ == "__main__":
    main()

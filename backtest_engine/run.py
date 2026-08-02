"""
TPT Backtesting Engine — CLI entry point.

Usage
-----
    python -m backtest_engine.run [--config config.yaml] [options]

All options can also be set in the YAML config file; CLI flags override YAML.
Run ``python -m backtest_engine.run --help`` for the full option list.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

import pandas as pd

from .config import Config, load_config
from .consistency import compute_consistency, profit_target_reached_after_days
from .drawdown import compute_drawdown
from .parser import parse_csv
from .reporting import (
    ScenarioResult,
    build_monthly_summary,
    compute_verdict,
    export_scenario_csv,
    export_summary_csv,
    generate_charts,
    print_recommendation,
    print_scenario_report,
)
from .session import filter_trades
from .sizing import (
    apply_dynamic_sizing,
    apply_static_sizing,
    check_position_limit,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m backtest_engine.run",
        description="TPT $150,000 Evaluation Backtesting Engine",
    )
    p.add_argument(
        "--config", metavar="FILE", default=None,
        help="Path to YAML config file (default: none).",
    )
    p.add_argument("--mgc1", metavar="FILE", help="MGC1! CSV file path.")
    p.add_argument("--m2k1", metavar="FILE", help="M2K1! CSV file path.")
    p.add_argument(
        "--session-mode", choices=["drop", "flatten"],
        help="How to handle session/weekend boundary violations.",
    )
    p.add_argument(
        "--dd-mode", choices=["eod", "close_to_close"],
        help="Trailing drawdown update mode.",
    )
    p.add_argument(
        "--sizes", nargs="+", type=int, metavar="N",
        help="M2K1! sizes to sweep (space-separated integers).",
    )
    p.add_argument(
        "--dynamic", action="store_true", default=None,
        help="Enable dynamic sizing in addition to static sweep.",
    )
    p.add_argument(
        "--charts", action="store_true", default=None,
        help="Generate and save matplotlib charts.",
    )
    p.add_argument(
        "--output-dir", metavar="DIR",
        help="Directory for CSV/chart outputs (default: reports/).",
    )
    p.add_argument(
        "--timezone", metavar="TZ",
        help="Timezone of the CSV data (default: US/Eastern).",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Core engine
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: Config) -> List[ScenarioResult]:
    """
    Execute the full backtest pipeline for all sizing scenarios.

    Returns the list of ScenarioResult objects (one per scenario).
    """
    # ── 1. Parse ──────────────────────────────────────────────────────────
    logger.info("Parsing MGC1! data from %s …", cfg.mgc1_file)
    mgc1_trades = parse_csv(cfg.mgc1_file, symbol="MGC1", data_timezone=cfg.data_timezone)

    logger.info("Parsing M2K1! data from %s …", cfg.m2k1_file)
    m2k1_trades = parse_csv(cfg.m2k1_file, symbol="M2K1", data_timezone=cfg.data_timezone)

    all_raw = pd.concat([mgc1_trades, m2k1_trades], ignore_index=True)
    logger.info(
        "Loaded %d MGC1 + %d M2K1 trade legs (%d total).",
        len(mgc1_trades), len(m2k1_trades), len(all_raw),
    )

    # ── 2. Session / weekend filter ───────────────────────────────────────
    logger.info("Applying session filter (mode=%s) …", cfg.session_mode)
    filtered, session_stats = filter_trades(all_raw, mode=cfg.session_mode)
    logger.info(
        "After filter: %d legs remain. Dropped=%d, Flattened=%d.",
        len(filtered), session_stats["dropped"], session_stats["flattened"],
    )

    if filtered.empty:
        print("ERROR: No trades remain after session filtering.", file=sys.stderr)
        return []

    # ── 3. Build scenarios ────────────────────────────────────────────────
    scenarios: List[ScenarioResult] = []

    # Static sweep
    for size in cfg.m2k1_sweep_sizes:
        label = f"M2K1_{size}lots_static"
        logger.info("Running scenario: %s …", label)
        scenario = _run_scenario(
            filtered_trades=filtered,
            cfg=cfg,
            m2k1_size=size,
            label=label,
            is_dynamic=False,
        )
        scenarios.append(scenario)

    # Dynamic sizing (if enabled)
    if cfg.dynamic_sizing_enabled:
        for size in cfg.m2k1_sweep_sizes:
            label = f"M2K1_{cfg.dynamic_sizing_start}-{size}lots_dynamic"
            logger.info("Running dynamic scenario: %s …", label)
            scenario = _run_dynamic_scenario(
                filtered_trades=filtered,
                cfg=cfg,
                step_size=size,
                label=label,
            )
            scenarios.append(scenario)

    # ── 4. Report ─────────────────────────────────────────────────────────
    for scenario in scenarios:
        print_scenario_report(scenario)
        export_scenario_csv(scenario, cfg.output_dir)
        if cfg.generate_charts:
            generate_charts(scenario, cfg.output_dir)

    export_summary_csv(scenarios, cfg.output_dir)
    print_recommendation(scenarios)

    return scenarios


def _run_scenario(
    filtered_trades: pd.DataFrame,
    cfg: Config,
    m2k1_size: int,
    label: str,
    is_dynamic: bool = False,
) -> ScenarioResult:
    """Run a single static-sizing scenario."""
    sized = apply_static_sizing(
        filtered_trades,
        symbol="M2K1",
        base_size=cfg.m2k1_base_size,
        target_size=m2k1_size,
        mgc1_multiplier=cfg.mgc1_size_multiplier,
    )
    check_position_limit(sized, max_micros=cfg.max_position_micros)
    return _build_scenario_result(sized, cfg, m2k1_size, label, is_dynamic)


def _run_dynamic_scenario(
    filtered_trades: pd.DataFrame,
    cfg: Config,
    step_size: int,
    label: str,
) -> ScenarioResult:
    """Run a single dynamic-sizing scenario."""
    sized = apply_dynamic_sizing(
        filtered_trades,
        base_size=cfg.m2k1_base_size,
        start_size=cfg.dynamic_sizing_start,
        step_size=step_size,
        trigger_profit=cfg.dynamic_sizing_trigger,
        initial_equity=cfg.account_size,
        mgc1_multiplier=cfg.mgc1_size_multiplier,
    )
    check_position_limit(sized, max_micros=cfg.max_position_micros)
    return _build_scenario_result(sized, cfg, step_size, label, is_dynamic=True)


def _build_scenario_result(
    sized: pd.DataFrame,
    cfg: Config,
    m2k1_size: int,
    label: str,
    is_dynamic: bool,
) -> ScenarioResult:
    """Compute drawdown, consistency, and build the ScenarioResult."""
    dd = compute_drawdown(
        sized,
        initial_equity=cfg.account_size,
        max_trailing_dd=cfg.max_trailing_dd,
        mode=cfg.trailing_dd_mode,
    )
    con = compute_consistency(
        sized,
        profit_target=cfg.profit_target,
        consistency_threshold=cfg.consistency_threshold,
        min_trading_days=cfg.min_trading_days,
    )

    total_net_pl = con.net_pl
    hits_target = total_net_pl >= cfg.profit_target
    if con.updated_profit_goal is not None:
        effective_target = con.updated_profit_goal
    else:
        effective_target = cfg.profit_target

    days_to_target = profit_target_reached_after_days(
        con.per_day_pnl, effective_target
    )

    verdict = compute_verdict(
        hits_target=total_net_pl >= effective_target,
        any_dd_breach=dd.any_breach,
        passes_consistency=con.passes_consistency,
        passes_min_days=con.passes_min_days,
        updated_profit_goal=con.updated_profit_goal,
        total_net_pl=total_net_pl,
    )

    monthly = build_monthly_summary(sized)

    return ScenarioResult(
        label=label,
        m2k1_size=m2k1_size,
        is_dynamic=is_dynamic,
        trades=sized,
        drawdown=dd,
        consistency=con,
        total_net_pl=total_net_pl,
        n_trading_days=con.n_trading_days,
        n_trades=len(sized),
        profit_target=effective_target,
        hits_target=total_net_pl >= effective_target,
        days_to_target=days_to_target,
        verdict=verdict,
        monthly_summary=monthly,
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

    overrides: dict = {}
    if args.mgc1:
        overrides["mgc1_file"] = args.mgc1
    if args.m2k1:
        overrides["m2k1_file"] = args.m2k1
    if args.session_mode:
        overrides["session_mode"] = args.session_mode
    if args.dd_mode:
        overrides["trailing_dd_mode"] = args.dd_mode
    if args.sizes:
        overrides["m2k1_sweep_sizes"] = args.sizes
    if args.dynamic:
        overrides["dynamic_sizing_enabled"] = True
    if args.charts:
        overrides["generate_charts"] = True
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.timezone:
        overrides["data_timezone"] = args.timezone

    cfg = load_config(path=args.config, overrides=overrides)

    logger.info("Configuration loaded. Starting backtest …")
    run(cfg)


if __name__ == "__main__":
    main()

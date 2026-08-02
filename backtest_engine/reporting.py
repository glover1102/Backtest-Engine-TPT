"""
TPT Backtesting Engine — reporting module.

Generates per-scenario reports:
* Console table (ASCII)
* CSV files under ``reports/``
* Optional matplotlib equity + drawdown charts (guarded by config flag)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from .consistency import ConsistencyResult, profit_target_reached_after_days
from .drawdown import DrawdownResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Per-scenario result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    """All computed results for one sizing scenario."""
    label: str                          # e.g. "M2K1_10lots_static"
    m2k1_size: int
    is_dynamic: bool
    trades: pd.DataFrame                # sized, filtered trades
    drawdown: DrawdownResult
    consistency: ConsistencyResult
    # Derived
    total_net_pl: float
    n_trading_days: int
    n_trades: int
    profit_target: float
    hits_target: bool
    days_to_target: Optional[int]
    verdict: str                        # "PASS" / "FAIL (target)" / "BREACH (drawdown)" / etc.
    monthly_summary: pd.DataFrame       # month × [MGC1_pnl, M2K1_pnl, combined_pnl]


# ─────────────────────────────────────────────────────────────────────────────
# Verdict logic
# ─────────────────────────────────────────────────────────────────────────────

def compute_verdict(
    *,
    hits_target: bool,
    any_dd_breach: bool,
    passes_consistency: bool,
    passes_min_days: bool,
    updated_profit_goal: Optional[float],
    total_net_pl: float,
) -> str:
    """
    Return a human-readable verdict string following TPT evaluation logic.

    Priority (worst → best):
    1. BREACH (drawdown) — $4,500 trailing DD breached → immediate fail.
    2. FAIL (min days) — fewer than 5 trading days.
    3. FAIL (target) — profit target not reached (or consistency-adjusted goal).
    4. FAIL (consistency) — target met but consistency rule forces updated goal.
    5. PASS — all conditions met.
    """
    if any_dd_breach:
        return "BREACH (drawdown)"
    if not passes_min_days:
        return "FAIL (min days)"
    if updated_profit_goal is not None:
        # Consistency failed → need net_pl ≥ updated_goal
        if total_net_pl >= updated_profit_goal:
            return "PASS (consistency-adjusted goal met)"
        return f"FAIL (consistency) — need ${ updated_profit_goal:,.0f}"
    if not hits_target:
        return "FAIL (target)"
    if not passes_consistency:
        return "FAIL (consistency)"
    return "PASS"


# ─────────────────────────────────────────────────────────────────────────────
# Monthly summary builder
# ─────────────────────────────────────────────────────────────────────────────

def build_monthly_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Build a month-by-month PnL table broken down by symbol and combined.

    Returns a DataFrame indexed by ``YYYY-MM`` with columns:
    ``MGC1_pnl``, ``M2K1_pnl``, ``combined_pnl``.
    """
    pnl_col = "effective_pnl" if "effective_pnl" in trades.columns else "net_pnl"
    df = trades.copy()
    df["month"] = pd.to_datetime(df["tpt_trading_day"]).dt.to_period("M").astype(str)

    pivot = (
        df.groupby(["month", "symbol"])[pnl_col]
        .sum()
        .unstack(fill_value=0.0)
        .rename(columns=lambda c: f"{c}_pnl")
    )

    # Ensure both symbol columns exist
    for sym in ("MGC1", "M2K1"):
        col = f"{sym}_pnl"
        if col not in pivot.columns:
            pivot[col] = 0.0

    pivot["combined_pnl"] = pivot["MGC1_pnl"] + pivot["M2K1_pnl"]
    return pivot.sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# Console reporter
# ─────────────────────────────────────────────────────────────────────────────

def print_scenario_report(scenario: ScenarioResult) -> None:
    """Print a human-readable report for one sizing scenario."""
    dd = scenario.drawdown
    con = scenario.consistency

    sep = "=" * 70

    print(f"\n{sep}")
    print(f"  SCENARIO: {scenario.label}")
    print(sep)

    # Monthly breakdown
    print("\n  Month-by-Month PnL:")
    print(f"  {'Month':<12} {'MGC1':>10} {'M2K1':>10} {'Combined':>12}")
    print("  " + "-" * 46)
    for month, row in scenario.monthly_summary.iterrows():
        mgc_p = row.get("MGC1_pnl", 0.0)
        m2k_p = row.get("M2K1_pnl", 0.0)
        comb = row.get("combined_pnl", 0.0)
        print(f"  {month:<12} {mgc_p:>10,.2f} {m2k_p:>10,.2f} {comb:>12,.2f}")
    print("  " + "-" * 46)
    print(
        f"  {'TOTAL':<12} "
        f"{scenario.monthly_summary.get('MGC1_pnl', pd.Series()).sum():>10,.2f} "
        f"{scenario.monthly_summary.get('M2K1_pnl', pd.Series()).sum():>10,.2f} "
        f"{scenario.total_net_pl:>12,.2f}"
    )

    # Summary stats
    print(f"\n  Total net P/L:      ${scenario.total_net_pl:>10,.2f}")
    print(f"  Trading days:       {scenario.n_trading_days:>10d}")
    print(f"  Total trade legs:   {scenario.n_trades:>10d}")
    avg = scenario.n_trades / scenario.n_trading_days if scenario.n_trading_days else 0
    print(f"  Avg trades/day:     {avg:>10.1f}")

    # Profit target
    target_str = f"${ scenario.profit_target:,.0f}"
    hit_str = "✓ YES" if scenario.hits_target else "✗ NO"
    print(f"\n  Profit target ({target_str}): {hit_str}", end="")
    if scenario.hits_target and scenario.days_to_target:
        print(f"  (reached after {scenario.days_to_target} trading days)")
    else:
        print()

    # Trailing drawdown
    print(f"\n  Trailing Drawdown ({dd.mode.upper()}):")
    print(f"    Peak equity:        ${dd.peak_equity:>10,.2f}")
    print(f"    Final equity:       ${dd.final_equity:>10,.2f}")
    print(f"    Max realized DD:    ${dd.max_realized_dd:>10,.2f}  (floor: ${ dd.max_trailing_dd:,.0f})")
    breach_str = "⛔ YES" if dd.any_breach else "✓ NO"
    print(f"    $4,500 breached:    {breach_str}")
    if dd.breach_days:
        for bd in dd.breach_days:
            print(f"      → breach on {bd}")

    if dd.ae_proxy_max_dd is not None:
        print(
            f"\n  ⚠️  AE proxy (approx worst-case floating DD): ${dd.ae_proxy_max_dd:,.2f}"
        )
        if dd.ae_proxy_breach_days:
            print("      AE-proxy breach days:", dd.ae_proxy_breach_days)
        print("      (This is an upper-bound estimate; not a confirmed breach.)")

    # Consistency rule
    print(f"\n  Profit Consistency Check (threshold < 50%):")
    print(f"    Highest profit day: ${con.highest_profit_day:>10,.2f}  ({con.highest_profit_day_date})")
    print(f"    Consistency %:      {con.consistency_pct * 100:>9.1f}%")
    pass_str = "✓ PASS" if con.passes_consistency else "✗ FAIL"
    print(f"    Result:             {pass_str}")
    if con.updated_profit_goal is not None:
        print(f"    Updated profit goal: ${con.updated_profit_goal:,.2f}")

    # Minimum trading days
    min_str = "✓" if con.passes_min_days else "✗"
    print(f"\n  Min 5 trading days: {min_str}  ({con.n_trading_days} days)")

    # Verdict
    print(f"\n  ► VERDICT: {scenario.verdict}")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def export_scenario_csv(scenario: ScenarioResult, output_dir: str) -> None:
    """Write per-scenario CSV files to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    safe_label = scenario.label.replace(" ", "_")

    # Monthly summary
    monthly_path = os.path.join(output_dir, f"{safe_label}_monthly.csv")
    scenario.monthly_summary.to_csv(monthly_path)
    logger.info("Wrote monthly summary to %s", monthly_path)

    # Per-day drawdown
    dd_rows = []
    for d in scenario.drawdown.days:
        row = {
            "trading_day": d.trading_day,
            "pnl": d.pnl,
            "equity_start": d.equity_start,
            "equity_end": d.equity_end,
            "peak": d.peak_end,
            "floor": d.floor,
            "drawdown_from_peak": d.drawdown_from_peak,
            "max_intraday_dd": d.max_drawdown_intraday,
            "breach": d.breach,
            "n_trades": d.n_trades,
        }
        if d.ae_proxy_dd_from_peak is not None:
            row["ae_proxy_dd"] = d.ae_proxy_dd_from_peak
        dd_rows.append(row)

    if dd_rows:
        dd_df = pd.DataFrame(dd_rows)
        dd_path = os.path.join(output_dir, f"{safe_label}_drawdown.csv")
        dd_df.to_csv(dd_path, index=False)
        logger.info("Wrote drawdown detail to %s", dd_path)

    # Per-day consistency
    con = scenario.consistency
    if con.per_day_pnl:
        con_df = pd.DataFrame(
            [{"trading_day": d, "daily_pnl": p} for d, p in sorted(con.per_day_pnl.items())]
        )
        con_df["cumulative_pnl"] = con_df["daily_pnl"].cumsum()
        con_path = os.path.join(output_dir, f"{safe_label}_daily_pnl.csv")
        con_df.to_csv(con_path, index=False)
        logger.info("Wrote daily P/L to %s", con_path)


def export_summary_csv(scenarios: List[ScenarioResult], output_dir: str) -> None:
    """Write a high-level summary CSV comparing all scenarios."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for s in scenarios:
        dd = s.drawdown
        con = s.consistency
        rows.append(
            {
                "scenario": s.label,
                "m2k1_size": s.m2k1_size,
                "dynamic": s.is_dynamic,
                "total_net_pl": s.total_net_pl,
                "n_trading_days": s.n_trading_days,
                "n_trades": s.n_trades,
                "hits_target": s.hits_target,
                "days_to_target": s.days_to_target,
                "peak_equity": dd.peak_equity,
                "final_equity": dd.final_equity,
                "max_realized_dd": dd.max_realized_dd,
                "dd_breached": dd.any_breach,
                "breach_days": "; ".join(str(d) for d in dd.breach_days),
                "ae_proxy_max_dd": dd.ae_proxy_max_dd,
                "consistency_pct": con.consistency_pct,
                "passes_consistency": con.passes_consistency,
                "updated_profit_goal": con.updated_profit_goal,
                "verdict": s.verdict,
            }
        )
    summary_df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "summary.csv")
    summary_df.to_csv(path, index=False)
    logger.info("Wrote scenario summary to %s", path)
    return summary_df


# ─────────────────────────────────────────────────────────────────────────────
# Optional charts
# ─────────────────────────────────────────────────────────────────────────────

def generate_charts(scenario: ScenarioResult, output_dir: str) -> None:
    """
    Generate and save equity-curve + drawdown underwater plots.

    Requires matplotlib.  Guarded so the engine runs without a display.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib not available — skipping charts.")
        return

    os.makedirs(output_dir, exist_ok=True)
    days = scenario.drawdown.days
    if not days:
        return

    dates = [d.trading_day for d in days]
    equity = [d.equity_end for d in days]
    dd = [d.drawdown_from_peak for d in days]
    floors = [d.floor for d in days]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Equity curve
    ax1.plot(dates, equity, color="steelblue", label="Account equity")
    ax1.plot(dates, floors, color="red", linestyle="--", alpha=0.7, label="DD floor ($4,500 below peak)")
    ax1.axhline(
        scenario.drawdown.initial_equity + scenario.consistency.net_pl * 0 + 9000 +
        scenario.drawdown.initial_equity,
        color="green", linestyle=":", alpha=0.5, label=f"Profit target (${scenario.profit_target:,.0f})"
    )
    ax1.set_ylabel("Account Equity ($)")
    ax1.set_title(f"Equity Curve — {scenario.label}")
    ax1.legend(fontsize=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.grid(True, alpha=0.3)

    # Mark breach days
    for d in scenario.drawdown.breach_days:
        ax1.axvline(d, color="red", alpha=0.5, linewidth=1)

    # Drawdown underwater
    dd_dollars = [d.drawdown_from_peak for d in days]
    ax2.fill_between(dates, dd_dollars, 0, color="salmon", alpha=0.6, label="Drawdown from peak")
    ax2.axhline(-scenario.drawdown.max_trailing_dd, color="red", linestyle="--",
                label=f"DD limit (${scenario.drawdown.max_trailing_dd:,.0f})")
    ax2.set_ylabel("Drawdown from Peak ($)")
    ax2.set_xlabel("Trading Day")
    ax2.legend(fontsize=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()

    safe_label = scenario.label.replace(" ", "_")
    chart_path = os.path.join(output_dir, f"{safe_label}_chart.png")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close(fig)
    logger.info("Saved chart to %s", chart_path)


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation helper
# ─────────────────────────────────────────────────────────────────────────────

def print_recommendation(scenarios: List[ScenarioResult]) -> None:
    """Print the recommended M2K1! size based on all scenario results."""
    print("\n" + "=" * 70)
    print("  RECOMMENDATION")
    print("=" * 70)

    passing = [
        s for s in scenarios
        if "PASS" in s.verdict and not s.is_dynamic
    ]

    if passing:
        best = max(passing, key=lambda s: s.m2k1_size)
        print(
            f"\n  ✓ Recommended M2K1! size: {best.m2k1_size} lots (static)."
        )
        print(
            f"    Net P/L: ${best.total_net_pl:,.2f}  |  Max DD: ${best.drawdown.max_realized_dd:,.2f}"
        )
        print(f"    Verdict: {best.verdict}")
    else:
        static = [s for s in scenarios if not s.is_dynamic]
        if not static:
            print("\n  ✗ No static scenarios evaluated.")
            return
        safest = min(static, key=lambda s: abs(s.drawdown.max_realized_dd))
        shortfall = max(0, safest.consistency.net_pl * 2 if not safest.consistency.passes_consistency
                        else 9000 - safest.total_net_pl)
        print(
            f"\n  ✗ No M2K1! size passes without breaching $4,500 drawdown."
        )
        print(f"    Safest option: {safest.m2k1_size} lots — net P/L ${safest.total_net_pl:,.2f}, "
              f"shortfall ${shortfall:,.2f}.")
        print(f"    Verdict: {safest.verdict}")

    # Dynamic scenarios
    dyn = [s for s in scenarios if s.is_dynamic and "PASS" in s.verdict]
    if dyn:
        best_dyn = max(dyn, key=lambda s: s.m2k1_size)
        print(
            f"\n  ✓ Best dynamic-sizing scenario: {best_dyn.label}"
        )
        print(f"    Net P/L: ${best_dyn.total_net_pl:,.2f}  |  Verdict: {best_dyn.verdict}")

    print("=" * 70 + "\n")

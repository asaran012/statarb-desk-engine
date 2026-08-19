"""Failure injection.

A monitoring system that has never been shown to fire is not a monitoring
system. Each scenario deliberately corrupts one thing and asserts that the
corresponding check moves off PASS. If a scenario ever fails to trip, the
demo exits non-zero, so the claim "monitoring catches X" stays true or the
build breaks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from ..monitoring.data_health import data_health_check
from ..monitoring.reconciliation import reconcile
from ..monitoring.strategy_health import risk_health_check


@dataclass
class Scenario:
    name: str
    description: str
    detected_by: str
    expected_status: str
    observed_status: str = ""
    detail: str = ""

    @property
    def caught(self) -> bool:
        return self.observed_status == self.expected_status

    def row(self) -> dict:
        return {"scenario": self.name, "injected": self.description,
                "detected_by": self.detected_by, "expected": self.expected_status,
                "observed": self.observed_status,
                "result": "CAUGHT" if self.caught else "MISSED",
                "detail": self.detail}


def _status_of(alerts, check: str) -> str:
    for a in alerts.alerts:
        if a.check == check:
            return a.status
    return "ABSENT"


def run_failure_demo(panel: pd.DataFrame, equity: pd.DataFrame,
                     expected_orders: pd.DataFrame,
                     executed_orders: pd.DataFrame) -> pd.DataFrame:
    scenarios: List[Scenario] = []
    syms = sorted(panel["symbol"].unique())

    # 1. missing symbol -----------------------------------------------------
    bad = panel[panel["symbol"] != syms[0]]
    a = data_health_check(bad, expected_symbols=syms)
    scenarios.append(Scenario(
        "missing_symbol", f"dropped all rows for {syms[0]}",
        "DATA / missing_symbols", "FAIL", _status_of(a, "missing_symbols"),
        f"{syms[0]} absent from feed"))

    # 2. duplicate rows -----------------------------------------------------
    bad = pd.concat([panel, panel.iloc[:25]], ignore_index=True)
    a = data_health_check(bad, expected_symbols=syms)
    scenarios.append(Scenario(
        "duplicate_rows", "re-appended 25 rows of the feed",
        "DATA / duplicate_observations", "FAIL",
        _status_of(a, "duplicate_observations"), "25 duplicated symbol-date keys"))

    # 3. stale prices -------------------------------------------------------
    bad = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    idx = bad.index[bad["symbol"] == syms[1]][200:230]
    for col in ("open", "high", "low", "close", "adjusted_close"):
        bad.loc[idx, col] = float(bad.loc[idx[0], col])
    a = data_health_check(bad, expected_symbols=syms)
    scenarios.append(Scenario(
        "stale_feed", f"froze 30 consecutive closes for {syms[1]}",
        "DATA / longest_stale_run_days", "FAIL",
        _status_of(a, "longest_stale_run_days"),
        "vendor feed stopped updating"))

    # 4. corrupt price ------------------------------------------------------
    bad = panel.copy()
    i = bad.index[bad["symbol"] == syms[2]][300]
    bad.loc[i, "adjusted_close"] = float(bad.loc[i, "adjusted_close"]) * 4.0
    a = data_health_check(bad, expected_symbols=syms)
    scenarios.append(Scenario(
        "corrupt_price", f"multiplied one {syms[2]} close by 4",
        "DATA / extreme_return_alerts", "WARNING",
        _status_of(a, "extreme_return_alerts"), "bad tick / unadjusted split"))

    # 5. negative price -----------------------------------------------------
    bad = panel.copy()
    bad.loc[bad.index[400], "close"] = -5.0
    a = data_health_check(bad, expected_symbols=syms)
    scenarios.append(Scenario(
        "negative_price", "set one close to -5.0",
        "DATA / non_positive_prices", "FAIL",
        _status_of(a, "non_positive_prices"), "log price would be undefined"))

    # 6. missing execution --------------------------------------------------
    dropped = executed_orders.iloc[1:]
    r = reconcile(expected_orders, dropped)
    scenarios.append(Scenario(
        "missing_fill", "removed one broker fill",
        "RECONCILIATION / missing_execution", "FAIL", r.status,
        f"{r.count('missing_execution')} unfilled order(s) -> naked leg"))

    # 7. duplicate execution ------------------------------------------------
    dup = pd.concat([executed_orders, executed_orders.iloc[[0]]], ignore_index=True)
    r = reconcile(expected_orders, dup)
    scenarios.append(Scenario(
        "duplicate_fill", "double-reported one fill",
        "RECONCILIATION / duplicate_execution", "FAIL", r.status,
        f"{r.count('duplicate_execution')} duplicate(s) -> doubled position"))

    # 8. wrong quantity -----------------------------------------------------
    wrong = executed_orders.copy()
    wrong.loc[wrong.index[0], "quantity"] = float(wrong.loc[wrong.index[0],
                                                            "quantity"]) * 0.5
    r = reconcile(expected_orders, wrong)
    scenarios.append(Scenario(
        "quantity_mismatch", "halved one fill quantity",
        "RECONCILIATION / quantity_mismatch", "WARNING", r.status,
        f"{r.count('quantity_mismatch')} partial fill -> unhedged residual"))

    # 9. wrong symbol -------------------------------------------------------
    wrong = executed_orders.copy()
    wrong.loc[wrong.index[2], "symbol"] = "WRONG"
    r = reconcile(expected_orders, wrong)
    scenarios.append(Scenario(
        "symbol_mismatch", "changed one fill's symbol",
        "RECONCILIATION / symbol_mismatch", "FAIL", r.status,
        f"{r.count('symbol_mismatch')} fat-finger -> wrong instrument held"))

    # 10. price deviation ---------------------------------------------------
    wrong = executed_orders.copy()
    wrong.loc[wrong.index[3], "fill_price"] = float(
        wrong.loc[wrong.index[3], "fill_price"]) * 1.2
    r = reconcile(expected_orders, wrong, price_tolerance_bps=50.0)
    scenarios.append(Scenario(
        "price_deviation", "filled one order 20% away from reference",
        "RECONCILIATION / price_deviation", "WARNING", r.status,
        f"{r.count('price_deviation')} bad fill -> hidden slippage"))

    # 11. drawdown breach ---------------------------------------------------
    eq = equity.copy()
    eq["equity"] = eq["equity"].astype(float)
    eq.iloc[-1, eq.columns.get_loc("equity")] = float(eq["equity"].max()) * 0.85
    a = risk_health_check(eq, drawdown_warn=0.05, drawdown_fail=0.10)
    scenarios.append(Scenario(
        "drawdown_breach", "marked the book down 15% from its peak",
        "RISK / current_drawdown_pct", "FAIL",
        _status_of(a, "current_drawdown_pct"), "loss limit breached"))

    # 12. gross exposure breach --------------------------------------------
    eq = equity.copy()
    eq.iloc[-1, eq.columns.get_loc("gross_exposure")] = \
        float(eq["equity"].iloc[-1]) * 1.4
    a = risk_health_check(eq, max_portfolio_gross=1.0)
    scenarios.append(Scenario(
        "gross_breach", "set gross exposure to 140% of equity",
        "RISK / gross_exposure_pct", "FAIL",
        _status_of(a, "gross_exposure_pct"), "leverage limit breached"))

    return pd.DataFrame([s.row() for s in scenarios])


def render(df: pd.DataFrame) -> str:
    lines = ["FAILURE INJECTION DEMO", "=" * 78,
             f"{'scenario':<20s}{'detected by':<38s}{'result':<10s}",
             "-" * 78]
    for _, r in df.iterrows():
        lines.append(f"{r['scenario']:<20s}{r['detected_by']:<38s}{r['result']:<10s}")
    lines.append("-" * 78)
    caught = int((df["result"] == "CAUGHT").sum())
    lines.append(f"{caught}/{len(df)} injected faults detected")
    return "\n".join(lines)

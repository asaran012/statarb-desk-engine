"""Accounting invariants, checkable at runtime and in tests.

These are cheap assertions that catch the class of bug that quietly inflates a
backtest.  They run inside :meth:`BacktestResult.check_invariants` after every
run, not only in the test suite, because a violated invariant means the reported
numbers are wrong and should not be shown at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


@dataclass
class InvariantReport:
    checks: List[dict] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append({"invariant": name, "status": "PASS" if ok else "FAIL",
                            "detail": detail})

    @property
    def all_pass(self) -> bool:
        return all(c["status"] == "PASS" for c in self.checks)

    @property
    def n_pass(self) -> int:
        return sum(1 for c in self.checks if c["status"] == "PASS")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.checks)

    def render(self) -> str:
        lines = ["ACCOUNTING INVARIANTS", "-" * 62]
        for c in self.checks:
            lines.append(f"  [{c['status']}] {c['invariant']:<42s}{c['detail']}")
        lines.append("-" * 62)
        lines.append(f"  {self.n_pass}/{len(self.checks)} passed")
        return "\n".join(lines)


def check_equity_reconciles(starting_equity: float, ending_equity: float,
                            trades: pd.DataFrame, rel_tol: float = 1e-8
                            ) -> tuple[bool, str]:
    """ending_equity == starting_equity + sum(net_pnl), all positions flat."""
    net = float(trades["net_pnl"].sum()) if len(trades) else 0.0
    expected = starting_equity + net
    scale = max(abs(expected), abs(starting_equity), 1.0)
    err = abs(ending_equity - expected)
    return err <= rel_tol * scale, (
        f"end={ending_equity:,.2f} expected={expected:,.2f} err={err:.6f}")


def check_costs_non_negative(trades: pd.DataFrame) -> tuple[bool, str]:
    if not len(trades):
        return True, "no trades"
    bad = int((trades["cost"] < 0).sum())
    return bad == 0, f"{bad} trades with negative cost"


def check_gross_net_consistency(trades: pd.DataFrame, tol: float = 1e-6
                                ) -> tuple[bool, str]:
    if not len(trades):
        return True, "no trades"
    err = float((trades["gross_pnl"] - trades["cost"] - trades["net_pnl"]).abs().max())
    return err <= tol, f"max |gross - cost - net| = {err:.10f}"


def check_timestamps_monotonic(equity: pd.Series) -> tuple[bool, str]:
    ok = equity.index.is_monotonic_increasing and equity.index.is_unique
    return ok, "equity index monotonic and unique" if ok else "index not monotonic/unique"


def check_execution_after_signal(trades: pd.DataFrame, min_delay: int = 1
                                 ) -> tuple[bool, str]:
    if not len(trades) or "entry_signal_date" not in trades.columns:
        return True, "no trades"
    bad = int((pd.to_datetime(trades["entry_date"]) <=
               pd.to_datetime(trades["entry_signal_date"])).sum())
    return bad == 0, f"{bad} fills at or before their own signal bar"


def check_exits_match_entries(trades: pd.DataFrame) -> tuple[bool, str]:
    if not len(trades):
        return True, "no trades"
    bad = int((pd.to_datetime(trades["exit_date"]) <
               pd.to_datetime(trades["entry_date"])).sum())
    unmatched = int(trades["exit_reason"].isna().sum() +
                    (trades["exit_reason"] == "").sum())
    return bad == 0 and unmatched == 0, (
        f"{bad} exits before entry, {unmatched} exits without a reason")


def check_flat_at_end(positions: dict, tol: float = 1e-9) -> tuple[bool, str]:
    live = {k: v for k, v in positions.items() if abs(v) > tol}
    return not live, f"{len(live)} symbols still held after liquidation"


def check_pre_trade_gross_limit(post_trade_ratios: list, limit: float,
                                tol: float = 1e-6) -> tuple[bool, str]:
    """Gross exposure immediately after each fill must respect the limit.

    This is the *enforceable* control: at the moment of trading, the desk knows
    its equity and its book and can size accordingly. It is a hard invariant.

    ``tol`` is 1e-6 of equity, i.e. one dollar in ten million: large enough to
    absorb float accumulation across dozens of legs, far too small to hide a
    sizing bug.
    """
    if not post_trade_ratios:
        return True, "no fills"
    worst = float(max(post_trade_ratios))
    return worst <= limit + tol, f"peak post-fill gross/equity = {worst:.4f} (limit {limit:.2f})"


def mtm_exposure_breaches(exposure: pd.DataFrame, limit: float) -> dict:
    """Mark-to-market drift past the gross limit *between* trades.

    Distinct from the pre-trade control and NOT an accounting invariant. Once a
    position is on, its gross notional moves with prices while equity moves with
    P&L, so the ratio drifts; a desk monitors this and de-risks, it does not
    pretend it cannot happen. Reported as a risk statistic and surfaced by the
    monitoring module, with the count of breach days rather than a silent pass.
    """
    if exposure.empty:
        return {"breach_days": 0, "peak_ratio": 0.0, "limit": limit}
    ratio = (exposure["gross_exposure"] / exposure["equity"]).replace(
        [np.inf, -np.inf], np.nan).dropna()
    return {"breach_days": int((ratio > limit).sum()),
            "peak_ratio": float(ratio.max()) if len(ratio) else 0.0,
            "limit": float(limit),
            "pct_days_in_breach": float((ratio > limit).mean() * 100) if len(ratio) else 0.0}


def check_no_position_without_trade(trades: pd.DataFrame,
                                    n_position_events: int) -> tuple[bool, str]:
    """Every open/close event must be represented by a trade record."""
    expected = 2 * len(trades)
    return expected == n_position_events, (
        f"{n_position_events} position events vs {expected} implied by "
        f"{len(trades)} trades")

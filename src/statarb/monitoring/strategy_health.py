"""Strategy, risk and trading monitoring, plus the daily desk report.

The report is the artifact a Desk Quant Analyst actually produces: a single
page, run before the open and after the close, that says whether the book is
behaving and whether anything needs chasing. Every number on it is computed
from the same objects the backtester produced, so the monitor cannot drift away
from the engine's view of the world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .alerts import AlertSet, FAIL, PASS, WARNING, combine
from .reconciliation import ReconciliationReport


def strategy_health_check(*, signal_states: pd.DataFrame, equity: pd.DataFrame,
                          as_of: pd.Timestamp, stop_z: float = 4.0) -> AlertSet:
    """``signal_states``: one row per active pair with columns pair, zscore, target."""
    a = AlertSet("STRATEGY")
    active = int((signal_states["target"] != 0).sum()) if len(signal_states) else 0
    a.add("active_pairs", PASS, active)
    a.add("monitored_pairs", PASS, int(len(signal_states)))
    if len(signal_states):
        new_signals = int(signal_states.get("new_signal", pd.Series(dtype=bool)).sum()) \
            if "new_signal" in signal_states else 0
        exits = int(signal_states.get("exiting", pd.Series(dtype=bool)).sum()) \
            if "exiting" in signal_states else 0
        a.add("new_signals", PASS, new_signals)
        a.add("exits", PASS, exits)
        wide = int((signal_states["zscore"].abs() > stop_z).sum())
        a.gauge(f"spread_beyond_{stop_z:g}_sigma", wide, warn=1, fail=3)
        stalled = int((signal_states["zscore"].isna()).sum())
        a.gauge("pairs_without_zscore", stalled, warn=1, fail=3)
    else:
        a.add("new_signals", PASS, 0)
        a.add("exits", PASS, 0)
    return a


def risk_health_check(equity: pd.DataFrame, *, max_portfolio_gross: float = 1.0,
                      max_symbol_exposure: float = 0.15,
                      symbol_exposures: Optional[Dict[str, float]] = None,
                      drawdown_warn: float = 0.05,
                      drawdown_fail: float = 0.10) -> AlertSet:
    from ..risk.metrics import drawdown_series
    a = AlertSet("RISK")
    if equity.empty:
        a.add("equity_history", FAIL, 0, detail="no equity history")
        return a
    last = equity.iloc[-1]
    eq = float(last["equity"])
    gross = float(last["gross_exposure"]) / eq if eq else np.nan
    net = float(last["net_exposure"]) / eq if eq else np.nan
    a.gauge("gross_exposure_pct", round(gross * 100, 2),
            warn=max_portfolio_gross * 100 * 0.95, fail=max_portfolio_gross * 100)
    a.add("net_exposure_pct", PASS, round(net * 100, 2))
    if symbol_exposures:
        top_sym = max(symbol_exposures, key=symbol_exposures.get)
        top = symbol_exposures[top_sym] / eq if eq else np.nan
        a.gauge("largest_symbol_exposure_pct", round(top * 100, 2),
                warn=max_symbol_exposure * 100 * 0.9,
                fail=max_symbol_exposure * 100, detail=top_sym)
    dd = float(drawdown_series(equity["equity"]).iloc[-1])
    a.gauge("current_drawdown_pct", round(abs(dd) * 100, 2),
            warn=drawdown_warn * 100, fail=drawdown_fail * 100)
    return a


def trading_health_check(expected_orders: pd.DataFrame,
                         executed_orders: pd.DataFrame,
                         recon: ReconciliationReport) -> AlertSet:
    a = AlertSet("TRADING")
    a.add("expected_orders", PASS, int(len(expected_orders)))
    a.add("executed_orders", PASS, int(len(executed_orders)))
    a.gauge("missing_fills", recon.count("missing_execution"), fail=1)
    a.gauge("unexpected_fills", recon.count("unexpected_execution"), fail=1)
    a.gauge("duplicate_fills", recon.count("duplicate_execution"), fail=1)
    a.gauge("quantity_mismatches", recon.count("quantity_mismatch"), warn=1, fail=3)
    a.gauge("symbol_mismatches", recon.count("symbol_mismatch"), fail=1)
    a.gauge("price_deviations", recon.count("price_deviation"), warn=1, fail=10)
    return a


@dataclass
class DeskReport:
    as_of: pd.Timestamp
    sections: List[AlertSet] = field(default_factory=list)
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return combine(*self.sections)

    def n_checks(self) -> int:
        return sum(len(s.alerts) for s in self.sections)

    def to_dict(self) -> dict:
        return {"as_of": str(pd.Timestamp(self.as_of).date()), "status": self.status,
                "n_checks": self.n_checks(),
                "sections": {s.section: [a.to_dict() for a in s.alerts]
                             for s in self.sections},
                **self.extra}

    def render(self) -> str:
        w = 46
        lines = ["=" * w, "STATARB DAILY DESK HEALTH REPORT", "=" * w,
                 f"DATE  {pd.Timestamp(self.as_of).date()}", ""]
        for s in self.sections:
            lines.append(s.section)
            for al in s.alerts:
                mark = {"PASS": " ", "WARNING": "!", "FAIL": "X"}[al.status]
                val = al.value
                sval = f"{val:,.2f}" if isinstance(val, float) else f"{val:,}" \
                    if isinstance(val, int) else str(val)
                detail = f"  <- {al.detail}" if al.detail else ""
                lines.append(f" {mark} {al.check:<32s}{sval:>10}{detail}")
            lines.append(f"   {'section status':<32s}{s.status:>10}")
            lines.append("")
        lines.append("=" * w)
        lines.append(f"OVERALL STATUS: {self.status}")
        lines.append("=" * w)
        return "\n".join(lines)


def build_desk_report(*, as_of, data: AlertSet, strategy: AlertSet, risk: AlertSet,
                      trading: AlertSet, recon: ReconciliationReport) -> DeskReport:
    rec = AlertSet("RECONCILIATION")
    rec.add("expected_positions", PASS, recon.n_expected)
    rec.add("actual_positions", PASS, recon.n_executed)
    rec.add("matched", PASS, recon.matched)
    rec.gauge("total_breaks", int(len(recon.breaks)), warn=1, fail=3)
    return DeskReport(as_of=as_of, sections=[data, strategy, risk, trading, rec])

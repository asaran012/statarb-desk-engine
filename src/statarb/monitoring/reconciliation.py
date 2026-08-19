"""Trade reconciliation.

Compares what the strategy *intended* to trade against what the broker says it
*did* trade.  On a real desk this runs after the close and any break is chased
before the next open, because an unreconciled position is an unhedged position.

Six break types are detected:

  ``missing_execution``   an expected order with no fill
  ``unexpected_execution`` a fill with no matching expected order
  ``duplicate_execution``  the same order id filled more than once
  ``quantity_mismatch``    filled quantity differs beyond tolerance
  ``symbol_mismatch``      the fill hit a different symbol than instructed
  ``price_deviation``      fill price outside the tolerance band

Matching is on ``order_id`` when present, which is what makes duplicate and
symbol breaks detectable at all: matching on (symbol, date) instead silently
merges a duplicate fill into a single larger position and reports "quantity
mismatch", losing the actual failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from .alerts import FAIL, PASS, WARNING

EXPECTED_COLUMNS = ["order_id", "date", "symbol", "side", "quantity", "limit_price"]
EXECUTED_COLUMNS = ["order_id", "date", "symbol", "side", "quantity", "fill_price"]


@dataclass
class ReconciliationReport:
    breaks: pd.DataFrame
    n_expected: int
    n_executed: int
    matched: int
    counts: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.breaks.empty:
            return PASS
        severe = {"missing_execution", "unexpected_execution", "duplicate_execution",
                  "symbol_mismatch"}
        return FAIL if set(self.breaks["break_type"]) & severe else WARNING

    def count(self, break_type: str) -> int:
        return int(self.counts.get(break_type, 0))

    def render(self) -> str:
        lines = ["TRADE RECONCILIATION", "-" * 52,
                 f"{'expected orders':<32s}{self.n_expected:>12}",
                 f"{'executed orders':<32s}{self.n_executed:>12}",
                 f"{'matched':<32s}{self.matched:>12}"]
        for k in ["missing_execution", "unexpected_execution", "duplicate_execution",
                  "quantity_mismatch", "symbol_mismatch", "price_deviation"]:
            lines.append(f"{'  ' + k:<32s}{self.count(k):>12}")
        lines.append("-" * 52)
        lines.append(f"{'status':<32s}{self.status:>12}")
        return "\n".join(lines)


def reconcile(expected: pd.DataFrame, executed: pd.DataFrame, *,
              qty_tolerance: float = 1e-6,
              price_tolerance_bps: float = 50.0) -> ReconciliationReport:
    exp = expected.copy()
    exe = executed.copy()
    for df, cols, name in ((exp, EXPECTED_COLUMNS, "expected"),
                           (exe, EXECUTED_COLUMNS, "executed")):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} orders missing columns {missing}")

    breaks: List[dict] = []

    dup_ids = exe["order_id"][exe["order_id"].duplicated(keep=False)].unique()
    for oid in dup_ids:
        rows = exe[exe["order_id"] == oid]
        breaks.append({"break_type": "duplicate_execution", "order_id": oid,
                       "detail": f"{len(rows)} fills for one order id",
                       "expected": np.nan, "executed": float(rows["quantity"].sum())})
    exe_unique = exe.drop_duplicates("order_id", keep="first")

    exp_ids = set(exp["order_id"])
    exe_ids = set(exe_unique["order_id"])

    for oid in sorted(exp_ids - exe_ids):
        r = exp[exp["order_id"] == oid].iloc[0]
        breaks.append({"break_type": "missing_execution", "order_id": oid,
                       "detail": f"{r['side']} {r['quantity']:g} {r['symbol']}",
                       "expected": float(r["quantity"]), "executed": np.nan})

    for oid in sorted(exe_ids - exp_ids):
        r = exe_unique[exe_unique["order_id"] == oid].iloc[0]
        breaks.append({"break_type": "unexpected_execution", "order_id": oid,
                       "detail": f"{r['side']} {r['quantity']:g} {r['symbol']}",
                       "expected": np.nan, "executed": float(r["quantity"])})

    merged = exp.merge(exe_unique, on="order_id", suffixes=("_exp", "_exe"))
    for _, r in merged.iterrows():
        if r["symbol_exp"] != r["symbol_exe"]:
            breaks.append({"break_type": "symbol_mismatch", "order_id": r["order_id"],
                           "detail": f"{r['symbol_exp']} -> {r['symbol_exe']}",
                           "expected": np.nan, "executed": np.nan})
            continue
        qexp = float(r["quantity_exp"]) * (1 if r["side_exp"] == "BUY" else -1)
        qexe = float(r["quantity_exe"]) * (1 if r["side_exe"] == "BUY" else -1)
        if abs(qexp - qexe) > qty_tolerance:
            breaks.append({"break_type": "quantity_mismatch", "order_id": r["order_id"],
                           "detail": f"{r['symbol_exp']} {qexp:g} vs {qexe:g}",
                           "expected": qexp, "executed": qexe})
        ref = float(r["limit_price"])
        fill = float(r["fill_price"])
        if ref > 0 and abs(fill - ref) / ref * 10_000 > price_tolerance_bps:
            breaks.append({"break_type": "price_deviation", "order_id": r["order_id"],
                           "detail": (f"{r['symbol_exp']} ref={ref:.4f} fill={fill:.4f} "
                                      f"({abs(fill-ref)/ref*1e4:.1f} bps)"),
                           "expected": ref, "executed": fill})

    bf = pd.DataFrame(breaks, columns=["break_type", "order_id", "detail",
                                       "expected", "executed"])
    counts = bf["break_type"].value_counts().to_dict() if not bf.empty else {}
    matched = len(merged) - int(counts.get("symbol_mismatch", 0))
    return ReconciliationReport(bf, len(exp), len(exe), matched, counts)


def orders_from_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Derive the expected-order blotter implied by a trade ledger.

    Each pair round trip becomes four single-leg orders (open A, open B,
    close A, close B), which is the granularity a broker actually reports.
    """
    rows: List[dict] = []
    for i, t in trades.reset_index(drop=True).iterrows():
        for leg, sym, qty, px in (("a", t["symbol_a"], t["qty_a"], t["entry_price_a"]),
                                  ("b", t["symbol_b"], t["qty_b"], t["entry_price_b"])):
            rows.append({"order_id": f"T{i:05d}-{leg}-open", "date": t["entry_date"],
                         "symbol": sym, "side": "BUY" if qty > 0 else "SELL",
                         "quantity": abs(float(qty)), "limit_price": float(px)})
        for leg, sym, qty, px in (("a", t["symbol_a"], t["qty_a"], t["exit_price_a"]),
                                  ("b", t["symbol_b"], t["qty_b"], t["exit_price_b"])):
            rows.append({"order_id": f"T{i:05d}-{leg}-close", "date": t["exit_date"],
                         "symbol": sym, "side": "SELL" if qty > 0 else "BUY",
                         "quantity": abs(float(qty)), "limit_price": float(px)})
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def simulate_executions(expected: pd.DataFrame, *, seed: int = 7,
                        slippage_bps: float = 2.0) -> pd.DataFrame:
    """A clean broker file: every order filled once, at a small random slippage."""
    rng = np.random.default_rng(seed)
    exe = expected.copy()
    drift = rng.normal(0.0, slippage_bps / 10_000.0, size=len(exe))
    exe["fill_price"] = exe["limit_price"] * (1.0 + drift)
    return exe[EXECUTED_COLUMNS]

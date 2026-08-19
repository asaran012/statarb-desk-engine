"""Monitoring + failure-injection demo on real data.

Mirrors statarb.cli's cmd_monitor / cmd_failures exactly, with one change:
selection_mode="correlation_only" instead of the CLI's implicit "full",
because the production ("full") pipeline selects zero real pairs in every
window (see reports/real/real_walkforward_summary.json) and these two demos
need an actual order/fill blotter to inject faults into. This does not touch
monitoring, reconciliation, or failure-injection logic -- only which already-
existing pair-selection ablation supplies the trades they operate on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from statarb.config import load_config
from statarb.data.loader import to_wide
from statarb.backtest.engine import run_backtest
from statarb.monitoring.data_health import data_health_check
from statarb.monitoring.failure_demo import render, run_failure_demo
from statarb.monitoring.reconciliation import orders_from_trades, reconcile, simulate_executions
from statarb.monitoring.strategy_health import (build_desk_report, risk_health_check,
                                                strategy_health_check, trading_health_check)
from statarb.research.pair_selection import select_pairs
from statarb.signals.spread import build_spread
from statarb.signals.strategy import generate_signals
from statarb.signals.zscore import rolling_zscore

cfg = load_config("config/real.yaml")
panel = pd.read_parquet("data/processed_real/panel.parquet")
panel["date"] = pd.to_datetime(panel["date"])
out = Path(cfg.reporting.reports_dir)
out.mkdir(parents=True, exist_ok=True)

train_end = pd.Timestamp(cfg.data.start_date) + pd.DateOffset(
    years=cfg.walkforward.train_years)
rep = select_pairs(panel[panel["date"] < train_end],
                   max_pairs=cfg.research.max_pairs_per_window,
                   correlation_threshold=cfg.research.correlation_threshold,
                   universe_kwargs=cfg.universe.model_dump(),
                   selection_mode="correlation_only")
test_start = train_end + pd.DateOffset(years=cfg.walkforward.validation_years)
test_end = test_start + pd.DateOffset(years=1)
bt = run_backtest(prices_close=to_wide(panel, cfg.data.price_field),
                  prices_open=to_wide(panel, "open"),
                  pairs=rep.pairs, test_start=test_start, test_end=test_end, config=cfg)
print(f"monitoring/failures test window: {test_start.date()}..{test_end.date()}  "
     f"pairs={len(rep.pairs)}  trades={len(bt.trades)}")

orders = orders_from_trades(bt.trades)
fills = simulate_executions(orders, seed=cfg.seed)
recon = reconcile(orders, fills)

# ---- monitoring -------------------------------------------------------------
as_of = bt.equity.index[-1]
close = to_wide(panel, cfg.data.price_field).loc[:as_of]
rows = []
for _, p in rep.pairs.iterrows():
    a, b = p["symbol_a"], p["symbol_b"]
    if a not in close.columns or b not in close.columns:
        continue
    spread = build_spread(np.log(close[a]), np.log(close[b]), float(p["beta"]))
    z = rolling_zscore(spread, cfg.strategy.zscore_lookback)["zscore"]
    sig = generate_signals(z, entry_z=cfg.strategy.entry_z, exit_z=cfg.strategy.exit_z,
                           stop_z=cfg.strategy.stop_z)
    rows.append({"pair": f"{a}/{b}", "zscore": float(z.iloc[-1]),
                "target": int(sig["target"].iloc[-1]),
                "new_signal": bool(sig["entry_flag"].iloc[-1]),
                "exiting": bool(sig["exit_flag"].iloc[-1])})
states = pd.DataFrame(rows)

d = data_health_check(panel, expected_symbols=sorted(panel["symbol"].unique()))
s = strategy_health_check(signal_states=states, equity=bt.equity, as_of=as_of,
                          stop_z=cfg.strategy.stop_z)
r = risk_health_check(bt.equity, max_portfolio_gross=cfg.risk.max_portfolio_gross,
                      max_symbol_exposure=cfg.risk.max_symbol_exposure)
t = trading_health_check(orders, fills, recon)
desk = build_desk_report(as_of=as_of, data=d, strategy=s, risk=r, trading=t, recon=recon)
print("\n" + desk.render())
print()
print(recon.render())
(out / "daily_report_real.txt").write_text(desk.render() + "\n\n" + recon.render())
(out / "daily_report_real.json").write_text(json.dumps(desk.to_dict(), indent=2, default=str))

n_checks = {"data": len(d.alerts), "strategy": len(s.alerts), "risk": len(r.alerts),
           "trading": len(t.alerts)}
print(f"\nmonitoring checks by category: {n_checks}  total={sum(n_checks.values())}")
with open(out / "monitoring_check_inventory_real.json", "w") as fh:
    json.dump({
        "by_category": n_checks,
        "total": sum(n_checks.values()),
        "checks": {
            "data": [a.check for a in d.alerts],
            "strategy": [a.check for a in s.alerts],
            "risk": [a.check for a in r.alerts],
            "trading": [a.check for a in t.alerts],
        },
    }, fh, indent=2)

# ---- failure injection --------------------------------------------------------
df = run_failure_demo(panel, bt.equity, orders, fills)
out_f = out
df.to_csv(out_f / "failure_demo_real.csv", index=False)
print("\n" + render(df))
missed = df[df["result"] != "CAUGHT"]
caught = int((df["result"] == "CAUGHT").sum())
print(f"\n{caught}/{len(df)} scenarios caught")
if len(missed):
    print("MISSED:")
    print(missed[["scenario", "expected", "observed"]].to_string(index=False))

with open(out_f / "reconciliation_results_real.json", "w") as fh:
    json.dump({"n_expected": recon.n_expected, "n_executed": recon.n_executed,
              "matched": recon.matched, "status": recon.status,
              "break_counts": recon.counts}, fh, indent=2, default=str)
print(f"\nwrote {out_f/'daily_report_real.txt'}, monitoring_check_inventory_real.json, "
     f"failure_demo_real.csv, reconciliation_results_real.json")

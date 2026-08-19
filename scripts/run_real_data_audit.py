"""Real-data walk-forward audit driver.

Runs the pipeline's existing (pre-built) selection_mode ablations against the
real-market panel and records what actually happens, including outright
failures. No research or strategy threshold is changed from config/real.yaml;
every mode exercised here (full, correlation_only, no_fdr, no_stability) is
already implemented in statarb.research.pair_selection for exactly this kind
of diagnostic comparison. Where an ablation still requires the "full"
selection stage as a prerequisite (C/D/E) and that stage finds zero pairs on
real data, the failure is recorded verbatim rather than silently swapped for
a mode that produces trades.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from statarb.config import load_config
from statarb.backtest.walkforward import run_walkforward
from statarb.research.ablations import EXPERIMENTS

CFG_PATH = "config/real.yaml"
OUT = Path("reports/real")
OUT.mkdir(parents=True, exist_ok=True)


def run_one(panel, cfg, *, selection_mode="full", hedge_ratio_method=None,
            risk_enabled=True, label="run"):
    t0 = time.perf_counter()
    try:
        res = run_walkforward(panel, cfg, selection_mode=selection_mode,
                              hedge_ratio_method=hedge_ratio_method,
                              risk_enabled=risk_enabled, verbose=False)
        return {"label": label, "status": "ok", "runtime_s": time.perf_counter() - t0,
                "result": res}
    except RuntimeError as e:
        return {"label": label, "status": "FAILED", "runtime_s": time.perf_counter() - t0,
                "error": str(e)}


def main():
    cfg = load_config(CFG_PATH)
    panel = pd.read_parquet("data/processed_real/panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])

    # ---- 1. production config, exactly as configured -----------------------
    print("=== production (full) selection_mode on real data ===")
    prod = run_one(panel, cfg, selection_mode="full", label="production_full")
    print(f"  status={prod['status']}"
          + (f" error={prod.get('error')}" if prod["status"] == "FAILED" else ""))

    # ---- 2. every pre-existing ablation experiment --------------------------
    print("\n=== ablation experiments (pre-existing selection_mode/overrides) ===")
    ablation_rows = []
    ablation_results = {}
    for exp in EXPERIMENTS:
        cfg_e = cfg
        if "costs" in exp.overrides:
            cfg_e = cfg_e.with_costs(**exp.overrides["costs"])
        out = run_one(panel, cfg_e, selection_mode=exp.selection_mode,
                      hedge_ratio_method=exp.overrides.get("hedge_ratio_method"),
                      risk_enabled=exp.overrides.get("risk_enabled", True),
                      label=exp.key)
        row = {"key": exp.key, "experiment": exp.experiment,
               "selection_mode": exp.selection_mode, "status": out["status"]}
        if out["status"] == "ok":
            res = out["result"]
            perf = res.summary
            row.update({
                "windows_with_pairs": len(res.windows),
                "pairs_selected_avg": float(np.mean(
                    [len(w.selection.pairs) for w in res.windows])),
                "trades": int(len(res.trades)),
                "cumulative_return": perf["cumulative_return"],
                "cagr": perf["cagr"], "sharpe": perf["sharpe"],
                "sortino": perf["sortino"], "max_drawdown": perf["max_drawdown"],
                "hit_rate": perf["hit_rate"],
                "total_costs": float(res.trades["cost"].sum()) if len(res.trades) else 0.0,
                "gross_pnl": float(res.trades["gross_pnl"].sum()) if len(res.trades) else 0.0,
                "net_pnl": float(res.trades["net_pnl"].sum()) if len(res.trades) else 0.0,
            })
            ablation_results[exp.key] = res
            print(f"  {exp.key:<20s} OK   windows={len(res.windows)} "
                  f"trades={len(res.trades)} sharpe={perf['sharpe']:.3f} "
                  f"cum_ret={perf['cumulative_return']:.4f}")
        else:
            row["error"] = out["error"]
            print(f"  {exp.key:<20s} FAILED  {out['error']}")
        ablation_rows.append(row)
    pd.DataFrame(ablation_rows).to_csv(OUT / "ablation_results_real.csv", index=False)

    # ---- 3. cost sensitivity, run against correlation_only (the only mode
    #         that produces trades in every window) since "full" fails at
    #         every cost level identically (0 pairs is cost-independent) -----
    print("\n=== cost sensitivity (correlation_only mode; full mode has 0 trades "
          "at every cost level since pair selection precedes costs) ===")
    cost_rows = []
    for bps in cfg.cost_sensitivity_bps:
        cfg_c = cfg.with_costs(commission_bps=bps / 3.0, slippage_bps=bps * 2.0 / 3.0)
        out = run_one(panel, cfg_c, selection_mode="correlation_only",
                      label=f"cost_{bps}")
        if out["status"] == "ok":
            res = out["result"]
            perf = res.summary
            cost_rows.append({"total_bps": bps, "sharpe": perf["sharpe"],
                              "cumulative_return": perf["cumulative_return"],
                              "cagr": perf["cagr"], "max_drawdown": perf["max_drawdown"],
                              "trades": int(len(res.trades)),
                              "total_costs": float(res.trades["cost"].sum())
                              if len(res.trades) else 0.0})
            print(f"  {bps:>4g} bps  trades={len(res.trades)} "
                  f"sharpe={perf['sharpe']:.3f} cum_ret={perf['cumulative_return']:.4f}")
        else:
            print(f"  {bps:>4g} bps  FAILED {out['error']}")
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(OUT / "cost_sensitivity_real.csv", index=False)

    # ---- 4. full detail on the correlation_only walk-forward (the only mode
    #         with trades in every window) for year-by-year / drawdown / trade
    #         stats / per-fold reporting ----------------------------------
    if "A_correlation_only" in ablation_results:
        res = ablation_results["A_correlation_only"]
        fold_rows = []
        for w in res.windows:
            t = w.test
            perf_fold = {
                "window": w.window.index,
                "train_start": str(w.window.train_start.date()),
                "train_end": str(w.window.train_end.date()),
                "valid_start": str(w.window.valid_start.date()),
                "valid_end": str(w.window.valid_end.date()),
                "test_start": str(w.window.test_start.date()),
                "test_end": str(w.window.test_end.date()),
                "pairs_selected": len(w.selection.pairs),
                "chosen_zscore_lookback": w.chosen_params["zscore_lookback"],
                "chosen_entry_z": w.chosen_params["entry_z"],
                "chosen_exit_z": w.chosen_params["exit_z"],
                "trades": len(t.trades),
                "gross_pnl": float(t.gross_pnl),
                "total_costs": float(t.total_costs),
                "net_pnl": float(t.net_pnl),
                "starting_equity": t.starting_equity,
                "ending_equity": t.ending_equity,
                "invariants_pass": t.invariants.all_pass,
                "n_invariants": len(t.invariants.checks),
            }
            rets = t.equity["equity"].pct_change().dropna()
            if len(rets) > 1 and rets.std() > 0:
                perf_fold["sharpe"] = float(rets.mean() / rets.std() * np.sqrt(252))
            else:
                perf_fold["sharpe"] = float("nan")
            eq = t.equity["equity"]
            perf_fold["max_drawdown"] = float((eq / eq.cummax() - 1.0).min())
            fold_rows.append(perf_fold)
        pd.DataFrame(fold_rows).to_csv(OUT / "walk_forward_results_real.csv", index=False)
        print(f"\nwrote {OUT/'walk_forward_results_real.csv'} ({len(fold_rows)} folds)")

        # year-by-year
        trades = res.trades.copy()
        equity = res.equity
        if len(trades):
            trades["exit_year"] = pd.to_datetime(trades["exit_date"]).dt.year
        yearly = []
        for yr, grp in equity.groupby(equity.index.year):
            yr_trades = trades[trades["exit_year"] == yr] if len(trades) else trades
            yearly.append({
                "year": int(yr),
                "trades": int(len(yr_trades)),
                "gross_pnl": float(yr_trades["gross_pnl"].sum()) if len(yr_trades) else 0.0,
                "net_pnl": float(yr_trades["net_pnl"].sum()) if len(yr_trades) else 0.0,
                "total_costs": float(yr_trades["cost"].sum()) if len(yr_trades) else 0.0,
                "start_equity": float(grp.iloc[0]), "end_equity": float(grp.iloc[-1]),
                "year_return": float(grp.iloc[-1] / grp.iloc[0] - 1.0),
            })
        pd.DataFrame(yearly).to_csv(OUT / "yearly_results_real.csv", index=False)
        print(f"wrote {OUT/'yearly_results_real.csv'}")

        trades.to_csv(OUT / "trades" / "trades_correlation_only.csv", index=False)
        equity.to_frame("equity").to_csv(OUT / "trades" / "equity_correlation_only.csv")

    summary = {
        "production_full_status": prod["status"],
        "production_full_error": prod.get("error"),
        "ablation_summary": ablation_rows,
        "cost_sensitivity_note": ("full-mode cost sensitivity not run: 0 pairs "
                                  "selected in every window makes it independent "
                                  "of cost level (see production_full_status)"),
    }
    with open(OUT / "real_walkforward_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nwrote {OUT/'real_walkforward_summary.json'}")


if __name__ == "__main__":
    main()

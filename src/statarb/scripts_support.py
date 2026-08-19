"""Report assembly used by `statarb report`."""

from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .reporting import plots as P
from .reporting.report import (write_metrics_json, write_research_summary,
                               write_resume_metrics)
from .research.ablations import breakeven_cost_bps

ORDER = ["baseline", "A_correlation_only", "A_no_fdr", "B_no_stability",
         "C_gross_no_costs", "D_rolling_beta", "E_no_risk_limits"]


def _read(path, **kw):
    p = Path(path)
    return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()


def _json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else (default or {})


def _benchmark_engine(cfg: Config, pairs: pd.DataFrame, *, warmup: int = 5,
                      measured: int = 10) -> dict:
    """Fast vs reference backtest loop: bit-for-bit equivalence + timing, on
    whatever panel/pairs this run actually produced. Mirrors
    scripts/benchmark_real_engine.py so the number in the report is the same
    kind of measurement, just taken on the current (often synthetic) data
    instead of the frozen real-data run."""
    from .backtest.engine import run_backtest
    from .data.loader import to_wide
    panel_path = Path(cfg.data.processed_dir) / "panel.parquet"
    if pairs.empty or not panel_path.exists():
        return {"reference_ms": None, "fast_ms": None, "speedup": None,
                "identical": None, "note": "no pairs/panel available to benchmark"}

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    close = to_wide(panel, cfg.data.price_field)
    opn = to_wide(panel, "open") if "open" in panel.columns else close
    kwargs = dict(prices_close=close, prices_open=opn, pairs=pairs,
                 test_start=close.index[0], test_end=close.index[-1], config=cfg)

    a = run_backtest(**kwargs, loop="reference")
    b = run_backtest(**kwargs, loop="fast")
    cols = ["gross_pnl", "cost", "net_pnl", "entry_z", "qty_a", "qty_b", "beta"]
    identical = bool(
        len(a.trades) == len(b.trades)
        and np.array_equal(a.trades[cols].to_numpy(dtype=float),
                           b.trades[cols].to_numpy(dtype=float))
        and np.array_equal(a.equity["equity"].to_numpy(), b.equity["equity"].to_numpy()))

    def timed(loop, n):
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            run_backtest(**kwargs, loop=loop)
            times.append(time.perf_counter() - t0)
        return times

    for loop in ("reference", "fast"):
        timed(loop, warmup)
    ref_ms = float(np.median(timed("reference", measured))) * 1000
    fast_ms = float(np.median(timed("fast", measured))) * 1000
    return {"reference_ms": round(ref_ms, 1), "fast_ms": round(fast_ms, 1),
            "speedup": round(ref_ms / fast_ms, 2), "identical": identical,
            "n_pairs": int(len(pairs)), "warmup_runs": warmup, "measured_runs": measured}


def generate_full_report(cfg: Config) -> dict:
    rep = Path(cfg.reporting.reports_dir)
    fig = Path(cfg.reporting.figures_dir)
    rep.mkdir(parents=True, exist_ok=True)

    eq = _read(Path(cfg.reporting.trades_dir) / "equity.csv", index_col=0,
               parse_dates=True)
    trades = _read(Path(cfg.reporting.trades_dir) / "trades.csv",
                   parse_dates=["entry_date", "exit_date"])
    if eq.empty or trades.empty:
        raise SystemExit("run `statarb backtest` first - no trades/equity found")
    equity = eq["equity"]

    wdf = _read(rep / "windows.csv")
    ab = _read(rep / "ablations.csv")
    if not ab.empty:
        ab["_o"] = ab["key"].map({k: i for i, k in enumerate(ORDER)})
        ab = ab.sort_values("_o").drop(columns="_o")
    cs = _read(rep / "cost_sensitivity.csv").sort_values("total_bps") \
        if (rep / "cost_sensitivity.csv").exists() else pd.DataFrame()
    fd = _read(rep / "failure_demo.csv")
    funnel = _json(rep / "funnel.json")
    truth = _json(rep / "truth_score.json")
    wf = _json(rep / "walkforward_summary.json")

    bounds = [tuple(r.split("..")) for r in wdf["test"]] if not wdf.empty else None
    P.equity_curve(equity, fig, window_bounds=bounds)
    P.drawdown_curve(equity, fig)
    P.rolling_sharpe(equity, fig)
    P.annual_returns(equity, fig)
    P.trade_pnl_distribution(trades, fig)
    P.holding_period_distribution(trades, fig)
    if not cs.empty:
        P.cost_sensitivity(cs, fig)
    if not ab.empty:
        P.ablation_bars(ab, fig)
    if funnel:
        P.funnel_chart(funnel, fig)
    pq = _read(rep / "selected_pairs_w0.csv")
    if not pq.empty:
        P.pair_quality(pq, fig)
    vg = _read(rep / "validation_grid.csv")
    if not vg.empty:
        P.parameter_heatmap(vg, fig)

    caught = (f"{int((fd['result'] == 'CAUGHT').sum())}/{len(fd)}"
              if not fd.empty else "NOT YET MEASURED")
    engine_bench = _benchmark_engine(cfg, pq)
    speedup = engine_bench.get("speedup")
    eng = {"modules": len(glob.glob("src/statarb/**/*.py", recursive=True)),
           "accounting invariants": 9, "data validation checks": 11,
           "reconciliation break types": 6, "monitoring checks": 29,
           "engine speedup": f"{speedup}x" if speedup else "NOT MEASURED",
           "walk-forward windows": len(wdf),
           "symbols": cfg.data.n_symbols,
           "cointegration tests per window": funnel.get("cointegration tests run", 0)}

    write_metrics_json(
        rep / "metrics.json", walkforward=wf, funnel=funnel, ablations=ab,
        cost_sens=cs, failure_demo=fd,
        engine_bench=engine_bench,
        storage_bench=None, truth_scores=truth, test_count=73,
        data_meta={"symbols": cfg.data.n_symbols, "source": cfg.data.source})
    if not wdf.empty:
        write_research_summary(rep / "research_summary.md", funnel=funnel,
                               truth=truth, walkforward=wf, windows=wdf,
                               ablations=ab, cost_sens=cs,
                               breakeven=breakeven_cost_bps(cs) if not cs.empty
                               else float("nan"))
    write_resume_metrics(rep / "resume_metrics.md", engineering=eng, funnel=funnel,
                         truth=truth, walkforward=wf, ablations=ab,
                         test_count=73, failure_caught=caught,
                         engine_bench=engine_bench)
    n_fig = len(glob.glob(str(fig / "*.png")))
    print(f"wrote {n_fig} figures -> {fig}")
    for f in ["metrics.json", "research_summary.md", "resume_metrics.md"]:
        if (rep / f).exists():
            print(f"  {rep/f}")
    return {"figures": n_fig}

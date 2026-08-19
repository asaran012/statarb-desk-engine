"""Fast vs reference engine: equivalence + timing benchmark on real data.

Uses the correlation_only pair set (the only real-data selection mode that
survives with a full pair book in every window) trading the FULL real date
range in one call, so the benchmark reflects production-scale data (69
symbols, ~2,400 trading sessions) rather than a toy fixture. Equivalence is
checked bit-for-bit on the trade ledger and equity curve; timing is 5
warm-up runs discarded, then 10 measured runs per loop, min/median/mean/std
reported, no config value touched.
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
from statarb.data.loader import to_wide
from statarb.backtest.engine import run_backtest
from statarb.research.pair_selection import select_pairs

CFG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config/real.yaml"
PANEL_PATH = sys.argv[2] if len(sys.argv) > 2 else "data/processed_real/panel.parquet"
OUT_PATH = sys.argv[3] if len(sys.argv) > 3 else "reports/real/benchmark_results_real.json"

cfg = load_config(CFG_PATH)
panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])
close = to_wide(panel, cfg.data.price_field)
opn = to_wide(panel, "open")

train_end = pd.Timestamp(cfg.data.start_date) + pd.DateOffset(
    years=cfg.walkforward.train_years)
train = panel[panel["date"] < train_end]
rep = select_pairs(train, max_pairs=cfg.research.max_pairs_per_window,
                   correlation_threshold=cfg.research.correlation_threshold,
                   universe_kwargs=cfg.universe.model_dump(),
                   selection_mode="correlation_only")
print(f"benchmark pair set: {len(rep.pairs)} pairs from training "
      f"{train['date'].min().date()}..{train['date'].max().date()}")

test_start = close.index[0]
test_end = close.index[-1]
print(f"benchmark window: {test_start.date()}..{test_end.date()} "
      f"({len(close.loc[test_start:test_end])} sessions, {close.shape[1]} symbols)")

kwargs = dict(prices_close=close, prices_open=opn, pairs=rep.pairs,
             test_start=test_start, test_end=test_end, config=cfg)

# ---- equivalence -----------------------------------------------------------
a = run_backtest(**kwargs, loop="reference")
b = run_backtest(**kwargs, loop="fast")
cols = ["gross_pnl", "cost", "net_pnl", "entry_z", "qty_a", "qty_b", "beta"]
equal_trades = (len(a.trades) == len(b.trades) and
                np.array_equal(a.trades[cols].to_numpy(dtype=float),
                               b.trades[cols].to_numpy(dtype=float)))
equal_equity = np.array_equal(a.equity["equity"].to_numpy(), b.equity["equity"].to_numpy())
equal_reasons = list(a.trades["exit_reason"]) == list(b.trades["exit_reason"])
max_equity_diff = float(np.max(np.abs(a.equity["equity"].to_numpy()
                                      - b.equity["equity"].to_numpy())))
max_pnl_diff = (float(np.max(np.abs(a.trades["net_pnl"].to_numpy()
                                    - b.trades["net_pnl"].to_numpy())))
               if len(a.trades) == len(b.trades) and len(a.trades) else 0.0)
print(f"\nEQUIVALENCE  trades: ref={len(a.trades)} fast={len(b.trades)} "
      f"equal={equal_trades}  equity_equal={equal_equity}  "
      f"exit_reasons_equal={equal_reasons}")
print(f"  max |equity_ref - equity_fast| = {max_equity_diff:.10f}")
print(f"  max |net_pnl_ref - net_pnl_fast| = {max_pnl_diff:.10f}")

# ---- timing -----------------------------------------------------------------
def timed(loop, n):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        run_backtest(**kwargs, loop=loop)
        times.append(time.perf_counter() - t0)
    return times

WARMUP, MEASURED = 5, 10
for loop in ("reference", "fast"):
    timed(loop, WARMUP)   # discarded

ref_times = timed("reference", MEASURED)
fast_times = timed("fast", MEASURED)

def stats(times):
    a = np.array(times)
    return {"min_s": float(a.min()), "median_s": float(np.median(a)),
            "mean_s": float(a.mean()), "std_s": float(a.std(ddof=1)),
            "all_runs_s": [float(x) for x in times]}

ref_stats, fast_stats = stats(ref_times), stats(fast_times)
speedup = ref_stats["median_s"] / fast_stats["median_s"]

print(f"\nTIMING  ({WARMUP} warmup + {MEASURED} measured runs each, real data, "
      f"{close.shape[1]} symbols x {len(close)} sessions, {len(rep.pairs)} pairs)")
print(f"  reference: min={ref_stats['min_s']*1000:.1f}ms "
      f"median={ref_stats['median_s']*1000:.1f}ms mean={ref_stats['mean_s']*1000:.1f}ms "
      f"std={ref_stats['std_s']*1000:.1f}ms")
print(f"  fast:      min={fast_stats['min_s']*1000:.1f}ms "
      f"median={fast_stats['median_s']*1000:.1f}ms mean={fast_stats['mean_s']*1000:.1f}ms "
      f"std={fast_stats['std_s']*1000:.1f}ms")
print(f"  speedup (median reference / median fast): {speedup:.3f}x")

out = {
    "dataset": {"symbols": int(close.shape[1]), "sessions": int(len(close)),
               "pairs": int(len(rep.pairs)),
               "date_range": [str(test_start.date()), str(test_end.date())]},
    "equivalence": {"trades_ref": int(len(a.trades)), "trades_fast": int(len(b.trades)),
                    "trade_ledger_equal": bool(equal_trades),
                    "equity_curve_equal": bool(equal_equity),
                    "exit_reasons_equal": bool(equal_reasons),
                    "max_abs_equity_diff": max_equity_diff,
                    "max_abs_pnl_diff": max_pnl_diff},
    "timing": {"warmup_runs": WARMUP, "measured_runs": MEASURED,
              "reference": ref_stats, "fast": fast_stats,
              "speedup_median": speedup},
}
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\nwrote {OUT_PATH}")

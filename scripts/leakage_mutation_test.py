"""Direct future-data mutation test on real data.

For a cutoff date T, compute rolling hedge ratio / z-score / signal / full
backtest against the real panel, then violently corrupt every observation
strictly AFTER T and recompute. If any quantity dated <= T changes, the
computation used future information. Run on real prices, not synthetic.
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
from statarb.research.hedge_ratio import rolling_ols_hedge_ratio
from statarb.research.pair_selection import select_pairs
from statarb.signals.strategy import pair_signal_frame
from statarb.backtest.engine import run_backtest

cfg = load_config("config/real.yaml")
panel = pd.read_parquet("data/processed_real/panel.parquet")
panel["date"] = pd.to_datetime(panel["date"])
close = to_wide(panel, cfg.data.price_field)
opn = to_wide(panel, "open")
log_close = np.log(close)

CUTOFF = pd.Timestamp("2022-06-30")
results = {}

train_end = pd.Timestamp(cfg.data.start_date) + pd.DateOffset(
    years=cfg.walkforward.train_years)
train = panel[panel["date"] < train_end]
rep = select_pairs(train, max_pairs=5, correlation_threshold=cfg.research.correlation_threshold,
                   universe_kwargs=cfg.universe.model_dump(), selection_mode="correlation_only")
a, b = rep.pairs.iloc[0]["symbol_a"], rep.pairs.iloc[0]["symbol_b"]
beta0, alpha0 = float(rep.pairs.iloc[0]["beta"]), float(rep.pairs.iloc[0]["alpha"])
print(f"mutation test pair: {a}/{b}  cutoff={CUTOFF.date()}")

# corrupt: multiply every post-cutoff close by a violent, time-varying factor
mut_close = close.copy()
future_mask = mut_close.index > CUTOFF
n_future = int(future_mask.sum())
rng = np.random.default_rng(999)
mut_close.loc[future_mask] = mut_close.loc[future_mask] * (
    1.0 + rng.uniform(0.5, 3.0, size=(n_future, mut_close.shape[1])))
mut_log_close = np.log(mut_close)
print(f"corrupted {n_future} post-cutoff sessions across all {mut_close.shape[1]} symbols "
      f"(factors drawn uniform[1.5x, 4.0x])")

pre_idx = log_close.index <= CUTOFF

# ---- 1. rolling hedge ratio -------------------------------------------------
rb_orig = rolling_ols_hedge_ratio(log_close[a], log_close[b],
                                  window=cfg.research.rolling_beta_window)
rb_mut = rolling_ols_hedge_ratio(mut_log_close[a], mut_log_close[b],
                                 window=cfg.research.rolling_beta_window)
beta_pre_orig = rb_orig.loc[pre_idx, "beta"].to_numpy()
beta_pre_mut = rb_mut.loc[pre_idx, "beta"].to_numpy()
ok = np.array_equal(np.nan_to_num(beta_pre_orig, nan=-999),
                    np.nan_to_num(beta_pre_mut, nan=-999))
results["rolling_hedge_ratio"] = {"PASS": bool(ok),
                                  "n_compared": int(pre_idx.sum()),
                                  "max_abs_diff": float(np.nanmax(np.abs(
                                      beta_pre_orig - beta_pre_mut)))}
print(f"[{'PASS' if ok else 'FAIL'}] rolling hedge ratio: pre-cutoff betas "
      f"{'identical' if ok else 'CHANGED'} ({pre_idx.sum()} obs compared)")

# ---- 2. z-score / signal / target position ----------------------------------
sf_orig = pair_signal_frame(log_close[a], log_close[b], beta=beta0, alpha=alpha0,
                            zscore_lookback=cfg.strategy.zscore_lookback,
                            entry_z=cfg.strategy.entry_z, exit_z=cfg.strategy.exit_z,
                            stop_z=cfg.strategy.stop_z,
                            max_holding_days=cfg.strategy.max_holding_days)
sf_mut = pair_signal_frame(mut_log_close[a], mut_log_close[b], beta=beta0, alpha=alpha0,
                           zscore_lookback=cfg.strategy.zscore_lookback,
                           entry_z=cfg.strategy.entry_z, exit_z=cfg.strategy.exit_z,
                           stop_z=cfg.strategy.stop_z,
                           max_holding_days=cfg.strategy.max_holding_days)
for col in ("zscore", "target"):
    o = sf_orig.loc[pre_idx, col].to_numpy(dtype=float)
    m = sf_mut.loc[pre_idx, col].to_numpy(dtype=float)
    ok = np.array_equal(np.nan_to_num(o, nan=-999), np.nan_to_num(m, nan=-999))
    results[col] = {"PASS": bool(ok), "n_compared": int(pre_idx.sum())}
    print(f"[{'PASS' if ok else 'FAIL'}] {col}: pre-cutoff values "
          f"{'identical' if ok else 'CHANGED'} ({pre_idx.sum()} obs compared)")

# ---- 3. full backtest: trades and positions up to cutoff --------------------
kwargs = dict(prices_close=close, prices_open=opn, pairs=rep.pairs,
             test_start=close.index[0], test_end=close.index[-1], config=cfg)
r_orig = run_backtest(**kwargs)
kwargs_mut = dict(kwargs, prices_close=mut_close, prices_open=opn)
r_mut = run_backtest(**kwargs_mut)

eq_orig_pre = r_orig.equity.loc[r_orig.equity.index <= CUTOFF, "equity"].to_numpy()
eq_mut_pre = r_mut.equity.loc[r_mut.equity.index <= CUTOFF, "equity"].to_numpy()
eq_ok = (len(eq_orig_pre) == len(eq_mut_pre)
        and np.array_equal(eq_orig_pre, eq_mut_pre))
results["backtest_equity_pre_cutoff"] = {"PASS": bool(eq_ok),
                                         "n_compared": int(len(eq_orig_pre))}
print(f"[{'PASS' if eq_ok else 'FAIL'}] full backtest equity curve: pre-cutoff "
      f"equity {'identical' if eq_ok else 'CHANGED'} ({len(eq_orig_pre)} obs compared)")

trades_orig_pre = r_orig.trades[pd.to_datetime(r_orig.trades["entry_date"]) <= CUTOFF]
trades_mut_pre = r_mut.trades[pd.to_datetime(r_mut.trades["entry_date"]) <= CUTOFF]
cols = ["entry_date", "symbol_a", "symbol_b", "qty_a", "qty_b", "entry_price_a",
       "entry_price_b"]
trades_ok = (len(trades_orig_pre) == len(trades_mut_pre) and
            trades_orig_pre[cols].reset_index(drop=True).equals(
                trades_mut_pre[cols].reset_index(drop=True)))
results["backtest_trades_pre_cutoff"] = {"PASS": bool(trades_ok),
                                         "n_compared": int(len(trades_orig_pre))}
print(f"[{'PASS' if trades_ok else 'FAIL'}] full backtest trades: pre-cutoff trade "
      f"records {'identical' if trades_ok else 'CHANGED'} "
      f"({len(trades_orig_pre)} trades compared)")

all_pass = all(v["PASS"] for v in results.values())
print(f"\nOVERALL: {'ALL PASS' if all_pass else 'AT LEAST ONE FAILURE'}")

Path("reports/real").mkdir(parents=True, exist_ok=True)
with open("reports/real/leakage_mutation_test.json", "w") as fh:
    json.dump({"cutoff": str(CUTOFF.date()), "pair": f"{a}/{b}",
              "all_pass": all_pass, "checks": results}, fh, indent=2)
print("wrote reports/real/leakage_mutation_test.json")

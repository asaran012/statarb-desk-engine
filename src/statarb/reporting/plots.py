"""Figures.

Chart honesty rules applied throughout:
  * equity axes start at a value that does not exaggerate the move
  * drawdowns are shown on their own panel rather than implied
  * test-window boundaries are shaded so the reader can see what was unseen
  * no smoothed or cherry-picked sub-periods
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
})

INK = "#1f2a44"
ACC = "#c2492e"
POS = "#2f7d5d"


def _save(fig, outdir: Path, name: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def equity_curve(equity: pd.Series, outdir: Path,
                 window_bounds: Optional[List[tuple]] = None,
                 name: str = "01_equity_curve.png") -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(equity.index, equity.to_numpy(), color=INK, lw=1.4)
    if window_bounds:
        for i, (s, e) in enumerate(window_bounds):
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                       color=ACC if i % 2 == 0 else INK, alpha=0.05)
            ax.axvline(pd.Timestamp(s), color="grey", lw=0.6, ls=":")
    ax.set_title("Walk-forward equity curve (out-of-sample test windows only)")
    ax.set_ylabel("Equity")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x/1e6:.2f}M")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return _save(fig, outdir, name)


def drawdown_curve(equity: pd.Series, outdir: Path,
                   name: str = "02_drawdown.png") -> Path:
    dd = equity / equity.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.fill_between(dd.index, dd.to_numpy() * 100, 0, color=ACC, alpha=0.35)
    ax.plot(dd.index, dd.to_numpy() * 100, color=ACC, lw=1.0)
    ax.set_title("Drawdown")
    ax.set_ylabel("%")
    return _save(fig, outdir, name)


def rolling_sharpe(equity: pd.Series, outdir: Path, window: int = 126,
                   trading_days: int = 252,
                   name: str = "03_rolling_sharpe.png") -> Path:
    r = equity.pct_change().dropna()
    rs = r.rolling(window).mean() / r.rolling(window).std(ddof=1) * np.sqrt(trading_days)
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.plot(rs.index, rs.to_numpy(), color=INK, lw=1.1)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(f"Rolling {window}-day annualised Sharpe")
    return _save(fig, outdir, name)


def annual_returns(equity: pd.Series, outdir: Path,
                   name: str = "04_annual_returns.png") -> Path:
    yearly = equity.resample("YE").last().pct_change()
    first_year = equity.index[0].year
    first_val = equity.iloc[0]
    y0 = equity[equity.index.year == first_year]
    if len(y0):
        yearly.iloc[0] = y0.iloc[-1] / first_val - 1.0
    yearly = yearly.dropna()
    fig, ax = plt.subplots(figsize=(7, 3.0))
    colors = [POS if v >= 0 else ACC for v in yearly]
    ax.bar([str(d.year) for d in yearly.index], yearly.to_numpy() * 100, color=colors)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("Annual return (test windows)")
    ax.set_ylabel("%")
    return _save(fig, outdir, name)


def spread_with_bands(spread: pd.Series, mean: pd.Series, std: pd.Series,
                      entry_z: float, outdir: Path, pair: str,
                      name: str = "05_spread_bands.png") -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(spread.index, spread.to_numpy(), color=INK, lw=1.0, label="log spread")
    ax.plot(mean.index, mean.to_numpy(), color="grey", lw=0.9, ls="--", label="rolling mean")
    ax.fill_between(spread.index, (mean - entry_z * std).to_numpy(),
                    (mean + entry_z * std).to_numpy(), color=ACC, alpha=0.10,
                    label=f"+/-{entry_z:g} sigma")
    ax.set_title(f"Spread and trading bands: {pair}")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    return _save(fig, outdir, name)


def zscore_with_trades(z: pd.Series, trades: pd.DataFrame, pair: str, outdir: Path,
                       entry_z: float, exit_z: float, stop_z: float,
                       name: str = "06_zscore_trades.png") -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(z.index, z.to_numpy(), color=INK, lw=1.0)
    for lvl, c, ls in ((entry_z, ACC, "--"), (-entry_z, ACC, "--"),
                       (exit_z, "grey", ":"), (-exit_z, "grey", ":"),
                       (stop_z, "black", "-."), (-stop_z, "black", "-.")):
        ax.axhline(lvl, color=c, lw=0.8, ls=ls)
    sub = trades[trades["pair"] == pair] if len(trades) else trades
    for _, t in sub.iterrows():
        ax.scatter(pd.Timestamp(t["entry_date"]), t["entry_z"], marker="^",
                   color=POS, s=26, zorder=3)
        if np.isfinite(t["exit_z"]):
            ax.scatter(pd.Timestamp(t["exit_date"]), t["exit_z"], marker="v",
                       color=ACC, s=26, zorder=3)
    ax.set_title(f"Z-score with entries (^) and exits (v): {pair}")
    return _save(fig, outdir, name)


def trade_pnl_distribution(trades: pd.DataFrame, outdir: Path,
                           name: str = "07_trade_pnl_hist.png") -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.0))
    v = trades["net_pnl"].to_numpy()
    ax.hist(v, bins=45, color=INK, alpha=0.8)
    ax.axvline(0, color=ACC, lw=1.0)
    ax.set_title(f"Net P&L per trade (n={len(v)})")
    ax.set_xlabel("net P&L")
    return _save(fig, outdir, name)


def holding_period_distribution(trades: pd.DataFrame, outdir: Path,
                                name: str = "08_holding_periods.png") -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.hist(trades["holding_period"].to_numpy(), bins=30, color=INK, alpha=0.8)
    ax.set_title("Holding period (business days)")
    return _save(fig, outdir, name)


def cost_sensitivity(df: pd.DataFrame, outdir: Path,
                     name: str = "09_cost_sensitivity.png") -> Path:
    fig, ax1 = plt.subplots(figsize=(7, 3.2))
    ax1.plot(df["total_bps"], df["sharpe"], "o-", color=INK, label="Sharpe")
    ax1.set_xlabel("round-trip cost (bps of notional, each side)")
    ax1.set_ylabel("Sharpe", color=INK)
    ax1.axhline(0, color="grey", lw=0.8)
    ax2 = ax1.twinx()
    ax2.plot(df["total_bps"], df["cumulative_return"] * 100, "s--", color=ACC,
             label="cumulative return")
    ax2.set_ylabel("cumulative return %", color=ACC)
    ax2.grid(False)
    ax1.set_title("Transaction-cost sensitivity")
    return _save(fig, outdir, name)


def pair_quality(pairs: pd.DataFrame, outdir: Path,
                 name: str = "10_pair_quality.png") -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    d = pairs.head(20)
    lbl = d["symbol_a"] + "/" + d["symbol_b"]
    ax.barh(range(len(d)), d["quality_score"].to_numpy(), color=INK, alpha=0.85)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(lbl, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("quality score (stability, beta CV, half-life, evidence)")
    ax.set_title("Selected pair quality")
    return _save(fig, outdir, name)


def parameter_heatmap(grid: pd.DataFrame, outdir: Path,
                      name: str = "11_param_heatmap.png") -> Path:
    """Validation Sharpe over (lookback, entry_z), averaged across exit_z and windows."""
    p = grid.pivot_table(index="zscore_lookback", columns="entry_z", values="sharpe",
                         aggfunc="mean")
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    im = ax.imshow(p.to_numpy(), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(p.columns)))
    ax.set_xticklabels([f"{c:g}" for c in p.columns])
    ax.set_yticks(range(len(p.index)))
    ax.set_yticklabels(p.index)
    ax.set_xlabel("entry_z")
    ax.set_ylabel("z-score lookback")
    ax.set_title("Mean VALIDATION Sharpe (not test)")
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.to_numpy()[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.grid(False)
    return _save(fig, outdir, name)


def exposure_over_time(equity: pd.DataFrame, outdir: Path,
                       name: str = "12_exposure.png") -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.0))
    eq = equity["equity"].replace(0, np.nan)
    ax.plot(equity.index, (equity["gross_exposure"] / eq * 100).to_numpy(),
            color=INK, lw=1.0, label="gross")
    ax.plot(equity.index, (equity["net_exposure"] / eq * 100).to_numpy(),
            color=ACC, lw=1.0, label="net")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("% of equity")
    ax.set_title("Gross and net exposure")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, outdir, name)


def ablation_bars(df: pd.DataFrame, outdir: Path,
                  name: str = "13_ablations.png") -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    d = df.copy()
    lbl = d["experiment"].to_numpy()
    axes[0].barh(range(len(d)), d["sharpe"].to_numpy(), color=INK, alpha=0.85)
    axes[0].set_yticks(range(len(d)))
    axes[0].set_yticklabels(lbl, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_title("Test Sharpe by configuration")
    axes[0].axvline(0, color="grey", lw=0.8)
    axes[1].barh(range(len(d)), (d["cumulative_return"] * 100).to_numpy(),
                 color=ACC, alpha=0.85)
    axes[1].set_yticks(range(len(d)))
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_title("Cumulative return %")
    axes[1].axvline(0, color="grey", lw=0.8)
    return _save(fig, outdir, name)


def funnel_chart(funnel: Dict[str, int], outdir: Path,
                 name: str = "14_research_funnel.png") -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    keys = list(funnel.keys())
    vals = [funnel[k] for k in keys]
    ax.barh(range(len(keys)), vals, color=INK, alpha=0.85)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xscale("symlog")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v}", va="center", fontsize=7.5)
    ax.set_title("Research funnel (window 0, training data only)")
    return _save(fig, outdir, name)

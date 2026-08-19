"""Post-trade analytics: attribution, regimes, cost decomposition."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def pnl_by_pair(trades: pd.DataFrame, top_n: Optional[int] = None) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    g = trades.groupby("pair").agg(
        trades=("net_pnl", "size"), gross_pnl=("gross_pnl", "sum"),
        cost=("cost", "sum"), net_pnl=("net_pnl", "sum"),
        hit_rate=("net_pnl", lambda s: float((s > 0).mean())),
        avg_holding=("holding_period", "mean")).sort_values("net_pnl", ascending=False)
    g["cost_pct_of_gross"] = np.where(g["gross_pnl"].abs() > 0,
                                      g["cost"] / g["gross_pnl"].abs() * 100, np.nan)
    return g.head(top_n) if top_n else g


def pnl_by_year(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["year"] = pd.to_datetime(t["exit_date"]).dt.year
    return t.groupby("year").agg(
        trades=("net_pnl", "size"), gross_pnl=("gross_pnl", "sum"),
        cost=("cost", "sum"), net_pnl=("net_pnl", "sum"),
        hit_rate=("net_pnl", lambda s: float((s > 0).mean())))


def exit_reason_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    g = trades.groupby("exit_reason").agg(
        count=("net_pnl", "size"), net_pnl=("net_pnl", "sum"),
        avg_pnl=("net_pnl", "mean"), avg_holding=("holding_period", "mean"))
    g["pct_of_trades"] = g["count"] / len(trades) * 100
    return g.sort_values("count", ascending=False)


def holding_period_distribution(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    bins = [0, 3, 5, 10, 20, 40, 10_000]
    labels = ["1-3d", "4-5d", "6-10d", "11-20d", "21-40d", "40d+"]
    t = trades.copy()
    t["bucket"] = pd.cut(t["holding_period"], bins=bins, labels=labels,
                         include_lowest=True)
    return t.groupby("bucket", observed=False).agg(
        count=("net_pnl", "size"), net_pnl=("net_pnl", "sum"),
        avg_pnl=("net_pnl", "mean"))


def performance_by_volatility_regime(equity: pd.Series, trades: pd.DataFrame,
                                     window: int = 60) -> pd.DataFrame:
    """Split trade P&L by the portfolio's own trailing volatility tercile.

    Uses *trailing* realised vol known at the trade's entry date, so the regime
    label is available at decision time rather than assigned with hindsight.
    """
    if trades is None or trades.empty or len(equity) < window + 5:
        return pd.DataFrame()
    rets = equity.pct_change()
    vol = rets.rolling(window, min_periods=window // 2).std() * np.sqrt(252)
    q = vol.quantile([1 / 3, 2 / 3]).to_numpy()
    t = trades.copy()
    t["entry_vol"] = vol.reindex(pd.to_datetime(t["entry_date"])).to_numpy()
    t["regime"] = np.where(t["entry_vol"] <= q[0], "low",
                           np.where(t["entry_vol"] <= q[1], "mid", "high"))
    return t.groupby("regime").agg(
        trades=("net_pnl", "size"), net_pnl=("net_pnl", "sum"),
        avg_pnl=("net_pnl", "mean"),
        hit_rate=("net_pnl", lambda s: float((s > 0).mean())))


def cost_decomposition(trades: pd.DataFrame) -> Dict[str, float]:
    if trades is None or trades.empty:
        return {}
    gross, cost = float(trades["gross_pnl"].sum()), float(trades["cost"].sum())
    return {"gross_pnl": gross, "total_costs": cost, "net_pnl": gross - cost,
            "cost_as_pct_of_gross": (cost / abs(gross) * 100) if gross else np.nan,
            "avg_cost_per_trade": cost / len(trades),
            "trades_gross_positive": int((trades["gross_pnl"] > 0).sum()),
            "trades_net_positive": int((trades["net_pnl"] > 0).sum()),
            "trades_flipped_by_costs": int(((trades["gross_pnl"] > 0) &
                                            (trades["net_pnl"] <= 0)).sum())}


def drawdown_attribution(equity: pd.Series, trades: pd.DataFrame,
                         start, trough) -> pd.DataFrame:
    """Which pairs lost money inside a given drawdown window."""
    if trades is None or trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    ex = pd.to_datetime(t["exit_date"])
    mask = (ex >= pd.Timestamp(start)) & (ex <= pd.Timestamp(trough))
    sub = t[mask]
    if sub.empty:
        return pd.DataFrame()
    return (sub.groupby("pair")
            .agg(trades=("net_pnl", "size"), net_pnl=("net_pnl", "sum"))
            .sort_values("net_pnl"))

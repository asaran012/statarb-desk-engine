"""Performance and risk metrics.

Stated assumptions, because every one of these is a place to get it wrong:

* 252 trading days per year (configurable). Annualising a daily Sharpe by
  ``sqrt(252)`` assumes i.i.d. daily returns; pair strategies have
  autocorrelated P&L (a position is held for a week or more), so the annualised
  figure is optimistic. The magnitude of that bias is not estimated here.
* CAGR is geometric and computed from the equity curve's endpoints over its
  actual elapsed calendar span, not from the mean of returns.
* Sharpe uses arithmetic mean excess return over the sample standard deviation
  of daily returns (ddof=1), then scales by ``sqrt(252)``.
* Sortino's denominator is downside deviation about zero, using the count of
  *all* observations rather than only the negative ones -- the convention
  differs across texts and changes the number materially.
* Max drawdown is peak-to-trough on the equity curve, not on returns.
* Turnover is annualised traded notional over average equity, counting both
  legs and both sides of a round trip.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def equity_to_returns(equity: pd.Series) -> pd.Series:
    return equity.astype("float64").pct_change().dropna()


def cumulative_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series, trading_days: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    ratio = equity.iloc[-1] / equity.iloc[0]
    if ratio <= 0:
        return float("nan")
    return float(ratio ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series, trading_days: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(trading_days))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0,
                 trading_days: int = 252) -> float:
    if len(returns) < 2:
        return float("nan")
    excess = returns - risk_free / trading_days
    sd = excess.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(trading_days))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0,
                  trading_days: int = 252) -> float:
    if len(returns) < 2:
        return float("nan")
    excess = returns - risk_free / trading_days
    downside = np.minimum(excess, 0.0)
    dd = np.sqrt(np.sum(downside ** 2) / len(excess))
    if dd == 0 or not np.isfinite(dd):
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(trading_days))


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    return float(drawdown_series(equity).min())


def calmar_ratio(equity: pd.Series, trading_days: int = 252) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd < 1e-12:
        return float("nan")
    return float(cagr(equity, trading_days) / mdd)


def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    if len(returns) < 20:
        return float("nan")
    return float(np.percentile(returns, (1 - level) * 100))


def expected_shortfall(returns: pd.Series, level: float = 0.95) -> float:
    if len(returns) < 20:
        return float("nan")
    var = value_at_risk(returns, level)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) else float("nan")


def drawdown_episodes(equity: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """Peak / trough / recovery decomposition of the largest drawdowns."""
    if len(equity) < 3:
        return pd.DataFrame(columns=["start", "trough", "recovery", "magnitude",
                                     "duration_days", "recovery_days"])
    dd = drawdown_series(equity)
    episodes, in_dd, start = [], False, None
    for date, val in dd.items():
        if not in_dd and val < -1e-12:
            in_dd, start = True, date
        elif in_dd and val >= -1e-12:
            seg = dd.loc[start:date]
            episodes.append({"start": start, "trough": seg.idxmin(),
                             "recovery": date, "magnitude": float(seg.min())})
            in_dd = False
    if in_dd:
        seg = dd.loc[start:]
        episodes.append({"start": start, "trough": seg.idxmin(), "recovery": pd.NaT,
                         "magnitude": float(seg.min())})
    if not episodes:
        return pd.DataFrame(columns=["start", "trough", "recovery", "magnitude",
                                     "duration_days", "recovery_days"])
    out = pd.DataFrame(episodes)
    out["duration_days"] = [
        (r.recovery - r.start).days if pd.notna(r.recovery)
        else (equity.index[-1] - r.start).days for r in out.itertuples()]
    out["recovery_days"] = [
        (r.recovery - r.trough).days if pd.notna(r.recovery) else np.nan
        for r in out.itertuples()]
    return out.sort_values("magnitude").head(top_n).reset_index(drop=True)


def trade_statistics(trades: pd.DataFrame) -> Dict[str, float]:
    if trades is None or trades.empty:
        return {"n_trades": 0, "hit_rate": float("nan"), "profit_factor": float("nan"),
                "avg_win": float("nan"), "avg_loss": float("nan"),
                "avg_holding_days": float("nan"), "gross_pnl": 0.0,
                "total_costs": 0.0, "net_pnl": 0.0, "avg_trade_pnl": float("nan"),
                "best_trade": float("nan"), "worst_trade": float("nan")}
    net = trades["net_pnl"]
    wins, losses = net[net > 0], net[net < 0]
    gross_win, gross_loss = float(wins.sum()), float(abs(losses.sum()))
    return {
        "n_trades": int(len(trades)),
        "hit_rate": float(len(wins) / len(trades)),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "avg_holding_days": float(trades["holding_period"].mean()),
        "median_holding_days": float(trades["holding_period"].median()),
        "gross_pnl": float(trades["gross_pnl"].sum()),
        "total_costs": float(trades["cost"].sum()),
        "net_pnl": float(net.sum()),
        "avg_trade_pnl": float(net.mean()),
        "best_trade": float(net.max()),
        "worst_trade": float(net.min()),
    }


def turnover(traded_notional: float, equity: pd.Series,
             trading_days: int = 252) -> float:
    """Annualised: total traded notional / average equity / years elapsed."""
    if len(equity) < 2:
        return float("nan")
    avg_eq = float(equity.mean())
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    if avg_eq <= 0:
        return float("nan")
    return float(traded_notional / avg_eq / years)


def performance_summary(equity: pd.Series, trades: Optional[pd.DataFrame] = None,
                        traded_notional: float = 0.0, risk_free: float = 0.0,
                        trading_days: int = 252) -> Dict[str, float]:
    rets = equity_to_returns(equity)
    out = {
        "start": str(equity.index[0].date()) if len(equity) else None,
        "end": str(equity.index[-1].date()) if len(equity) else None,
        "n_days": int(len(equity)),
        "starting_equity": float(equity.iloc[0]) if len(equity) else np.nan,
        "ending_equity": float(equity.iloc[-1]) if len(equity) else np.nan,
        "cumulative_return": cumulative_return(equity),
        "cagr": cagr(equity, trading_days),
        "annualized_volatility": annualized_volatility(rets, trading_days),
        "sharpe": sharpe_ratio(rets, risk_free, trading_days),
        "sortino": sortino_ratio(rets, risk_free, trading_days),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(equity, trading_days),
        "skewness": float(rets.skew()) if len(rets) > 3 else np.nan,
        "kurtosis": float(rets.kurtosis()) if len(rets) > 3 else np.nan,
        "var_95_daily": value_at_risk(rets, 0.95),
        "cvar_95_daily": expected_shortfall(rets, 0.95),
        "turnover_annual": turnover(traded_notional, equity, trading_days),
    }
    out.update(trade_statistics(trades))
    return out

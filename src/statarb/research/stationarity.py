"""Stationarity diagnostics for the spread.

ADF null hypothesis: the series contains a unit root (non-stationary).  A small
p-value is *evidence against* a unit root.  It is not proof of mean reversion,
and with hundreds of pairs tested it is not even strong evidence — see
:mod:`statarb.research.multiple_testing`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


@dataclass
class ADFResult:
    statistic: float
    pvalue: float
    used_lag: int
    n_obs: int
    critical_values: Dict[str, float]
    regression: str

    @property
    def stationary_at_5pct(self) -> bool:
        return self.pvalue < 0.05

    def to_dict(self) -> dict:
        return {"adf_stat": self.statistic, "adf_pvalue": self.pvalue,
                "adf_lag": self.used_lag, "adf_nobs": self.n_obs,
                "adf_crit_5pct": self.critical_values.get("5%", np.nan)}


def adf_test(series: pd.Series | np.ndarray, regression: str = "c",
             autolag: Optional[str] = "AIC", maxlag: Optional[int] = None) -> ADFResult:
    """Augmented Dickey-Fuller test.

    ``regression='c'``: constant, no trend.  We do not include a deterministic
    trend in the spread regression: a mean-reverting spread with a time trend is
    not something a market-neutral book can hold, so allowing one would flatter
    the test without being tradeable.
    """
    x = np.asarray(pd.Series(series).dropna(), dtype="float64")
    if x.size < 20:
        raise ValueError(f"ADF needs >= 20 observations, got {x.size}")
    if np.allclose(x, x[0]):
        return ADFResult(np.nan, 1.0, 0, x.size, {}, regression)
    stat, pval, usedlag, nobs, crit, _ = adfuller(
        x, regression=regression, autolag=autolag, maxlag=maxlag)
    return ADFResult(float(stat), float(pval), int(usedlag), int(nobs),
                     {k: float(v) for k, v in crit.items()}, regression)


def hurst_exponent(series: pd.Series | np.ndarray,
                   min_lag: int = 2, max_lag: int = 100) -> float:
    """Rescaled-variance estimate of the Hurst exponent.

    Regress ``log Var(x_{t+k} - x_t)`` on ``log k``; the slope is ``2H``.
    H < 0.5 suggests mean reversion, H ~ 0.5 a random walk, H > 0.5 trending.

    Reported as a *descriptive* statistic only.  The estimator is biased in
    small samples, sensitive to the lag range, and has no null distribution
    attached here, so it never gates pair selection.
    """
    x = np.asarray(pd.Series(series).dropna(), dtype="float64")
    max_lag = int(min(max_lag, x.size // 2))
    if max_lag <= min_lag + 2:
        return float("nan")
    lags = np.arange(min_lag, max_lag)
    var = np.array([np.var(x[k:] - x[:-k], ddof=1) for k in lags])
    ok = var > 0
    if ok.sum() < 3:
        return float("nan")
    slope = np.polyfit(np.log(lags[ok]), np.log(var[ok]), 1)[0]
    return float(slope / 2.0)


def variance_ratio(series: pd.Series | np.ndarray, k: int = 5) -> float:
    """Lo-MacKinlay variance ratio. VR < 1 is consistent with mean reversion."""
    x = np.asarray(pd.Series(series).dropna(), dtype="float64")
    d = np.diff(x)
    if d.size < k * 3:
        return float("nan")
    v1 = np.var(d, ddof=1)
    vk = np.var(x[k:] - x[:-k], ddof=1) / k
    return float(vk / v1) if v1 > 0 else float("nan")

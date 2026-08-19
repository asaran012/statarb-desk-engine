"""Engle-Granger cointegration testing.

Two I(1) series are cointegrated if some linear combination of them is I(0).
Correlation is a statement about *co-movement of returns*; cointegration is a
statement about *the levels not drifting apart*.  Only the second one justifies
holding a spread and expecting it to come back.

Engle-Granger, two steps:
  1. OLS ``log A ~ alpha + beta * log B``
  2. unit-root test on the residual

Critical values are **not** standard ADF values, because the residual comes from
an estimated cointegrating vector.  We use ``statsmodels.tsa.stattools.coint``,
which applies the correct MacKinnon distribution, rather than hand-rolling step
2 with ``adfuller`` on the residual (a common and materially wrong shortcut that
overstates significance).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from .hedge_ratio import static_ols_hedge_ratio
from .stationarity import adf_test


@dataclass
class CointegrationResult:
    symbol_a: str
    symbol_b: str
    stat: float
    pvalue: float
    critical_values: Dict[str, float]
    beta: float
    alpha: float
    adf_pvalue: float
    adf_stat: float
    n_obs: int

    @property
    def passes(self) -> bool:
        return np.isfinite(self.pvalue) and self.pvalue < 0.05

    def to_dict(self) -> dict:
        return {
            "symbol_a": self.symbol_a, "symbol_b": self.symbol_b,
            "coint_stat": self.stat, "coint_pvalue": self.pvalue,
            "coint_crit_5pct": self.critical_values.get("5%", np.nan),
            "beta": self.beta, "alpha": self.alpha,
            "adf_pvalue": self.adf_pvalue, "adf_stat": self.adf_stat,
            "n_obs": self.n_obs,
        }


def engle_granger(log_a: pd.Series, log_b: pd.Series,
                  symbol_a: str = "A", symbol_b: str = "B",
                  trend: str = "c", maxlag: Optional[int] = None
                  ) -> CointegrationResult:
    """Engle-Granger test plus the fitted cointegrating vector.

    ``trend='c'``: constant in the cointegrating regression, no time trend.
    """
    df = pd.concat([log_a.rename("a"), log_b.rename("b")], axis=1).dropna()
    if len(df) < 60:
        raise ValueError(f"cointegration test needs >= 60 obs, got {len(df)}")

    stat, pval, crit = coint(df["a"].to_numpy(), df["b"].to_numpy(),
                             trend=trend, maxlag=maxlag, autolag="AIC")
    hr = static_ols_hedge_ratio(df["a"], df["b"])
    resid = df["a"] - hr.alpha - hr.beta * df["b"]
    try:
        adf = adf_test(resid, regression="c")
        adf_p, adf_s = adf.pvalue, adf.statistic
    except ValueError:
        adf_p, adf_s = float("nan"), float("nan")

    return CointegrationResult(
        symbol_a=symbol_a, symbol_b=symbol_b, stat=float(stat), pvalue=float(pval),
        critical_values={"1%": float(crit[0]), "5%": float(crit[1]),
                         "10%": float(crit[2])},
        beta=hr.beta, alpha=hr.alpha, adf_pvalue=float(adf_p),
        adf_stat=float(adf_s), n_obs=len(df))


def test_pairs(log_px: pd.DataFrame, pairs, progress: bool = False):
    """Run Engle-Granger over a list of (a, b) tuples. Returns a tidy frame."""
    rows = []
    for i, (a, b) in enumerate(pairs):
        if a not in log_px.columns or b not in log_px.columns:
            continue
        try:
            res = engle_granger(log_px[a], log_px[b], a, b)
        except (ValueError, np.linalg.LinAlgError):
            continue
        rows.append(res.to_dict())
        if progress and (i + 1) % 100 == 0:
            print(f"  cointegration: {i + 1}/{len(pairs)}", flush=True)
    return pd.DataFrame(rows)

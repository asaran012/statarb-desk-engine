"""Hedge-ratio estimation.

The hedge ratio beta answers: for one dollar of exposure to A, how many dollars
of B offset the shared factor?  Because we work in log prices, beta is an
elasticity, and the tradeable consequence is that a dollar-neutral book is only
hedge-neutral when beta == 1.

Two estimators are supported.  Both are *causal by construction*: the rolling
version's value at index t is computed from observations <= t only, and the
static version is fitted on the training slice and then frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class HedgeRatio:
    alpha: float
    beta: float
    resid_std: float
    r_squared: float
    n_obs: int
    method: str = "static_ols"

    def spread(self, log_a: pd.Series, log_b: pd.Series,
               include_alpha: bool = True) -> pd.Series:
        s = log_a - self.beta * log_b
        return s - self.alpha if include_alpha else s

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta,
                "resid_std": self.resid_std, "r_squared": self.r_squared,
                "hedge_n_obs": self.n_obs, "hedge_method": self.method}


def static_ols_hedge_ratio(log_a: pd.Series, log_b: pd.Series) -> HedgeRatio:
    """OLS of log A on log B with an intercept.

    Note the asymmetry: regressing A on B and B on A give betas that are not
    reciprocals unless R^2 == 1.  We fix the ordering upstream (in
    :mod:`statarb.research.pair_selection`) by always regressing the
    lexicographically first symbol on the second, so a pair's spread definition
    never depends on iteration order.
    """
    df = pd.concat([log_a, log_b], axis=1).dropna()
    if len(df) < 30:
        raise ValueError(f"hedge ratio needs >= 30 observations, got {len(df)}")
    y = df.iloc[:, 0].to_numpy()
    X = sm.add_constant(df.iloc[:, 1].to_numpy())
    fit = sm.OLS(y, X).fit()
    return HedgeRatio(alpha=float(fit.params[0]), beta=float(fit.params[1]),
                      resid_std=float(np.std(fit.resid, ddof=2)),
                      r_squared=float(fit.rsquared), n_obs=len(df),
                      method="static_ols")


def rolling_ols_hedge_ratio(log_a: pd.Series, log_b: pd.Series, window: int = 120,
                            min_periods: Optional[int] = None) -> pd.DataFrame:
    """Causal rolling OLS beta and alpha.

    Closed form on rolling moments, so it is O(n) rather than O(n * window):

        beta_t  = Cov_t(a, b) / Var_t(b)
        alpha_t = mean_t(a) - beta_t * mean_t(b)

    Every moment at index t uses the trailing ``window`` observations *ending at
    t inclusive*.  No shift is applied here; the caller decides the execution
    lag.  ``tests/test_signals.py::test_rolling_beta_is_causal`` pins this by
    mutating future values and asserting earlier betas are unchanged.
    """
    df = pd.concat([log_a.rename("a"), log_b.rename("b")], axis=1).dropna()
    mp = min_periods or window
    r = df.rolling(window=window, min_periods=mp)
    mean_a, mean_b = r["a"].mean(), r["b"].mean()
    var_b = r["b"].var(ddof=1)
    cov_ab = r.cov().unstack()["a"]["b"] if False else (
        df["a"].rolling(window, min_periods=mp).cov(df["b"]))
    beta = cov_ab / var_b.replace(0.0, np.nan)
    alpha = mean_a - beta * mean_b
    out = pd.DataFrame({"alpha": alpha, "beta": beta})
    out["spread"] = df["a"] - out["beta"] * df["b"] - out["alpha"]
    return out


def beta_stability(betas: pd.Series) -> dict:
    """Dispersion of a beta path. Lower ``cv`` means a more trustworthy hedge."""
    b = betas.dropna()
    if len(b) < 5:
        return {"beta_mean": np.nan, "beta_std": np.nan, "beta_cv": np.nan,
                "beta_min": np.nan, "beta_max": np.nan}
    mean = float(b.mean())
    std = float(b.std(ddof=1))
    return {"beta_mean": mean, "beta_std": std,
            "beta_cv": float(std / abs(mean)) if abs(mean) > 1e-9 else np.inf,
            "beta_min": float(b.min()), "beta_max": float(b.max())}

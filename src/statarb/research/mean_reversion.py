"""Speed-of-reversion diagnostics.

Half-life comes from an AR(1) approximation to an Ornstein-Uhlenbeck process:

    dS_t = alpha + lambda * S_{t-1} + eps_t          (OLS on differences)
    S_t  = alpha + (1 + lambda) * S_{t-1} + eps_t    (equivalently, phi = 1+lambda)

Mean reversion requires ``-2 < lambda < 0``.  Then a shock decays by half after

    half_life = -ln(2) / ln(1 + lambda)

The naive form ``-ln(2)/lambda`` is the continuous-time limit and is only a good
approximation for ``|lambda| << 1``; both are returned so the discrepancy is
visible rather than hidden.  Non-mean-reverting fits return NaN, never a
number that would silently pass a downstream filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class HalfLifeResult:
    half_life: float
    half_life_continuous: float
    lam: float
    phi: float
    tstat: float
    n_obs: int
    mean_reverting: bool

    def to_dict(self) -> dict:
        return {"half_life": self.half_life, "ar1_lambda": self.lam,
                "ar1_phi": self.phi, "ar1_tstat": self.tstat,
                "mean_reverting": self.mean_reverting}


def half_life(spread: pd.Series | np.ndarray) -> HalfLifeResult:
    s = pd.Series(spread).dropna().astype("float64")
    if len(s) < 30:
        raise ValueError(f"half-life needs >= 30 observations, got {len(s)}")
    lag = s.shift(1)
    delta = (s - lag).iloc[1:]
    lag = lag.iloc[1:]
    X = sm.add_constant(lag.to_numpy())
    fit = sm.OLS(delta.to_numpy(), X).fit()
    lam = float(fit.params[1])
    tstat = float(fit.tvalues[1])
    phi = 1.0 + lam

    reverting = (-2.0 < lam < 0.0) and (0.0 < phi < 1.0)
    if reverting:
        hl = float(-np.log(2.0) / np.log(phi))
        hl_cont = float(-np.log(2.0) / lam)
    else:
        hl = float("nan")
        hl_cont = float("nan")
    return HalfLifeResult(hl, hl_cont, lam, phi, tstat, len(s), reverting)


def ou_parameters(spread: pd.Series | np.ndarray) -> dict:
    """Long-run mean and stationary sigma implied by the AR(1) fit."""
    s = pd.Series(spread).dropna().astype("float64")
    hl = half_life(s)
    if not hl.mean_reverting:
        return {"mu": float(s.mean()), "sigma_eq": float(s.std(ddof=1)),
                "kappa": float("nan"), "half_life": float("nan")}
    lag = s.shift(1).iloc[1:]
    delta = (s - s.shift(1)).iloc[1:]
    X = sm.add_constant(lag.to_numpy())
    fit = sm.OLS(delta.to_numpy(), X).fit()
    a, lam = float(fit.params[0]), float(fit.params[1])
    mu = -a / lam
    resid_sd = float(np.std(fit.resid, ddof=2))
    phi = 1.0 + lam
    sigma_eq = resid_sd / np.sqrt(max(1.0 - phi ** 2, 1e-12))
    return {"mu": mu, "sigma_eq": float(sigma_eq),
            "kappa": float(-np.log(phi)), "half_life": hl.half_life}

"""Spread construction.

    S_t = log(P_A,t) - alpha - beta * log(P_B,t)

Log prices, not raw prices, for three reasons an interviewer will ask about:

1. beta is then an elasticity, so the hedge is expressed in *dollar* terms
   (one dollar of A against beta dollars of B) and does not change meaning when
   a stock's price level drifts or it splits.
2. Log price differences are returns, so the spread's P&L maps directly onto
   a dollar-neutral-ish book without a Jacobian correction.
3. Raw-price spreads inherit the price level: a $400 name against a $40 name
   produces a spread dominated by the expensive leg.

The intercept alpha is subtracted so the spread is centred on its training-
period mean.  It is estimated on training data and frozen; the rolling z-score
re-centres anyway, but keeping alpha makes the raw spread plot interpretable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_spread(log_a: pd.Series, log_b: pd.Series, beta: float,
                 alpha: float = 0.0) -> pd.Series:
    return (log_a - alpha - beta * log_b).rename("spread")


def build_spread_rolling(log_a: pd.Series, log_b: pd.Series,
                         beta_path: pd.Series, alpha_path: pd.Series | None = None
                         ) -> pd.Series:
    """Spread under a time-varying hedge ratio.

    ``beta_path`` must already be causal (value at t uses data <= t).  No shift
    is applied here: the execution lag is the backtester's job, and applying it
    in two places would double-count it.
    """
    a = alpha_path if alpha_path is not None else 0.0
    return (log_a - a - beta_path * log_b).rename("spread")


def spread_dollar_legs(price_a: float, price_b: float, beta: float,
                       notional: float) -> tuple[float, float]:
    """Dollar exposures for one unit of long spread, hedge-ratio neutral.

    P&L of the spread is ``V * dS = V * (dP_A/P_A - beta * dP_B/P_B)``, so the
    B leg must carry ``beta`` times the A leg's dollar exposure, with the
    opposite sign.  Gross exposure is therefore ``V * (1 + |beta|)`` and the
    caller sizes V so that stays inside the per-pair limit.
    """
    v = notional / (1.0 + abs(beta))
    return v, -beta * v

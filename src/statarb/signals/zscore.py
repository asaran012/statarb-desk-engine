"""Rolling z-score of the spread.

    z_t = (S_t - mean_{t-L+1..t}(S)) / std_{t-L+1..t}(S)

The window *includes* t.  That is not look-ahead: at the close of day t, S_t is
observed, so a statistic over the trailing L observations ending at t is
information available at t.  The look-ahead question is about *execution*, and
it is answered in :mod:`statarb.backtest.execution` by requiring a delay of at
least one bar.

An expanding-window variant is provided for the first L bars of a test period.
We deliberately do NOT use it by default: seeding the rolling statistics from
the tail of the training window is both realistic (on a live desk you have
yesterday's data) and leak-free (training data precedes the test period).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(spread: pd.Series, lookback: int, min_periods: int | None = None
                   ) -> pd.DataFrame:
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    mp = min_periods or lookback
    r = spread.rolling(window=lookback, min_periods=mp)
    mean = r.mean()
    std = r.std(ddof=1)
    z = (spread - mean) / std.replace(0.0, np.nan)
    return pd.DataFrame({"spread": spread, "mean": mean, "std": std, "zscore": z})


def expanding_zscore(spread: pd.Series, min_periods: int = 30) -> pd.DataFrame:
    e = spread.expanding(min_periods=min_periods)
    mean, std = e.mean(), e.std(ddof=1)
    z = (spread - mean) / std.replace(0.0, np.nan)
    return pd.DataFrame({"spread": spread, "mean": mean, "std": std, "zscore": z})

"""Exposure analytics derived from the equity history."""

from __future__ import annotations

import numpy as np
import pandas as pd


def exposure_summary(equity_frame: pd.DataFrame) -> dict:
    if equity_frame.empty:
        return {}
    eq = equity_frame["equity"].replace(0.0, np.nan)
    gross = equity_frame["gross_exposure"] / eq
    net = equity_frame["net_exposure"] / eq
    return {
        "avg_gross_exposure_pct": float(gross.mean() * 100),
        "peak_gross_exposure_pct": float(gross.max() * 100),
        "avg_abs_net_exposure_pct": float(net.abs().mean() * 100),
        "peak_abs_net_exposure_pct": float(net.abs().max() * 100),
        "avg_open_pairs": float(equity_frame["n_open_pairs"].mean()),
        "max_open_pairs": int(equity_frame["n_open_pairs"].max()),
        "pct_days_invested": float((equity_frame["n_open_pairs"] > 0).mean() * 100),
    }


def market_neutrality(equity_frame: pd.DataFrame) -> dict:
    """How close to dollar-neutral the book actually ran.

    Hedge-ratio-neutral sizing is dollar-neutral only when beta == 1, so a
    non-zero net exposure is expected, not a bug. Quantify it rather than
    claiming "market neutral" and hoping nobody checks.
    """
    if equity_frame.empty:
        return {}
    net = (equity_frame["net_exposure"] / equity_frame["equity"].replace(0.0, np.nan))
    gross = (equity_frame["gross_exposure"] /
             equity_frame["equity"].replace(0.0, np.nan))
    ratio = (net.abs() / gross.replace(0.0, np.nan)).dropna()
    return {"mean_net_over_gross_pct": float(ratio.mean() * 100) if len(ratio) else 0.0,
            "max_net_over_gross_pct": float(ratio.max() * 100) if len(ratio) else 0.0}

"""Execution timing.

The single most important line in this repository:

    a signal computed from the close of bar t may not be filled before bar
    t + delay_days, and delay_days >= 1 is enforced by the config schema.

Same-bar execution -- computing a z-score from today's close and filling at
today's close -- is the most common way a pairs backtest manufactures returns
that do not exist.  It is impossible here because :func:`apply_execution_delay`
is the only way a target position reaches the portfolio, and it shifts.

``fill_price`` selects which field of the *execution* bar is used:
  ``open``   signal at close t, fill at the open of t+delay (most realistic
             for an overnight-decision desk)
  ``close``  signal at close t, fill at the close of t+delay (default;
             conservative because the whole day's move happens before the fill)
"""

from __future__ import annotations

import pandas as pd


def apply_execution_delay(signals: pd.DataFrame, delay_days: int = 1) -> pd.DataFrame:
    """Shift target positions forward so nothing is actionable on its own bar."""
    if delay_days < 1:
        raise ValueError("delay_days must be >= 1; same-bar execution is not allowed")
    out = signals.copy()
    out["exec_target"] = signals["target"].shift(delay_days).fillna(0).astype("int8")
    out["exec_reason"] = signals["exit_reason"].shift(delay_days).fillna("")
    out["signal_date"] = pd.Series(signals.index, index=signals.index).shift(delay_days)
    out["exec_zscore"] = signals["zscore"].shift(delay_days)
    return out


def fill_price(bars_open: pd.DataFrame, bars_close: pd.DataFrame,
               date: pd.Timestamp, symbol: str, which: str = "close") -> float:
    src = bars_open if which == "open" else bars_close
    return float(src.at[date, symbol])

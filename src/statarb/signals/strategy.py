"""Signal state machine.

Emits *target positions*, not orders.  Whether and when a target becomes a fill
is entirely the backtester's decision, which is what keeps execution
assumptions in one auditable place.

State is explicit and single-valued per pair:

    FLAT --(z >  entry)--> SHORT_SPREAD   (short A, long B)
    FLAT --(z < -entry)--> LONG_SPREAD    (long A, short B)
    OPEN --(|z| <  exit)--> FLAT          exit_reason=mean_reversion
    OPEN --(|z| >  stop)--> FLAT          exit_reason=stop_loss
    OPEN --(held > max)---> FLAT          exit_reason=max_holding

Two behaviours that are easy to get wrong and are pinned by tests:

* **No re-entry while open.** A raw ``z > entry`` mask re-fires on every bar the
  spread stays wide, which in a naive backtest manufactures a fresh round-trip
  (and a fresh cost) every day. Position state gates entries.
* **Stop above entry, not below.** ``|z| > stop`` means the spread diverged
  *further*. Exiting there is a bet that the relationship is broken rather than
  stretched. It caps loss per episode; it does not improve expectancy, and the
  README says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

import numpy as np
import pandas as pd


class Position(IntEnum):
    SHORT_SPREAD = -1
    FLAT = 0
    LONG_SPREAD = 1


@dataclass
class SignalEvent:
    date: pd.Timestamp
    action: str                # "enter" | "exit"
    target: int
    zscore: float
    reason: str


def generate_signals(zscore: pd.Series, *, entry_z: float = 2.0, exit_z: float = 0.5,
                     stop_z: float = 4.0, max_holding_days: int = 60
                     ) -> pd.DataFrame:
    """Run the state machine over a z-score path.

    Returns a frame indexed like ``zscore`` with:
      ``target``       desired position for the bar (state after processing t)
      ``entry_flag``   True on the bar whose close triggered an entry
      ``exit_flag``    True on the bar whose close triggered an exit
      ``exit_reason``  populated on exit bars
      ``bars_held``    bars since entry
    """
    if exit_z >= entry_z:
        raise ValueError("exit_z must be < entry_z")
    if stop_z <= entry_z:
        raise ValueError("stop_z must be > entry_z")

    z = zscore.to_numpy(dtype="float64")
    n = z.size
    target = np.zeros(n, dtype="int8")
    entry_flag = np.zeros(n, dtype=bool)
    exit_flag = np.zeros(n, dtype=bool)
    bars_held = np.zeros(n, dtype="int32")
    reasons: List[str] = [""] * n

    state = int(Position.FLAT)
    held = 0

    for t in range(n):
        zt = z[t]
        if not np.isfinite(zt):
            target[t] = state
            bars_held[t] = held
            if state != 0:
                held += 1
            continue

        if state == Position.FLAT:
            if zt > entry_z:
                state, held = int(Position.SHORT_SPREAD), 0
                entry_flag[t] = True
            elif zt < -entry_z:
                state, held = int(Position.LONG_SPREAD), 0
                entry_flag[t] = True
        else:
            held += 1
            if abs(zt) > stop_z:
                exit_flag[t], reasons[t] = True, "stop_loss"
                state, held = int(Position.FLAT), 0
            elif abs(zt) < exit_z:
                exit_flag[t], reasons[t] = True, "mean_reversion"
                state, held = int(Position.FLAT), 0
            elif held >= max_holding_days:
                exit_flag[t], reasons[t] = True, "max_holding"
                state, held = int(Position.FLAT), 0

        target[t] = state
        bars_held[t] = held

    return pd.DataFrame({"zscore": zscore.to_numpy(), "target": target,
                         "entry_flag": entry_flag, "exit_flag": exit_flag,
                         "exit_reason": reasons, "bars_held": bars_held},
                        index=zscore.index)


def pair_signal_frame(log_a: pd.Series, log_b: pd.Series, beta: float, alpha: float,
                      *, zscore_lookback: int, entry_z: float, exit_z: float,
                      stop_z: float, max_holding_days: int,
                      beta_path: Optional[pd.Series] = None,
                      alpha_path: Optional[pd.Series] = None) -> pd.DataFrame:
    """Spread -> z-score -> signals, in one call. Static or rolling hedge ratio."""
    from .spread import build_spread, build_spread_rolling
    from .zscore import rolling_zscore

    if beta_path is not None:
        spread = build_spread_rolling(log_a, log_b, beta_path, alpha_path)
        betas = beta_path.reindex(spread.index)
    else:
        spread = build_spread(log_a, log_b, beta, alpha)
        betas = pd.Series(beta, index=spread.index)

    zs = rolling_zscore(spread, zscore_lookback)
    sig = generate_signals(zs["zscore"], entry_z=entry_z, exit_z=exit_z,
                           stop_z=stop_z, max_holding_days=max_holding_days)
    out = zs.join(sig[["target", "entry_flag", "exit_flag", "exit_reason",
                       "bars_held"]])
    out["beta"] = betas
    return out

"""Rolling-window stability of a candidate pair.

A pair that is cointegrated over one 8-year window may have been cointegrated
for two of those years and drifting for six.  Engle-Granger on the full window
cannot tell you which, because it reports a single number for the whole sample.
So we re-run the test inside K sub-windows and ask how *often* it holds, how
much beta moves, and how much the half-life moves.

This is the single highest-value filter in the pipeline on this dataset: the
synthetic market contains regime-break pairs that pass full-window
cointegration 100% of the time and are still untradeable out of sample.  If a
reviewer asks "what stops you selecting a pair that used to work", this module
is the answer, and Ablation B measures what it is worth.

Everything here consumes training data only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .cointegration import engle_granger
from .mean_reversion import half_life


@dataclass
class StabilityResult:
    n_windows: int
    n_tested: int
    pass_rate: float
    beta_mean: float
    beta_std: float
    beta_cv: float
    half_life_mean: float
    half_life_std: float
    spread_vol_mean: float
    spread_vol_cv: float
    pvalues: List[float]

    def to_dict(self) -> dict:
        return {
            "stability_windows": self.n_tested,
            "stability_pass_rate": self.pass_rate,
            "beta_mean": self.beta_mean, "beta_std": self.beta_std,
            "beta_cv": self.beta_cv,
            "half_life_mean": self.half_life_mean,
            "half_life_std": self.half_life_std,
            "spread_vol_mean": self.spread_vol_mean,
            "spread_vol_cv": self.spread_vol_cv,
        }


def _window_bounds(n: int, n_windows: int, window_days: int) -> List[tuple]:
    """Evenly spaced rolling sub-window [start, end) index pairs."""
    if n < window_days:
        window_days = max(60, n // 2)
    if n_windows <= 1 or n <= window_days:
        return [(0, n)]
    step = (n - window_days) / (n_windows - 1)
    return [(int(round(i * step)), int(round(i * step)) + window_days)
            for i in range(n_windows)]


def pair_stability(log_a: pd.Series, log_b: pd.Series, *,
                   n_windows: int = 6, window_days: int = 252,
                   alpha: float = 0.05,
                   symbol_a: str = "A", symbol_b: str = "B") -> StabilityResult:
    df = pd.concat([log_a.rename("a"), log_b.rename("b")], axis=1).dropna()
    n = len(df)
    bounds = _window_bounds(n, n_windows, window_days)

    pvals: List[float] = []
    betas: List[float] = []
    hls: List[float] = []
    vols: List[float] = []

    for lo, hi in bounds:
        sub = df.iloc[lo:hi]
        if len(sub) < 60:
            continue
        try:
            res = engle_granger(sub["a"], sub["b"], symbol_a, symbol_b)
        except (ValueError, np.linalg.LinAlgError):
            continue
        pvals.append(res.pvalue)
        betas.append(res.beta)
        spread = sub["a"] - res.alpha - res.beta * sub["b"]
        vols.append(float(spread.std(ddof=1)))
        try:
            hl = half_life(spread)
            if hl.mean_reverting and np.isfinite(hl.half_life):
                hls.append(hl.half_life)
        except ValueError:
            pass

    n_tested = len(pvals)
    if n_tested == 0:
        return StabilityResult(len(bounds), 0, 0.0, *([np.nan] * 7), [])

    pass_rate = float(np.mean(np.asarray(pvals) < alpha))
    b = np.asarray(betas, dtype="float64")
    b_mean, b_std = float(b.mean()), float(b.std(ddof=1)) if b.size > 1 else 0.0
    b_cv = float(b_std / abs(b_mean)) if abs(b_mean) > 1e-9 else np.inf
    v = np.asarray(vols, dtype="float64")
    v_mean = float(v.mean())
    v_cv = float(v.std(ddof=1) / v_mean) if v.size > 1 and v_mean > 1e-12 else 0.0
    h = np.asarray(hls, dtype="float64")

    return StabilityResult(
        n_windows=len(bounds), n_tested=n_tested, pass_rate=pass_rate,
        beta_mean=b_mean, beta_std=b_std, beta_cv=b_cv,
        half_life_mean=float(h.mean()) if h.size else np.nan,
        half_life_std=float(h.std(ddof=1)) if h.size > 1 else 0.0,
        spread_vol_mean=v_mean, spread_vol_cv=v_cv, pvalues=pvals)


def quality_score(row: pd.Series, *, half_life_min: float = 2.0,
                  half_life_max: float = 60.0) -> float:
    """Explainable 0-1 pair score.  Four equally weighted, bounded components.

    No optimiser, no learned weights: a score that cannot be explained on a
    whiteboard is a score that cannot be defended when a pair loses money.

      * ``s_stability``  fraction of sub-windows in which cointegration held
      * ``s_beta``       1 - coefficient of variation of the rolling beta
      * ``s_half_life``  triangular preference for a tradeable reversion speed
      * ``s_evidence``   1 - full-window Engle-Granger p-value

    Every input is computed on training data only; the test window is never
    touched here.
    """
    s_stab = float(np.clip(row.get("stability_pass_rate", 0.0), 0.0, 1.0))

    cv = row.get("beta_cv", np.inf)
    s_beta = float(np.clip(1.0 - cv, 0.0, 1.0)) if np.isfinite(cv) else 0.0

    hl = row.get("half_life", np.nan)
    if not np.isfinite(hl) or hl <= half_life_min or hl >= half_life_max:
        s_hl = 0.0
    else:
        ideal = 0.5 * (half_life_min + half_life_max) * 0.4   # ~12 days by default
        s_hl = float(np.clip(1.0 - abs(hl - ideal) / (half_life_max - half_life_min),
                             0.0, 1.0))

    p = row.get("coint_pvalue", 1.0)
    s_ev = float(np.clip(1.0 - p, 0.0, 1.0)) if np.isfinite(p) else 0.0

    return float(np.mean([s_stab, s_beta, s_hl, s_ev]))

"""Return-correlation screening.

Correlation is used **only** to shortlist candidates cheaply: an all-pairs
Engle-Granger sweep over a 60-name universe is 1,770 regressions per window, and
correlation prunes that to the economically plausible subset first.

    High correlation is NOT equivalent to cointegration.

Two names can co-move almost perfectly in returns while their price levels
diverge without bound (each carrying its own random-walk component); that pair
is correlated and untradeable.  Conversely a genuinely cointegrated pair can
show modest return correlation.  This repository measures the gap explicitly in
Ablation A rather than asserting it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def correlation_matrix(returns: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    return returns.corr(method=method, min_periods=60)


def candidate_pairs(
    returns: pd.DataFrame,
    threshold: float = 0.60,
    sectors: Optional[Dict[str, str]] = None,
    restrict_to_same_sector: bool = True,
    max_pairs: Optional[int] = None,
) -> pd.DataFrame:
    """Shortlist ordered ``(a, b)`` pairs with ``a < b`` lexicographically.

    Fixing the ordering matters: OLS of A on B and of B on A give different
    betas, so an unstable pair ordering would make the spread definition depend
    on dictionary iteration order.
    """
    corr = correlation_matrix(returns)
    syms = list(corr.columns)
    rows: List[dict] = []
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            a, b = sorted((syms[i], syms[j]))
            c = corr.loc[a, b]
            if not np.isfinite(c) or c < threshold:
                continue
            if restrict_to_same_sector and sectors:
                if sectors.get(a) != sectors.get(b):
                    continue
            rows.append({"symbol_a": a, "symbol_b": b, "correlation": float(c),
                         "sector": (sectors or {}).get(a, "")})
    out = pd.DataFrame(rows, columns=["symbol_a", "symbol_b", "correlation", "sector"])
    if out.empty:
        return out
    out = out.sort_values("correlation", ascending=False).reset_index(drop=True)
    return out.head(max_pairs) if max_pairs else out


def as_tuples(pairs: pd.DataFrame) -> List[Tuple[str, str]]:
    return list(zip(pairs["symbol_a"], pairs["symbol_b"]))

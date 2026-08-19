"""Multiple-hypothesis correction.

Screening 400 pairs at alpha=0.05 yields ~20 "significant" results even when
every null is true.  Family-wise methods (Bonferroni) control the probability of
*any* false positive and are far too conservative for research screening.
Benjamini-Hochberg instead controls the expected *proportion* of false
discoveries among the rejected set, which is the quantity a researcher building
a pair book actually cares about.

BH procedure: sort p-values ascending, find the largest k with
``p_(k) <= k/m * alpha``, reject the k smallest.  Implemented directly (rather
than imported) so the step-up logic is visible and unit-testable.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05
                       ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return ``(reject, qvalues, threshold)`` in the caller's original order.

    ``qvalues`` are the standard step-up adjusted p-values, enforced monotone
    non-decreasing and clipped at 1.
    """
    p = np.asarray(pvalues, dtype="float64")
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0), 0.0

    finite = np.isfinite(p)
    reject = np.zeros(m, dtype=bool)
    q = np.ones(m, dtype="float64")
    pf = p[finite]
    mf = pf.size
    if mf == 0:
        return reject, q, 0.0

    order = np.argsort(pf, kind="mergesort")
    ps = pf[order]
    ranks = np.arange(1, mf + 1)

    below = ps <= (ranks / mf) * alpha
    k = int(np.max(np.where(below)[0]) + 1) if below.any() else 0
    threshold = float(ps[k - 1]) if k > 0 else 0.0

    qs = np.minimum.accumulate((ps * mf / ranks)[::-1])[::-1]
    qs = np.clip(qs, 0.0, 1.0)

    rej_sorted = np.zeros(mf, dtype=bool)
    rej_sorted[:k] = True

    q_unsorted = np.empty(mf); q_unsorted[order] = qs
    r_unsorted = np.zeros(mf, dtype=bool); r_unsorted[order] = rej_sorted
    q[finite] = q_unsorted
    reject[finite] = r_unsorted
    return reject, q, threshold


def bonferroni(pvalues: Sequence[float], alpha: float = 0.05
               ) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pvalues, dtype="float64")
    m = max(p.size, 1)
    q = np.clip(p * m, 0.0, 1.0)
    return q <= alpha, q


def expected_false_positives(n_tests: int, alpha: float) -> float:
    """How many rejections a naive threshold produces under a complete null."""
    return float(n_tests) * float(alpha)

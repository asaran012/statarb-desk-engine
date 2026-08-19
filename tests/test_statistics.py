"""Statistical procedures must be correct on inputs whose answer is known."""
import numpy as np
import pandas as pd
import pytest

from statarb.data.synthetic import (cointegrated_pair, independent_random_walks,
                                    ou_process, regime_breaking_pair)
from statarb.research.cointegration import engle_granger
from statarb.research.hedge_ratio import (rolling_ols_hedge_ratio,
                                          static_ols_hedge_ratio)
from statarb.research.mean_reversion import half_life
from statarb.research.multiple_testing import (benjamini_hochberg, bonferroni,
                                               expected_false_positives)
from statarb.research.stationarity import adf_test


def test_cointegrated_pair_is_detected():
    df = cointegrated_pair(n=1200, beta=1.0, half_life=12.0, seed=1)
    r = engle_granger(np.log(df["A"]), np.log(df["B"]))
    assert r.pvalue < 0.05, f"known cointegrated pair not detected (p={r.pvalue:.4f})"


def test_independent_random_walks_are_not_cointegrated():
    """The critical negative control: without it a broken test passes everything."""
    rejections = 0
    for s in range(20):
        df = independent_random_walks(n=1200, seed=s)
        if engle_granger(np.log(df["A"]), np.log(df["B"])).pvalue < 0.05:
            rejections += 1
    # Under the null we expect ~5% rejections; allow generous slack for 20 draws.
    assert rejections <= 4, f"{rejections}/20 spurious rejections - test is too loose"


def test_correlated_but_not_cointegrated_is_rejected():
    """High correlation must not by itself produce a cointegration rejection."""
    rejections = 0
    for s in range(20):
        df = independent_random_walks(n=1200, seed=s, rho=0.85)
        ra = np.log(df["A"]).diff().dropna()
        rb = np.log(df["B"]).diff().dropna()
        assert ra.corr(rb) > 0.6, "fixture is not actually correlated"
        if engle_granger(np.log(df["A"]), np.log(df["B"])).pvalue < 0.05:
            rejections += 1
    assert rejections <= 4, f"correlation alone produced {rejections}/20 rejections"


def test_adf_rejects_stationary_and_not_random_walk():
    ou = pd.Series(ou_process(2000, 10.0, 0.02, np.random.default_rng(3)))
    rw = pd.Series(np.random.default_rng(3).normal(0, 0.02, 2000).cumsum())
    assert adf_test(ou).pvalue < 0.05
    assert adf_test(rw).pvalue > 0.05


def test_half_life_recovers_known_value():
    for target in (5.0, 12.0, 30.0):
        s = pd.Series(ou_process(20_000, target, 0.02, np.random.default_rng(11)))
        est = half_life(s).half_life_continuous
        assert abs(est - target) / target < 0.20, f"{target} -> {est}"


def test_half_life_is_nan_for_random_walk():
    rw = pd.Series(np.random.default_rng(5).normal(0, 0.02, 3000).cumsum())
    hl = half_life(rw).half_life_continuous
    assert (not np.isfinite(hl)) or hl > 200, f"random walk gave half-life {hl}"


def test_hedge_ratio_recovers_known_beta():
    for beta in (0.5, 1.0, 1.8):
        df = cointegrated_pair(n=3000, beta=beta, seed=7)
        est = static_ols_hedge_ratio(np.log(df["A"]), np.log(df["B"])).beta
        assert abs(est - beta) < 0.05, f"beta {beta} -> {est}"


def test_rolling_beta_is_causal():
    """Mutating a future price must not change any earlier beta."""
    df = cointegrated_pair(n=600, beta=1.2, seed=9)
    la, lb = np.log(df["A"]), np.log(df["B"])
    base = rolling_ols_hedge_ratio(la, lb, window=120)
    la2 = la.copy()
    la2.iloc[400:] *= 1.75          # violently corrupt the future
    after = rolling_ols_hedge_ratio(la2, lb, window=120)
    pd.testing.assert_series_equal(base["beta"].iloc[:400], after["beta"].iloc[:400])
    assert not np.allclose(base["beta"].iloc[450:].dropna(),
                           after["beta"].iloc[450:].dropna()), \
        "corruption had no effect at all - the test is not exercising anything"


def test_benjamini_hochberg_all_null_rejects_almost_nothing():
    p = np.random.default_rng(0).uniform(size=500)
    reject, q, _ = benjamini_hochberg(p, alpha=0.05)
    assert reject.sum() <= 2
    assert np.all((q >= 0) & (q <= 1))


def test_benjamini_hochberg_is_monotone_and_less_strict_than_bonferroni():
    p = np.array([0.001, 0.008, 0.02, 0.04, 0.2, 0.5])
    r_bh, q, _ = benjamini_hochberg(p, alpha=0.05)
    r_bo, _ = bonferroni(p, alpha=0.05)
    assert r_bh.sum() >= r_bo.sum()
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12), "q-values not monotone in p"


def test_benjamini_hochberg_known_worked_example():
    # Classic step-up example: with alpha=0.05, m=4, p=(0.01,0.02,0.03,0.9)
    # thresholds are 0.0125,0.025,0.0375,0.05 -> largest k satisfying p_k<=k*a/m is 3.
    p = np.array([0.01, 0.02, 0.03, 0.9])
    reject, _, _ = benjamini_hochberg(p, alpha=0.05)
    assert list(reject) == [True, True, True, False]


def test_expected_false_positives():
    assert expected_false_positives(100, 0.05) == pytest.approx(5.0)


def test_regime_break_pair_breaks_in_the_second_half():
    df = regime_breaking_pair(n=1500, break_frac=0.6, seed=2)
    cut = int(1500 * 0.6)
    pre = engle_granger(np.log(df["A"][:cut]), np.log(df["B"][:cut])).pvalue
    post = engle_granger(np.log(df["A"][cut:]), np.log(df["B"][cut:])).pvalue
    assert pre < 0.05 < post, f"pre={pre:.4f} post={post:.4f}"

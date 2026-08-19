"""Signal construction: z-scores and the entry/exit state machine."""
import numpy as np
import pandas as pd
import pytest

from statarb.signals.spread import spread_dollar_legs
from statarb.signals.strategy import Position, generate_signals
from statarb.signals.zscore import rolling_zscore


def test_rolling_zscore_known_values():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = rolling_zscore(s, lookback=3)
    # window [1,2,3]: mean 2, sd 1 -> (3-2)/1 = 1.0
    assert z["zscore"].iloc[2] == pytest.approx(1.0)
    assert z["zscore"].iloc[3] == pytest.approx(1.0)
    assert np.isnan(z["zscore"].iloc[0])


def test_rolling_zscore_constant_series_is_not_infinite():
    z = rolling_zscore(pd.Series([2.0] * 40), lookback=10)["zscore"]
    assert not np.isinf(z.dropna()).any(), "zero variance produced infinite z"


def test_rolling_zscore_uses_only_past_and_present():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=300))
    base = rolling_zscore(s, lookback=60)["zscore"]
    s2 = s.copy()
    s2.iloc[200:] += 50.0
    after = rolling_zscore(s2, lookback=60)["zscore"]
    pd.testing.assert_series_equal(base.iloc[:200], after.iloc[:200])


def test_signals_enter_short_above_entry_and_exit_at_exit_band():
    z = pd.Series([0.0, 0.5, 2.5, 1.5, 0.4, 0.1])
    sig = generate_signals(z, entry_z=2.0, exit_z=0.5, stop_z=4.0)
    pos = sig["target"].to_numpy()
    assert pos[2] == Position.SHORT_SPREAD   # z above +2 -> short the spread
    assert pos[3] == Position.SHORT_SPREAD   # still open between bands
    assert pos[4] == Position.FLAT           # |z| < 0.5 -> exit


def test_signals_enter_long_below_negative_entry():
    z = pd.Series([0.0, -2.5, -1.0, -0.2])
    pos = generate_signals(z, entry_z=2.0, exit_z=0.5)["target"].to_numpy()
    assert pos[1] == Position.LONG_SPREAD
    assert pos[3] == Position.FLAT


def test_stop_loss_fires_beyond_stop_band():
    z = pd.Series([0.0, 2.5, 3.0, 4.5, 4.6])
    sig = generate_signals(z, entry_z=2.0, exit_z=0.5, stop_z=4.0)
    assert sig["target"].iloc[3] == Position.FLAT
    assert sig["exit_reason"].iloc[3] == "stop_loss"


def test_no_reentry_while_position_open():
    """Re-crossing the entry band while already open must not double up."""
    z = pd.Series([0.0, 2.5, 2.1, 2.6, 2.2, 0.1])
    sig = generate_signals(z, entry_z=2.0, exit_z=0.5, stop_z=4.0)
    entries = int(sig["entry_flag"].sum())
    assert entries == 1, f"expected one entry, got {entries}"


def test_max_holding_days_forces_exit():
    z = pd.Series([0.0] + [2.5] * 40)
    sig = generate_signals(z, entry_z=2.0, exit_z=0.5, stop_z=10.0, max_holding_days=10)
    assert "max_holding" in set(sig["exit_reason"])


def test_nan_zscore_never_opens_a_position():
    z = pd.Series([np.nan, np.nan, 2.5, np.nan, 0.1])
    sig = generate_signals(z, entry_z=2.0, exit_z=0.5)
    assert sig["target"].iloc[0] == Position.FLAT
    assert sig["target"].iloc[1] == Position.FLAT


def test_spread_legs_are_beta_weighted():
    """Legs are dollar exposures: B must carry beta times A, opposite sign."""
    va, vb = spread_dollar_legs(price_a=50.0, price_b=25.0, beta=1.5,
                                notional=100_000.0)
    assert va > 0 and vb < 0
    assert abs(vb) / abs(va) == pytest.approx(1.5, rel=1e-9)
    assert abs(va) + abs(vb) == pytest.approx(100_000.0, rel=1e-9), \
        "gross exposure must equal the notional the risk layer sized"


def test_spread_legs_gross_is_invariant_to_price_level():
    """Sizing is in dollars, so a stock split must not change gross exposure."""
    g1 = sum(abs(x) for x in spread_dollar_legs(price_a=50.0, price_b=25.0,
                                                beta=1.2, notional=1e5))
    g2 = sum(abs(x) for x in spread_dollar_legs(price_a=5.0, price_b=250.0,
                                                beta=1.2, notional=1e5))
    assert g1 == pytest.approx(g2)

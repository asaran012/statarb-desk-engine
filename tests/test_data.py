"""Data validation must *detect* faults, not merely run without error."""
import numpy as np
import pandas as pd
import pytest

from statarb.data.calendar import expected_sessions, trading_calendar
from statarb.data.loader import log_returns, to_wide
from statarb.data.validator import validate_panel


def _fault(panel, fn):
    p = panel.copy()
    return fn(p)


def test_clean_panel_passes_all_checks(market):
    panel, _ = market
    rep = validate_panel(panel)
    failed = [c.name for c in rep.results if c.status == "FAIL"]
    assert not failed, f"clean data failed: {failed}"


def test_detects_duplicate_symbol_date(market):
    panel, _ = market
    bad = pd.concat([panel, panel.iloc[[10]]], ignore_index=True)
    assert validate_panel(bad).count("duplicate_symbol_date") >= 1


def test_detects_null_prices(market):
    panel, _ = market
    bad = panel.copy()
    bad.loc[bad.index[5:9], "close"] = np.nan
    assert validate_panel(bad).count("null_prices") >= 4


def test_detects_non_positive_prices(market):
    panel, _ = market
    bad = panel.copy()
    bad.loc[bad.index[3], "close"] = -1.0
    bad.loc[bad.index[4], "close"] = 0.0
    assert validate_panel(bad).count("non_positive_prices") >= 2


def test_detects_impossible_ohlc(market):
    panel, _ = market
    bad = panel.copy()
    bad.loc[bad.index[7], "low"] = bad.loc[bad.index[7], "high"] * 2.0
    assert validate_panel(bad).count("impossible_ohlc") >= 1


def test_detects_negative_volume(market):
    panel, _ = market
    bad = panel.copy()
    bad.loc[bad.index[11], "volume"] = -100
    assert validate_panel(bad).count("negative_volume") >= 1


def test_detects_stale_prices(market):
    panel, _ = market
    bad = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    sym = bad["symbol"].iloc[0]
    idx = bad.index[bad["symbol"] == sym][50:65]
    bad.loc[idx, "close"] = float(bad.loc[idx[0], "close"])
    assert validate_panel(bad).count("stale_prices") >= 1


def test_detects_extreme_returns(market):
    panel, _ = market
    bad = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    sym = bad["symbol"].iloc[0]
    i = bad.index[bad["symbol"] == sym][100]
    # returns are computed from adjusted_close, which is the correct column
    bad.loc[i, "adjusted_close"] = float(bad.loc[i, "adjusted_close"]) * 3.0
    assert validate_panel(bad).count("extreme_returns") >= 1


def test_detects_missing_symbol(market):
    panel, _ = market
    syms = sorted(panel["symbol"].unique())
    bad = panel[panel["symbol"] != syms[0]]
    assert validate_panel(bad, expected_symbols=syms).count("missing_symbols") >= 1


def test_validator_does_not_mutate_input(market):
    panel, _ = market
    before = panel.copy()
    validate_panel(panel)
    pd.testing.assert_frame_equal(panel, before)


def test_trading_calendar_excludes_weekends_and_holidays():
    cal = trading_calendar("2024-01-01", "2024-12-31")
    assert pd.Timestamp("2024-01-01") not in cal      # New Year's Day
    assert pd.Timestamp("2024-07-04") not in cal      # Independence Day
    assert pd.Timestamp("2024-12-25") not in cal      # Christmas
    assert pd.Timestamp("2024-03-29") not in cal      # Good Friday
    assert pd.Timestamp("2024-06-19") not in cal      # Juneteenth
    assert all(d.weekday() < 5 for d in cal)


def test_juneteenth_only_from_2022():
    assert pd.Timestamp("2021-06-18") in trading_calendar("2021-06-01", "2021-06-30")
    assert pd.Timestamp("2022-06-20") not in trading_calendar("2022-06-01", "2022-06-30")


def test_expected_session_count_is_plausible():
    n = len(trading_calendar("2024-01-01", "2024-12-31"))
    assert 248 <= n <= 254, f"{n} sessions in 2024 is not plausible"


def test_to_wide_is_aligned_and_sorted(market):
    panel, _ = market
    w = to_wide(panel, "close")
    assert w.index.is_monotonic_increasing
    assert not w.index.duplicated().any()
    assert list(w.columns) == sorted(w.columns)


def test_log_returns_have_no_look_ahead(market):
    panel, _ = market
    w = to_wide(panel, "close")
    r = log_returns(w)
    assert r.iloc[0].isna().all(), "first return must be NaN, not fabricated"

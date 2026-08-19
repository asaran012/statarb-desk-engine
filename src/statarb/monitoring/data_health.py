"""Pre-open data monitoring.

Wraps the validator into a daily desk check with explicit PASS/WARNING/FAIL
rules. The distinction that matters operationally: a FAIL means *do not trade
on this data*; a WARNING means trade but investigate. Duplicate rows and
missing symbols are FAIL because they silently corrupt position sizing; a stale
quote is a WARNING because one flat print is common and usually benign.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd


from ..data.validator import validate_panel
from .alerts import AlertSet, PASS


def longest_stale_run(panel: pd.DataFrame, price_col: str = "adjusted_close") -> int:
    """Longest run of identical consecutive prices for any single symbol.

    The validator counts stale *runs*; a single 30-day frozen feed is one run
    and would only ever raise a WARNING. Severity has to scale with how long
    the feed has been dead, because a 30-day freeze silently zeroes 30 days of
    returns and every downstream beta and z-score computed from them.
    """
    if price_col not in panel.columns or panel.empty:
        return 0
    longest = 0
    d = panel.sort_values(["symbol", "date"])
    for _, g in d.groupby("symbol", sort=False):
        p = g[price_col].to_numpy()
        if len(p) < 2:
            continue
        same = p[1:] == p[:-1]
        run = best = 0
        for flag in same:
            run = run + 1 if flag else 0
            best = max(best, run)
        longest = max(longest, best + 1 if best else 0)
    return int(longest)


def data_health_check(panel: pd.DataFrame, *,
                      expected_symbols: Optional[List[str]] = None,
                      stale_price_days: int = 5,
                      extreme_return_abs: float = 0.25) -> AlertSet:
    rep = validate_panel(panel, expected_symbols=expected_symbols,
                         stale_price_days=stale_price_days,
                         extreme_return_abs=extreme_return_abs)
    a = AlertSet("DATA")
    a.add("symbols_received", PASS, rep.meta.get("Symbols received", 0))
    a.add("rows", PASS, rep.meta.get("Rows", 0))
    a.gauge("duplicate_observations", rep.count("duplicate_symbol_date"), fail=1)
    a.gauge("null_prices", rep.count("null_prices"), warn=1, fail=10)
    a.gauge("non_positive_prices", rep.count("non_positive_prices"), fail=1)
    a.gauge("impossible_ohlc", rep.count("impossible_ohlc"), fail=1)
    a.gauge("stale_prices", rep.count("stale_prices"), warn=1, fail=10)
    a.gauge("longest_stale_run_days", longest_stale_run(panel),
            warn=stale_price_days, fail=2 * stale_price_days,
            detail="consecutive identical closes for one symbol")
    a.gauge("extreme_return_alerts", rep.count("extreme_returns"), warn=1, fail=25)
    a.gauge("missing_symbols", rep.count("missing_symbols"), fail=1)
    a.gauge("symbols_missing_sessions", rep.count("symbols_with_missing_sessions"),
            warn=1, fail=5)
    a.gauge("calendar_mismatch", rep.count("calendar_mismatch"), warn=1, fail=10)
    return a

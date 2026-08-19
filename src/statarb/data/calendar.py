"""A minimal NYSE-style trading calendar.

Deliberately dependency-free.  It is not exchange-exact (no half days, no
one-off closures such as 2018-12-05), and the README says so.  Its purpose is
(a) to give the synthetic panel a realistic index and (b) to give the validator
something to compare a vendor's calendar against, which is how missing-session
bugs actually get caught on a desk.
"""

from __future__ import annotations

import datetime as _dt
from functools import lru_cache
from typing import List

import pandas as pd


def easter_sunday(year: int) -> _dt.date:
    """Anonymous Gregorian algorithm (Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return _dt.date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> _dt.date:
    """n-th (1-indexed) `weekday` of a month; n=-1 means last."""
    if n > 0:
        d = _dt.date(year, month, 1)
        offset = (weekday - d.weekday()) % 7
        return d + _dt.timedelta(days=offset + 7 * (n - 1))
    last = _dt.date(year, month, 28)
    while (last + _dt.timedelta(days=1)).month == month:
        last += _dt.timedelta(days=1)
    offset = (last.weekday() - weekday) % 7
    return last - _dt.timedelta(days=offset)


def _observed(d: _dt.date) -> _dt.date:
    if d.weekday() == 5:
        return d - _dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + _dt.timedelta(days=1)
    return d


@lru_cache(maxsize=64)
def market_holidays(year: int) -> frozenset:
    hol = {
        _observed(_dt.date(year, 1, 1)),                    # New Year's Day
        _nth_weekday(year, 1, 0, 3),                        # MLK Jr.
        _nth_weekday(year, 2, 0, 3),                        # Washington's Birthday
        easter_sunday(year) - _dt.timedelta(days=2),        # Good Friday
        _nth_weekday(year, 5, 0, -1),                       # Memorial Day
        _observed(_dt.date(year, 7, 4)),                    # Independence Day
        _nth_weekday(year, 9, 0, 1),                        # Labor Day
        _nth_weekday(year, 11, 3, 4),                       # Thanksgiving
        _observed(_dt.date(year, 12, 25)),                  # Christmas
    }
    if year >= 2022:
        hol.add(_observed(_dt.date(year, 6, 19)))           # Juneteenth
    return frozenset(hol)


def trading_calendar(start: str, end: str) -> pd.DatetimeIndex:
    """Business days between ``start`` and ``end`` minus US market holidays."""
    days = pd.bdate_range(start=start, end=end)
    hol = set()
    for yr in range(days[0].year, days[-1].year + 1):
        hol |= set(market_holidays(yr))
    keep = [d for d in days if d.date() not in hol]
    return pd.DatetimeIndex(keep)


def expected_sessions(start, end) -> List[pd.Timestamp]:
    return list(trading_calendar(str(pd.Timestamp(start).date()),
                                 str(pd.Timestamp(end).date())))

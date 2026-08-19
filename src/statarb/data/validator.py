"""Market-data validation.

Design rule taken straight from desk practice: **the validator never mutates
data**.  It classifies and counts.  Removal is a separate, explicit decision
made by :mod:`statarb.data.cleaner`, which records what it dropped.  Silent
repair is how a data bug becomes a P&L bug six weeks later.

Ten checks are implemented; each returns a :class:`CheckResult` carrying a
status, a count and an example frame so an analyst can eyeball the offending
rows rather than trust a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .calendar import trading_calendar

PASS, WARNING, FAIL = "PASS", "WARNING", "FAIL"
_RANK = {PASS: 0, WARNING: 1, FAIL: 2}

REQUIRED_COLUMNS = ["date", "symbol", "open", "high", "low", "close",
                    "adjusted_close", "volume"]


@dataclass
class CheckResult:
    name: str
    status: str
    count: int
    message: str
    examples: Optional[pd.DataFrame] = None

    def to_dict(self) -> dict:
        return {"check": self.name, "status": self.status,
                "count": int(self.count), "message": self.message}


@dataclass
class DataHealthReport:
    results: List[CheckResult] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)

    def add(self, r: CheckResult) -> None:
        self.results.append(r)

    @property
    def status(self) -> str:
        if not self.results:
            return PASS
        worst = max(_RANK[r.status] for r in self.results)
        return {0: PASS, 1: WARNING, 2: FAIL}[worst]

    def get(self, name: str) -> Optional[CheckResult]:
        return next((r for r in self.results if r.name == name), None)

    def count(self, name: str) -> int:
        r = self.get(name)
        return r.count if r else 0

    def to_dict(self) -> dict:
        return {"status": self.status, "meta": self.meta,
                "checks": [r.to_dict() for r in self.results]}

    def render(self) -> str:
        lines = ["=" * 52, "DATA HEALTH REPORT", "=" * 52]
        for k, v in self.meta.items():
            lines.append(f"{k+':':32s}{v}")
        lines.append("-" * 52)
        for r in self.results:
            lines.append(f"{r.name+':':32s}{r.count:>8}  [{r.status}]")
        lines.append("-" * 52)
        lines.append(f"{'Overall status:':32s}{self.status}")
        lines.append("=" * 52)
        return "\n".join(lines)


def _flag(count: int, warn_at: int = 1, fail_at: Optional[int] = None) -> str:
    if fail_at is not None and count >= fail_at:
        return FAIL
    return WARNING if count >= warn_at else PASS


def validate_panel(
    df: pd.DataFrame,
    *,
    expected_symbols: Optional[List[str]] = None,
    stale_price_days: int = 5,
    extreme_return_abs: float = 0.25,
    max_missing_pct: float = 0.02,
    ohlc_tolerance: float = 1e-8,
    check_calendar: bool = True,
) -> DataHealthReport:
    rep = DataHealthReport()

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    rep.add(CheckResult("schema_missing_columns", FAIL if missing_cols else PASS,
                        len(missing_cols),
                        f"missing: {missing_cols}" if missing_cols else "schema ok"))
    if missing_cols:
        return rep

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    symbols = sorted(df["symbol"].unique())

    rep.meta["Symbols received"] = len(symbols)
    rep.meta["Rows"] = len(df)
    rep.meta["Start"] = str(df["date"].min().date())
    rep.meta["End"] = str(df["date"].max().date())

    # 1. duplicates ---------------------------------------------------------
    dup_mask = df.duplicated(subset=["symbol", "date"], keep=False)
    rep.add(CheckResult("duplicate_symbol_date", _flag(int(dup_mask.sum()), fail_at=1),
                        int(dup_mask.sum()), "duplicate (symbol, date) rows",
                        df.loc[dup_mask].head(10)))

    # 2. null prices --------------------------------------------------------
    price_cols = ["open", "high", "low", "close", "adjusted_close"]
    null_mask = df[price_cols].isna().any(axis=1)
    rep.add(CheckResult("null_prices", _flag(int(null_mask.sum())), int(null_mask.sum()),
                        "rows with a null price field", df.loc[null_mask].head(10)))

    # 3. non-positive prices ------------------------------------------------
    nonpos = (df[price_cols] <= 0).any(axis=1) & ~null_mask
    rep.add(CheckResult("non_positive_prices", _flag(int(nonpos.sum()), fail_at=1),
                        int(nonpos.sum()), "rows with price <= 0",
                        df.loc[nonpos].head(10)))

    # 4. OHLC consistency ---------------------------------------------------
    tol = ohlc_tolerance
    ok = ~null_mask & ~nonpos
    bad_ohlc = ok & (
        (df["high"] + tol < df["low"])
        | (df["high"] + tol < df["open"]) | (df["high"] + tol < df["close"])
        | (df["low"] - tol > df["open"]) | (df["low"] - tol > df["close"])
    )
    rep.add(CheckResult("impossible_ohlc", _flag(int(bad_ohlc.sum()), fail_at=1),
                        int(bad_ohlc.sum()), "high<low or open/close outside [low,high]",
                        df.loc[bad_ohlc].head(10)))

    # 5. negative / zero volume --------------------------------------------
    bad_vol = df["volume"] < 0
    rep.add(CheckResult("negative_volume", _flag(int(bad_vol.sum()), fail_at=1),
                        int(bad_vol.sum()), "rows with negative volume",
                        df.loc[bad_vol].head(10)))

    # 6. stale prices -------------------------------------------------------
    stale_rows, stale_total = [], 0
    for sym, g in df.sort_values("date").groupby("symbol", sort=False):
        c = g["close"].to_numpy(dtype="float64")
        if c.size < 2:
            continue
        same = np.isclose(c[1:], c[:-1], rtol=0, atol=1e-10)
        run, longest = 0, 0
        for i, s in enumerate(same):
            run = run + 1 if s else 0
            longest = max(longest, run)
            if run == stale_price_days - 1:
                stale_total += 1
                stale_rows.append({"symbol": sym,
                                   "date": g["date"].iloc[i + 1],
                                   "close": c[i + 1],
                                   "run_length": stale_price_days})
    rep.add(CheckResult("stale_prices", _flag(stale_total), stale_total,
                        f">= {stale_price_days} identical consecutive closes",
                        pd.DataFrame(stale_rows).head(10) if stale_rows else None))

    # 7. extreme returns ----------------------------------------------------
    d = df.sort_values(["symbol", "date"]).copy()
    d["logret"] = np.log(d["adjusted_close"]) - np.log(
        d.groupby("symbol", sort=False)["adjusted_close"].shift(1))
    extreme = d["logret"].abs() > extreme_return_abs
    rep.add(CheckResult("extreme_returns", _flag(int(extreme.sum())), int(extreme.sum()),
                        f"|log return| > {extreme_return_abs}",
                        d.loc[extreme, ["symbol", "date", "adjusted_close",
                                        "logret"]].head(10)))

    # 8. missing observations vs the union of observed sessions -------------
    all_dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    counts = df.groupby("symbol", sort=False)["date"].nunique()
    miss_pct = 1.0 - counts / len(all_dates)
    offenders = miss_pct[miss_pct > max_missing_pct]
    rep.add(CheckResult("symbols_with_missing_sessions", _flag(len(offenders)),
                        int(len(offenders)),
                        f"symbols missing > {max_missing_pct:.1%} of sessions",
                        offenders.rename("missing_pct").to_frame().head(10)
                        if len(offenders) else None))

    # 9. unexpectedly absent symbols ---------------------------------------
    if expected_symbols is not None:
        absent = sorted(set(expected_symbols) - set(symbols))
        rep.meta["Symbols expected"] = len(expected_symbols)
        rep.add(CheckResult("missing_symbols", _flag(len(absent), fail_at=1), len(absent),
                            f"expected but not received: {absent[:8]}"))
    else:
        rep.add(CheckResult("missing_symbols", PASS, 0, "no expected-symbol list supplied"))

    # 10. trading-calendar agreement ---------------------------------------
    if check_calendar:
        cal = trading_calendar(str(all_dates[0].date()), str(all_dates[-1].date()))
        unexpected = sorted(set(all_dates) - set(cal))
        absent_sessions = sorted(set(cal) - set(all_dates))
        n_cal = len(unexpected) + len(absent_sessions)
        rep.add(CheckResult(
            "calendar_mismatch", _flag(n_cal), n_cal,
            f"{len(absent_sessions)} expected sessions absent, "
            f"{len(unexpected)} unexpected sessions present"))
    else:
        rep.add(CheckResult("calendar_mismatch", PASS, 0, "calendar check disabled"))

    return rep

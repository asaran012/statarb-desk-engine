"""Explicit, logged cleaning.

The validator classifies; this module is the only place allowed to *change*
rows, and every change is appended to a :class:`CleaningLog` that ships with the
processed dataset.  "How many rows did you drop and why?" is a standard desk
question and the answer must be a file, not a recollection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class CleaningLog:
    actions: List[Dict[str, object]] = field(default_factory=list)

    def record(self, action: str, rows: int, detail: str = "") -> None:
        self.actions.append({"action": action, "rows": int(rows), "detail": detail})

    @property
    def total_rows_removed(self) -> int:
        return sum(a["rows"] for a in self.actions if a["action"].startswith("drop"))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.actions) if self.actions else pd.DataFrame(
            columns=["action", "rows", "detail"])

    def render(self) -> str:
        if not self.actions:
            return "CLEANING LOG: no modifications applied."
        lines = ["CLEANING LOG", "-" * 52]
        for a in self.actions:
            lines.append(f"{a['action']:<28s}{a['rows']:>8}  {a['detail']}")
        lines.append("-" * 52)
        lines.append(f"{'total rows removed':<28s}{self.total_rows_removed:>8}")
        return "\n".join(lines)


def clean_panel(df: pd.DataFrame, *, forward_fill_limit: int = 0
                ) -> tuple[pd.DataFrame, CleaningLog]:
    """Deterministic cleaning: dedupe -> drop unusable rows -> optional ffill.

    ``forward_fill_limit=0`` (the default) means *no* interpolation.  Filling a
    price gap manufactures a return of exactly zero, which biases volatility
    down and can create phantom mean reversion; we would rather carry the hole
    and let the universe filter drop the symbol.
    """
    log = CleaningLog()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    n0 = len(out)

    dup = out.duplicated(subset=["symbol", "date"], keep="last")
    if dup.any():
        out = out.loc[~dup]
        log.record("drop_duplicates", int(dup.sum()), "kept last observation")

    price_cols = ["open", "high", "low", "close", "adjusted_close"]
    bad = out[price_cols].isna().any(axis=1) | (out[price_cols] <= 0).any(axis=1)
    if bad.any():
        out = out.loc[~bad]
        log.record("drop_invalid_prices", int(bad.sum()), "null or non-positive price")

    if "volume" in out.columns:
        negv = out["volume"] < 0
        if negv.any():
            out.loc[negv, "volume"] = np.nan
            log.record("nullify_negative_volume", int(negv.sum()),
                       "volume set to NaN, row retained")

    if forward_fill_limit > 0:
        before = out[price_cols].isna().sum().sum()
        out = out.sort_values(["symbol", "date"])
        out[price_cols] = out.groupby("symbol", sort=False)[price_cols].ffill(
            limit=forward_fill_limit)
        after = out[price_cols].isna().sum().sum()
        log.record("forward_fill_prices", int(before - after),
                   f"limit={forward_fill_limit} (biases realised vol downward)")

    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    log.record("rows_in", n0, "")
    log.record("rows_out", len(out), "")
    return out, log

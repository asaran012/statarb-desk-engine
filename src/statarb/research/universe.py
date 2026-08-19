"""Research-universe construction.

Filters applied *inside each walk-forward training window* using only that
window's data.  Building the universe once over the full history would import
survivorship and liquidity information from the future into every earlier test
period -- a subtle leak that is easy to miss and fatal to the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class UniverseReport:
    eligible: List[str]
    stats: pd.DataFrame
    rejections: Dict[str, int]

    def render(self) -> str:
        lines = ["UNIVERSE CONSTRUCTION", "-" * 46,
                 f"{'candidates':<32s}{len(self.stats):>10}"]
        for k, v in self.rejections.items():
            lines.append(f"{'  rejected: ' + k:<32s}{v:>10}")
        lines.append(f"{'eligible':<32s}{len(self.eligible):>10}")
        return "\n".join(lines)


def symbol_statistics(panel: pd.DataFrame, price_field: str = "adjusted_close"
                      ) -> pd.DataFrame:
    d = panel.sort_values(["symbol", "date"]).copy()
    d["logret"] = np.log(d[price_field]) - np.log(
        d.groupby("symbol", sort=False)[price_field].shift(1))
    d["dollar_volume"] = d[price_field] * d["volume"]
    n_sessions = d["date"].nunique()
    g = d.groupby("symbol", sort=False)
    stats = pd.DataFrame({
        "observations": g["date"].nunique(),
        "avg_dollar_volume": g["dollar_volume"].mean(),
        "median_price": g[price_field].median(),
        "min_price": g[price_field].min(),
        "ann_volatility": g["logret"].std(ddof=1) * np.sqrt(252),
        "null_prices": g[price_field].apply(lambda s: int(s.isna().sum())),
    })
    stats["missing_pct"] = 1.0 - stats["observations"] / max(n_sessions, 1)
    if "sector" in panel.columns:
        stats["sector"] = panel.drop_duplicates("symbol").set_index("symbol")["sector"]
    return stats.sort_index()


def build_universe(panel: pd.DataFrame, *, min_history_days: int = 750,
                   min_average_dollar_volume: float = 5e6,
                   max_missing_pct: float = 0.02, min_price: float = 5.0,
                   price_field: str = "adjusted_close") -> UniverseReport:
    stats = symbol_statistics(panel, price_field)
    rej: Dict[str, int] = {}

    m_hist = stats["observations"] >= min_history_days
    rej["insufficient history"] = int((~m_hist).sum())
    m_liq = stats["avg_dollar_volume"] >= min_average_dollar_volume
    rej["illiquid"] = int((~m_liq).sum())
    m_miss = stats["missing_pct"] <= max_missing_pct
    rej["too many missing sessions"] = int((~m_miss).sum())
    m_px = stats["min_price"] >= min_price
    rej["price below floor"] = int((~m_px).sum())
    m_vol = stats["ann_volatility"].notna() & (stats["ann_volatility"] > 0)
    rej["degenerate volatility"] = int((~m_vol).sum())

    keep = m_hist & m_liq & m_miss & m_px & m_vol
    return UniverseReport(eligible=sorted(stats.index[keep].tolist()),
                          stats=stats, rejections=rej)

"""Loading and reshaping the tidy panel."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

PANEL_COLUMNS = ["date", "symbol", "open", "high", "low", "close",
                 "adjusted_close", "volume"]


def load_panel(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif path.suffix in (".csv", ".txt"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported extension: {path.suffix}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def save_panel(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def to_wide(df: pd.DataFrame, field: str = "adjusted_close") -> pd.DataFrame:
    """Tidy long -> wide (index=date, columns=symbol)."""
    w = df.pivot_table(index="date", columns="symbol", values=field, aggfunc="last")
    return w.sort_index()


def log_prices(wide: pd.DataFrame) -> pd.DataFrame:
    return np.log(wide)


def log_returns(wide: pd.DataFrame) -> pd.DataFrame:
    return np.log(wide).diff()


def slice_dates(df: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Inclusive date slice that works for both wide and tidy frames."""
    if isinstance(df.index, pd.DatetimeIndex):
        lo = pd.Timestamp(start) if start is not None else df.index.min()
        hi = pd.Timestamp(end) if end is not None else df.index.max()
        return df.loc[(df.index >= lo) & (df.index <= hi)]
    lo = pd.Timestamp(start) if start is not None else df["date"].min()
    hi = pd.Timestamp(end) if end is not None else df["date"].max()
    return df.loc[(df["date"] >= lo) & (df["date"] <= hi)]


def sector_map(df: pd.DataFrame) -> dict:
    if "sector" not in df.columns:
        return {}
    return (df.drop_duplicates("symbol").set_index("symbol")["sector"].to_dict())


def available_symbols(df: pd.DataFrame) -> List[str]:
    return sorted(df["symbol"].unique().tolist())

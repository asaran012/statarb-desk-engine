"""Data-source abstraction.

The engine only ever asks a :class:`DataSource` for a tidy panel, so swapping
Yahoo for a vendor feed (or a parquet dump of one) is a one-class change and
touches nothing downstream.

``YahooSource`` is implemented but **was not exercised**: the build environment
has no egress to any market-data host.  It is left in place, unmocked and
untested, and the README says so rather than pretending otherwise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .loader import load_panel
from .synthetic import generate_market

SCHEMA = ["date", "symbol", "open", "high", "low", "close", "adjusted_close", "volume"]


class DataSource(ABC):
    @abstractmethod
    def fetch(self, symbols: List[str], start: str, end: str) -> pd.DataFrame: ...

    @staticmethod
    def _conform(df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in SCHEMA if c not in df.columns]
        if missing:
            raise ValueError(f"source returned a panel missing {missing}")
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values(["date", "symbol"]).reset_index(drop=True)


class SyntheticSource(DataSource):
    """Deterministic labelled market. Used for every run in this repository."""

    def __init__(self, n_symbols: int = 60, seed: int = 20260818):
        self.n_symbols, self.seed = n_symbols, seed
        self.ground_truth = None

    def fetch(self, symbols, start: str, end: str) -> pd.DataFrame:
        panel, gt = generate_market(start, end, n_symbols=self.n_symbols, seed=self.seed)
        self.ground_truth = gt
        return self._conform(panel)


class ParquetSource(DataSource):
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def fetch(self, symbols, start, end) -> pd.DataFrame:
        df = self._conform(load_panel(self.path))
        if symbols:
            df = df[df["symbol"].isin(symbols)]
        return df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]


class CSVSource(ParquetSource):
    pass


class YahooSource(DataSource):
    """yfinance adapter.  NOT EXERCISED in this build (no network egress)."""

    def fetch(self, symbols: List[str], start: str, end: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as e:  # pragma: no cover - unreachable offline
            raise RuntimeError(
                "yfinance is not installed. `pip install '.[market]'` then rerun "
                "`python -m statarb.cli download`.") from e
        raw = yf.download(symbols, start=start, end=end, auto_adjust=False,
                          progress=False, group_by="ticker", threads=True)
        frames = []
        for sym in symbols:                                   # pragma: no cover
            try:
                sub = raw[sym].dropna(how="all").reset_index()
            except (KeyError, TypeError):
                continue
            sub.columns = [str(c).lower().replace(" ", "_") for c in sub.columns]
            sub = sub.rename(columns={"adj_close": "adjusted_close"})
            sub["symbol"] = sym
            frames.append(sub)
        if not frames:
            raise RuntimeError("Yahoo returned no usable rows for the requested symbols")
        out = pd.concat(frames, ignore_index=True)
        return self._conform(out[SCHEMA])


def make_source(kind: str, *, path: Optional[str] = None, n_symbols: int = 60,
                seed: int = 0) -> DataSource:
    kind = kind.lower()
    if kind == "synthetic":
        return SyntheticSource(n_symbols=n_symbols, seed=seed)
    if kind == "yahoo":
        return YahooSource()
    if kind in ("parquet", "csv"):
        if path is None:
            raise ValueError(f"source '{kind}' requires a path")
        return ParquetSource(path) if kind == "parquet" else CSVSource(path)
    raise ValueError(f"unknown data source '{kind}'")

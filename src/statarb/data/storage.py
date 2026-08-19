"""Parquet + DuckDB research store.

Parquet is the on-disk format; DuckDB is the query engine pointed at it.  The
point of the abstraction is that research queries are expressed in SQL against
columnar files rather than as ad-hoc pandas chains, which is how the analytics
half of a desk actually works.

``benchmark_aggregation`` measures one honest comparison (a per-symbol
aggregate) between pandas and DuckDB.  It reports wall-clock numbers from the
machine it ran on and claims nothing else.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional

import duckdb
import pandas as pd


class ResearchStore:
    def __init__(self, duckdb_path: str | Path, parquet_path: str | Path):
        self.duckdb_path = Path(duckdb_path)
        self.parquet_path = Path(parquet_path)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[duckdb.DuckDBPyConnection]:
        con = duckdb.connect(str(self.duckdb_path))
        try:
            con.execute(
                f"CREATE OR REPLACE VIEW prices AS "
                f"SELECT * FROM read_parquet('{self.parquet_path.as_posix()}')")
            yield con
        finally:
            con.close()

    # ---- demonstration queries -------------------------------------------
    def query(self, sql: str) -> pd.DataFrame:
        with self.connect() as con:
            return con.execute(sql).df()

    def daily_prices(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        return self.query(f"""
            SELECT date, symbol, open, high, low, close, adjusted_close, volume
            FROM prices
            WHERE symbol = '{symbol}' AND date BETWEEN '{start}' AND '{end}'
            ORDER BY date""")

    def symbol_statistics(self) -> pd.DataFrame:
        """Per-symbol liquidity / coverage / volatility summary, computed in SQL."""
        return self.query("""
            WITH r AS (
              SELECT symbol, date, adjusted_close, volume,
                     ln(adjusted_close / LAG(adjusted_close)
                        OVER (PARTITION BY symbol ORDER BY date)) AS logret,
                     adjusted_close * volume AS dollar_volume
              FROM prices)
            SELECT symbol,
                   COUNT(*)                       AS observations,
                   MIN(date)                      AS first_date,
                   MAX(date)                      AS last_date,
                   AVG(dollar_volume)             AS avg_dollar_volume,
                   MIN(adjusted_close)            AS min_price,
                   AVG(adjusted_close)            AS avg_price,
                   STDDEV_SAMP(logret) * SQRT(252) AS ann_volatility,
                   SUM(CASE WHEN adjusted_close IS NULL THEN 1 ELSE 0 END) AS null_prices
            FROM r GROUP BY symbol ORDER BY symbol""")

    def data_quality_summary(self) -> pd.DataFrame:
        return self.query("""
            SELECT COUNT(*) AS rows,
                   COUNT(DISTINCT symbol) AS symbols,
                   COUNT(DISTINCT date)   AS sessions,
                   MIN(date) AS start_date, MAX(date) AS end_date,
                   SUM(CASE WHEN close <= 0 THEN 1 ELSE 0 END) AS non_positive_close,
                   SUM(CASE WHEN high < low THEN 1 ELSE 0 END) AS bad_ohlc
            FROM prices""")

    def date_range(self) -> tuple[str, str]:
        df = self.query("SELECT MIN(date) a, MAX(date) b FROM prices")
        return str(df.iloc[0]["a"])[:10], str(df.iloc[0]["b"])[:10]

    def top_volume(self, n: int = 10) -> pd.DataFrame:
        return self.query(f"""
            SELECT symbol, AVG(adjusted_close * volume) AS avg_dollar_volume
            FROM prices GROUP BY symbol ORDER BY avg_dollar_volume DESC LIMIT {n}""")

    # ---- benchmark --------------------------------------------------------
    def benchmark_aggregation(self, panel: Optional[pd.DataFrame] = None,
                              repeats: int = 3) -> Dict[str, float]:
        """Per-symbol aggregate: pandas in-memory vs DuckDB-over-Parquet.

        The DuckDB timing *includes* reading the Parquet file from disk; the
        pandas timing assumes the frame is already materialised in RAM, which
        deliberately favours pandas.  Report it that way.
        """
        if panel is None:
            panel = pd.read_parquet(self.parquet_path)

        def _pandas() -> pd.DataFrame:
            d = panel.sort_values(["symbol", "date"]).copy()
            d["dv"] = d["adjusted_close"] * d["volume"]
            d["lr"] = d.groupby("symbol", sort=False)["adjusted_close"].transform(
                lambda s: (s / s.shift(1)).apply("log"))
            return d.groupby("symbol").agg(
                observations=("date", "size"), avg_dollar_volume=("dv", "mean"),
                vol=("lr", "std")).reset_index()

        sql = """
            WITH r AS (SELECT symbol, adjusted_close * volume AS dv,
                       ln(adjusted_close / LAG(adjusted_close)
                          OVER (PARTITION BY symbol ORDER BY date)) AS lr
                       FROM prices)
            SELECT symbol, COUNT(*) observations, AVG(dv) avg_dollar_volume,
                   STDDEV_SAMP(lr) vol FROM r GROUP BY symbol"""

        t_pd = []
        for _ in range(repeats):
            t0 = time.perf_counter(); _pandas(); t_pd.append(time.perf_counter() - t0)
        t_db = []
        for _ in range(repeats):
            t0 = time.perf_counter(); self.query(sql); t_db.append(time.perf_counter() - t0)

        pandas_s, duck_s = min(t_pd), min(t_db)
        return {"rows": int(len(panel)), "repeats": repeats,
                "pandas_seconds": round(pandas_s, 4),
                "duckdb_seconds": round(duck_s, 4),
                "speedup_duckdb_vs_pandas": round(pandas_s / duck_s, 3)}

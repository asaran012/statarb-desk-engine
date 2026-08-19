"""Typed, validated configuration.

Every strategy / research / execution constant lives in YAML and is parsed into
pydantic models here.  Nothing in ``src/statarb`` may hard-code a threshold that
belongs in this file: an interviewer's first probe on a backtest is usually
"where did that 2.0 come from?", and the answer must be "config, and it was
swept on validation data only".
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    source: Literal["synthetic", "yahoo", "csv", "parquet"] = "synthetic"
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    duckdb_path: Path = Path("data/processed/statarb.duckdb")
    start_date: str
    end_date: str
    n_symbols: int = 60
    price_field: str = "adjusted_close"


class ValidationConfig(BaseModel):
    stale_price_days: int = 5
    extreme_return_abs: float = 0.25
    max_missing_pct: float = 0.02
    ohlc_tolerance: float = 1e-8


class UniverseConfig(BaseModel):
    min_history_days: int = 750
    min_average_dollar_volume: float = 5_000_000
    max_missing_pct: float = 0.02
    min_price: float = 5.0


class ResearchConfig(BaseModel):
    correlation_threshold: float = 0.60
    cointegration_alpha: float = 0.05
    adf_alpha: float = 0.05
    fdr_alpha: float = 0.05
    stability_windows: int = 6
    stability_window_days: int = 252
    min_stability_pass_rate: float = 0.50
    half_life_min_days: float = 2.0
    half_life_max_days: float = 60.0
    max_pairs_per_window: int = 20
    restrict_to_same_sector: bool = True
    hedge_ratio_method: Literal["static", "rolling"] = "static"
    rolling_beta_window: int = 120


class StrategyConfig(BaseModel):
    zscore_lookback: int = Field(60, gt=1)
    entry_z: float = Field(2.0, gt=0)
    exit_z: float = Field(0.5, ge=0)
    stop_z: float = Field(4.0, gt=0)
    max_holding_days: int = 60

    @model_validator(mode="after")
    def _ordering(self) -> "StrategyConfig":
        if self.exit_z >= self.entry_z:
            raise ValueError("exit_z must be strictly below entry_z")
        if self.stop_z <= self.entry_z:
            raise ValueError("stop_z must be strictly above entry_z")
        return self


class ExecutionConfig(BaseModel):
    delay_days: int = Field(1, ge=1, description="Same-bar fills are forbidden: delay >= 1")
    fill_price: Literal["open", "close"] = "close"
    commission_bps: float = Field(1.0, ge=0)
    slippage_bps: float = Field(2.0, ge=0)

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.slippage_bps


class RiskConfig(BaseModel):
    initial_capital: float = 10_000_000
    max_pair_gross: float = Field(0.10, gt=0, le=1.0)
    max_symbol_exposure: float = Field(0.15, gt=0, le=1.0)
    max_portfolio_gross: float = Field(1.00, gt=0)
    trading_days_per_year: int = 252
    risk_free_rate: float = 0.0


class WalkForwardConfig(BaseModel):
    train_years: int = 3
    validation_years: int = 1
    test_years: int = 1
    step_years: int = 1


class ParamGrid(BaseModel):
    zscore_lookback: List[int] = [20, 40, 60, 90]
    entry_z: List[float] = [1.5, 2.0, 2.5]
    exit_z: List[float] = [0.0, 0.5, 1.0]

    def combinations(self) -> List[dict]:
        out = []
        for lb in self.zscore_lookback:
            for ez in self.entry_z:
                for xz in self.exit_z:
                    if xz >= ez:
                        continue  # invalid: exit band must sit inside entry band
                    out.append({"zscore_lookback": lb, "entry_z": ez, "exit_z": xz})
        return out


class ReportingConfig(BaseModel):
    figures_dir: Path = Path("reports/figures")
    trades_dir: Path = Path("reports/trades")
    reports_dir: Path = Path("reports")


class Config(BaseModel):
    seed: int = 20260818
    data: DataConfig
    validation: ValidationConfig = ValidationConfig()
    universe: UniverseConfig = UniverseConfig()
    research: ResearchConfig = ResearchConfig()
    strategy: StrategyConfig = StrategyConfig()
    execution: ExecutionConfig = ExecutionConfig()
    risk: RiskConfig = RiskConfig()
    walkforward: WalkForwardConfig = WalkForwardConfig()
    param_grid: ParamGrid = ParamGrid()
    cost_sensitivity_bps: List[float] = [0, 2, 5, 10, 20]
    reporting: ReportingConfig = ReportingConfig()

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh)
        return cls(**raw)

    def with_strategy(self, **overrides) -> "Config":
        """Return a copy with strategy parameters replaced (used by the sweep)."""
        new = self.model_copy(deep=True)
        new.strategy = StrategyConfig(**{**self.strategy.model_dump(), **overrides})
        return new

    def with_costs(self, commission_bps: float | None = None,
                   slippage_bps: float | None = None) -> "Config":
        new = self.model_copy(deep=True)
        d = self.execution.model_dump()
        if commission_bps is not None:
            d["commission_bps"] = commission_bps
        if slippage_bps is not None:
            d["slippage_bps"] = slippage_bps
        new.execution = ExecutionConfig(**d)
        return new


def load_config(path: str | Path = "config/default.yaml") -> Config:
    return Config.load(path)

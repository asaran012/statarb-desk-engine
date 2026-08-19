"""Walk-forward orchestration.

The contract, per window:

    TRAIN       universe, pair screening, cointegration, hedge ratios, stability
    VALIDATION  strategy-parameter sweep; the winner is frozen
    TEST        traded once, with everything above already fixed

Nothing selected or fitted in TRAIN or VALIDATION may be revised after seeing
TEST, and TEST bars are never an input to any estimator.  The enforcement is
structural: :class:`Window` hands each stage a *pre-sliced* frame, so a stage
physically cannot reach a bar outside its slice.

Windows are chained: the equity of window k+1 starts where window k ended, and
the book is force-liquidated at each boundary.  That mirrors a desk re-striking
its pair book annually, and it keeps per-window accounting closed so the
equity-reconciliation invariant is checkable window by window.

Parameter selection ranks validation configurations by Sharpe **with a minimum
trade count**, because a configuration that took three trades can post a
spectacular validation Sharpe and mean nothing.  Ties break toward the smaller
lookback and lower entry threshold (more trades, more evidence).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..data.loader import slice_dates, to_wide
from ..research.hedge_ratio import rolling_ols_hedge_ratio
from ..research.pair_selection import SelectionReport, select_pairs
from ..risk.metrics import performance_summary, sharpe_ratio, equity_to_returns
from .engine import BacktestResult, run_backtest

MIN_VALIDATION_TRADES = 8


@dataclass
class Window:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def label(self) -> str:
        return (f"W{self.index}: train {self.train_start.date()}..{self.train_end.date()}"
                f" | valid {self.valid_start.year} | test {self.test_start.year}")

    def to_dict(self) -> dict:
        return {"window": self.index,
                "train_start": str(self.train_start.date()),
                "train_end": str(self.train_end.date()),
                "valid_start": str(self.valid_start.date()),
                "valid_end": str(self.valid_end.date()),
                "test_start": str(self.test_start.date()),
                "test_end": str(self.test_end.date())}


def make_windows(dates: pd.DatetimeIndex, *, train_years: int = 3,
                 validation_years: int = 1, test_years: int = 1,
                 step_years: int = 1) -> List[Window]:
    start, end = dates.min(), dates.max()
    windows: List[Window] = []
    i, cursor = 0, start
    while True:
        tr_s = cursor
        tr_e = tr_s + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        va_s = tr_e + pd.Timedelta(days=1)
        va_e = va_s + pd.DateOffset(years=validation_years) - pd.Timedelta(days=1)
        te_s = va_e + pd.Timedelta(days=1)
        te_e = te_s + pd.DateOffset(years=test_years) - pd.Timedelta(days=1)
        if te_s > end:
            break
        te_e = min(te_e, end)
        if len(dates[(dates >= te_s) & (dates <= te_e)]) < 60:
            break
        windows.append(Window(i, tr_s, tr_e, va_s, va_e, te_s, te_e))
        i += 1
        cursor = cursor + pd.DateOffset(years=step_years)
    return windows


@dataclass
class WindowResult:
    window: Window
    selection: SelectionReport
    chosen_params: dict
    validation_grid: pd.DataFrame
    test: BacktestResult
    research_seconds: float
    truth_score: Optional[dict] = None


@dataclass
class WalkForwardResult:
    windows: List[WindowResult] = field(default_factory=list)
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict = field(default_factory=dict)
    runtime_seconds: float = 0.0

    @property
    def all_invariants_pass(self) -> bool:
        return all(w.test.invariants.all_pass for w in self.windows)

    def invariant_frame(self) -> pd.DataFrame:
        rows = []
        for w in self.windows:
            f = w.test.invariants.to_frame()
            f.insert(0, "window", w.window.index)
            rows.append(f)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


class ResearchCache:
    """Pair selection depends only on (training slice, mode) -- never on strategy
    parameters -- so it is computed once per window per mode and reused by the
    parameter sweep and by every ablation that shares that mode."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, str], Tuple[SelectionReport, float]] = {}

    def get(self, key: Tuple[int, str], factory) -> Tuple[SelectionReport, float]:
        if key not in self._cache:
            t0 = time.perf_counter()
            self._cache[key] = (factory(), 0.0)
            rep, _ = self._cache[key]
            self._cache[key] = (rep, time.perf_counter() - t0)
        return self._cache[key]


def _rolling_betas(panel: pd.DataFrame, pairs: pd.DataFrame, window: int,
                   price_field: str) -> Dict[str, pd.Series]:
    wide = to_wide(panel, price_field)
    log_px = np.log(wide)
    out: Dict[str, pd.Series] = {}
    for _, r in pairs.iterrows():
        a, b = r["symbol_a"], r["symbol_b"]
        if a in log_px.columns and b in log_px.columns:
            rb = rolling_ols_hedge_ratio(log_px[a], log_px[b], window=window)
            out[f"{a}/{b}"] = rb["beta"].ffill()
    return out


def run_walkforward(
    panel: pd.DataFrame,
    config: Config,
    *,
    selection_mode: str = "full",
    hedge_ratio_method: Optional[str] = None,
    risk_enabled: bool = True,
    sweep_parameters: bool = True,
    cache: Optional[ResearchCache] = None,
    ground_truth=None,
    verbose: bool = True,
) -> WalkForwardResult:
    t_start = time.perf_counter()
    cfg = config
    pf_field = cfg.data.price_field
    hedge_method = hedge_ratio_method or cfg.research.hedge_ratio_method
    cache = cache if cache is not None else ResearchCache()

    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    wide_close = to_wide(panel, pf_field)
    wide_open = to_wide(panel, "open") if "open" in panel.columns else wide_close
    dates = wide_close.index

    windows = make_windows(dates, train_years=cfg.walkforward.train_years,
                           validation_years=cfg.walkforward.validation_years,
                           test_years=cfg.walkforward.test_years,
                           step_years=cfg.walkforward.step_years)
    if not windows:
        raise ValueError("history too short for the configured walk-forward scheme")

    universe_kwargs = cfg.universe.model_dump()
    results: List[WindowResult] = []
    equity_pieces: List[pd.Series] = []
    trade_pieces: List[pd.DataFrame] = []
    equity_carry = cfg.risk.initial_capital

    for w in windows:
        if verbose:
            print(f"  {w.label()}", flush=True)

        # ---------------- TRAIN: research only ----------------------------
        train_panel = slice_dates(panel, w.train_start, w.train_end)
        rep, research_s = cache.get(
            (w.index, selection_mode),
            lambda: select_pairs(
                train_panel, price_field=pf_field,
                correlation_threshold=cfg.research.correlation_threshold,
                cointegration_alpha=cfg.research.cointegration_alpha,
                fdr_alpha=cfg.research.fdr_alpha, adf_alpha=cfg.research.adf_alpha,
                half_life_min=cfg.research.half_life_min_days,
                half_life_max=cfg.research.half_life_max_days,
                stability_windows=cfg.research.stability_windows,
                stability_window_days=cfg.research.stability_window_days,
                min_stability_pass_rate=cfg.research.min_stability_pass_rate,
                max_pairs=cfg.research.max_pairs_per_window,
                restrict_to_same_sector=cfg.research.restrict_to_same_sector,
                universe_kwargs=universe_kwargs, selection_mode=selection_mode))

        if rep.pairs.empty:
            if verbose:
                print("    no pairs selected; window skipped", flush=True)
            continue

        truth = None
        if ground_truth is not None:
            from ..research.pair_selection import score_against_truth
            truth = score_against_truth(rep.pairs, ground_truth)

        rolling_beta = None
        if hedge_method == "rolling":
            hist = slice_dates(panel, w.train_start, w.test_end)
            rolling_beta = _rolling_betas(hist, rep.pairs,
                                          cfg.research.rolling_beta_window, pf_field)

        # ---------------- VALIDATION: parameter sweep ---------------------
        grid_rows: List[dict] = []
        chosen = cfg.strategy.model_dump()
        if sweep_parameters:
            for combo in cfg.param_grid.combinations():
                trial = cfg.with_strategy(**combo)
                try:
                    r = run_backtest(
                        prices_close=wide_close, prices_open=wide_open,
                        pairs=rep.pairs, test_start=w.valid_start,
                        test_end=w.valid_end, config=trial,
                        starting_equity=cfg.risk.initial_capital,
                        risk_enabled=risk_enabled, rolling_beta=rolling_beta)
                except (ValueError, KeyError):
                    continue
                rets = equity_to_returns(r.equity["equity"])
                sh = sharpe_ratio(rets, cfg.risk.risk_free_rate,
                                  cfg.risk.trading_days_per_year)
                grid_rows.append({**combo, "sharpe": sh, "n_trades": len(r.trades),
                                  "net_pnl": r.net_pnl,
                                  "return": r.net_pnl / r.starting_equity})
            grid = pd.DataFrame(grid_rows)
            if not grid.empty:
                elig = grid[(grid["n_trades"] >= MIN_VALIDATION_TRADES)
                            & grid["sharpe"].notna()]
                if elig.empty:
                    elig = grid[grid["sharpe"].notna()]
                if not elig.empty:
                    elig = elig.sort_values(
                        ["sharpe", "zscore_lookback", "entry_z"],
                        ascending=[False, True, True])
                    best = elig.iloc[0]
                    chosen = {**chosen,
                              "zscore_lookback": int(best["zscore_lookback"]),
                              "entry_z": float(best["entry_z"]),
                              "exit_z": float(best["exit_z"])}
        else:
            grid = pd.DataFrame(grid_rows)

        # ---------------- TEST: one run, frozen everything ----------------
        test_cfg = cfg.with_strategy(**{k: chosen[k] for k in
                                        ("zscore_lookback", "entry_z", "exit_z")})
        test_res = run_backtest(
            prices_close=wide_close, prices_open=wide_open, pairs=rep.pairs,
            test_start=w.test_start, test_end=w.test_end, config=test_cfg,
            starting_equity=equity_carry, risk_enabled=risk_enabled,
            rolling_beta=rolling_beta)

        equity_carry = test_res.ending_equity
        eq = test_res.equity["equity"]
        equity_pieces.append(eq)
        if len(test_res.trades):
            t = test_res.trades.copy()
            t.insert(0, "window", w.index)
            trade_pieces.append(t)

        results.append(WindowResult(window=w, selection=rep, chosen_params=chosen,
                                    validation_grid=grid, test=test_res,
                                    research_seconds=research_s, truth_score=truth))
        if verbose:
            print(f"    pairs={len(rep.pairs)} params={{lb:{chosen['zscore_lookback']},"
                  f" entry:{chosen['entry_z']}, exit:{chosen['exit_z']}}} "
                  f"trades={len(test_res.trades)} "
                  f"net={test_res.net_pnl:+,.0f} "
                  f"invariants={'PASS' if test_res.invariants.all_pass else 'FAIL'}",
                  flush=True)

    if not results:
        raise RuntimeError("walk-forward produced no tradeable windows")

    equity = pd.concat(equity_pieces).sort_index()
    equity = equity[~equity.index.duplicated(keep="last")]
    trades = (pd.concat(trade_pieces, ignore_index=True) if trade_pieces
              else pd.DataFrame())
    traded_notional = sum(w.test.total_traded_notional for w in results)

    summary = performance_summary(equity, trades, traded_notional,
                                  cfg.risk.risk_free_rate,
                                  cfg.risk.trading_days_per_year)
    summary["windows"] = len(results)
    summary["selection_mode"] = selection_mode
    summary["hedge_ratio_method"] = hedge_method
    summary["risk_limits_enabled"] = risk_enabled
    summary["total_traded_notional"] = traded_notional

    return WalkForwardResult(windows=results, equity=equity, trades=trades,
                             summary=summary,
                             runtime_seconds=time.perf_counter() - t_start)

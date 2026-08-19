"""Backtest engine.

Event-ordered daily loop.  Within each bar the sequence is fixed and is the
whole ballgame:

    1. read the *already shifted* execution target for this bar
    2. close pairs whose target went flat            (fills at this bar's price)
    3. open pairs whose target went non-flat         (fills at this bar's price)
    4. mark the book to this bar's close

Step 1 consumes ``exec_target``, produced by
:func:`statarb.backtest.execution.apply_execution_delay`, which is the only
route from a signal to a position.  A target seen on bar t was computed from the
close of bar ``t - delay``, so no fill can ever use information from its own
bar.  ``tests/test_backtester.py::test_no_same_bar_execution`` pins this by
corrupting a single close and asserting the fill price is unchanged.

Signals are computed over ``[test_start - buffer, test_end]`` where ``buffer``
is the z-score lookback.  Seeding trailing statistics from the tail of the
*training* window is realistic (a live desk has yesterday's data) and leak-free
(training strictly precedes test).  Trades are only opened on test-window bars.

Two interchangeable inner loops
-------------------------------
``loop="reference"`` indexes pandas objects directly and is the readable
specification.  ``loop="fast"`` (default) flattens the same data to numpy
arrays, after profiling showed pandas scalar ``.at`` lookups consuming ~36% of
runtime.  They share all setup, sizing, risk and accounting code -- only the
lookup mechanics differ -- and ``test_fast_and_reference_engines_agree``
asserts the two produce identical trade ledgers and equity curves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..risk.limits import RiskLimits
from ..signals.spread import spread_dollar_legs
from ..signals.strategy import pair_signal_frame
from .accounting import (InvariantReport, check_costs_non_negative,
                         check_equity_reconciles, check_execution_after_signal,
                         check_exits_match_entries, check_flat_at_end,
                         check_gross_net_consistency,
                         check_no_position_without_trade,
                         check_pre_trade_gross_limit, check_timestamps_monotonic,
                         mtm_exposure_breaches)
from .costs import CostModel
from .execution import apply_execution_delay
from .portfolio import Portfolio


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    rejected: pd.DataFrame
    invariants: InvariantReport
    starting_equity: float
    ending_equity: float
    total_costs: float
    total_traded_notional: float
    pairs_traded: List[str] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def net_pnl(self) -> float:
        return self.ending_equity - self.starting_equity

    @property
    def gross_pnl(self) -> float:
        return float(self.trades["gross_pnl"].sum()) if len(self.trades) else 0.0


def _sizing_quantities(direction: int, beta: float, price_a: float, price_b: float,
                       notional: float, mode: str = "hedge_neutral"):
    """Quantities for one pair trade.

    ``hedge_neutral`` (default)
        Dollar exposure in B is ``-beta`` times that in A, so position P&L
        tracks ``V * dS`` where ``S`` is the log spread. Gross = ``notional``.
    ``dollar_neutral``
        Equal and opposite dollar legs. Simpler and exactly market-neutral in
        dollars, but only hedge-neutral when ``beta == 1``; otherwise the
        position retains factor exposure proportional to ``|1 - beta|``.
    """
    if mode == "dollar_neutral":
        v_a, v_b = notional / 2.0, -notional / 2.0
    else:
        v_a, v_b = spread_dollar_legs(price_a, price_b, beta, notional)
    return direction * v_a / price_a, direction * v_b / price_b


def _try_open(pf: Portfolio, limits: RiskLimits, cost_model: CostModel, cfg: Config,
              *, key: str, a: str, b: str, tgt: int, beta: float, pa: float,
              pb: float, date, signal_date, zscore: float, close_row,
              sizing_mode: str) -> None:
    """Size, risk-check and (if it survives) open one pair. Shared by both loops."""
    # Recomputed per candidate: each fill earlier in this bar has already
    # charged its cost against equity, so a value cached at the top of the bar
    # would overstate the available risk budget on any multi-entry day.
    equity_now = pf.equity(close_row)
    intended = cfg.risk.max_pair_gross * equity_now
    qty_a, qty_b = _sizing_quantities(tgt, beta, pa, pb, intended, sizing_mode)
    leg_notionals = {a: abs(qty_a * pa), b: abs(qty_b * pb)}
    gross_intended = sum(leg_notionals.values())

    # Size against equity NET of the cost this fill will incur. Costs are
    # charged at the fill, so a position sized to exactly the limit times
    # *pre-trade* equity breaches it against *post-trade* equity by precisely
    # the cost. Reserving it is conservative and removes the breach.
    equity_for_sizing = equity_now - cost_model.charge(gross_intended)
    scale, binding = limits.scale_for_trade(
        equity=equity_for_sizing, intended_notional=gross_intended,
        current_gross=pf.gross_exposure(close_row),
        symbol_exposures=pf.symbol_exposure(close_row),
        leg_notionals=leg_notionals)
    if scale <= 0.0:
        pf.reject(key, date, binding or "risk_limit", f"intended={gross_intended:,.0f}")
        return

    pf.open_pair(pair=key, symbol_a=a, symbol_b=b, direction=tgt,
                 qty_a=qty_a * scale, qty_b=qty_b * scale, price_a=pa, price_b=pb,
                 date=date, signal_date=signal_date, zscore=zscore, beta=beta,
                 notional=gross_intended * scale)
    pf.record_post_trade_exposure(close_row)
    if scale < 1.0:
        pf.reject(key, date, f"scaled_{binding}", f"scale={scale:.3f}")


def _loop_reference(pf, limits, cost_model, cfg, *, signal_frames, pair_meta,
                    prices_close, fill_src, test_idx, sizing_mode) -> None:
    """Readable pandas-indexed specification of the daily loop."""
    for date in test_idx:
        close_row = prices_close.loc[date]
        fill_row = fill_src.loc[date]

        for key in list(pf.open_trades.keys()):
            sf = signal_frames.get(key)
            if sf is None or date not in sf.index:
                continue
            if int(sf.at[date, "exec_target"]) != 0:
                continue
            m = pair_meta[key]
            pa, pb = float(fill_row[m["symbol_a"]]), float(fill_row[m["symbol_b"]])
            if not (np.isfinite(pa) and np.isfinite(pb)):
                continue
            pf.close_pair(key, price_a=pa, price_b=pb, date=date,
                          zscore=float(sf.at[date, "exec_zscore"]),
                          reason=sf.at[date, "exec_reason"] or "mean_reversion")

        for key, sf in signal_frames.items():
            if key in pf.open_trades or date not in sf.index:
                continue
            tgt = int(sf.at[date, "exec_target"])
            if tgt == 0:
                continue
            pos = sf.index.get_loc(date)
            if pos >= 1 and int(sf.iloc[pos - 1]["exec_target"]) == tgt:
                continue
            m = pair_meta[key]
            a, b = m["symbol_a"], m["symbol_b"]
            pa, pb = float(fill_row[a]), float(fill_row[b])
            if not (np.isfinite(pa) and np.isfinite(pb)) or pa <= 0 or pb <= 0:
                pf.reject(key, date, "bad_price", f"{a}={pa} {b}={pb}")
                continue
            bk = sf.at[date, "beta"]
            _try_open(pf, limits, cost_model, cfg, key=key, a=a, b=b, tgt=tgt,
                      beta=float(bk) if np.isfinite(bk) else m["beta"], pa=pa, pb=pb,
                      date=date, signal_date=sf.at[date, "signal_date"],
                      zscore=float(sf.at[date, "exec_zscore"]),
                      close_row=close_row, sizing_mode=sizing_mode)

        pf.mark(date, close_row)


def _loop_fast(pf, limits, cost_model, cfg, *, signal_frames, pair_meta,
               prices_close, fill_src, test_idx, sizing_mode) -> None:
    """Numerically identical loop over pre-flattened numpy arrays."""
    sig: Dict[str, dict] = {}
    for key, sf in signal_frames.items():
        sig[key] = {
            "offset": sf.index.get_loc(test_idx[0]),
            "target": sf["exec_target"].to_numpy(dtype="int8"),
            "zscore": sf["exec_zscore"].to_numpy(dtype="float64"),
            "beta": sf["beta"].to_numpy(dtype="float64"),
            "reason": sf["exec_reason"].to_numpy(dtype=object),
            "signal_date": sf["signal_date"].to_numpy(),
        }
    col = {s: i for i, s in enumerate(prices_close.columns)}
    close_arr = prices_close.to_numpy(dtype="float64")
    fill_arr = fill_src.to_numpy(dtype="float64")
    bar_pos = [prices_close.index.get_loc(d) for d in test_idx]
    pair_keys = list(signal_frames.keys())

    for i, date in enumerate(test_idx):
        t = bar_pos[i]
        close_row = {s: close_arr[t, j] for s, j in col.items()}
        fill_row = fill_arr[t]

        for key in list(pf.open_trades.keys()):
            s = sig.get(key)
            if s is None:
                continue
            k = s["offset"] + i
            if k < 0 or k >= s["target"].size or s["target"][k] != 0:
                continue
            m = pair_meta[key]
            pa = fill_row[col[m["symbol_a"]]]
            pb = fill_row[col[m["symbol_b"]]]
            if not (np.isfinite(pa) and np.isfinite(pb)):
                continue
            pf.close_pair(key, price_a=float(pa), price_b=float(pb), date=date,
                          zscore=float(s["zscore"][k]),
                          reason=str(s["reason"][k] or "mean_reversion"))

        for key in pair_keys:
            if key in pf.open_trades:
                continue
            s = sig[key]
            k = s["offset"] + i
            if k < 0 or k >= s["target"].size:
                continue
            tgt = int(s["target"][k])
            if tgt == 0 or (k >= 1 and int(s["target"][k - 1]) == tgt):
                continue
            m = pair_meta[key]
            a, b = m["symbol_a"], m["symbol_b"]
            pa, pb = float(fill_row[col[a]]), float(fill_row[col[b]])
            if not (np.isfinite(pa) and np.isfinite(pb)) or pa <= 0 or pb <= 0:
                pf.reject(key, date, "bad_price", f"{a}={pa} {b}={pb}")
                continue
            bk = s["beta"][k]
            _try_open(pf, limits, cost_model, cfg, key=key, a=a, b=b, tgt=tgt,
                      beta=float(bk) if np.isfinite(bk) else m["beta"], pa=pa, pb=pb,
                      date=date, signal_date=pd.Timestamp(s["signal_date"][k]),
                      zscore=float(s["zscore"][k]), close_row=close_row,
                      sizing_mode=sizing_mode)

        pf.mark(date, close_row)


def run_backtest(
    *,
    prices_close: pd.DataFrame,
    prices_open: pd.DataFrame,
    pairs: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    config: Config,
    starting_equity: Optional[float] = None,
    risk_enabled: bool = True,
    sizing_mode: str = "hedge_neutral",
    rolling_beta: Optional[Dict[str, pd.Series]] = None,
    liquidate_at_end: bool = True,
    loop: str = "fast",
) -> BacktestResult:
    """Trade a frozen pair set over one unseen window."""
    cfg = config
    cost_model = CostModel(cfg.execution.commission_bps, cfg.execution.slippage_bps)
    equity0 = float(starting_equity if starting_equity is not None
                    else cfg.risk.initial_capital)
    pf = Portfolio(equity0, cost_model)
    limits = RiskLimits(cfg.risk.max_pair_gross, cfg.risk.max_symbol_exposure,
                        cfg.risk.max_portfolio_gross, enabled=risk_enabled)

    log_px = np.log(prices_close)
    buffer = int(cfg.strategy.zscore_lookback + cfg.execution.delay_days + 5)
    idx = prices_close.index
    test_idx = idx[(idx >= test_start) & (idx <= test_end)]
    if len(test_idx) == 0:
        raise ValueError(f"no bars in test window {test_start}..{test_end}")
    first_pos = idx.get_loc(test_idx[0])
    warm_start = idx[max(0, first_pos - buffer)]

    signal_frames: Dict[str, pd.DataFrame] = {}
    pair_meta: Dict[str, dict] = {}
    for _, row in pairs.iterrows():
        a, b = row["symbol_a"], row["symbol_b"]
        if a not in log_px.columns or b not in log_px.columns:
            continue
        key = f"{a}/{b}"
        window = (log_px.index >= warm_start) & (log_px.index <= test_end)
        la, lb = log_px.loc[window, a], log_px.loc[window, b]
        bpath = apath = None
        if rolling_beta is not None and key in rolling_beta:
            bpath = rolling_beta[key].reindex(la.index)
            apath = pd.Series(0.0, index=la.index)
        sf = pair_signal_frame(
            la, lb, beta=float(row["beta"]), alpha=float(row.get("alpha", 0.0)),
            zscore_lookback=cfg.strategy.zscore_lookback,
            entry_z=cfg.strategy.entry_z, exit_z=cfg.strategy.exit_z,
            stop_z=cfg.strategy.stop_z,
            max_holding_days=cfg.strategy.max_holding_days,
            beta_path=bpath, alpha_path=apath)
        signal_frames[key] = apply_execution_delay(sf, cfg.execution.delay_days)
        pair_meta[key] = {"symbol_a": a, "symbol_b": b, "beta": float(row["beta"])}

    fill_src = prices_open if cfg.execution.fill_price == "open" else prices_close
    runner = _loop_fast if loop == "fast" else _loop_reference
    runner(pf, limits, cost_model, cfg, signal_frames=signal_frames,
           pair_meta=pair_meta, prices_close=prices_close, fill_src=fill_src,
           test_idx=test_idx, sizing_mode=sizing_mode)

    # ---- forced liquidation at window end -----------------------------------
    if liquidate_at_end and pf.open_trades:
        last = test_idx[-1]
        row = prices_close.loc[last].to_dict()
        for key in list(pf.open_trades.keys()):
            m = pair_meta[key]
            pf.close_pair(key, price_a=float(row[m["symbol_a"]]),
                          price_b=float(row[m["symbol_b"]]), date=last,
                          zscore=np.nan, reason="end_of_period")
        pf.equity_history[-1] = {
            **pf.equity_history[-1], "equity": pf.equity(row), "cash": pf.cash,
            "gross_exposure": pf.gross_exposure(row),
            "net_exposure": pf.net_exposure(row), "n_positions": len(pf.positions),
            "n_open_pairs": 0}

    trades = pf.trades_frame()
    equity = pf.equity_frame()
    ending = float(equity["equity"].iloc[-1]) if len(equity) else equity0

    inv = InvariantReport()
    inv.add("ending_equity == start + sum(net_pnl)",
            *check_equity_reconciles(equity0, ending, trades))
    inv.add("gross_pnl - cost == net_pnl", *check_gross_net_consistency(trades))
    inv.add("all costs >= 0", *check_costs_non_negative(trades))
    inv.add("equity timestamps monotonic", *check_timestamps_monotonic(equity["equity"]))
    inv.add("fills strictly after signal bar", *check_execution_after_signal(trades))
    inv.add("every exit matches an entry", *check_exits_match_entries(trades))
    inv.add("book flat after liquidation", *check_flat_at_end(pf.positions))
    inv.add("no position without a trade record",
            *check_no_position_without_trade(trades, pf.position_events))
    if risk_enabled:
        inv.add("post-fill gross exposure within limit",
                *check_pre_trade_gross_limit(pf.post_trade_gross_ratios,
                                             cfg.risk.max_portfolio_gross))
    mtm = mtm_exposure_breaches(equity, cfg.risk.max_portfolio_gross)

    return BacktestResult(
        trades=trades, equity=equity, rejected=pf.rejected_frame(), invariants=inv,
        starting_equity=equity0, ending_equity=ending, total_costs=pf.total_costs,
        total_traded_notional=pf.total_traded_notional,
        pairs_traded=sorted(signal_frames.keys()),
        meta={"test_start": str(pd.Timestamp(test_start).date()),
              "test_end": str(pd.Timestamp(test_end).date()),
              "n_pairs": len(signal_frames), "bars": len(test_idx),
              "risk_enabled": risk_enabled, "sizing_mode": sizing_mode,
              "loop": loop,
              "commission_bps": cfg.execution.commission_bps,
              "slippage_bps": cfg.execution.slippage_bps,
              "mtm_gross_breach_days": mtm["breach_days"],
              "mtm_peak_gross_ratio": mtm["peak_ratio"]})

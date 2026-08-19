"""Accounting invariants and risk limits.

These are the tests that would catch a silently wrong backtest. A P&L number
that does not reconcile to the equity curve is not a P&L number.
"""
import numpy as np
import pandas as pd
import pytest

from statarb.backtest.costs import CostModel
from statarb.backtest.engine import run_backtest
from statarb.risk.limits import RiskLimits


def _run(wide, selection, config, **kw):
    close, opn = wide
    return run_backtest(prices_close=close, prices_open=opn, pairs=selection.pairs,
                        test_start=pd.Timestamp("2020-01-01"),
                        test_end=pd.Timestamp("2020-12-31"), config=config, **kw)


def test_all_invariants_pass(wide, selection, config):
    r = _run(wide, selection, config)
    failed = [c["invariant"] for c in r.invariants.checks if c["status"] != "PASS"]
    assert not failed, f"invariant failures: {failed}"
    assert r.invariants.all_pass and r.invariants.n_pass == len(r.invariants.checks)


def test_equity_reconciles_to_trade_ledger(wide, selection, config):
    """final equity - starting equity must equal the sum of net trade P&L."""
    r = _run(wide, selection, config)
    start = float(r.equity["equity"].iloc[0])
    end = float(r.equity["equity"].iloc[-1])
    ledger = float(r.trades["net_pnl"].sum())
    assert (end - start) == pytest.approx(ledger, rel=1e-9, abs=1e-6), \
        f"equity moved {end - start:.6f} but ledger says {ledger:.6f}"


def test_gross_pnl_minus_costs_equals_net(wide, selection, config):
    r = _run(wide, selection, config)
    d = r.trades["gross_pnl"] - r.trades["cost"] - r.trades["net_pnl"]
    assert float(d.abs().max()) < 1e-9


def test_costs_are_never_negative(wide, selection, config):
    r = _run(wide, selection, config)
    assert float(r.trades["cost"].min()) >= 0.0


def test_gross_exposure_limit_never_breached_at_trade_time(wide, selection, config):
    r = _run(wide, selection, config)
    named = {c["invariant"]: c for c in r.invariants.checks}
    key = next(k for k in named if "gross" in k and "limit" in k)
    assert named[key]["status"] == "PASS", named[key]["detail"]


def test_every_entry_has_exactly_one_exit(wide, selection, config):
    r = _run(wide, selection, config)
    assert r.trades["exit_date"].notna().all()
    assert (pd.to_datetime(r.trades["exit_date"]) >=
            pd.to_datetime(r.trades["entry_date"])).all()


def test_equity_curve_has_no_gaps_or_nans(wide, selection, config):
    r = _run(wide, selection, config)
    assert r.equity["equity"].notna().all()
    assert r.equity.index.is_monotonic_increasing
    assert not r.equity.index.duplicated().any()


def test_cost_model_is_proportional_to_notional():
    cm = CostModel(commission_bps=1.0, slippage_bps=2.0)
    assert cm.charge(1_000_000.0) == pytest.approx(1_000_000.0 * 3.0 / 10_000.0)
    assert cm.charge(2_000_000.0) == pytest.approx(2 * cm.charge(1_000_000.0))
    assert cm.charge(0.0) == 0.0


def test_cost_model_ignores_sign_of_notional():
    cm = CostModel(commission_bps=1.0, slippage_bps=2.0)
    assert cm.charge(-500_000.0) == pytest.approx(cm.charge(500_000.0))


def test_risk_limit_scales_down_oversized_trade():
    lim = RiskLimits(max_pair_gross=0.10, max_symbol_exposure=0.15,
                     max_portfolio_gross=1.00)
    scale, binding = lim.scale_for_trade(
        equity=1_000_000.0, intended_notional=500_000.0, current_gross=0.0,
        symbol_exposures={}, leg_notionals={"A": 250_000.0, "B": -250_000.0})
    assert scale < 1.0 and binding is not None
    assert 500_000.0 * scale <= 0.10 * 1_000_000.0 + 1e-6


def test_risk_limit_leaves_small_trade_untouched():
    lim = RiskLimits(max_pair_gross=0.10, max_symbol_exposure=0.15,
                     max_portfolio_gross=1.00)
    scale, binding = lim.scale_for_trade(
        equity=1_000_000.0, intended_notional=50_000.0, current_gross=0.0,
        symbol_exposures={}, leg_notionals={"A": 25_000.0, "B": -25_000.0})
    assert scale == pytest.approx(1.0) and binding is None


def test_portfolio_gross_limit_binds_when_book_is_full():
    lim = RiskLimits(max_pair_gross=0.10, max_symbol_exposure=0.15,
                     max_portfolio_gross=1.00)
    scale, binding = lim.scale_for_trade(
        equity=1_000_000.0, intended_notional=100_000.0, current_gross=980_000.0,
        symbol_exposures={}, leg_notionals={"A": 50_000.0, "B": -50_000.0})
    assert scale < 1.0
    assert 980_000.0 + 100_000.0 * scale <= 1_000_000.0 + 1e-6


def test_symbol_concentration_limit_binds():
    lim = RiskLimits(max_pair_gross=0.50, max_symbol_exposure=0.15,
                     max_portfolio_gross=1.00)
    scale, binding = lim.scale_for_trade(
        equity=1_000_000.0, intended_notional=200_000.0, current_gross=0.0,
        symbol_exposures={"A": 100_000.0},
        leg_notionals={"A": 100_000.0, "B": -100_000.0})
    assert scale < 1.0
    assert 100_000.0 + 100_000.0 * scale <= 0.15 * 1_000_000.0 + 1e-6

"""Reconciliation must catch every break type it claims to catch."""
import pandas as pd
import pytest

from statarb.monitoring.reconciliation import (orders_from_trades, reconcile,
                                               simulate_executions)


@pytest.fixture
def blotter():
    exp = pd.DataFrame({
        "order_id": ["O1", "O2", "O3", "O4"],
        "date": pd.to_datetime(["2024-01-02"] * 4),
        "symbol": ["AAA", "BBB", "CCC", "DDD"],
        "side": ["BUY", "SELL", "BUY", "SELL"],
        "quantity": [100.0, 200.0, 50.0, 75.0],
        "limit_price": [10.0, 20.0, 30.0, 40.0],
    })
    exe = exp.copy()
    exe["fill_price"] = exe["limit_price"]
    return exp, exe[["order_id", "date", "symbol", "side", "quantity", "fill_price"]]


def test_clean_blotter_reconciles(blotter):
    exp, exe = blotter
    r = reconcile(exp, exe)
    assert r.status == "PASS"
    assert r.breaks.empty and r.matched == 4


def test_detects_missing_execution(blotter):
    exp, exe = blotter
    r = reconcile(exp, exe[exe["order_id"] != "O2"])
    assert r.count("missing_execution") == 1
    assert r.status == "FAIL"


def test_detects_unexpected_execution(blotter):
    exp, exe = blotter
    extra = exe.iloc[[0]].copy()
    extra["order_id"] = "O99"
    r = reconcile(exp, pd.concat([exe, extra], ignore_index=True))
    assert r.count("unexpected_execution") == 1
    assert r.status == "FAIL"


def test_detects_duplicate_execution(blotter):
    """A double fill must not be silently netted into a quantity mismatch."""
    exp, exe = blotter
    r = reconcile(exp, pd.concat([exe, exe.iloc[[1]]], ignore_index=True))
    assert r.count("duplicate_execution") == 1
    assert r.count("quantity_mismatch") == 0
    assert r.status == "FAIL"


def test_detects_quantity_mismatch(blotter):
    exp, exe = blotter
    bad = exe.copy()
    bad.loc[0, "quantity"] = 90.0
    r = reconcile(exp, bad)
    assert r.count("quantity_mismatch") == 1


def test_detects_symbol_mismatch(blotter):
    exp, exe = blotter
    bad = exe.copy()
    bad.loc[2, "symbol"] = "ZZZ"
    r = reconcile(exp, bad)
    assert r.count("symbol_mismatch") == 1
    assert r.status == "FAIL"


def test_detects_price_deviation(blotter):
    exp, exe = blotter
    bad = exe.copy()
    bad.loc[3, "fill_price"] = 44.0          # 1000 bps away
    r = reconcile(exp, bad, price_tolerance_bps=50.0)
    assert r.count("price_deviation") == 1


def test_side_flip_is_a_quantity_break(blotter):
    """Buying when told to sell is a signed-quantity break, not a match."""
    exp, exe = blotter
    bad = exe.copy()
    bad.loc[0, "side"] = "SELL"
    assert reconcile(exp, bad).count("quantity_mismatch") == 1


def test_small_slippage_within_tolerance_is_clean(blotter):
    exp, exe = blotter
    bad = exe.copy()
    bad["fill_price"] = bad["fill_price"] * 1.0002    # 2 bps
    assert reconcile(exp, bad, price_tolerance_bps=50.0).breaks.empty


def test_missing_columns_raise(blotter):
    exp, exe = blotter
    with pytest.raises(ValueError):
        reconcile(exp.drop(columns=["limit_price"]), exe)


def test_orders_from_trades_produces_four_legs_per_trade(wide, selection, config):
    from statarb.backtest.engine import run_backtest
    close, opn = wide
    r = run_backtest(prices_close=close, prices_open=opn, pairs=selection.pairs,
                     test_start=pd.Timestamp("2020-01-01"),
                     test_end=pd.Timestamp("2020-12-31"), config=config)
    orders = orders_from_trades(r.trades)
    assert len(orders) == 4 * len(r.trades)
    assert orders["order_id"].is_unique


def test_round_trip_through_simulated_broker_is_clean(wide, selection, config):
    from statarb.backtest.engine import run_backtest
    close, opn = wide
    r = run_backtest(prices_close=close, prices_open=opn, pairs=selection.pairs,
                     test_start=pd.Timestamp("2020-01-01"),
                     test_end=pd.Timestamp("2020-12-31"), config=config)
    orders = orders_from_trades(r.trades)
    fills = simulate_executions(orders, seed=3, slippage_bps=2.0)
    rec = reconcile(orders, fills, price_tolerance_bps=50.0)
    assert rec.status == "PASS", rec.render()

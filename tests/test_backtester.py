"""Backtest engine: look-ahead, cost monotonicity, engine equivalence."""
import numpy as np
import pandas as pd
import pytest

from statarb.backtest.engine import run_backtest


def _run(wide, selection, config, **over):
    close, opn = wide
    cfg = config
    if "strategy" in over:
        cfg = cfg.with_strategy(**over.pop("strategy"))
    if "costs" in over:
        cfg = cfg.with_costs(**over.pop("costs"))
    return run_backtest(prices_close=close, prices_open=opn, pairs=selection.pairs,
                        test_start=pd.Timestamp("2020-01-01"),
                        test_end=pd.Timestamp("2020-12-31"), config=cfg, **over)


def test_backtest_produces_trades(wide, selection, config):
    r = _run(wide, selection, config)
    assert len(r.trades) > 0
    assert len(r.equity) > 200


def test_no_same_bar_execution(wide, selection, config):
    """Corrupting a close must not change the fill price recorded on that date.

    This is the strongest available look-ahead test: if the engine ever peeked
    at the signal-date close to fill, the corruption would move fills.
    """
    close, opn = wide
    base = _run((close, opn), selection, config)
    assert len(base.trades) > 0
    t0 = base.trades.iloc[0]
    sym = t0["symbol_a"]
    sig_date = pd.Timestamp(t0["entry_date"]) - pd.Timedelta(days=1)
    close2 = close.copy()
    idx = close2.index[close2.index < pd.Timestamp(t0["entry_date"])]
    if len(idx):
        close2.loc[idx[-1], sym] *= 1.5   # corrupt the bar the signal was formed on
    after = _run((close2, opn), selection, config)
    assert after.trades.iloc[0]["entry_price_a"] == pytest.approx(
        t0["entry_price_a"]), "fill price moved when only a prior close changed"


def test_execution_is_strictly_after_signal(wide, selection, config):
    r = _run(wide, selection, config)
    assert (pd.to_datetime(r.trades["entry_date"]) >
            pd.to_datetime(r.trades["entry_signal_date"])).all()


def test_higher_costs_never_increase_net_pnl(wide, selection, config):
    """Monotonicity: costs are a strict drag, so net P&L cannot rise with bps."""
    nets = []
    for bps in (0.0, 2.0, 5.0, 10.0, 20.0):
        r = _run(wide, selection, config,
                 costs={"commission_bps": bps, "slippage_bps": 0.0})
        nets.append(float(r.equity["equity"].iloc[-1]))
    assert all(nets[i] >= nets[i + 1] - 1e-6 for i in range(len(nets) - 1)), nets


def test_zero_cost_equals_gross(wide, selection, config):
    r = _run(wide, selection, config,
             costs={"commission_bps": 0.0, "slippage_bps": 0.0})
    assert float(r.trades["cost"].sum()) == pytest.approx(0.0, abs=1e-9)
    assert float(r.trades["net_pnl"].sum()) == pytest.approx(
        float(r.trades["gross_pnl"].sum()), rel=1e-9)


def test_fast_and_reference_engines_agree(wide, selection, config):
    """Two independent implementations of the same spec must agree exactly."""
    a = _run(wide, selection, config, loop="reference")
    b = _run(wide, selection, config, loop="fast")
    assert len(a.trades) == len(b.trades)
    cols = ["gross_pnl", "cost", "net_pnl", "entry_z", "qty_a", "qty_b", "beta"]
    np.testing.assert_array_equal(a.trades[cols].to_numpy(dtype=float),
                                  b.trades[cols].to_numpy(dtype=float))
    np.testing.assert_array_equal(a.equity["equity"].to_numpy(),
                                  b.equity["equity"].to_numpy())
    assert list(a.trades["exit_reason"]) == list(b.trades["exit_reason"])


def test_no_trades_when_no_pairs(wide, selection, config):
    close, opn = wide
    empty = selection.pairs.iloc[0:0]
    r = run_backtest(prices_close=close, prices_open=opn, pairs=empty,
                     test_start=pd.Timestamp("2020-01-01"),
                     test_end=pd.Timestamp("2020-12-31"), config=config)
    assert len(r.trades) == 0
    assert float(r.equity["equity"].std()) == pytest.approx(0.0, abs=1e-9)


def test_book_is_flat_at_end(wide, selection, config):
    r = _run(wide, selection, config)
    assert float(r.equity["gross_exposure"].iloc[-1]) == pytest.approx(0.0, abs=1e-6)


def test_tighter_entry_threshold_trades_less(wide, selection, config):
    loose = _run(wide, selection, config, strategy={"entry_z": 1.5})
    tight = _run(wide, selection, config, strategy={"entry_z": 3.0})
    assert len(tight.trades) <= len(loose.trades)

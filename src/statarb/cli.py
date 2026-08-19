"""Command line interface.

    statarb synth       generate the labelled synthetic market
    statarb download    fetch real prices (requires network + [market] extra)
    statarb validate    run the data health report
    statarb research    run pair selection on one training window
    statarb backtest    run the full walk-forward
    statarb ablations   run the ablation study and cost sweep
    statarb monitor     produce the daily desk health report
    statarb failures    run the failure-injection demo
    statarb report      assemble figures and reports from a completed run
    statarb all         everything, in order
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config, load_config

DEFAULT_CONFIG = "config/default.yaml"


# --------------------------------------------------------------------------
def _panel(cfg: Config, force_synth: bool = False) -> pd.DataFrame:
    """Load the processed panel, generating it if absent."""
    from .data.loader import load_panel
    p = Path(cfg.data.processed_dir) / "panel.parquet"
    if p.exists() and not force_synth:
        return load_panel(p)
    return cmd_synth(cfg)


def cmd_synth(cfg: Config, **_) -> pd.DataFrame:
    from .data.loader import save_panel
    from .data.synthetic import generate_market
    t0 = time.time()
    panel, gt = generate_market(cfg.data.start_date, cfg.data.end_date,
                                cfg.data.n_symbols, cfg.seed)
    out = Path(cfg.data.processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_panel(panel, out / "panel.parquet")
    gt.save(out / "ground_truth.json")
    print(f"generated {len(panel):,} rows, {panel['symbol'].nunique()} symbols, "
          f"{panel['date'].nunique():,} sessions in {time.time()-t0:.2f}s")
    print(f"  -> {out/'panel.parquet'}")
    print(f"  -> {out/'ground_truth.json'} "
          f"({len(gt.cointegrated_pairs)} cointegrated, "
          f"{len(gt.correlated_only_pairs)} decoys, "
          f"{len(gt.regime_break_pairs)} regime-break)")
    return panel


def cmd_download(cfg: Config, **_):
    from .data.downloader import make_source
    from .data.loader import save_panel
    if not cfg.data.universe_file:
        raise ValueError(
            "data.universe_file is not set; real downloads need an explicit "
            "(symbol, sector) list, e.g. config/real.yaml + data/real_universe.csv")
    uni = pd.read_csv(cfg.data.universe_file)
    symbols = sorted(uni["symbol"].unique().tolist())
    sector_of = dict(zip(uni["symbol"], uni["sector"]))
    print(f"downloading {len(symbols)} symbols {cfg.data.start_date}..{cfg.data.end_date} "
          f"from source={cfg.data.source!r}")
    src = make_source(cfg.data.source, seed=cfg.seed)
    panel = src.fetch(symbols, cfg.data.start_date, cfg.data.end_date)
    panel["sector"] = panel["symbol"].map(sector_of)
    got = sorted(panel["symbol"].unique().tolist())
    missing = sorted(set(symbols) - set(got))
    if missing:
        print(f"  WARNING: {len(missing)} requested symbols returned no rows: {missing}")
    out = Path(cfg.data.processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_panel(panel, out / "panel.parquet")
    print(f"downloaded {len(panel):,} rows, {len(got)}/{len(symbols)} symbols "
          f"-> {out/'panel.parquet'}")


def cmd_validate(cfg: Config, **_):
    from .data.validator import validate_panel
    panel = _panel(cfg)
    rep = validate_panel(panel, expected_symbols=sorted(panel["symbol"].unique()))
    print(rep.render())
    return rep


def cmd_research(cfg: Config, **_):
    from .research.pair_selection import select_pairs, score_against_truth
    from .data.synthetic import GroundTruth
    panel = _panel(cfg)
    train_end = pd.Timestamp(cfg.data.start_date) + pd.DateOffset(
        years=cfg.walkforward.train_years)
    train = panel[panel["date"] < train_end]
    t0 = time.time()
    rep = select_pairs(train, max_pairs=cfg.research.max_pairs_per_window,
                       correlation_threshold=cfg.research.correlation_threshold,
                       universe_kwargs=cfg.universe.model_dump())
    print(f"research on {train['date'].min().date()}..{train['date'].max().date()} "
          f"in {time.time()-t0:.1f}s\n")
    print(rep.render() if hasattr(rep, "render") else "")
    for k, v in rep.funnel.items():
        print(f"  {k:<34s}{v:>8,}")
    for n in rep.notes:
        print(f"  note: {n}")
    gt_path = Path(cfg.data.processed_dir) / "ground_truth.json"
    if gt_path.exists():
        gt = GroundTruth.load(gt_path)
        score = score_against_truth(rep.pairs, gt)
        print("\n  selection vs ground truth:")
        for k, v in score.items():
            print(f"    {k:<32s}{v}")
    return rep


def cmd_backtest(cfg: Config, **kw):
    from .backtest.walkforward import run_walkforward
    from .data.synthetic import GroundTruth
    panel = _panel(cfg)
    gt_path = Path(cfg.data.processed_dir) / "ground_truth.json"
    ground_truth = GroundTruth.load(gt_path) if gt_path.exists() else None
    res = run_walkforward(panel, cfg, ground_truth=ground_truth, verbose=True)
    out = Path(cfg.reporting.trades_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.trades.to_csv(out / "trades.csv", index=False)
    res.equity.to_frame("equity").to_csv(out / "equity.csv")
    print("\n" + "-" * 60)
    for k, v in res.summary.items():
        print(f"  {k:<28s}{v if isinstance(v,str) else f'{v:,.4f}' if isinstance(v,float) else f'{v:,}'}")
    print("-" * 60)
    print(f"  wrote {out/'trades.csv'} and {out/'equity.csv'}")
    _write_report_inputs(cfg, res)
    return res


def _write_report_inputs(cfg: Config, res) -> None:
    """Persist the per-window artifacts `report` needs, straight off the
    WalkForwardResult that just computed them. `report` used to read these
    from files nothing wrote, so it silently rendered whatever a run months
    ago had left on disk."""
    out = Path(cfg.reporting.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    w0 = res.windows[0]

    rows = []
    for w in res.windows:
        win = w.window
        rows.append({
            "window": win.index,
            "train": f"{win.train_start.date()}..{win.train_end.date()}",
            "validate": f"{win.valid_start.date()}..{win.valid_end.date()}",
            "test": f"{win.test_start.date()}..{win.test_end.date()}",
            "n_pairs": len(w.selection.pairs),
            "params": (f"lb:{w.chosen_params['zscore_lookback']}, "
                      f"entry:{w.chosen_params['entry_z']}, "
                      f"exit:{w.chosen_params['exit_z']}"),
            "n_trades": len(w.test.trades),
            "net_pnl": w.test.net_pnl,
        })
    pd.DataFrame(rows).to_csv(out / "windows.csv", index=False)
    (out / "funnel.json").write_text(json.dumps(w0.selection.funnel, indent=2))
    (out / "truth_score.json").write_text(
        json.dumps(w0.truth_score or {}, indent=2, default=str))
    (out / "walkforward_summary.json").write_text(
        json.dumps(res.summary, indent=2, default=str))
    w0.selection.pairs.to_csv(out / "selected_pairs_w0.csv", index=False)
    w0.validation_grid.to_csv(out / "validation_grid.csv", index=False)


def cmd_ablations(cfg: Config, **_):
    from .research.ablations import (breakeven_cost_bps, cost_sensitivity,
                                     run_ablations)
    panel = _panel(cfg)
    out = Path(cfg.reporting.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    ab = run_ablations(panel, cfg)
    ab.to_csv(out / "ablations.csv", index=False)
    cs = cost_sensitivity(panel, cfg)
    cs.to_csv(out / "cost_sensitivity.csv", index=False)
    print(ab[["experiment", "trades", "cumulative_return", "sharpe",
              "max_drawdown"]].to_string(index=False))
    print(f"\nbreakeven cost: {breakeven_cost_bps(cs)}")
    return ab, cs


def cmd_monitor(cfg: Config, **_):
    from .backtest.engine import run_backtest
    from .data.loader import to_wide
    from .monitoring.data_health import data_health_check
    from .monitoring.reconciliation import (orders_from_trades, reconcile,
                                            simulate_executions)
    from .monitoring.strategy_health import (build_desk_report, risk_health_check,
                                             strategy_health_check,
                                             trading_health_check)
    from .research.pair_selection import select_pairs
    panel = _panel(cfg)
    train_end = pd.Timestamp(cfg.data.start_date) + pd.DateOffset(
        years=cfg.walkforward.train_years)
    rep = select_pairs(panel[panel["date"] < train_end],
                       max_pairs=cfg.research.max_pairs_per_window,
                       universe_kwargs=cfg.universe.model_dump())
    test_start = train_end + pd.DateOffset(years=cfg.walkforward.validation_years)
    test_end = test_start + pd.DateOffset(years=1)
    bt = run_backtest(prices_close=to_wide(panel), prices_open=to_wide(panel, "open"),
                      pairs=rep.pairs, test_start=test_start, test_end=test_end,
                      config=cfg)
    orders = orders_from_trades(bt.trades)
    fills = simulate_executions(orders, seed=cfg.seed)
    recon = reconcile(orders, fills)
    as_of = bt.equity.index[-1]
    states = _live_signal_states(panel, rep.pairs, as_of, cfg)
    d = data_health_check(panel, expected_symbols=sorted(panel["symbol"].unique()))
    s = strategy_health_check(signal_states=states, equity=bt.equity, as_of=as_of,
                              stop_z=cfg.strategy.stop_z)
    r = risk_health_check(bt.equity,
                          max_portfolio_gross=cfg.risk.max_portfolio_gross,
                          max_symbol_exposure=cfg.risk.max_symbol_exposure)
    t = trading_health_check(orders, fills, recon)
    desk = build_desk_report(as_of=as_of, data=d, strategy=s, risk=r, trading=t,
                             recon=recon)
    print(desk.render())
    print()
    print(recon.render())
    out = Path(cfg.reporting.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "daily_report.txt").write_text(desk.render() + "\n\n" + recon.render())
    orders.to_csv(out / "expected_orders.csv", index=False)
    fills.to_csv(out / "executed_orders.csv", index=False)
    (out / "daily_report.json").write_text(json.dumps(desk.to_dict(), indent=2,
                                                      default=str))
    print(f"\n  wrote {out/'daily_report.txt'}, expected_orders.csv, "
          f"executed_orders.csv")
    return desk


def _live_signal_states(panel: pd.DataFrame, pairs: pd.DataFrame,
                        as_of: pd.Timestamp, cfg: Config) -> pd.DataFrame:
    """Current z-score and desired position for every monitored pair.

    Recomputed from prices rather than read off the trade ledger, so the
    monitor reflects what the strategy *wants* right now, which is what a desk
    checks before the open.
    """
    from .data.loader import to_wide
    from .signals.spread import build_spread
    from .signals.strategy import generate_signals
    from .signals.zscore import rolling_zscore
    close = to_wide(panel).loc[:as_of]
    rows = []
    for _, p in pairs.iterrows():
        a, b = p["symbol_a"], p["symbol_b"]
        if a not in close.columns or b not in close.columns:
            continue
        spread = build_spread(np.log(close[a]), np.log(close[b]), float(p["beta"]))
        z = rolling_zscore(spread, cfg.strategy.zscore_lookback)["zscore"]
        sig = generate_signals(z, entry_z=cfg.strategy.entry_z,
                               exit_z=cfg.strategy.exit_z,
                               stop_z=cfg.strategy.stop_z)
        rows.append({"pair": f"{a}/{b}", "zscore": float(z.iloc[-1]),
                     "target": int(sig["target"].iloc[-1]),
                     "new_signal": bool(sig["entry_flag"].iloc[-1]),
                     "exiting": bool(sig["exit_flag"].iloc[-1])})
    return pd.DataFrame(rows)


def cmd_failures(cfg: Config, **_):
    from .backtest.engine import run_backtest
    from .data.loader import to_wide
    from .monitoring.failure_demo import render, run_failure_demo
    from .monitoring.reconciliation import orders_from_trades, simulate_executions
    from .research.pair_selection import select_pairs
    panel = _panel(cfg)
    train_end = pd.Timestamp(cfg.data.start_date) + pd.DateOffset(
        years=cfg.walkforward.train_years)
    rep = select_pairs(panel[panel["date"] < train_end], max_pairs=10,
                       universe_kwargs=cfg.universe.model_dump())
    test_start = train_end + pd.DateOffset(years=cfg.walkforward.validation_years)
    bt = run_backtest(prices_close=to_wide(panel), prices_open=to_wide(panel, "open"),
                      pairs=rep.pairs, test_start=test_start,
                      test_end=test_start + pd.DateOffset(years=1), config=cfg)
    orders = orders_from_trades(bt.trades)
    fills = simulate_executions(orders, seed=cfg.seed)
    df = run_failure_demo(panel, bt.equity, orders, fills)
    out = Path(cfg.reporting.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "failure_demo.csv", index=False)
    print(render(df))
    missed = df[df["result"] != "CAUGHT"]
    if len(missed):
        print("\nMISSED:")
        print(missed[["scenario", "expected", "observed"]].to_string(index=False))
        return df
    return df


def cmd_report(cfg: Config, **_):
    from .scripts_support import generate_full_report
    return generate_full_report(cfg)


def cmd_all(cfg: Config, **kw):
    cmd_synth(cfg)
    cmd_validate(cfg)
    cmd_research(cfg)
    cmd_backtest(cfg)
    cmd_monitor(cfg)
    cmd_failures(cfg)
    cmd_ablations(cfg)
    cmd_report(cfg)


COMMANDS = {"synth": cmd_synth, "download": cmd_download, "validate": cmd_validate,
            "research": cmd_research, "backtest": cmd_backtest,
            "ablations": cmd_ablations, "monitor": cmd_monitor,
            "failures": cmd_failures, "report": cmd_report, "all": cmd_all}


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="statarb", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg = cfg.model_copy(update={"seed": args.seed})
    COMMANDS[args.command](cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

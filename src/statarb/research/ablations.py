"""Ablation study.

Each experiment removes exactly one component and re-runs the identical
walk-forward protocol. The point is to establish that each piece of the
pipeline is load-bearing, rather than asserting it in a README.

A. correlation only vs full cointegration funnel
B. cointegration without vs with rolling stability screening
C. gross vs net of transaction costs
D. static vs rolling hedge ratio
E. with vs without risk limits

Experiments C, D and E reuse the ``full`` pair selection, so only A and B pay
for an extra research pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..backtest.walkforward import run_walkforward
from ..config import Config
from ..risk.metrics import performance_summary


@dataclass
class Experiment:
    key: str
    experiment: str
    hypothesis: str
    selection_mode: str = "full"
    overrides: Dict = field(default_factory=dict)


EXPERIMENTS: List[Experiment] = [
    Experiment("baseline", "Full pipeline (baseline)",
               "All components active."),
    Experiment("A_correlation_only", "A: correlation only (no cointegration)",
               "Correlation alone admits pairs with no equilibrium, so spreads "
               "drift and stops fire instead of reversions.",
               selection_mode="correlation_only"),
    Experiment("A_no_fdr", "A2: cointegration without FDR control",
               "Without multiple-testing control, ~5% of tested pairs enter on "
               "noise alone.",
               selection_mode="no_fdr"),
    Experiment("B_no_stability", "B: no rolling stability screen",
               "Full-sample cointegration admits pairs whose relationship broke "
               "mid-sample; stability screening should remove them.",
               selection_mode="no_stability"),
    Experiment("C_gross_no_costs", "C: gross of transaction costs",
               "Costs are a first-order drag on a high-turnover strategy.",
               overrides={"costs": {"commission_bps": 0.0, "slippage_bps": 0.0}}),
    # NOTE: the baseline uses a STATIC beta fitted on the training window
    # (config default), so the meaningful experiment is adding the rolling
    # re-estimation, not removing it. An earlier version of this file ran
    # "static" against a static baseline and produced bit-identical numbers -
    # a silent no-op that looked like a finding.
    Experiment("D_rolling_beta", "D: rolling hedge ratio (vs static baseline)",
               "Re-estimating beta on a rolling window should track drift in the "
               "relationship, but also injects estimation noise into the spread "
               "and forces re-hedging turnover.",
               overrides={"hedge_ratio_method": "rolling"}),
    Experiment("E_no_risk_limits", "E: no risk limits",
               "Removing concentration and gross caps raises exposure and "
               "should raise drawdown more than it raises return.",
               overrides={"risk_enabled": False}),
]


def _apply(config: Config, exp: Experiment) -> Config:
    cfg = config
    ov = exp.overrides
    if "costs" in ov:
        cfg = cfg.with_costs(**ov["costs"])
    return cfg


def run_ablations(panel: pd.DataFrame, config: Config, *,
                  experiments: Optional[List[Experiment]] = None,
                  verbose: bool = True) -> pd.DataFrame:
    rows = []
    for exp in (experiments or EXPERIMENTS):
        cfg = _apply(config, exp)
        if verbose:
            print(f"  running {exp.experiment} ...", flush=True)
        res = run_walkforward(
            panel, cfg, selection_mode=exp.selection_mode,
            hedge_ratio_method=exp.overrides.get("hedge_ratio_method"),
            risk_enabled=exp.overrides.get("risk_enabled", True),
            verbose=False)
        perf = res.summary
        rows.append({
            "key": exp.key, "experiment": exp.experiment,
            "pairs_selected": int(np.mean([len(w.selection.pairs) for w in res.windows])),
            "trades": int(len(res.trades)),
            "cumulative_return": perf["cumulative_return"],
            "cagr": perf["cagr"], "sharpe": perf["sharpe"],
            "max_drawdown": perf["max_drawdown"],
            "hit_rate": perf["hit_rate"],
            "total_costs": float(res.trades["cost"].sum()) if len(res.trades) else 0.0,
            "peak_gross_pct": float(np.nanmax(
                [w.test.equity["gross_exposure"].div(
                    w.test.equity["equity"]).max() for w in res.windows]) * 100),
            "hypothesis": exp.hypothesis,
        })
    return pd.DataFrame(rows)


def cost_sensitivity(panel: pd.DataFrame, config: Config,
                     bps_grid=(0.0, 2.0, 5.0, 10.0, 20.0),
                     verbose: bool = True) -> pd.DataFrame:
    """Sweep total round-trip cost and find the level that kills the edge."""
    rows = []
    for bps in bps_grid:
        cfg = config.with_costs(commission_bps=bps / 3.0, slippage_bps=bps * 2.0 / 3.0)
        if verbose:
            print(f"  cost sweep {bps:g} bps ...", flush=True)
        res = run_walkforward(panel, cfg, verbose=False)
        perf = res.summary
        rows.append({"total_bps": bps, "sharpe": perf["sharpe"],
                     "cumulative_return": perf["cumulative_return"],
                     "cagr": perf["cagr"], "max_drawdown": perf["max_drawdown"],
                     "trades": int(len(res.trades)),
                     "total_costs": float(res.trades["cost"].sum())
                     if len(res.trades) else 0.0})
    df = pd.DataFrame(rows)
    return df


def breakeven_cost_bps(df: pd.DataFrame) -> float:
    """Linear interpolation of the cost level at which cumulative return hits 0."""
    x, y = df["total_bps"].to_numpy(), df["cumulative_return"].to_numpy()
    for i in range(len(x) - 1):
        if y[i] > 0 >= y[i + 1]:
            return float(x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1]))
    return float("nan")

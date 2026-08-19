# Research Summary

> **These performance figures come from synthetic data.**
> This environment has no market-data egress (Yahoo and Stooq both return HTTP
> 403), so the pipeline was exercised on a labelled synthetic market generated
> by `statarb.data.synthetic`. That generator builds spreads from an
> Ornstein-Uhlenbeck process with known half-lives, which is exactly the model
> the strategy assumes. **Performance measured on data drawn from the
> strategy's own assumptions is not evidence of alpha, and the Sharpe ratio
> below must not be read as one.** The synthetic run demonstrates that the
> engine is correct, the invariants hold, and the research procedure recovers
> known ground truth. Real-market metrics are marked NOT YET MEASURED and can
> be produced with the reproduction command in the README.

## 1. Research funnel

Pair selection runs **inside each training window only**. Nothing below
was computed with knowledge of the test period.

| stage | pairs surviving |
|---|---|
| universe eligible | 60 |
| all possible pairs | 1,770 |
| corr >= 0.6 | 78 |
| cointegration tests run | 78 |
| naive p < 0.05 | 43 |
| BH-FDR q < 0.05 | 34 |
| residual ADF p < 0.05 | 34 |
| half-life in [2,60]d | 34 |
| stability >= 50% | 18 |
| selected | 18 |

## 2. Selection against known ground truth

The synthetic market is labelled, so the selection procedure can be
scored rather than admired.

| metric | value |
|---|---|
| selected | 18.0000 |
| true_positives | 9.0000 |
| decoys_selected | 0.0000 |
| decoy_admission_rate | 0.0000 |
| regime_break_selected | 3.0000 |
| unclassified_selected | 6.0000 |
| precision | 0.5000 |
| recall | 0.3000 |
| n_true_pairs | 30.0000 |
| n_decoy_pairs | 12.0000 |

The decoy admission rate is the number that matters: these are
pairs engineered to have *higher* return correlation than the true
cointegrated pairs while sharing no equilibrium. A funnel that admits
them is selecting on correlation, not cointegration.

## 3. Walk-forward protocol

Rolling 3y train / 1y validate / 1y test, stepping 1 year. Parameters
are chosen on validation and then frozen; the test window is touched
exactly once. Positions are liquidated at every window boundary so no
position spans a re-fit.

| window | train | validate | test | pairs | params | trades | net P&L |
|---|---|---|---|---|---|---|---|
| 0 | 2016-01-04..2019-01-03 | 2019-01-04..2020-01-03 | 2020-01-04..2021-01-03 | 18 | {"zscore_lookback": 90, "entry_z": 1.5, "exit_z": 1.0, "stop_z": 4.0, "max_holding_days": 60} | 177 | 1,190,951 |
| 1 | 2017-01-04..2020-01-03 | 2020-01-04..2021-01-03 | 2021-01-04..2022-01-03 | 15 | {"zscore_lookback": 40, "entry_z": 2.0, "exit_z": 1.0, "stop_z": 4.0, "max_holding_days": 60} | 120 | 713,666 |
| 2 | 2018-01-04..2021-01-03 | 2021-01-04..2022-01-03 | 2022-01-04..2023-01-03 | 8 | {"zscore_lookback": 60, "entry_z": 1.5, "exit_z": 0.5, "stop_z": 4.0, "max_holding_days": 60} | 81 | 913,034 |
| 3 | 2019-01-04..2022-01-03 | 2022-01-04..2023-01-03 | 2023-01-04..2024-01-03 | 7 | {"zscore_lookback": 90, "entry_z": 1.5, "exit_z": 0.5, "stop_z": 4.0, "max_holding_days": 60} | 50 | 713,210 |
| 4 | 2020-01-04..2023-01-03 | 2023-01-04..2024-01-03 | 2024-01-04..2025-01-03 | 7 | {"zscore_lookback": 60, "entry_z": 2.0, "exit_z": 0.5, "stop_z": 4.0, "max_holding_days": 60} | 38 | 732,576 |
| 5 | 2021-01-04..2024-01-03 | 2024-01-04..2025-01-03 | 2025-01-04..2025-12-31 | 5 | {"zscore_lookback": 60, "entry_z": 2.0, "exit_z": 0.5, "stop_z": 4.0, "max_holding_days": 60} | 25 | 461,491 |

## 4. Aggregate performance (SYNTHETIC)

| metric | value |
|---|---|
| n_days | 1,506 |
| n_trades | 491 |
| cumulative_return | 0.4725 |
| cagr | 0.0668 |
| annualized_volatility | 0.0185 |
| sharpe | 3.5216 |
| sortino | 6.3805 |
| max_drawdown | -0.0102 |
| calmar | 6.5303 |
| hit_rate | 0.7902 |
| profit_factor | 5.2910 |
| avg_holding_days | 10.0305 |
| turnover_annual | 15.3280 |

The Sharpe ratio here is an artifact of generating spreads from the
same OU model the strategy assumes. It is reported for completeness and
is **not** a performance claim.

## 5. Ablations

Each row removes exactly one component and re-runs the identical
walk-forward protocol.

| configuration | pairs | trades | cum. return | Sharpe | max DD | peak gross % |
|---|---|---|---|---|---|---|
| Full pipeline (baseline) | 10 | 491 | 0.4725 | 3.52 | -0.0102 | 100.0 |
| A: correlation only (no cointegration) | 20 | 342 | 0.1057 | 1.16 | -0.0177 | 104.7 |
| A2: cointegration without FDR control | 10 | 505 | 0.4815 | 3.50 | -0.0102 | 100.0 |
| B: no rolling stability screen | 20 | 876 | 0.8332 | 3.98 | -0.0208 | 103.8 |
| C: gross of transaction costs | 10 | 491 | 0.5163 | 3.79 | -0.0100 | 100.0 |
| D: rolling hedge ratio (vs static baseline) | 10 | 216 | 0.0327 | 0.23 | -0.0408 | 109.7 |
| E: no risk limits | 10 | 492 | 0.4777 | 3.54 | -0.0102 | 110.6 |

Hypotheses under test:

- **A: correlation only (no cointegration)** — Correlation alone admits pairs with no equilibrium, so spreads drift and stops fire instead of reversions.
- **A2: cointegration without FDR control** — Without multiple-testing control, ~5% of tested pairs enter on noise alone.
- **B: no rolling stability screen** — Full-sample cointegration admits pairs whose relationship broke mid-sample; stability screening should remove them.
- **C: gross of transaction costs** — Costs are a first-order drag on a high-turnover strategy.
- **D: rolling hedge ratio (vs static baseline)** — Re-estimating beta on a rolling window should track drift in the relationship, but also injects estimation noise into the spread and forces re-hedging turnover.
- **E: no risk limits** — Removing concentration and gross caps raises exposure and should raise drawdown more than it raises return.

## 6. Transaction-cost sensitivity

| round-trip bps | Sharpe | cumulative return | trades | total costs |
|---|---|---|---|---|
| 0 | 3.79 | 0.5163 | 491 | 0 |
| 2 | 3.61 | 0.4870 | 491 | 233,339 |
| 5 | 3.34 | 0.4441 | 491 | 574,412 |
| 10 | 3.16 | 0.3832 | 400 | 919,258 |
| 20 | 2.50 | 0.2732 | 290 | 1,284,479 |

Breakeven cost level: does not break even within the tested range.


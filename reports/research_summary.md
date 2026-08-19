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
| corr >= 0.6 | 75 |
| cointegration tests run | 75 |
| naive p < 0.05 | 45 |
| BH-FDR q < 0.05 | 40 |
| residual ADF p < 0.05 | 40 |
| half-life in [2,60]d | 40 |
| stability >= 50% | 14 |
| selected | 14 |

## 2. Selection against known ground truth

The synthetic market is labelled, so the selection procedure can be
scored rather than admired.

| metric | value |
|---|---|
| selected | 14 |
| true_positives | 7 |
| decoys_selected | 0 |
| decoy_admission_rate | 0.0000 |
| regime_break_selected | 3 |
| unclassified_selected | 4 |
| precision | 0.5000 |
| recall | 0.2333 |
| n_true_pairs | 30 |
| n_decoy_pairs | 12 |

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
| 0 | 2016-01-04..2019-01-03 | 2019-01-04..2020-01-03 | 2020-01-04..2021-01-03 | 14 | lb:90, entry:1.5, exit:1.0 | 139 | 981,554 |
| 1 | 2017-01-04..2020-01-03 | 2020-01-04..2021-01-03 | 2021-01-04..2022-01-03 | 16 | lb:60, entry:1.5, exit:1.0 | 179 | 1,616,919 |
| 2 | 2018-01-04..2021-01-03 | 2021-01-04..2022-01-03 | 2022-01-04..2023-01-03 | 13 | lb:90, entry:1.5, exit:0.5 | 100 | 1,309,004 |
| 3 | 2019-01-04..2022-01-03 | 2022-01-04..2023-01-03 | 2023-01-04..2024-01-03 | 9 | lb:90, entry:1.5, exit:0.5 | 65 | 1,004,066 |
| 4 | 2020-01-04..2023-01-03 | 2023-01-04..2024-01-03 | 2024-01-04..2025-01-03 | 10 | lb:90, entry:1.5, exit:0.5 | 79 | 1,736,584 |
| 5 | 2021-01-04..2024-01-03 | 2024-01-04..2025-01-03 | 2025-01-04..2025-12-31 | 11 | lb:90, entry:2.0, exit:0.5 | 46 | 1,182,181 |

## 4. Aggregate performance (SYNTHETIC)

| metric | value |
|---|---|
| n_days | 1,507 |
| n_trades | 608 |
| cumulative_return | 0.7830 |
| cagr | 0.1015 |
| annualized_volatility | 0.0241 |
| sharpe | 4.0353 |
| sortino | 7.1919 |
| max_drawdown | -0.0193 |
| calmar | 5.2481 |
| hit_rate | 0.8092 |
| profit_factor | 7.6000 |
| avg_holding_days | 10.3914 |
| turnover_annual | 19.0179 |

The Sharpe ratio here is an artifact of generating spreads from the
same OU model the strategy assumes. It is reported for completeness and
is **not** a performance claim.

## 5. Ablations

Each row removes exactly one component and re-runs the identical
walk-forward protocol.

| configuration | pairs | trades | cum. return | Sharpe | max DD | peak gross % |
|---|---|---|---|---|---|---|
| Full pipeline (baseline) | 12 | 608 | 0.7830 | 4.04 | -0.0193 | 102.9 |
| A: correlation only (no cointegration) | 20 | 305 | 0.0696 | 0.64 | -0.0552 | 115.1 |
| A2: cointegration without FDR control | 12 | 641 | 0.7012 | 3.75 | -0.0193 | 102.9 |
| B: no rolling stability screen | 19 | 779 | 1.1165 | 4.59 | -0.0110 | 102.9 |
| C: gross of transaction costs | 12 | 663 | 0.8554 | 4.29 | -0.0186 | 102.9 |
| D: rolling hedge ratio (vs static baseline) | 12 | 236 | -0.0974 | -0.67 | -0.1073 | 103.8 |
| E: no risk limits | 12 | 608 | 0.7925 | 4.02 | -0.0192 | 106.1 |

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
| 0 | 4.29 | 0.8554 | 663 | 0 |
| 2 | 4.12 | 0.8047 | 608 | 315,243 |
| 5 | 3.87 | 0.7405 | 608 | 772,750 |
| 10 | 3.44 | 0.6386 | 608 | 1,495,990 |
| 20 | 2.64 | 0.4266 | 395 | 1,877,424 |

Breakeven cost level: does not break even within the tested range.


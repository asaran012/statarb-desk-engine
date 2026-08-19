# Resume Metrics

Every number here came from an actual run. Numbers are split into two
groups, because only one group is safe to put on a resume.

## Safe to claim: engineering and statistical scale

These describe what was built and measured. They are reproducible and
do not depend on the data being real.

| metric | measured value |
|---|---|
| modules | 48 |
| accounting invariants | 9 |
| data validation checks | 11 |
| reconciliation break types | 6 |
| monitoring checks | 29 |
| engine speedup | 2.39x |
| walk-forward windows | 6 |
| symbols | 60 |
| cointegration tests per window | 78 |
| research funnel | 1,770 possible pairs -> 78 cointegration tests -> 18 selected |
| tests passing | 73 |
| failure scenarios detected | 12/12 |
| decoy admission rate | 0.0000 |

## NOT safe to claim: performance

| metric | synthetic value | real-market value |
|---|---|---|
| sharpe | 3.5216 | NOT YET MEASURED |
| cagr | 0.0668 | NOT YET MEASURED |
| cumulative_return | 0.4725 | NOT YET MEASURED |
| max_drawdown | -0.0102 | NOT YET MEASURED |
| hit_rate | 0.7902 | NOT YET MEASURED |

**Do not put the synthetic Sharpe on a resume.** It was produced on
spreads generated from an Ornstein-Uhlenbeck process, which is the
exact model the strategy assumes mean-reverts. The first question a
desk quant will ask is 'on what data?', and the honest answer ends the
conversation badly if the number is presented as alpha.

## Suggested resume bullets

Each of these is backed by a measured number above.

1. Built a statistical-arbitrage research platform in Python (48 modules, 73 tests): Engle-Granger cointegration with Benjamini-Hochberg FDR control across 1,770 candidate pairs, rolling-window stability screening, and a walk-forward harness with strict train/validate/test separation.

2. Implemented a vectorised backtesting engine validated by 9 accounting invariants (equity-to-ledger reconciliation, cost non-negativity, pre-trade gross limits, execution-strictly-after-signal); profiling identified pandas scalar indexing as 36% of runtime, and a NumPy rewrite delivered a 2.39x speedup (433ms -> 181ms) with output verified bit-identical against the reference implementation.

3. Built a desk monitoring layer with 11 data-quality checks and 6 trade-reconciliation break types, proven by a failure-injection harness in which 12/12 deliberately corrupted scenarios were detected; ablation showed correlation-only selection changed Sharpe from 3.52 to 1.16.

## One-line project title

StatArb Desk Engine — cointegration research, walk-forward backtesting and desk monitoring for equity pairs trading (Python)

## GitHub description (<= 150 chars)

Statistical arbitrage research platform: cointegration + FDR pair selection, walk-forward backtesting, and desk monitoring in Python.


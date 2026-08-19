# StatArb Desk Engine — Final Validation Report

Audit date: 2026-08-19. Machine: Apple M5 Pro, 24GB RAM, macOS (Darwin
25.5.0 arm64). Repository: `/Users/saranay12/Documents/statarb-desk-engine`
(no git history pre-audit; a repository was initialized at the start of this
audit at commit `27db6585a090ba39e6be210569fb0ce01849645a` to track every
change made during it).

```
OVERALL VERDICT

Project runs end-to-end:               YES
All tests pass:                         YES (73/73)
Real market data successfully processed: YES (69/69 symbols, 166,842 rows)
Walk-forward backtest completed:        YES, with a genuine negative result
                                         under production config (0 pairs
                                         selected in 6/6 windows -- see below)
Leakage audit passed:                   YES (5/5 mutation-test checks, real data)
Accounting audit passed:                YES (54/54 invariant checks, real data)
Fast/reference engines equivalent:      YES (bit-identical, real and synthetic)
Benchmarks reproducible:                YES (script included, 15 runs each)
Monitoring validated:                   YES (29 checks, 12/12 failure scenarios caught)
Reconciliation validated:               YES (6/6 break categories individually tested)

RESUME READY: YES WITH CAVEATS
```

The caveat is not a weakness in the engineering — it is the central honest
finding of this audit: **the production statistical-arbitrage strategy,
run with its exact shipped configuration against 69 real large-cap US
equities over 2017-2026, found zero pairs that survived cointegration + BH-
FDR + ADF + half-life + stability screening, in every one of its 6
walk-forward windows.** It never placed a real trade. This is not a bug —
it was independently reproduced, the BH-FDR implementation was verified
against `statsmodels`, and the zero-pair result (0-1 raw hits against ~68
candidates/window, below the ~3.4 expected under a complete null) is
statistically unsurprising for a diversified large-cap universe under strict
multiple-testing control. What can be claimed is the *engineering and
research-methodology* work end-to-end on real data; what cannot be claimed
is a positive (or any) real trading result under production settings. See
`resume_verified_metrics.md` for the exact split.

## What was actually run this session

- Full test suite, 6 times (73/73 passing every time; one isolated
  transient failure investigated and not reproduced, see `limitations.md`).
- Coverage report (48% overall; accounting/engine/portfolio all 89-99%).
- Real-data download (Yahoo via yfinance) for 69 hand-picked, diversified,
  liquid large-cap equities, 2017-01-03 to 2026-08-17 — after fixing a
  genuine plumbing bug that made this path completely non-functional before
  this audit.
- Real-data validation (11 checks; WARNING status from 12 extreme-return
  days and, after a genuine calendar-bug fix, 2 remaining documented one-off
  NYSE closures).
- Real-data research funnel on all 6 walk-forward training windows, under
  4 pre-existing selection modes (full / no_fdr / no_stability /
  correlation_only), with zero threshold changes.
- Full walk-forward backtest attempts under the production config and every
  pre-existing ablation (`baseline`, `A_correlation_only`, `A_no_fdr`,
  `B_no_stability`, `C_gross_no_costs`, `D_rolling_beta`,
  `E_no_risk_limits`) — 4 failed for the same underlying reason (0 pairs),
  3 succeeded and were fully reported, failures recorded rather than
  papered over.
- Cost-sensitivity sweep (0/2/5/10/20 bps) on the one ablation mode with
  trades in every window.
- A direct future-data mutation test across 4 independent computations
  (rolling hedge ratio, z-score, signal/target, full backtest) on real data.
- A from-scratch benchmark (fast vs reference engine, 5 warm-up + 10
  measured runs) on both real and synthetic data, with bit-for-bit
  equivalence checked before timing.
- Monitoring + failure-injection demo on real data (29 named checks, 12/12
  injected faults caught).
- Reconciliation break-category check (6 categories, each independently
  unit-tested).
- A DuckDB-vs-pandas benchmark on the real 166,842-row dataset.
- A lint pass (`ruff`, pyflakes rules only) — 26 genuine findings (25 unused
  imports, 1 unused variable), all fixed, full suite re-verified green
  after each fix.

## Bugs found and fixed during this audit

1. **`cmd_download` was completely broken** (`src/statarb/cli.py`):
   `make_source(cfg.data.source, cfg)` passed an extra positional argument to
   a keyword-only function signature, and `src.fetch()` was called with none
   of its three required arguments. This is why the pre-audit project's
   `FINAL_STATUS.md` was correct that the Yahoo path was "implemented but
   never exercised" — it could not have run. Fixed by wiring a proper
   ticker/sector universe file through to `fetch(symbols, start, end)`.
2. **NYSE trading calendar closed the market on a real trading day**
   (`src/statarb/data/calendar.py`): the Saturday-holiday observance rule
   was applied uniformly to all holidays, including New Year's Day, which
   incorrectly marked 2021-12-31 as a holiday. Real downloaded data proved
   NYSE was open that day (year-end settlement convention), while it was
   correctly closed on the Friday before a Saturday July 4th and a Saturday
   Christmas. Fixed by special-casing New Year's Day; `calendar_mismatch`
   warnings dropped from 3 to 2 (the 2 remaining are genuine, documented,
   un-modeled one-off closures, not bugs).
3. **25 unused imports + 1 unused local variable**, found by `ruff`
   (pyflakes rules), fixed mechanically, full suite re-verified after.

No strategy parameter (entry/exit threshold, lookback, pair-selection rule,
transaction cost, train/test period) was changed at any point in this audit.

## Files changed

See `git diff` (repository was initialized at the start of this audit
specifically to track this). Summary by category:

- **Bug fix**: `src/statarb/cli.py`, `src/statarb/data/downloader.py`
  (download plumbing); `src/statarb/data/calendar.py` (holiday logic).
- **Environment/config compatibility**: `src/statarb/config.py` (added
  `data.universe_file`, additive, defaults to `None`).
- **Real-data configuration (new, additive)**: `config/real.yaml`,
  `data/real_universe.csv`.
- **Benchmark/audit instrumentation (new)**: `scripts/benchmark_real_engine.py`,
  `scripts/leakage_mutation_test.py`, `scripts/run_real_data_audit.py`,
  `scripts/run_real_monitoring_and_failures.py`.
- **Lint/cleanup**: 25 unused-import removals + 1 unused-variable removal
  across `src/`, `scripts/`, `tests/` (mechanical, `ruff --fix`, zero
  behavior change, re-verified by full test suite).
- **Reporting**: everything under `reports/real/` and `reports/final_audit/`.

## Files to send for independent review

```
reports/final_audit/final_validation_report.md
reports/final_audit/resume_verified_metrics.md
reports/final_audit/real_data_summary.json
reports/final_audit/trade_metrics.json
reports/final_audit/benchmark_results.json
reports/final_audit/walk_forward_results.csv
reports/final_audit/cost_sensitivity.csv
reports/final_audit/leakage_audit.md
reports/final_audit/accounting_audit.md
reports/final_audit/statistical_audit.md
reports/final_audit/limitations.md
reports/final_audit/ablation_results.csv
reports/final_audit/pair_research.json
```

## Three candidate resume bullets (safe-to-claim metrics only)

**Research**: Built a statistical-arbitrage research pipeline (cointegration,
Engle-Granger, Benjamini-Hochberg FDR verified exact against `statsmodels`,
ADF stationarity, rolling stability screening) and ran it on 69 real US
equities across 10 sectors (2017-2026, 2,418 sessions) through 6 rolling
walk-forward windows with a structurally-enforced train/validation/test
split; independently confirmed 0/12 engineered decoy pairs pass on synthetic
ground truth and the FDR screen correctly rejects noise on real data (0
pairs survive against ~68 candidates/window, below the ~3.4 expected under a
null).

**Backtesting/systems**: Implemented a T+1-delayed-execution backtest engine
with 9 runtime accounting invariants (54/54 passing on real trade data
across 6 windows), pre-trade gross/symbol/pair exposure limits, and
transaction-cost modeling; verified bit-identical output between two
independent engine implementations (856/856 trades, exact equity match) at a
measured 7.49x speedup (reproducible 15-run benchmark); passed a direct
future-data mutation test across 4 independent computations; 73/73 unit
tests passing.

**Desk operations**: Built an 11-check data-validation layer, a 6-category
trade-reconciliation engine, and a 29-check daily monitoring report (data,
strategy, risk, execution) covering real trading data; independently
verified 12/12 deliberately-injected failure scenarios (bad prices, stale
feeds, missing/duplicate/mismatched fills, drawdown and exposure breaches)
are correctly detected.

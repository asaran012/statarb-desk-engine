# Resume-verified metrics

Every number below was produced by a command run during this audit
(2026-08-19, this repository, this machine). Where a number differs from
existing repo documentation, the difference and the reason are stated.

---

## A. Definitely safe to claim

### Codebase / testing

| Metric | Value | Command | Source |
|---|---|---|---|
| Python source files | 48 | `find src -name "*.py" \| wc -l` | `src/statarb/` |
| Python source lines | 5,570 | `find src -name "*.py" -exec cat {} + \| wc -l` | `src/statarb/` |
| Test files | 6 | `find tests -name "test_*.py" \| wc -l` | `tests/` |
| Test functions | 73 | `grep -rc "^def test_" tests/*.py` | `tests/` |
| Tests passing | 73/73 | `pytest tests -v` | `test_results.txt` |
| Test suite runtime | ~2.3-2.5s | `pytest tests -v` | `test_results.txt` |
| Overall statement coverage | 48% (2,904 stmts, 1,508 missed) | `pytest --cov=src/statarb --cov-report=term-missing` | `coverage.txt` |
| Accounting/engine coverage | `accounting.py` 89%, `engine.py` 90%, `portfolio.py` 99% | same | `coverage.txt` |

*Why defensible*: directly counted / directly executed, this session.

### Real-market dataset

| Metric | Value | Command | Source |
|---|---|---|---|
| Equities requested | 69 | `data/real_universe.csv` | static, hand-picked pre-download |
| Equities downloaded | 69/69 (100%) | `statarb download --config config/real.yaml` | `real_data_summary.json` |
| Sectors represented | 10 | `data/real_universe.csv` | — |
| Date range | 2017-01-03 -> 2026-08-17 | — | `real_data_summary.json` |
| Years spanned | 9.62 | — | `real_data_summary.json` |
| Trading sessions | 2,418 | — | `real_data_summary.json` |
| Raw rows | 166,842 | — | `real_data_summary.json` |
| Missing / duplicate observations | 0 / 0 | `statarb validate --config config/real.yaml` | `real_data_summary.json` |
| Data validation checks (real data) | 11 checks, status WARNING (2 categories) | same | `real_data_summary.json` |
| Parquet size on disk | 4,697,149 bytes (4.5 MB) | `du -h` | `real_data_summary.json` |
| DuckDB size on disk | 274,432 bytes | same | `real_data_summary.json` |

*Why defensible*: `data.source: yahoo` end-to-end run this session, real
vendor (Yahoo via yfinance 1.6.0), verified row/date/null counts by direct
pandas inspection, not read off a printed log.

### Real pair-research funnel (all 6 walk-forward training windows)

| Metric | Value | Command | Source |
|---|---|---|---|
| Candidate universe (eligible) | 69 (every window) | `select_pairs(...)` per window | `pair_research.json` |
| Theoretical pair count (69 choose 2) | 2,346 | `69*68/2` | — |
| Sector-restricted, corr>=0.6 candidates | 68 (window 0; 60-70 typical across windows) | `select_pairs` funnel | `pair_research.json` |
| Cointegration tests run | matches candidate count each window | Engle-Granger | `pair_research.json` |
| Naive p<0.05 hits | 0-1 per window | — | `pair_research.json` |
| BH-FDR q<0.05 survivors (production) | **0 in every one of 6 windows** | `statarb research`/`backtest --config config/real.yaml` | `pair_research.json`, `real_walkforward_summary.json` |

*Why defensible*: this is the actual, reproducible output of the shipped
pipeline against real data with zero threshold changes. It is a genuine,
negative finding, not a bug — see `statistical_audit.md`.

### Engineering invariants / controls

| Metric | Value | Command | Source |
|---|---|---|---|
| Accounting invariants (named, implemented) | 9 | code inspection, `accounting.py` | `accounting_audit.md` |
| Invariant checks passing on real data | 54/54 (9 invariants x 6 windows) | `scripts/run_real_data_audit.py` | `walk_forward_results.csv` |
| Data validation checks (named) | 11 | code inspection + `validate_panel` | `leakage_audit.md`/code |
| Reconciliation break categories (named) | 6 | code inspection, `reconciliation.py` | `accounting_audit.md` |
| Monitoring checks (named, live count) | 29 (data 12, strategy 6, risk 3, trading 8) | `scripts/run_real_monitoring_and_failures.py` | `monitoring_results.json` |
| Failure-injection scenarios | 12/12 detected, on real data | same | `reports/real/failure_demo_real.csv` |
| Leakage mutation test | 5/5 checks PASS (hedge ratio, z-score, target, equity, trades), real data | `scripts/leakage_mutation_test.py` | `leakage_mutation_test.json` |
| BH-FDR vs. `statsmodels` reference | exact match (200 hypotheses, identical rejects, q-values to 1e-16) | ad hoc verification script | `statistical_audit.md` |

*Why defensible*: every count above comes from either reading the source
directly (invariant/check/break-type names) or a live run this session
(pass/fail counts), never from documentation.

### Engine correctness & performance

| Metric | Value | Command | Source |
|---|---|---|---|
| Fast vs reference engine equivalence, real data | bit-identical: 856/856 trades, equity curves, exit reasons; max abs diff = 0.0 | `scripts/benchmark_real_engine.py` | `benchmark_results.json` |
| Fast vs reference speedup, real data (median of 10 runs, 5 warmup) | **7.49x** (643.3ms -> 85.9ms) | same | `benchmark_results.json` |
| Fast vs reference speedup, synthetic data | 7.46x (666.8ms -> 89.4ms), also bit-identical | same, `config/default.yaml` | `benchmark_results.json` |
| DuckDB vs pandas, real data (per-symbol aggregate, 10 reps) | 1.24x (14.5ms -> 11.8ms) | `ResearchStore.benchmark_aggregation` | `duckdb_benchmark.json` |

*Why defensible*: newly measured, this machine, this session, with
warm-up runs and equivalence checked bit-for-bit before timing.
**Supersedes** the previously documented, unreproducible 2.39x figure — see
Section C.

---

## B. Safe but requires context

| Metric | Value | Data regime | Context required |
|---|---|---|---|
| Sharpe 3.52, cum. return 47.3%, max DD -1.02% | as documented pre-audit | **SYNTHETIC DATA** | Spreads were generated from an Ornstein-Uhlenbeck process, the exact model the strategy assumes. This measures implementation correctness, not trading edge. Reconfirmed reproducible this session (`statarb research --config config/default.yaml` decoy_admission_rate=0.0 unchanged). |
| Sharpe -0.42, cum. return -8.58%, 398 trades, 2021-2026 | **REAL DATA, `A_correlation_only` ABLATION** | Cointegration + FDR screen deliberately removed. This is a diagnostic to exercise the engine on real trades, not the production strategy. It lost money. |
| Sharpe -0.27, cum. return -0.59%, 28 trades | **REAL DATA, `A_no_fdr` ablation**, only 3/6 windows had a pair | Thin sample; directionally consistent with correlation_only (both negative). |
| Sharpe 1.79, cum. return +1.35%, 10 trades, 1 window | **REAL DATA, `B_no_stability` ablation** | Single window, 10 trades. Not statistically meaningful; do not present as a result. |
| Production strategy real-market Sharpe/return/drawdown | **NOT MEASURABLE** | **REAL DATA, PRODUCTION CONFIG** | Zero pairs selected in 6/6 windows; the strategy never traded. This is the headline honest real-market finding, not a gap in measurement. |
| Cost sensitivity 0-20bps (correlation_only ablation, real data) | cum. return -6.6% (0bps) to -10.6% (20bps); 398->328 trades | **REAL DATA, ABLATION** | See `cost_sensitivity.csv`. Not run against the production config because production has 0 trades at every cost level (pair selection precedes cost application). |

---

## C. Do not claim

| Claim (as previously documented) | Status | Reason |
|---|---|---|
| "2.39x speedup (433ms -> 181ms), bit-identical" | **DO NOT CLAIM** | No benchmark script existed in the repo to reproduce it; hardware/dataset scale that produced it is undocumented. Superseded by this audit's 7.49x (real) / 7.46x (synthetic), both bit-identical, both with a reproducible script (`scripts/benchmark_real_engine.py`) and 10 measured + 5 warm-up runs. |
| "real-market Sharpe / CAGR / cumulative return / max drawdown / hit rate" as a single production number | **DO NOT CLAIM AS PRODUCTION PERFORMANCE** | The production configuration selected zero real pairs; there is no production trade blotter to compute these from. Only ablation-mode numbers exist, and they are explicitly ablations (Section B). |
| "Synthetic Sharpe of 3.52" as a resume performance figure | **DO NOT CLAIM** (unchanged from the project's own pre-audit position) | Measures the implementation against its own generating assumptions. The project's own README already says this and this audit agrees. |
| Any causal explanation for the 12 real extreme-return days | **DO NOT CLAIM CAUSALITY** | Flagged by the validator, not causally investigated (see `limitations.md`). |
| "18 pairs selected" / any specific real-data pair count under the production config | **DO NOT CLAIM** | Real production selection is 0 pairs in every window; 18 was a synthetic-data figure from a different (single-window) run. |

---

## D. Strongest resume metrics, ranked

1. **Built and independently re-verified a from-scratch Benjamini-Hochberg FDR correction against `statsmodels`** (exact match, 200 synthetic hypotheses) and used it in a real pair-discovery pipeline that correctly rejects false signal: 0/12 engineered decoy pairs admitted on synthetic ground truth; 0 pairs survive FDR on 6/6 real walk-forward windows against ~68 correlation-screened candidates/window (below the ~3.4 false positives expected under a complete null).
2. **Ran a full real-market pipeline** — 69 liquid US equities, 10 sectors, 2017-2026, 2,418 sessions, 166,842 rows, 0 missing/duplicate observations, sourced live via Yahoo Finance — through 6 rolling walk-forward windows with a strict train/validation/test split enforced structurally, not by convention.
3. **Passed a direct future-data mutation test** on real prices: corrupted every post-cutoff observation across all 69 symbols by 1.5-4x and confirmed rolling hedge ratio, z-score, signal, full-engine equity curve, and full-engine trade ledger were bit-for-bit unchanged before the cutoff.
4. **9 accounting invariants, 54/54 checks passing on real trades** across 6 walk-forward windows (equity reconciliation, gross/net/cost identity, execution-after-signal, exposure limits enforced pre-trade, forced liquidation, no orphan positions).
5. **Measured a 7.49x backtest engine speedup with bit-identical output** (856/856 trades, exact equity curve match) between two independent implementations of the same event loop, on real market data, with a reproducible 15-run (5 warm-up + 10 measured) benchmark script.
6. **73/73 unit tests passing**, including statistical controls (ADF null hypothesis, cointegration critical values, hedge-ratio/half-life recovery on known synthetic parameters).
7. **12/12 independently-injected failure scenarios detected** on real data across data-quality, execution-reconciliation, and risk-limit monitoring layers (29 named monitoring checks across 4 categories).
8. **Found, root-caused, and fixed 3 genuine bugs during the audit**: (a) the real-data download path was completely non-functional (wrong function arity, missing required arguments — confirmed via code, not assumption); (b) the hand-rolled NYSE trading calendar incorrectly closed the market on Dec 31 of a year whose New Year's Day fell on Saturday, contradicted by real trading data; (c) 25 unused-import lint findings across the codebase, fixed mechanically and re-verified against the full test suite.
9. **6 reconciliation break categories independently exercised** (missing/unexpected/duplicate execution, quantity/symbol mismatch, price deviation) and unit-tested individually.
10. **Ran a genuine Pandas-vs-DuckDB analytical benchmark** on the real 166,842-row dataset (not a synthetic toy), reporting an honest, unglamorous 1.24x — refused to inflate a modest result.

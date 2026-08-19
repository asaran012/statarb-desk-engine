# Limitations

Honest list of what would have to be true, or what remains unverified, before
any number in this audit should be trusted beyond what is claimed for it.

## Real-market performance is not a validated trading result

The **production configuration** (cointegration + BH-FDR + ADF + half-life +
stability, all thresholds exactly as shipped in `config/real.yaml`, no value
changed for this audit) selected **zero pairs in all 6 real walk-forward
windows** (2017-2026, 69 large-cap US equities, 10 sectors). There is no
real-market Sharpe, return, or drawdown number for the production strategy —
not because it lost money, but because it never traded. Any Sharpe/return
number attributed to "real data" in this repository comes from the
`A_correlation_only` **ablation** (cointegration and FDR removed), which is
explicitly a diagnostic, not the strategy, and it lost money
(cum. return -8.6%, Sharpe -0.42, 2021-2026).

## Universe construction

69 tickers were hand-picked *before* any download or backtest, for liquidity
and sector diversity, not filtered by any result. But the universe is
**current, listed, liquid large caps** — every company that was delisted,
acquired, or failed since 2017 is absent by construction. This is
survivorship bias and this audit cannot correct it; it can only disclose it
(the project's own `data/README.md` flagged this risk before real data was
ever available).

## Small real signal, wide confidence

With ~68 correlation-screened candidates per window and 0-1 naive
significant hits, the real-data funnel result is consistent with "no
detectable cointegration signal in this universe/period," not with "we
proved there is none." A different universe (smaller-cap, more sector pairs,
non-US, futures/ETF pairs) or a longer/shorter training window could produce
different results. This audit did not vary the universe, because doing so
after seeing the zero-pair result would be exactly the kind of post-hoc
retuning Rule 3 of this audit forbids.

## Ablation results are thin samples in some modes

`B_no_stability` (stability screen removed) produced pairs in only 1 of 6
windows (10 trades total). A positive Sharpe (1.79) on 10 trades in one
window is not evidence of anything and must not be reported as a finding —
it is noted in `ablation_results.csv` for completeness, not endorsed.
`A_no_fdr` produced pairs in only 3 of 6 windows (28 trades total) — thin,
though directionally consistent with `A_correlation_only` (both net
negative).

## Benchmark numbers are single-machine

The 7.49x (real data) / 7.46x (synthetic data) fast-vs-reference speedup was
measured once, on one machine (Apple M5 Pro, 24GB, this audit's session),
with 5 warm-up + 10 measured runs. It supersedes the previously documented
2.39x figure (unreproducible: unknown hardware, unknown dataset scale, no
script in the repo to regenerate it) but is itself a single-machine number,
not a guarantee of the same ratio elsewhere. Both loops were confirmed
bit-identical on both datasets, which is the load-bearing claim; the exact
multiplier is secondary and hardware-dependent.

## Extreme-return warnings on real data were not causally investigated

12 real-data sessions crossed the |log-return| > 0.25 threshold (COVID-week
energy names, single-name earnings shocks, etc. — see `real_data_summary.json`
for dates). Per this audit's own rule (Phase 19), no causal story is asserted
for any of them beyond what is directly checkable from the price series
itself; they are flagged, not explained.

## Data vendor risk

Real prices came from Yahoo Finance via `yfinance` in a single
download session. Adjusted-close history is retroactively restated by
vendors after corporate actions, so re-running `statarb download` next month
is not guaranteed to reproduce today's exact numbers bit-for-bit (the
project's own `data/README.md` says this and this audit did not test it,
since it would require re-downloading on a future date).

## `YahooSource` had a real, now-fixed plumbing bug

Before this audit, `cmd_download` called `make_source(cfg.data.source, cfg)`
(wrong arity) and `src.fetch()` (missing required arguments) — the real-data
path had never actually been exercised, exactly as the pre-existing
`FINAL_STATUS.md` disclosed. Both are fixed in this audit
(`src/statarb/cli.py`, `src/statarb/data/downloader.py`) and now verified
working end-to-end (69/69 symbols downloaded). This means the pre-audit
project's "real-market metrics: NOT YET MEASURED" status was accurate for a
reason beyond "no network access" — the code to do it did not run either.

## One flaky test observation

`tests/test_backtester.py::test_no_same_bar_execution` failed once, in a run
executed concurrently with another CPU-bound process (a real-data `validate`
call in the same batch). It passed on 2 immediate reruns in isolation and on
5 subsequent full-suite runs (73/73 each time). Root cause not identified;
most likely explanation is resource contention affecting floating-point
reduction order in a multi-threaded BLAS call, not a logic bug — but this is
a plausible explanation, not a proven one. Documented rather than dismissed.

## Coverage gaps

Unit-test coverage is 48% of `src/statarb` by statement count. The 0%-
covered modules (`walkforward.py`, `cli.py`, `cleaner.py`, `storage.py`,
`ablations.py`, `reporting/*`, `risk/metrics.py`, `risk/exposure.py`,
`monitoring/data_health.py`, `monitoring/failure_demo.py`,
`monitoring/strategy_health.py`, `scripts_support.py`) are exercised by this
audit's end-to-end CLI/script runs (which all completed successfully and
produced internally-consistent output), not by `pytest`. That is evidence
these modules work on the paths actually exercised — the specific inputs
used in this audit — not a substitute for direct unit tests of their edge
cases.

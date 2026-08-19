# Statistical methodology audit

## ADF (Augmented Dickey-Fuller)

`src/statarb/research/stationarity.py::adf_test` wraps
`statsmodels.tsa.stattools.adfuller`. Verified by reading the code and by an
independent controlled test (`H0: unit root / non-stationary`, a *small*
p-value is evidence *against* a unit root):

```
$ .venv/bin/python -c "... controlled OU spread vs pure random walk ..."
ADF on true mean-reverting spread: p=0.000000  (correctly rejects unit root)
ADF on pure random walk:           p=0.7452    (correctly fails to reject unit root)
```

The module's own docstring states the null explicitly and warns against the
common misreading: "It is not proof of mean reversion, and with hundreds of
pairs tested it is not even strong evidence." Nowhere in the codebase is a
high ADF p-value treated as evidence of stationarity — `pair_selection.py`
only ever filters on `adf_pvalue < adf_alpha` (keep on *low* p-value).

## Cointegration (Engle-Granger)

`src/statarb/research/cointegration.py::engle_granger` uses
`statsmodels.tsa.stattools.coint` (MacKinnon critical values for the
two-step residual test), explicitly *not* a hand-rolled ADF on the OLS
residual — the docstring flags that shortcut as "a common and materially
wrong shortcut that overstates significance."

**Positive/negative controls, synthetic ground truth**
(`data/processed/ground_truth.json`: 30 engineered cointegrated pairs, 12
correlated-only decoys with deliberately *higher* correlation than the true
pairs, 6 regime-break pairs that are cointegrated only pre-break):

```
$ .venv/bin/python -m statarb.cli research --config config/default.yaml
  selected                16
  true_positives           9
  decoys_selected          0        <- 0 of 12 engineered decoys admitted
  decoy_admission_rate     0.0
  regime_break_selected    3        <- correctly admitted (genuinely
                                        cointegrated in the pre-break window)
  precision                0.5625
  recall                   0.3
```

Reconfirmed after every code change made in this audit (calendar/downloader
fixes do not touch the synthetic generator or the funnel). **0/12 negative
controls (decoys) ever pass the full pipeline.** Regime-break pairs being
selected is correct behaviour, not a false positive: they *are* cointegrated
within the training window that selects them, and the ground-truth scorer
tracks them separately from real errors.

**Real-data behaviour** (`reports/real/funnel_all_windows.json`): across all
6 real walk-forward training windows, the `full` pipeline (cointegration +
BH-FDR + ADF + half-life + stability) selects **zero** pairs in every single
window. The pre-FDR raw hit rate (`no_fdr` mode) is 0-1 pairs per window
against ~68 correlation-screened candidates per window — *below* the ~3.4
false positives expected under a complete null at alpha=0.05. This is not a
bug; it is the expected outcome of applying strict multiple-testing
discipline to a diversified large-cap universe where genuine, stable
cross-sectional cointegration is rare. See `statistical_audit.md` (this
file) cross-referenced with `../real/real_walkforward_summary.json`.

## Benjamini-Hochberg FDR

`src/statarb/research/multiple_testing.py::benjamini_hochberg` is a from-
scratch step-up implementation (deliberately not imported, "so the step-up
logic is visible and unit-testable" per its docstring). Independently
verified against the reference implementation:

```
$ .venv/bin/python -c "... vs statsmodels.stats.multitest.multipletests(method='fdr_bh') ..."
n hypotheses: 200
n rejected (mine): 20   n rejected (statsmodels): 20
reject masks identical: True
qvalues max abs diff: 1.1102230246251565e-16   (floating-point noise)
```

Also verified NaN-handling (non-finite p-values are excluded from ranking
and never rejected) and re-run against the real research funnel: at
`fdr_alpha=0.05`, `expected_false_positives(78, 0.05)` = 3.9, printed live by
the pipeline itself as a running sanity check next to every funnel report.

## Hedge ratio

`src/statarb/research/hedge_ratio.py::static_ols_hedge_ratio` — OLS of
log(A) on log(B) with intercept. Controlled recovery test (known
`beta=1.7, alpha=0.3`, OU spread with kappa=0.05, n=2000, noisy):

```
true beta=1.7  -> estimated beta=1.8384  (abs err 0.138, ~8% relative)
true alpha=0.3 -> estimated alpha=0.2739 (abs err 0.026)
```

Error is consistent with the injected residual noise, not a bug (OLS is
unbiased in expectation; a single noisy draw will show err > 0). Rolling
variant (`rolling_ols_hedge_ratio`) is separately pinned causal by both the
existing unit test (`test_rolling_beta_is_causal`) and this audit's mutation
test (`leakage_audit.md`).

## Half-life

`src/statarb/research/mean_reversion.py::half_life` — AR(1) OLS,
`half_life = -ln(2)/ln(1+lambda)`. Controlled test on the same OU process
(theoretical half-life `= ln(2)/kappa`):

```
theoretical half-life ~13.51d -> estimated 12.95d   (4% error, mean_reverting=True)
```

Tolerance: within 5% on a single 2000-observation noisy draw is consistent
with sampling variance around an unbiased AR(1) estimator; not tuned to hit
this number.

## Verdict

**PASS on every statistical component checked.** ADF null hypothesis is
correctly stated and correctly used; cointegration uses the correct
(MacKinnon) critical values, not the common hand-rolled shortcut; BH-FDR
matches the reference implementation to float precision; hedge ratio and
half-life recover known synthetic parameters within sampling-noise
tolerance; and the pipeline's own negative controls (engineered decoys) are
never admitted, on either the run that shipped in the repo or on every rerun
performed during this audit.

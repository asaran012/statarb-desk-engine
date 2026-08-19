# Methodology

## Pair selection

Runs inside each training window, never across the full sample.

1. **Universe.** Liquidity, price and history filters, with five explicit
   rejection reasons recorded per symbol. Re-run per window so the universe
   itself carries no forward information.
2. **Correlation pre-filter.** Pairwise return correlation ≥ 0.6, optionally
   restricted to same sector. This is a *computational* filter, not a
   statistical one — it exists to avoid running 1,770 cointegration tests when
   78 will do. Correlation is explicitly not evidence of cointegration, and the
   ablation study measures exactly that.
3. **Engle-Granger.** `statsmodels.tsa.stattools.coint`, which applies
   MacKinnon critical values appropriate to a residual from an estimated
   cointegrating vector. Running plain ADF on an OLS residual instead would
   overstate significance.
4. **Benjamini-Hochberg FDR.** Hand-implemented step-up procedure with monotone
   q-values. Controls the expected proportion of false discoveries among
   rejections. The funnel prints the expected false-positive count under the
   complete null so the reader can see the scale of the multiple-testing
   problem (78 tests at α = 0.05 → ~3.9 expected false positives).
5. **ADF on the spread** and **half-life in [2, 60] days.** Half-life is
   returned in both discrete `−ln2/ln(1+λ)` and continuous `−ln2/λ` forms, NaN
   when `λ ≥ 0`.
6. **Rolling stability.** Engle-Granger re-run in K sub-windows; a pair must
   cointegrate in at least 50% of them. Intended to remove relationships that
   held once and then broke. **See the research audit — the ablation for this
   stage was confounded and did not support the hypothesis.**
7. **Quality score.** Explainable 0–1 composite of four equally weighted
   components: stability pass rate, 1 − beta coefficient of variation, a
   triangular preference over half-life, and 1 − p-value. Deliberately not a
   fitted model, because a fitted selection model is one more thing to overfit.

## Signal construction

Spread is `log P_A − β·log P_B`. Z-score is rolling over the configured
lookback, using a window that ends at *t* inclusive. That is not look-ahead:
the close of day *t* is known when the signal is formed on day *t*, and
execution is deferred to *t+1*.

The state machine is explicit rather than vectorised comparison, because the
rules are stateful: no re-entry while a position is open, the stop sits *above*
the entry band, and `max_holding_days` forces an exit. Vectorising this is how
people accidentally allow same-bar round trips.

## Execution

`delay_days ≥ 1` is enforced at config validation, not by convention. Fills are
at the next open with proportional slippage. The trade ledger records the
signal date and the execution date separately so the invariant
`fills strictly after signal bar` can be checked rather than assumed.

## Sizing and risk

Sizing is in dollar exposures, not share counts, so it is invariant to price
level and splits. For one unit of long spread the B leg carries `β` times the
A leg's dollar exposure with opposite sign, so gross is `V·(1 + |β|)`.

Limits scale a trade down rather than rejecting it, subject to a minimum
fraction of 0.10 below which the trade is dropped. Sizing targets
`equity_now − cost(gross_intended)` because costs are charged at the fill —
see the research audit for the bug this fixed.

## Walk-forward protocol

Rolling 3 years train / 1 year validate / 1 year test, stepping 1 year, giving
6 windows over 2016–2025. Parameters are swept on validation, frozen, then
applied once to test. `MIN_VALIDATION_TRADES = 8` rejects parameter sets with
too few validation trades to mean anything. Positions are liquidated at every
window boundary so no position spans a re-fit. Equity chains across windows.

## What the synthetic generator does

Six sectors, 60 symbols. Anchored so follower prices start at a realistic level
rather than compounding the leader's price into six figures — an early bug
produced $592,167 share prices.

Three labelled families:
- **Cointegrated** — follower = β·leader + OU residual, constant β.
- **Correlated-only decoys** — `r_t = ρ·r_partner + √(1−ρ²)·σε`, giving high
  return correlation with an I(1) residual and no shared equilibrium.
- **Regime-break** — cointegrated up to a break point, then not.

Ground truth includes derived follower-follower pairs, because two followers of
the same leader are transitively cointegrated; omitting them penalised the
funnel for correct selections.

Measured properties on the 2016–2018 training window:

| family | n | mean abs corr | median coint p | fraction p < 0.05 |
|---|---|---|---|---|
| true cointegrated | 24 | 0.817 | 0.0044 | 0.83 |
| correlated only | 12 | 0.867 | 0.2924 | 0.08 |
| regime break | 6 | 0.813 | 0.0002 | 1.00 |

The decoys having *higher* correlation than the true pairs is the whole point:
it makes "correlation is not cointegration" a measured result rather than a
slogan.

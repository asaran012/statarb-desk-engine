# Project Decisions

A log of choices made and bugs found, kept because the bugs are the most
informative artifact this project produced. Each entry records what happened,
why it mattered, and what changed.

## Bugs found during development

### 1. Follower intercept double-counting
**Symptom:** synthetic share prices reached $592,167.
**Cause:** `log P_f = log(P0) + β·log(P_lead) + resid` compounds the leader's
price *level* into the follower, not just its returns.
**Fix:** solve analytically for the intercept α so the follower starts at a
target price. Range is now $23–$3,310.
**Lesson:** a generator bug that produces implausible output is lucky. The
dangerous ones produce plausible output.

### 2. The decoys weren't decoys
**Symptom:** none — everything looked fine.
**Cause:** "correlated but not cointegrated" names loaded on an independent
shared shock, giving them ~zero correlation with their nominal partner. The
correlation-vs-cointegration ablation would have been vacuous while appearing
to confirm the hypothesis.
**Fix:** `r_t = ρ·r_partner + √(1−ρ²)·σε`. Decoys now carry mean |corr| 0.867
versus 0.817 for true pairs, with only 8% passing cointegration versus 83%.
**Lesson:** validate that your negative controls are actually negative controls.
An ablation against a broken control proves nothing and looks like proof.

### 3. Incomplete ground truth
**Cause:** two followers of the same leader are transitively cointegrated (both
are β·L + OU), so correct selections were being scored as false positives.
**Fix:** derive follower-follower pairs into the labelled positive set.

### 4. Truth scoring conflated categories
**Cause:** regime-break pairs are *legitimately* cointegrated during training
and only break later. Counting their selection as a precision failure punishes
the funnel for being right at the time it decided.
**Fix:** report `decoys_selected` / `decoy_admission_rate` separately from
`regime_break_selected`, and never fold the latter into precision.

### 5. Gross exposure invariant failure (1.0096 vs 1.00 limit)
**Cause, part one:** transaction costs are charged at the fill, so sizing to
exactly `limit × pre-trade equity` breaches the limit post-trade by exactly the
cost.
**Cause, part two:** `equity_now` was cached at the top of each bar, so on
multi-entry days the error accumulated across fills.
**Fix:** size against `equity_now − cost(gross_intended)`, and recompute equity
per candidate. Peak post-fill ratio is now exactly 1.0000.
**Structural fix:** split `check_pre_trade_gross_limit` (hard invariant) from
`mtm_exposure_breaches` (reported statistic). Conflating them either produces
false alarms or hides real breaches.

### 6. Stale-feed monitoring gap
**Symptom:** the failure-injection demo reported 11/12 — a 30-day frozen price
feed only raised WARNING.
**Cause:** the validator counts stale *runs*. A 30-day freeze is one run, and
one run is a warning.
**Fix:** added `longest_stale_run_days`, escalating on run *length* (warn at 5,
fail at 10). Severity should scale with how long the feed has been dead, since
30 frozen days silently zero out 30 days of returns and every beta and z-score
downstream. Clean data still reports PASS at length 2. Now 12/12.

### 7. Ablation D was a silent no-op
**Symptom:** the "static hedge ratio" ablation returned results bit-identical
to baseline — same Sharpe to four decimals, same 491 trades, same costs to the
cent.
**Cause:** the config default was *already* `static`. The experiment compared
static against static.
**Fix:** re-specified the experiment to test `rolling`. Sharpe then collapsed
from 3.52 to 0.23.
**Lesson:** bit-identical output across a supposed configuration change is a
bug signal, not a finding. Reporting it naively would have been a false claim.

### 8. Monitor fed ledger data instead of live state
**Symptom:** `pairs_without_zscore = 3`, section status FAIL.
**Cause:** the CLI passed `exit_z` from the trade ledger as the "current"
z-score; end-of-period exits have NaN there.
**Fix:** `_live_signal_states()` recomputes z-scores from prices as of the
report date. A monitor should reflect what the strategy wants *now*, which is
what a desk checks pre-open — not a summary of what it already did.

## Deliberate choices

**Dual backtest loops.** A readable pandas reference implementation and a fast
NumPy one, held bit-identical by test. Without the reference, "2.39× speedup"
would be an unverifiable claim about a rewrite.

**Hand-implemented Benjamini-Hochberg.** Available in statsmodels, but the
procedure is short, the monotonicity requirement is easy to get wrong, and
being able to derive it on a whiteboard is worth more than the import.

**Quality score is not a fitted model.** Four equally weighted, individually
explainable components. A fitted selection model is one more surface to overfit
and one more thing that can't be explained under questioning.

**`forward_fill_limit` defaults to 0.** Filling a price gap manufactures a fake
zero return that then propagates into every downstream beta and z-score.
Missing data should be visibly missing.

**Synthetic data is labelled.** The generator's value is not that it looks like
a market — it doesn't. It's that ground truth exists, so the selection
procedure can be *scored* instead of admired.

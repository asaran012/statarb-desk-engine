# Interview Guide

## Read this first

This project can be described in a way that gets you hired or in a way that
gets you caught. The difference is entirely about which numbers you claim —
and, since the real-data audit, about which of *two* data regimes you're
talking about.

There are now two separate stories, and they must never be conflated:

1. **Synthetic data** — proves the engine and the statistical procedure are
   correct, against known ground truth. Sharpe 3.52 lives here. It is a
   correctness check, not a performance number.
2. **Real data** — 69 real US equities, 2017–2026, run through the exact
   production pipeline with zero threshold changes. **Zero pairs survived
   screening in every one of 6 walk-forward windows. It never placed a
   trade.** There is no real Sharpe, positive or negative.

**Never say:** "It achieved a Sharpe of 3.5."
**Never say:** "I tested it on real data and it worked" / "it didn't work."
**Say:** "On synthetic data it's correct by construction — I built the
ground truth myself. On real data, the same pipeline found zero tradeable
pairs in every window, which is a genuine statistical result, not a bug."

A desk quant's first question about any backtest is *what data*. If you lead
with the honest answer — including the real-data null result — you look like
someone who understands backtests and multiple-testing discipline. If you
lead with the Sharpe and they extract the real story, you look like someone
who doesn't, or worse, someone who does and hoped they wouldn't ask.

---

## The 30-second pitch

> I built a statistical-arbitrage research platform for equity pairs — pair
> selection through cointegration testing with false-discovery-rate control,
> a walk-forward backtester with strict train/validate/test separation, and a
> desk monitoring layer with trade reconciliation. I validated it two ways:
> against a labelled synthetic market I generated myself, where I could score
> the selection procedure against known ground truth, and then against 69
> real US equities over nine years. The real run is the more interesting
> result — the production screens rejected every single candidate pair, in
> all six walk-forward windows. That's not a failure, it's the FDR control
> doing exactly its job on a diversified large-cap universe, and I can show
> three diagnostic runs where loosening those screens did produce trades —
> and lost money.

## The 2-minute version

Add:

- **The funnel.** 1,770 possible pairs per window (synthetic) → correlation
  pre-filter → Engle-Granger → Benjamini-Hochberg → rolling stability. Each
  stage runs on training data only, re-run per window. On synthetic data with
  known ground truth this selects 18 pairs and admits **0 of 12 engineered
  decoys** that were built to have *higher* return correlation than the true
  pairs (0.867 vs 0.817).
- **The same funnel on real data finds nothing — and that's the point.** 69
  real equities, ~68 correlation-screened candidates per window, 0–1 raw
  cointegration hits per window (below the ~3.4 false positives expected
  under a complete null at α = 0.05), **0 pairs survive BH-FDR in 6/6
  windows.** Reproduced independently; the FDR implementation matches
  `statsmodels` exactly. I have three diagnostic ablations (screens
  deliberately weakened) that prove the engine executes real fills
  correctly — all three lost money, which rules out a hidden profitable
  strategy sitting just under the screens.
- **The engineering.** Two independent implementations of the backtest inner
  loop — a readable pandas reference and a NumPy version — that a test
  asserts produce bit-identical output on both real and synthetic data
  (856/856 trades exact match on real data). Measured **7.49× speedup**
  (643ms → 86ms) on real data, 7.46× on synthetic, both bit-identical,
  reproducible via `scripts/benchmark_real_engine.py`.
- **The honesty layer.** 9 accounting invariants, 54/54 passing across 6
  real-data windows, 73 tests, and a failure-injection harness where 12
  deliberately corrupted scenarios all get caught — on real data.

## The 5-minute version

Add the failures below — those are the most valuable 90 seconds you have.

---

## Talk about the real-data result first

This is the single most important thing to get right in this interview. It
is a *negative* result and it is the strongest thing in the project, because
it's reproducible, independently checked, and statistically expected rather
than convenient.

**What happened:** the production config — no threshold changes — selected
zero pairs in 6/6 real walk-forward windows. Pre-FDR, the raw cointegration
hit rate was 0–1 per window against ~68 candidates, which is *below* what a
complete null would produce by chance (~3.4 false positives at α = 0.05). So
the honest reading isn't "the screens are too strict," it's "there was
essentially nothing there to find, in this universe, over this period."

**How I know it isn't a bug:**
- Reproduced across multiple runs, same result every time.
- BH-FDR verified against `statsmodels.stats.multitest.multipletests` on 200
  synthetic hypotheses — identical rejections, q-values match to 1e-16.
- Three ablations that weaken the screens *do* produce trades on the same
  real data (correlation-only: 398 trades; no-FDR: 28 trades in 3/6
  windows), which proves the pipeline can and does execute real fills — the
  zero is specific to the strict production screen, not a plumbing failure.

**What I'd do next, and say so unprompted:** the 69-name universe was
deliberately diversified across 10 sectors for data-quality reasons. That
diversification works against finding cointegration, which is more common
within an industry than across one. The natural next experiment is a
same-industry universe (regional banks, refiners, airlines) before
concluding anything about real-market viability generally.

---

## Talk about the failures

Interviewers discount claims and credit self-criticism. You have six genuine
ones — two from building the strategy, four from the real-data audit that
followed. Lead with them.

### 1. The risk-limit bug the invariants caught

The gross-exposure invariant failed at 1.0096 against a 1.00 limit. It looked
like floating-point noise. It wasn't.

**Cause:** transaction costs are charged at the fill. Sizing a position to
exactly 100% of *pre-trade* equity necessarily breaches 100% of *post-trade*
equity, by exactly the cost. Compounded by a second bug — equity was cached at
the top of the bar, so on days with multiple entries the drift accumulated.

**Fix:** size against `equity_now − cost_model.charge(gross_intended)`, and
recompute equity per candidate within the bar. Peak post-fill gross/equity is
now exactly 1.0000.

**The lesson to state out loud:** I also split the concepts. A pre-trade limit
is a hard invariant. Mark-to-market drift on *existing* positions between bars
is a different thing and is now reported as a monitoring statistic rather
than silently folded into the invariant. Conflating them is how a limit
breach gets hidden.

### 2. My decoys weren't decoys

The synthetic "correlated but not cointegrated" names originally loaded on an
independent shock, so they had near-zero correlation with their partners. The
entire correlation-vs-cointegration ablation would have been vacuous — it would
have "proved" the point while testing nothing.

**Fix:** `r_t = ρ·r_partner + √(1−ρ²)·σε`, giving high return correlation with
an I(1) residual. Now the decoys have *higher* correlation than the true pairs
and only 8% pass cointegration versus 83%. The ablation became measurable
instead of assertable.

### 3. An ablation falsified my hypothesis — and the ablation was confounded

Removing the rolling-stability screen *improved* Sharpe (3.52 → 3.98) and
nearly doubled cumulative return, on synthetic data. My hypothesis said it
should hurt.

**Why:** the screen halves the book from 20 pairs to 10, so the experiment
conflates screening quality with breadth. More pairs means more independent
bets means higher Sharpe from diversification alone — and max drawdown
correspondingly doubled (−1.02% → −2.08%). To isolate the screen I would need
to hold pair count fixed and vary only which pairs are admitted. **I have not
run that experiment.** Saying so is the point.

### 4. A silent no-op that looked like a finding

The hedge-ratio ablation initially returned numbers bit-identical to baseline
— same Sharpe to four decimals, same 491 trades, same costs to the cent.
Reported naively that reads as "static and rolling betas are equivalent",
which would be a false claim.

**Cause:** the config default was already `static`, so I was comparing static
against static. Once corrected to test *rolling*, Sharpe collapsed 3.52 → 0.23
and hit rate fell to 0.52.

**The interesting part is why:** the generator uses a **constant** β. Rolling
re-estimation adds pure variance to a parameter that doesn't move. It's a clean
bias-variance demonstration and tells you nothing about real markets, where
betas do drift. Bit-identical results across a supposed configuration change
should always be treated as a bug until proven otherwise.

### 5. The real-data download path was completely non-functional

Found during the real-data audit, not before: `cli.py` called
`make_source(cfg.data.source, cfg)` with an extra positional argument against
a keyword-only signature, and then called `src.fetch()` with none of its
three required arguments. It could not have run. This is exactly why the
project's own pre-audit status doc said the Yahoo path was "implemented but
never exercised" — the code had never been executed, and it showed. Fixed by
wiring a proper ticker/sector universe file through to
`fetch(symbols, start, end)`.

**The lesson:** "implemented" and "implemented correctly" are different
claims, and the only way to close the gap is to actually run the path, not
to read the code and conclude it looks right.

### 6. The NYSE calendar closed the market on a day it was open

The Saturday-holiday observance rule was applied uniformly to every holiday,
including New Year's Day, which incorrectly marked 2021-12-31 as a market
holiday. Real downloaded data proved NYSE was open that day (a year-end
settlement convention), while correctly closed the Friday before a Saturday
July 4th and a Saturday Christmas. Fixed by special-casing New Year's Day.

**The lesson:** a hand-rolled calendar is a set of special cases pretending
to be a rule. Real data is what exposed this — the synthetic calendar never
would have, because the synthetic generator never encoded the exception in
the first place.

---

## Questions you should expect

**"Walk me through the real-data run."**
See "Talk about the real-data result first" above. Lead with the number (0
pairs, 6/6 windows), then the statistical reasoning for why it's not a bug,
then the three ablations that prove the engine still executes real fills
correctly, then the honest next step (same-industry universe).

**"Why is the real-data result zero and not, say, a handful of marginal
pairs?"**
Because a diversified, 10-sector, large-cap universe is close to the
worst case for finding genuine cross-sectional cointegration — shared
economic drivers, which cointegration needs, are far more common within an
industry. 0–1 raw hits against ~68 candidates per window is actually
*consistent* with, not contradicted by, that expectation.

**"Why Engle-Granger and not Johansen?"**
EG is a two-step estimator, is asymmetric in which leg is regressed on which,
and only finds one cointegrating vector. Johansen handles n > 2 assets and is
invariant to ordering. I chose EG because the strategy is explicitly pairwise
and EG's output maps directly onto a hedge ratio I can trade. I fix pair
ordering lexicographically so the asymmetry is at least deterministic. Johansen
is the honest upgrade path.

**"Why not just run ADF on the OLS residual?"**
Because the residual comes from an *estimated* cointegrating vector, so its
distribution under the null isn't the standard Dickey-Fuller distribution.
Using DF critical values overstates significance. `statsmodels.coint` applies
MacKinnon critical values that account for the estimation step.

**"Derive the half-life."**
Fit `ΔS_t = λ·S_{t−1} + ε`. With `φ = 1 + λ`, the AR(1) decays as `φ^k`. Set
`φ^k = ½` → `k = −ln2 / ln(1+λ)` (discrete). The continuous OU form is
`−ln2 / λ`. They agree for small |λ| and diverge as mean reversion gets fast.
I return both, and NaN when `λ ≥ 0` because there's no mean reversion to
measure. Expect to be asked to do this on a whiteboard.

**"How do you know there's no look-ahead?"**
Three mechanisms. Config validation forces `delay_days ≥ 1`. A test corrupts
the signal-date close and asserts the fill price doesn't move. A second test
multiplies all future prices by 1.5–4× on real data across all 69 symbols and
asserts every earlier rolling beta, z-score, signal, and full engine
equity/trade output is bit-for-bit unchanged before the cutoff — *and*
asserts later values did change, so the test can't pass vacuously. That last
clause is the part most people forget.

**"Your Sharpe is 3.5. Is that real?"**
No. It's synthetic data generated from an Ornstein-Uhlenbeck process, which is
exactly the model the strategy assumes mean-reverts. Measuring profitability
under your own assumptions tests the implementation, not the hypothesis. On
real data the strategy selected zero pairs in every one of 6 windows, so
there is no real Sharpe at all — not a bad one, none. Separately, even the
synthetic figure's √252 annualisation assumes serially uncorrelated returns
and positions are held ~10 days, so the arithmetic is biased upward before
you even get to the data question.

**"What's the biggest weakness?"**
Two, and I'd name both. First, the real universe is hand-picked and
deliberately diversified across 10 sectors, which works against finding the
within-industry relationships cointegration actually needs — I haven't yet
tried a concentrated, same-industry universe, which is the obvious next
experiment. Second, survivorship bias: a universe of currently-listed tickers
has already discarded every company that failed. On the engineering side, no
test suite can close researcher degrees of freedom — I ran the test windows
many times while fixing bugs, and every look is a small overfit.

**"Why is FDR control there if it just rejects everything on real data?"**
That's exactly what it should do when there's nothing to find, and it's
weaker evidence in the other direction than it looks: in the synthetic
generator, true pairs have median p = 0.0044 and decoys median p = 0.2924 —
the populations barely overlap, so a raw α = 0.05 cut already separates them
there. FDR is insurance for the regime where signal and noise overlap, which
is real markets — and the real-data run is one data point consistent with
that regime being exactly where I am now.

**"What does BH actually do?"**
Sort p-values ascending, find the largest k with `p_(k) ≤ k·α/m`, reject all
hypotheses up to k. It controls the expected *proportion* of false discoveries
among rejections, unlike Bonferroni which controls the probability of *any*
false discovery and is far more conservative. I implemented it by hand,
including q-value monotonicity, verified it against a worked example, and
independently re-verified it against `statsmodels` (200 hypotheses, identical
rejections, q-values match to 1e-16).

**"Costs — what are you missing?"**
Market impact, borrow cost on the short leg, financing, and the correlation
between illiquidity and stop-outs. The last one matters most: liquidity dries
up precisely when your stops fire. My model is proportional-to-notional
commission plus slippage, swept 0–40 bps on synthetic data — moot on the
production real-data result since nothing traded, but exercised on the
diagnostic ablations' real fills.

**"How would you take this to production?"**
Given the real-data result, the honest order is: first, test a concentrated
same-industry universe instead of the diversified one, since that's the more
likely place to find genuine cointegration. Only if that produces real
candidates does the rest of the roadmap matter: point-in-time universe data
to remove survivorship bias, real fills to calibrate the cost model, capacity
analysis, and intraday de-levering for the MTM drift I currently only report.

**"How much of this did you write?"**
Answer this one honestly and specifically. Name the bugs in the "Talk about
the failures" section above — they are yours in the sense that matters,
because you can explain the mechanism, including the two found only once
real data forced the code to actually run (the broken download path, the
calendar bug).

---

## Things not to say

- "It's production-ready." It isn't. No positive real-data result, no impact
  model, no capacity analysis.
- "The strategy is profitable." It has no real trades to be profitable or
  unprofitable — say that precisely, don't round it to either "it works" or
  "it failed."
- "I tested it on 10 years of real data and it didn't work." It ran
  correctly and found nothing; conflating "found nothing" with "broken" is
  the exact mistake this project's own results argue against.
- "I tested it on 10 years of data" (unqualified) for the synthetic Sharpe.
  You tested it on 10 years of *synthetic* data. The word is load-bearing.
- Any number from `reports/metrics.json` under a key ending `_SYNTHETIC`,
  without the qualifier attached.
- Presenting the three real-data ablations (`A_correlation_only`, `A_no_fdr`,
  `B_no_stability`) as anything but diagnostics. They lost money; they exist
  to prove the engine works, not to suggest a looser strategy would be
  profitable.

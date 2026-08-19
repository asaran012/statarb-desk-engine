# Interview Guide

## Read this first

This project can be described in a way that gets you hired or in a way that
gets you caught. The difference is entirely about which numbers you claim.

**Never say:** "It achieved a Sharpe of 3.5."
**Say:** "I ran it on synthetic data because I had no market feed, so the
performance numbers are meaningless as alpha — what I can defend is the
engineering and the statistical procedure."

A desk quant's first question about any backtest is *what data*. If you lead
with the honest answer you look like someone who understands backtests. If you
lead with the Sharpe and they extract the answer, you look like someone who
doesn't — or worse, someone who does and hoped they wouldn't ask.

---

## The 30-second pitch

> I built a statistical-arbitrage research platform for equity pairs — pair
> selection through cointegration testing with false-discovery-rate control,
> a walk-forward backtester with strict train/validate/test separation, and a
> desk monitoring layer with trade reconciliation. I had no market-data access
> in the environment I built it in, so I wrote a labelled synthetic market
> generator instead, which let me score the selection procedure against known
> ground truth. The interesting parts are the failures: an ablation falsified
> one of my own design hypotheses, and the accounting invariants caught a
> risk-limit bug I'd never have found by eye.

## The 2-minute version

Add:

- **The funnel.** 1,770 possible pairs per window → 78 pass a 0.6 correlation
  pre-filter → 43 pass Engle-Granger at α = 0.05 → 34 survive Benjamini-Hochberg
  → 18 survive rolling-stability screening. Each stage runs on training data
  only, re-run per window.
- **Why synthetic data is a feature here.** The generator labels every pair:
  30 genuinely cointegrated, 12 decoys engineered to have *higher* return
  correlation than the true pairs (0.867 vs 0.817) but no shared equilibrium,
  and 6 that cointegrate then break mid-sample. That means I can score the
  funnel instead of admiring it. **Decoy admission rate: 0.0.**
- **The engineering.** Two independent implementations of the backtest inner
  loop — a readable pandas reference and a NumPy version — that a test asserts
  produce bit-identical output. Profiling showed pandas scalar `.at[]` lookups
  were 36% of runtime; the rewrite gave 2.39× (433ms → 181ms).
- **The honesty layer.** 9 accounting invariants after every run, 73 tests,
  and a failure-injection harness where 12 deliberately corrupted scenarios all
  get caught.

## The 5-minute version

Add the three results that went wrong — see "Talk about the failures" below.
Those are the most valuable 90 seconds you have.

---

## Talk about the failures

Interviewers discount claims and credit self-criticism. You have four genuine
ones. Lead with them.

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
is a different thing and is now reported as a monitoring statistic (3 breach
days, peak 1.00955) rather than silently folded into the invariant. Conflating
them is how a limit breach gets hidden.

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
nearly doubled cumulative return. My hypothesis said it should hurt.

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

---

## Questions you should expect

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
multiplies all future prices by 1.75 and asserts every earlier rolling beta is
bit-identical — *and* asserts later betas did change, so the test can't pass
vacuously. That last clause is the part most people forget.

**"Your Sharpe is 3.5. Is that real?"**
No. It's synthetic data generated from an Ornstein-Uhlenbeck process, which is
exactly the model the strategy assumes mean-reverts. Measuring profitability
under your own assumptions tests the implementation, not the hypothesis.
Separately, the √252 annualisation assumes serially uncorrelated returns and
positions are held ~10 days, so even the arithmetic is biased upward. No
Newey-West correction is applied.

**"What's the biggest weakness?"**
Survivorship bias would be, on real data — a universe of currently-listed
tickers has already discarded every company that failed. On this build, it's
that no test suite can close researcher degrees of freedom: I ran the test
windows many times while fixing bugs, and every look is a small overfit.

**"Why is FDR control there if the ablation showed no effect?"**
Because the ablation can't measure it. In the generator, true pairs have median
p = 0.0044 and decoys median p = 0.2924 — the populations barely overlap, so a
raw α = 0.05 cut already separates them. FDR is insurance for the regime where
signal and noise overlap, which is real markets. The null result is a fact
about my generator, not about multiple-testing correction.

**"What does BH actually do?"**
Sort p-values ascending, find the largest k with `p_(k) ≤ k·α/m`, reject all
hypotheses up to k. It controls the expected *proportion* of false discoveries
among rejections, unlike Bonferroni which controls the probability of *any*
false discovery and is far more conservative. I implemented it by hand,
including q-value monotonicity, and tested it against a worked example.

**"Costs — what are you missing?"**
Market impact, borrow cost on the short leg, financing, and the correlation
between illiquidity and stop-outs. The last one matters most: liquidity dries
up precisely when your stops fire. My model is proportional-to-notional
commission plus slippage, swept 0–40 bps.

**"How would you take this to production?"**
Point-in-time universe data first — nothing else matters if the universe is
survivorship-biased. Then real fills to calibrate the cost model, then capacity
analysis, then intraday de-levering for the MTM drift I currently only report.

**"How much of this did you write?"**
Answer this one honestly and specifically. Name the bugs in §1–4 above — they
are yours in the sense that matters, because you can explain the mechanism.
Being vague here is worse than any technical gap.

---

## Things not to say

- "It's production-ready." It isn't. No real data, no impact model, no capacity
  analysis.
- "The strategy is profitable." Unmeasured.
- "I tested it on 10 years of data." You tested it on 10 years of *synthetic*
  data. The word is load-bearing.
- Any number from `reports/metrics.json` under a key ending `_SYNTHETIC`,
  without the qualifier attached.

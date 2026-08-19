# Limitations

Ordered by how much they should change your reading of the results.

## 1. No real market data

The build environment had no market-data egress (HTTP 403 from Yahoo and
Stooq). Every performance figure is synthetic. The strategy assumes mean
reversion; the generator produces mean reversion. **Performance measured under
your own assumptions is not evidence.** See `reports/research_audit.md` §0.

The `YahooSource` adapter is implemented but was **never executed**. Failures
on that path should be expected and are more likely to be API-shape issues than
logic errors.

## 2. Survivorship bias is unaddressed for real data

The synthetic universe has no delistings. A real universe built from currently
listed tickers has already discarded every company that failed, which flatters
any long-biased strategy and distorts pair selection. Point-in-time constituent
data is required and is not free.

## 3. The Sharpe is biased upward even arithmetically

√252 annualisation assumes serially uncorrelated returns. Positions are held
~10 days, so returns are autocorrelated. No Newey-West or block-bootstrap
correction is applied.

## 4. Cost model omits the costs that matter in a crisis

No market impact, no borrow cost on the short leg, no financing, and no
modelling of the correlation between illiquidity and stop-outs. The last is the
serious one: liquidity disappears exactly when stops fire.

## 5. Fills are assumed perfect

Every order fills in full at the next open. No volume caps, no partial fills,
no gapping through stops.

## 6. Researcher degrees of freedom

The walk-forward is honest within a run, but the test windows were executed
many times during development while bugs were found and fixed. The correlation
threshold (0.6), half-life band (2–60d), stability threshold (50%), grid
boundaries and universe size were all chosen by hand.

## 7. Two experiments that have not been run

- The stability ablation, re-run holding pair count fixed, to disentangle
  screening quality from breadth.
- Capacity analysis: at what AUM does the strategy's own impact eat the edge?

## 8. Statistical method choices

Engle-Granger is asymmetric and finds at most one cointegrating vector.
Johansen would be stronger and would extend to baskets. Half-life estimates on
~750 observations are noisy and are used only as a coarse filter.

## 9. Mark-to-market exposure drift is reported, not enforced

Pre-trade gross limits are hard. Drift between bars is monitored (3 breach
days, peak 1.00955 in the 2020 window) but not acted upon. A real desk would
de-lever or explicitly accept it.

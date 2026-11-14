# Limitations

Ordered by how much they should change your reading of the results.

## 1. No real-market trading evidence — the production strategy found nothing to trade

The production pipeline was run against 69 real, liquid, sector-diversified
US equities (2017–2026, 166,842 rows, Yahoo Finance) through all 6
walk-forward windows with zero threshold changes. **Zero pairs survived
cointegration + BH-FDR + ADF + half-life + stability screening, in every
window.** It never placed a trade.

This is not a data-access problem — real data was successfully downloaded,
validated, and run end-to-end. It is a genuine negative result: there is no
real-market Sharpe, CAGR, drawdown, or hit rate to report, positive or
negative, because nothing was ever selected to trade. The 0–1 raw
cointegration hits per window against ~68 candidates is below the ~3.4 false
positives expected under a complete null at α = 0.05, so the screens are
behaving correctly, not failing. See
`reports/final_audit/statistical_audit.md` and
`reports/real/real_walkforward_summary.json`.

Three diagnostic ablations (screens deliberately weakened, solely to give the
engine real fills to process) did trade — and lost money in every case with a
meaningful sample (`A_correlation_only`: −8.58% over 398 trades;
`A_no_fdr`: −0.59% over 28 trades in 3/6 windows). They are evidence the
engine handles real fills correctly, not evidence of a hidden profitable
strategy underneath the screens.

## 2. The real universe is hand-picked and diversified by construction

69 currently-listed tickers, chosen by hand across 10 sectors. Two separate
effects follow from this, both pushing toward fewer, not more, real
cointegrated pairs being found:

- **Survivorship bias.** A universe built from currently listed tickers has
  already discarded every company that failed. Point-in-time constituent
  data is required to remove this and is not free.
- **Diversification suppresses the base rate.** Genuine, stable
  cross-sectional cointegration is more common *within* an industry (shared
  input costs, regulation, demand drivers) than across 10 deliberately
  spread sectors. Choosing a diversified universe for data-quality reasons
  (avoiding a single-sector blowup skewing validation) may have been in
  tension with choosing a universe likely to contain the relationships the
  strategy is looking for. This has not been tested — a same-industry
  universe (e.g., regional banks, refiners, or airlines) is the natural next
  experiment before concluding the strategy finds nothing in real markets.

## 3. The only Sharpe that exists is biased upward even arithmetically

Since the production strategy has no real trades, the only Sharpe figure
available is the synthetic one (3.52), which is not a performance claim (see
`reports/research_audit.md` §0) and is separately biased: √252 annualisation
assumes serially uncorrelated returns, positions are held ~10 days, and no
Newey-West or block-bootstrap correction is applied.

## 4. Cost model omits the costs that matter in a crisis

No market impact, no borrow cost on the short leg, no financing, and no
modelling of the correlation between illiquidity and stop-outs. The last is
the serious one: liquidity disappears exactly when stops fire.

## 5. Fills are assumed perfect

Every order fills in full at the next open. No volume caps, no partial
fills, no gapping through stops. This applies to the real-fill ablations
too — the fills were real prices, but execution assumptions were not
relaxed.

## 6. Researcher degrees of freedom

The walk-forward is honest within a run, but the test windows — both
synthetic and real — were executed many times during development and during
the real-data audit while bugs were found and fixed (a broken download
path, a calendar bug). The correlation threshold (0.6), half-life band
(2–60d), stability threshold (50%), grid boundaries and universe size were
all chosen by hand and never re-tuned against real data, which is the right
discipline but means they were never validated against it either.

## 7. Two experiments that have not been run

- The stability ablation, re-run holding pair count fixed, to disentangle
  screening quality from breadth.
- Capacity analysis: at what AUM does the strategy's own impact eat the
  edge? Moot under the current real-data result (nothing to size), but
  applies immediately if a same-industry universe (see §2) produces real
  candidates.

## 8. Statistical method choices

Engle-Granger is asymmetric and finds at most one cointegrating vector.
Johansen would be stronger and would extend to baskets. Half-life estimates
on ~750 observations are noisy and are used only as a coarse filter.

## 9. Mark-to-market exposure drift is reported, not enforced

Pre-trade gross limits are hard. Drift between bars is monitored (3 breach
days, peak 1.00955 in the 2020 synthetic window) but not acted upon. A real
desk would de-lever or explicitly accept it. Not exercised on real data
since the production strategy never held a position.

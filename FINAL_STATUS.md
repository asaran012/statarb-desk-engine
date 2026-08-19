# Final Status

## Build environment constraint (determines everything below)

**No market-data egress.** Yahoo Finance and Stooq both returned HTTP 403; the
proxy allowlist covers PyPI, GitHub, npm and crates only. Per the
specification's Rule 3, the implementation was completed in full and run
against a labelled synthetic market. All real-market metrics are marked
`NOT YET MEASURED`.

## Reproduction

```bash
pip install -e . && make all && pytest
```

For real data:

```bash
pip install '.[market]'
# set data.source: yahoo in config/default.yaml
python -m statarb.cli download && make all
```

## What was measured

| item | status | value |
|---|---|---|
| tests | ✅ | 73 passing |
| accounting invariants | ✅ | 9/9 on every window |
| engine speedup | ✅ | 2.39× (433ms → 181ms), bit-identical output |
| data validation checks | ✅ | 11 |
| reconciliation break types | ✅ | 6 |
| failure injection | ✅ | 12/12 detected |
| walk-forward | ✅ | 6 windows, 2016–2025, 491 trades |
| research funnel | ✅ | 1,770 pairs → 78 tested → 18 selected |
| decoy admission rate | ✅ | 0.0 |
| ablations | ✅ | 7 configurations |
| cost sensitivity | ✅ | 0–40 bps, no breakeven in range |
| figures | ✅ | 14 |
| **real-market performance** | ❌ | **NOT YET MEASURED** |
| capacity analysis | ❌ | not run |
| stability ablation with pair count held fixed | ❌ | not run |
| `YahooSource` adapter | ⚠️ | implemented, **never executed** |

## Deliverables

- `README.md` — headline results with the synthetic-data warning first
- `reports/research_audit.md` — leakage/bias audit and contradicted hypotheses
- `reports/resume_metrics.md` — metrics split safe-to-claim vs not
- `reports/research_summary.md`, `reports/metrics.json`
- `docs/` — methodology, architecture, project decisions, interview guide, limitations
- `reports/figures/` — 14 figures
- `tests/` — 73 tests

## The one thing to carry forward

The synthetic Sharpe of 3.52 is not evidence of anything. It was produced on
spreads generated from an Ornstein-Uhlenbeck process, which is precisely the
model the strategy assumes mean-reverts. It must not appear on a resume.

The defensible claims are the engineering and statistical ones: 9 invariants,
73 tests, 2.39× verified-identical speedup, 12/12 fault detection, 0.0 decoy
admission rate, and four documented bugs whose mechanisms can be explained
under questioning.

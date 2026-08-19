# Backtest accounting audit

## Runtime invariants: the actual count, by name

`src/statarb/backtest/accounting.py` + the calls made in
`src/statarb/backtest/engine.py::run_backtest` (not the count from any
README). Nine checks run after **every** backtest call, not only in tests:

| # | invariant | function | condition |
|---|---|---|---|
| 1 | `ending_equity == start + sum(net_pnl)` | `check_equity_reconciles` | `abs(end - (start+net)) <= 1e-8 * scale` |
| 2 | `gross_pnl - cost == net_pnl` | `check_gross_net_consistency` | max abs error `<= 1e-6` |
| 3 | all costs `>= 0` | `check_costs_non_negative` | zero trades with `cost < 0` |
| 4 | equity timestamps monotonic | `check_timestamps_monotonic` | index strictly increasing and unique |
| 5 | fills strictly after signal bar | `check_execution_after_signal` | zero trades with `entry_date <= entry_signal_date` |
| 6 | every exit matches an entry | `check_exits_match_entries` | zero exits before entry, zero missing exit reasons |
| 7 | book flat after liquidation | `check_flat_at_end` | zero symbols with `abs(position) > 1e-9` after forced liquidation |
| 8 | no position without a trade record | `check_no_position_without_trade` | `n_position_events == 2 * n_trades` |
| 9 | post-fill gross exposure within limit | `check_pre_trade_gross_limit` | worst post-trade gross/equity ratio `<= limit + 1e-6` (only when risk limits enabled) |

A tenth quantity, mark-to-market gross-exposure breach (`mtm_exposure_breaches`),
is deliberately **not** an invariant — its own docstring explains why: gross
notional drifts with prices between trades while equity moves with P&L, so
drift past the limit *between* fills is expected desk behaviour, not a bug.
It is reported as a risk statistic instead (breach days, peak ratio), which
this audit also captured (see `walk_forward_results.csv`, `peak_gross_pct`
in ablation results).

## Independent verification, real data

Ran across all 6 real-data walk-forward windows under the `A_correlation_only`
ablation (the only real-data mode with trades in every window —
production/`full` selected zero pairs, see `statistical_audit.md`):

```
window  test_start  test_end    pairs  trades   net_pnl        invariants_pass  n_invariants
0       2021-01-03  2022-01-02  20     60      -250,035.74     True             9
1       2022-01-03  2023-01-02  20     93       -46,126.00     True             9
2       2023-01-03  2024-01-02  20     47      -459,892.16     True             9
3       2024-01-03  2025-01-02  20     67       -47,686.01     True             9
4       2025-01-03  2026-01-02  20     88        36,923.47     True             9
5       2026-01-03  2026-08-17  20     43       -91,029.77     True             9
```

**54/54 invariant checks pass** (9 invariants x 6 windows), on real trades,
real prices, real transaction costs. Reproduce:
`.venv/bin/python scripts/run_real_data_audit.py` →
`reports/real/walk_forward_results_real.csv`.

Also independently re-derived (not read off the report) for the full
398-trade real blotter (`reports/real/trades/trades_correlation_only.csv`):

```
gross_pnl.sum()               = -656,567.26
total_costs.sum()             =  201,278.94
gross_pnl - total_costs       = -857,846.21   [== net_pnl.sum() exactly]
ending_equity - initial_capital(10,000,000)   = -857,846.21   [diff 1.8e-8, float rounding]
```

(Note: the *first* row of the concatenated equity series, 9,999,700.01, is
already net of window 0's first-bar trading costs, not the pre-trade capital
— comparing against that row instead of the true `10,000,000` starting
capital produces a spurious ~$300 "discrepancy". This audit caught and
corrected that exact mistake in its own draft before publishing this
document; the correct baseline is `cfg.risk.initial_capital`, the value
actually passed into the first window's `run_backtest`.)

## Other invariants checked directly

- **No orphan positions**: `check_flat_at_end` / `check_no_position_without_trade`
  both pass on every window; `liquidate_at_end=True` forces every open pair
  closed at the final test-window bar (`run_backtest`, "forced liquidation"
  section).
- **No trades before signals**: `check_execution_after_signal` passes on
  every window (54/54); independently, every trade's `entry_date` is
  strictly after `entry_signal_date` by construction of
  `apply_execution_delay`.
- **No negative transaction costs**: `check_costs_non_negative` passes on
  every window.
- **Forced liquidation works**: confirmed by `check_flat_at_end` passing
  post-liquidation, and by inspecting the `liquidate_at_end` block in
  `engine.py`, which closes every remaining open pair at the last test bar's
  close before the equity series is finalized.
- **Gross exposure limit respected at the moment of trading**:
  `check_pre_trade_gross_limit` passes on every window (`max_portfolio_gross
  = 1.00` in `config/real.yaml`, unchanged from the production config).
- **Symbol / pair exposure limits**: enforced by `RiskLimits.scale_for_trade`
  (`src/statarb/risk/limits.py`) before any fill is recorded; a trade that
  would breach `max_pair_gross` or `max_symbol_exposure` is scaled down or
  rejected (`pf.reject(...)`), never silently allowed through. Confirmed by
  reading `_try_open` in `engine.py`, which computes `scale` from
  `limits.scale_for_trade` before calling `pf.open_pair`.

## Verdict

**PASS.** All 9 documented invariants hold on every real-data window tested
(54/54 checks), the gross/net/cost identity re-derives exactly by hand from
the raw trade ledger, and every risk/exposure control is enforced
pre-trade rather than reported after the fact.

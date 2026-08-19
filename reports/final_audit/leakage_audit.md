# Look-ahead / data-leakage audit

Scope: independent code inspection plus a direct mutation test, both against
real market data (`data/processed_real/panel.parquet`, 69 symbols, 2017-01-03
to 2026-08-17). All findings below were produced during this audit; commands
to reproduce are given inline.

## 1. Rolling calculations

**Rolling hedge ratio** (`src/statarb/research/hedge_ratio.py::rolling_ols_hedge_ratio`)
uses pandas `.rolling(window).cov/.var`, which by construction only consumes
observations `[t-window+1, t]`. No shift is applied inside the function; the
execution-delay shift is applied later by the engine, so the value computed
*for* bar `t` is genuinely known *at* bar `t`, and the trade built from it is
still delayed a further `execution.delay_days` bars.

**Rolling z-score** (`src/statarb/signals/zscore.py`) — same pattern, backed
by pandas `.rolling().mean()/.std()`.

**Universe construction** (`src/statarb/research/universe.py`) is rebuilt
from scratch inside every walk-forward training slice
(`src/statarb/backtest/walkforward.py::run_walkforward`, "TRAIN" section)
rather than once over the full history. The module docstring calls this out
explicitly: building it once would "import survivorship and liquidity
information from the future into every earlier test period." Confirmed by
reading `run_walkforward`: `train_panel = slice_dates(panel, w.train_start,
w.train_end)` is the only input `select_pairs` ever receives.

## 2. Pair selection

`select_pairs` (`src/statarb/research/pair_selection.py`) takes a single
price frame and has no other way to reach data — "Leakage control is
structural, not conventional" per its own docstring. The walk-forward driver
slices the training window *before* calling it. Verified by reading every
call site in `walkforward.py`: `select_pairs(train_panel, ...)` where
`train_panel` is bounded by `w.train_start`/`w.train_end`.

The validation-only parameter sweep (`param_grid.combinations()`) is scored
by Sharpe on the **validation** window only (`w.valid_start`..`w.valid_end`);
the winning `(zscore_lookback, entry_z, exit_z)` combination is then frozen
and used for exactly one untouched run on the **test** window. Test-window
performance never feeds back into parameter choice — verified by reading the
window loop: the `grid` DataFrame is built entirely from
`test_start=w.valid_start, test_end=w.valid_end` backtests, and `chosen`
is fixed before `test_cfg`/`test_res` are computed.

## 3. Signal execution timing

`apply_execution_delay` (`src/statarb/backtest/execution.py`) is the **only**
path from a signal to a fillable target: `exec_target =
target.shift(delay_days)`, and `ExecutionConfig.delay_days` is
schema-validated `>= 1` (`src/statarb/config.py`), so same-bar execution is
structurally impossible, not just discouraged by convention. With the
project's default `fill_price: close`, a signal read from bar `t`'s close
fills no earlier than bar `t+1`'s close.

## 4. Mutation test (real data)

`scripts/leakage_mutation_test.py`: computed rolling hedge ratio, z-score,
target position, and a full backtest (trades + equity curve) on the real
BAC/JPM pair through 2026-08-17, then multiplied **every** post-2022-06-30
price for **all 69 symbols** by a random factor in [1.5x, 4.0x] and
recomputed everything from scratch.

```
$ .venv/bin/python scripts/leakage_mutation_test.py
mutation test pair: BAC/JPM  cutoff=2022-06-30
corrupted 1035 post-cutoff sessions across all 69 symbols (factors drawn uniform[1.5x, 4.0x])
[PASS] rolling hedge ratio: pre-cutoff betas identical (1383 obs compared)
[PASS] zscore: pre-cutoff values identical (1383 obs compared)
[PASS] target: pre-cutoff values identical (1383 obs compared)
[PASS] full backtest equity curve: pre-cutoff equity identical (1383 obs compared)
[PASS] full backtest trades: pre-cutoff trade records identical (125 trades compared)

OVERALL: ALL PASS
```

Raw result: `reports/real/leakage_mutation_test.json`.

**PASS on all five checks.** Corrupting the future left every pre-cutoff
number — hedge ratio, z-score, target position, full-engine equity curve, and
the full-engine trade ledger — bit-for-bit unchanged. This is a stronger test
than the pre-existing unit test
(`tests/test_backtester.py::test_no_same_bar_execution`, which perturbs one
bar for one symbol); this audit's version corrupts the entire remaining
dataset and checks four independent code paths, not just the engine.

## 5. Existing test-suite coverage of the same claim

- `tests/test_signals.py::test_rolling_beta_is_causal` — pins the rolling
  hedge ratio's causality with a synthetic-data mutation.
- `tests/test_backtester.py::test_no_same_bar_execution` — pins execution
  timing with a synthetic-data mutation.
- `tests/test_backtester.py::test_execution_is_strictly_after_signal` —
  asserts `entry_date > entry_signal_date` on every trade.

All three pass (see `test_results.txt`). This audit's mutation test extends
the same claim to real data and to the hedge-ratio/z-score/signal layer, not
only the engine.

## Verdict

**PASS.** No look-ahead path was found by inspection, and the direct
mutation test found none empirically, on real data, across every layer of
the pipeline that touches price history.

"""Synthetic market generator with **ground-truth labels**.

Why this module exists
----------------------
The build environment has no egress to any market-data vendor, so the pipeline
must still be exercisable end-to-end.  But a synthetic market is more than a
fallback: because we *know* which pairs are truly cointegrated, we can measure
the pair-selection procedure's precision and recall.  That is a statement about
the method, not about alpha, and it survives interview scrutiny in a way a
synthetic Sharpe ratio never will.

Four generative regimes are produced deliberately:

``cointegrated``
    ``log P_follow = a + b * log P_lead + OU_t`` with a stationary
    Ornstein-Uhlenbeck residual.  Ground-truth positive.

``correlated_only``
    Two names loading on the same sector factor **in returns** but each carrying
    its own independent random-walk component.  High return correlation, no
    common stochastic trend, therefore *not* cointegrated.  These are the
    adversarial cases that break correlation-only pair selection.

``regime_break``
    Cointegrated for the first ``break_frac`` of the sample, after which the
    residual becomes a pure random walk.  Ground-truth "unstable": the honest
    answer is that stability filtering should down-rank these.

``independent``
    Plain correlated-market random walks.

All randomness flows through a single seeded ``numpy.random.Generator``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .calendar import trading_calendar

SECTORS = ["TECH", "FIN", "ENER", "HLTH", "INDU", "CONS"]


@dataclass
class GroundTruth:
    """Labels the generator knows and the research pipeline must not see."""

    cointegrated_pairs: List[Tuple[str, str]] = field(default_factory=list)
    correlated_only_pairs: List[Tuple[str, str]] = field(default_factory=list)
    regime_break_pairs: List[Tuple[str, str]] = field(default_factory=list)
    half_lives: Dict[str, float] = field(default_factory=dict)
    betas: Dict[str, float] = field(default_factory=dict)
    sectors: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def key(a: str, b: str) -> str:
        return "|".join(sorted((a, b)))

    def positive_set(self) -> set:
        return {self.key(*p) for p in self.cointegrated_pairs}

    def negative_set(self) -> set:
        return {self.key(*p) for p in self.correlated_only_pairs}

    def unstable_set(self) -> set:
        return {self.key(*p) for p in self.regime_break_pairs}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path: Path) -> "GroundTruth":
        with open(path) as fh:
            raw = json.load(fh)
        raw["cointegrated_pairs"] = [tuple(p) for p in raw["cointegrated_pairs"]]
        raw["correlated_only_pairs"] = [tuple(p) for p in raw["correlated_only_pairs"]]
        raw["regime_break_pairs"] = [tuple(p) for p in raw["regime_break_pairs"]]
        return cls(**raw)


def ou_process(n: int, half_life: float, sigma: float,
               rng: np.random.Generator, x0: float = 0.0) -> np.ndarray:
    """Discrete Ornstein-Uhlenbeck path with the requested half-life in bars.

    ``x_t = phi * x_{t-1} + eps``  with ``phi = exp(-ln2 / half_life)``.
    The stationary standard deviation is ``sigma``; the innovation scale is
    therefore ``sigma * sqrt(1 - phi**2)``.
    """
    phi = float(np.exp(-np.log(2.0) / half_life))
    eps_scale = sigma * np.sqrt(max(1.0 - phi ** 2, 1e-12))
    x = np.empty(n)
    x[0] = x0 if x0 != 0.0 else rng.normal(0.0, sigma)
    shocks = rng.normal(0.0, eps_scale, size=n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + shocks[t]
    return x


def _ohlcv_from_close(close: np.ndarray, rng: np.random.Generator,
                      base_volume: float) -> Dict[str, np.ndarray]:
    """Build a self-consistent OHLCV bar set around a close series."""
    n = close.size
    prev_close = np.concatenate([[close[0]], close[:-1]])
    gap = rng.normal(0.0, 0.002, size=n)
    open_ = prev_close * (1.0 + gap)
    intraday = np.abs(rng.normal(0.0, 0.008, size=n)) + 0.001
    hi_base = np.maximum(open_, close)
    lo_base = np.minimum(open_, close)
    high = hi_base * (1.0 + intraday * rng.uniform(0.3, 1.0, size=n))
    low = lo_base * (1.0 - intraday * rng.uniform(0.3, 1.0, size=n))
    low = np.minimum(low, lo_base)
    high = np.maximum(high, hi_base)
    volume = np.round(base_volume * np.exp(rng.normal(0.0, 0.35, size=n)))
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def _anchor(lp_partner: np.ndarray, beta: float, resid: np.ndarray,
            target_p0: float) -> np.ndarray:
    """Intercept that makes the follower start at ``target_p0``.

    Naively writing ``log P_f = log(P0) + beta * log P_lead + resid`` adds the
    partner's *level* on top of a fresh base price and compounds it, producing
    six-figure share prices.  Solve for alpha instead.
    """
    alpha = np.log(target_p0) - beta * lp_partner[0] - resid[0]
    return alpha + beta * lp_partner + resid


def _correlated_not_cointegrated(lp_partner: np.ndarray, rho: float,
                                 rng: np.random.Generator,
                                 target_p0: float) -> np.ndarray:
    """High return correlation, no common stochastic trend.

    ``r_t = rho * r_partner,t + sqrt(1-rho^2) * sigma * eps_t``.  Integrating
    gives ``log P = rho * log P_partner + I(1) noise``: the regression residual
    is a random walk whose variance grows linearly in t, so the pair is
    correlated but **not** cointegrated.  These are the adversarial cases that
    a correlation-only screen accepts and a cointegration screen rejects.
    """
    d = np.diff(lp_partner, prepend=lp_partner[0])
    sigma = float(np.std(d[1:])) or 0.01
    r = rho * d + np.sqrt(max(1 - rho ** 2, 0.0)) * rng.normal(0.0, sigma, size=d.size)
    r[0] = 0.0
    return np.log(target_p0) + np.cumsum(r)


def generate_market(
    start: str,
    end: str,
    n_symbols: int = 60,
    seed: int = 20260818,
    frac_cointegrated: float = 0.30,
    frac_correlated_only: float = 0.20,
    frac_regime_break: float = 0.10,
) -> Tuple[pd.DataFrame, GroundTruth]:
    """Generate a labelled synthetic equity panel.

    Returns a tidy long DataFrame with the schema the rest of the pipeline
    expects, plus the :class:`GroundTruth` label set.
    """
    rng = np.random.default_rng(seed)
    dates = trading_calendar(start, end)
    n = len(dates)
    if n < 400:
        raise ValueError("need at least ~400 trading days for meaningful research")

    gt = GroundTruth()
    market = np.cumsum(rng.normal(0.0003, 0.009, size=n))  # common log-price factor

    per_sector = max(4, n_symbols // len(SECTORS))
    symbols: List[str] = []
    log_prices: Dict[str, np.ndarray] = {}
    base_vol: Dict[str, float] = {}

    sym_idx = 0
    for sec in SECTORS:
        if sym_idx >= n_symbols:
            break
        sector_factor = np.cumsum(rng.normal(0.0001, 0.006, size=n))
        sector_syms: List[str] = []
        k = min(per_sector, n_symbols - sym_idx)

        # --- leaders: independent random walks anchored on market + sector ----
        n_leaders = max(2, k // 3)
        for j in range(n_leaders):
            sym = f"{sec}{j:02d}"
            idio = np.cumsum(rng.normal(0.0, 0.011, size=n))
            lp = np.log(rng.uniform(25, 250)) + 0.9 * market + 0.7 * sector_factor + idio
            log_prices[sym] = lp
            symbols.append(sym)
            sector_syms.append(sym)
            gt.sectors[sym] = sec
            base_vol[sym] = rng.uniform(1.5e6, 9e6)
        sym_idx += n_leaders

        remaining = k - n_leaders
        total_frac = frac_cointegrated + frac_correlated_only + frac_regime_break
        n_coint = max(1, int(round(remaining * frac_cointegrated / total_frac)))
        n_break = int(round(remaining * frac_regime_break / total_frac))
        n_coint = min(n_coint, remaining)
        n_break = min(n_break, remaining - n_coint)
        n_corr = remaining - n_coint - n_break

        # --- cointegrated followers -----------------------------------------
        for j in range(n_coint):
            lead = sector_syms[j % n_leaders]
            sym = f"{sec}{n_leaders + j:02d}"
            beta = float(rng.uniform(0.80, 1.20))
            hl = float(rng.uniform(6.0, 25.0))
            sigma = float(rng.uniform(0.020, 0.045))
            resid = ou_process(n, hl, sigma, rng)
            log_prices[sym] = _anchor(log_prices[lead], beta, resid, rng.uniform(25, 250))
            symbols.append(sym)
            gt.sectors[sym] = sec
            base_vol[sym] = rng.uniform(1.5e6, 9e6)
            gt.cointegrated_pairs.append((sym, lead))
            gt.half_lives[GroundTruth.key(sym, lead)] = hl
            gt.betas[GroundTruth.key(sym, lead)] = beta
        sym_idx += n_coint

        # --- regime-break followers ------------------------------------------
        for j in range(n_break):
            lead = sector_syms[j % n_leaders]
            sym = f"{sec}{n_leaders + n_coint + j:02d}"
            beta = float(rng.uniform(0.85, 1.15))
            hl = float(rng.uniform(8.0, 20.0))
            sigma = float(rng.uniform(0.02, 0.04))
            brk = int(n * 0.60)
            resid = np.empty(n)
            resid[:brk] = ou_process(brk, hl, sigma, rng)
            drift = np.cumsum(rng.normal(0.0, sigma * 0.28, size=n - brk))
            resid[brk:] = resid[brk - 1] + drift          # cointegration dies here
            log_prices[sym] = _anchor(log_prices[lead], beta, resid, rng.uniform(25, 250))
            symbols.append(sym)
            gt.sectors[sym] = sec
            base_vol[sym] = rng.uniform(1.5e6, 9e6)
            gt.regime_break_pairs.append((sym, lead))
        sym_idx += n_break

        # --- correlated-but-not-cointegrated ---------------------------------
        for j in range(n_corr):
            partner = sector_syms[j % n_leaders]
            sym = f"{sec}{n_leaders + n_coint + n_break + j:02d}"
            log_prices[sym] = _correlated_not_cointegrated(
                log_prices[partner], rho=float(rng.uniform(0.80, 0.90)),
                rng=rng, target_p0=float(rng.uniform(25, 250)))
            symbols.append(sym)
            gt.sectors[sym] = sec
            base_vol[sym] = rng.uniform(1.5e6, 9e6)
            gt.correlated_only_pairs.append((sym, partner))
        sym_idx += n_corr

    # ---- transitive positives ------------------------------------------------
    # Two followers of the same leader are cointegrated with each other:
    #   log P1 = a1 + b1*L + OU1,  log P2 = a2 + b2*L + OU2
    #   =>  log P1 - (b1/b2) log P2  is stationary.
    # Omitting these from the label set would count correct selections as false
    # positives and understate the screen's precision.
    followers: Dict[str, List[str]] = {}
    for follow, lead in gt.cointegrated_pairs:
        followers.setdefault(lead, []).append(follow)
    derived: List[Tuple[str, str]] = []
    for lead, fs in followers.items():
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                derived.append((fs[i], fs[j]))
    gt.cointegrated_pairs.extend(derived)

    # ---- assemble tidy panel -------------------------------------------------
    frames = []
    for sym in symbols:
        close = np.exp(log_prices[sym])
        bars = _ohlcv_from_close(close, rng, base_vol[sym])
        df = pd.DataFrame({
            "date": dates,
            "symbol": sym,
            "open": bars["open"],
            "high": bars["high"],
            "low": bars["low"],
            "close": bars["close"],
            "adjusted_close": bars["close"],
            "volume": bars["volume"],
            "sector": gt.sectors[sym],
        })
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    for c in ["open", "high", "low", "close", "adjusted_close"]:
        panel[c] = panel[c].astype("float64").round(4)
    panel["volume"] = panel["volume"].astype("int64")
    return panel, gt


# --------------------------------------------------------------------------- #
# Small deterministic generators used directly by the statistical unit tests.
# --------------------------------------------------------------------------- #

def cointegrated_pair(n: int = 1000, beta: float = 1.0, half_life: float = 12.0,
                      sigma: float = 0.03, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lead = np.cumsum(rng.normal(0.0002, 0.01, size=n)) + np.log(100)
    resid = ou_process(n, half_life, sigma, rng)
    follow = np.log(50) + beta * lead + resid
    return pd.DataFrame({"A": np.exp(follow), "B": np.exp(lead)})


def independent_random_walks(n: int = 1000, seed: int = 0,
                             rho: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    e1 = rng.normal(0.0002, 0.01, size=n)
    e2 = rho * e1 + np.sqrt(max(1 - rho ** 2, 0.0)) * rng.normal(0.0002, 0.01, size=n)
    return pd.DataFrame({"A": np.exp(np.log(100) + np.cumsum(e1)),
                         "B": np.exp(np.log(80) + np.cumsum(e2))})


def regime_breaking_pair(n: int = 1200, break_frac: float = 0.6,
                         half_life: float = 12.0, sigma: float = 0.03,
                         seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lead = np.cumsum(rng.normal(0.0002, 0.01, size=n)) + np.log(100)
    brk = int(n * break_frac)
    resid = np.empty(n)
    resid[:brk] = ou_process(brk, half_life, sigma, rng)
    resid[brk:] = resid[brk - 1] + np.cumsum(rng.normal(0, sigma * 0.3, size=n - brk))
    follow = np.log(50) + lead + resid
    return pd.DataFrame({"A": np.exp(follow), "B": np.exp(lead)})

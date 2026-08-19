"""The research funnel, run once per walk-forward training window.

    universe -> correlation shortlist -> Engle-Granger -> BH-FDR
             -> half-life filter -> rolling stability -> score -> top N

Leakage control is structural, not conventional: :func:`select_pairs` accepts a
single price frame and physically cannot see anything else.  The walk-forward
driver slices the training window before calling it, so there is no code path
in which a future observation reaches this function.  ``selection_mode`` exists
so the ablations can disable individual stages and measure their contribution
rather than assert it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .cointegration import test_pairs
from .correlation import as_tuples, candidate_pairs
from .mean_reversion import half_life
from .multiple_testing import benjamini_hochberg, expected_false_positives
from .stability import pair_stability, quality_score
from .universe import build_universe


@dataclass
class SelectionReport:
    pairs: pd.DataFrame
    funnel: Dict[str, int] = field(default_factory=dict)
    universe_size: int = 0
    notes: List[str] = field(default_factory=list)

    def render(self) -> str:
        lines = ["RESEARCH FUNNEL", "-" * 52,
                 f"{'eligible universe':<36s}{self.universe_size:>12}"]
        for k, v in self.funnel.items():
            lines.append(f"{k:<36s}{v:>12}")
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)

    def top(self, n: int = 10) -> pd.DataFrame:
        cols = ["symbol_a", "symbol_b", "correlation", "coint_pvalue", "coint_qvalue",
                "adf_pvalue", "half_life", "stability_pass_rate", "beta",
                "quality_score"]
        cols = [c for c in cols if c in self.pairs.columns]
        return self.pairs.head(n)[cols]


def select_pairs(
    panel: pd.DataFrame,
    *,
    price_field: str = "adjusted_close",
    correlation_threshold: float = 0.60,
    cointegration_alpha: float = 0.05,
    fdr_alpha: float = 0.05,
    adf_alpha: float = 0.05,
    half_life_min: float = 2.0,
    half_life_max: float = 60.0,
    stability_windows: int = 6,
    stability_window_days: int = 252,
    min_stability_pass_rate: float = 0.50,
    max_pairs: int = 20,
    restrict_to_same_sector: bool = True,
    universe_kwargs: Optional[dict] = None,
    selection_mode: str = "full",
) -> SelectionReport:
    """Run the funnel on ONE training slice.

    ``selection_mode``:
      ``full``            every stage (the production configuration)
      ``no_stability``    skip rolling-stability filtering (Ablation B)
      ``no_fdr``          naive p < alpha instead of BH (multiple-testing demo)
      ``correlation_only`` rank by |correlation| alone (Ablation A)
    """
    from ..data.loader import log_returns, to_wide

    funnel: Dict[str, int] = {}
    notes: List[str] = []

    ureport = build_universe(panel, price_field=price_field, **(universe_kwargs or {}))
    eligible = ureport.eligible
    funnel["universe eligible"] = len(eligible)
    if len(eligible) < 2:
        return SelectionReport(pd.DataFrame(), funnel, len(eligible),
                               ["universe too small"])

    sub = panel[panel["symbol"].isin(eligible)]
    wide = to_wide(sub, price_field)
    log_px = np.log(wide)
    rets = log_returns(wide)
    sectors = (sub.drop_duplicates("symbol").set_index("symbol")["sector"].to_dict()
               if "sector" in sub.columns else None)

    n_possible = len(eligible) * (len(eligible) - 1) // 2
    funnel["all possible pairs"] = n_possible

    cands = candidate_pairs(rets, threshold=correlation_threshold, sectors=sectors,
                            restrict_to_same_sector=restrict_to_same_sector)
    funnel[f"corr >= {correlation_threshold}"] = len(cands)
    if cands.empty:
        return SelectionReport(pd.DataFrame(), funnel, len(eligible),
                               ["no correlated candidates"])

    if selection_mode == "correlation_only":
        out = cands.copy()
        out["quality_score"] = out["correlation"]
        out = out.sort_values("correlation", ascending=False).head(max_pairs)
        # Hedge ratios are still needed to define a tradeable spread.
        betas, alphas = [], []
        for a, b in as_tuples(out):
            from .hedge_ratio import static_ols_hedge_ratio
            hr = static_ols_hedge_ratio(log_px[a], log_px[b])
            betas.append(hr.beta); alphas.append(hr.alpha)
        out["beta"], out["alpha"] = betas, alphas
        funnel["selected (correlation only)"] = len(out)
        notes.append("correlation-only mode: no cointegration/FDR/stability filter")
        return SelectionReport(out.reset_index(drop=True), funnel, len(eligible), notes)

    ct = test_pairs(log_px, as_tuples(cands))
    funnel["cointegration tests run"] = len(ct)
    if ct.empty:
        return SelectionReport(pd.DataFrame(), funnel, len(eligible),
                               ["no pair produced a valid test"])

    ct = ct.merge(cands[["symbol_a", "symbol_b", "correlation", "sector"]],
                  on=["symbol_a", "symbol_b"], how="left")

    naive = int((ct["coint_pvalue"] < cointegration_alpha).sum())
    funnel[f"naive p < {cointegration_alpha}"] = naive
    exp_fp = expected_false_positives(len(ct), cointegration_alpha)
    notes.append(f"under a complete null, {len(ct)} tests at alpha="
                 f"{cointegration_alpha} would yield ~{exp_fp:.1f} false positives")

    reject, qvals, _ = benjamini_hochberg(ct["coint_pvalue"].to_numpy(), fdr_alpha)
    ct["coint_qvalue"] = qvals
    ct["fdr_reject"] = reject
    funnel[f"BH-FDR q < {fdr_alpha}"] = int(reject.sum())

    kept = ct[ct["coint_pvalue"] < cointegration_alpha] if selection_mode == "no_fdr" \
        else ct[ct["fdr_reject"]]
    if selection_mode == "no_fdr":
        notes.append("no_fdr mode: naive p-value threshold, no multiplicity control")
    if kept.empty:
        return SelectionReport(pd.DataFrame(), funnel, len(eligible),
                               notes + ["no pair survived multiple-testing control"])

    kept = kept[kept["adf_pvalue"] < adf_alpha].copy()
    funnel[f"residual ADF p < {adf_alpha}"] = len(kept)
    if kept.empty:
        return SelectionReport(pd.DataFrame(), funnel, len(eligible),
                               notes + ["no pair passed the residual ADF check"])

    hls, hl_ok = [], []
    for _, r in kept.iterrows():
        spread = log_px[r["symbol_a"]] - r["alpha"] - r["beta"] * log_px[r["symbol_b"]]
        try:
            hl = half_life(spread.dropna())
            hls.append(hl.half_life if hl.mean_reverting else np.nan)
        except ValueError:
            hls.append(np.nan)
        hl_ok.append(np.isfinite(hls[-1]) and half_life_min <= hls[-1] <= half_life_max)
    kept["half_life"] = hls
    kept = kept[pd.Series(hl_ok, index=kept.index)].copy()
    funnel[f"half-life in [{half_life_min:g},{half_life_max:g}]d"] = len(kept)
    if kept.empty:
        return SelectionReport(pd.DataFrame(), funnel, len(eligible),
                               notes + ["no pair had a tradeable half-life"])

    if selection_mode == "no_stability":
        for c in ["stability_pass_rate", "beta_cv"]:
            kept[c] = np.nan
        kept["quality_score"] = 1.0 - kept["coint_pvalue"]
        notes.append("no_stability mode: rolling-window stability filter disabled")
    else:
        srows = []
        for _, r in kept.iterrows():
            st = pair_stability(log_px[r["symbol_a"]], log_px[r["symbol_b"]],
                                n_windows=stability_windows,
                                window_days=stability_window_days,
                                alpha=cointegration_alpha,
                                symbol_a=r["symbol_a"], symbol_b=r["symbol_b"])
            srows.append(st.to_dict())
        kept = pd.concat([kept.reset_index(drop=True),
                          pd.DataFrame(srows)], axis=1)
        kept = kept[kept["stability_pass_rate"] >= min_stability_pass_rate].copy()
        funnel[f"stability >= {min_stability_pass_rate:.0%}"] = len(kept)
        if kept.empty:
            return SelectionReport(pd.DataFrame(), funnel, len(eligible),
                                   notes + ["no pair was stable across sub-windows"])
        kept["quality_score"] = kept.apply(
            lambda r: quality_score(r, half_life_min=half_life_min,
                                    half_life_max=half_life_max), axis=1)

    kept = kept.sort_values("quality_score", ascending=False).head(max_pairs)
    funnel["selected"] = len(kept)
    return SelectionReport(kept.reset_index(drop=True), funnel, len(eligible), notes)


def score_against_truth(selected: pd.DataFrame, ground_truth) -> dict:
    """Precision / recall of pair selection against synthetic ground truth.

    Only meaningful in synthetic mode.  This is a statement about the *selection
    procedure*, not about profitability, and it is the one performance-shaped
    number in this repository that survives having been produced on simulated
    data -- because the data-generating process is known rather than assumed.
    """
    if selected is None or selected.empty:
        return {"selected": 0, "true_positives": 0, "decoys_selected": 0,
                "precision": float("nan"), "recall": 0.0,
                "decoy_admission_rate": 0.0, "regime_break_selected": 0}
    key = ground_truth.key
    chosen = {key(a, b) for a, b in zip(selected["symbol_a"], selected["symbol_b"])}
    pos, neg, unstable = (ground_truth.positive_set(), ground_truth.negative_set(),
                          ground_truth.unstable_set())
    tp = len(chosen & pos)
    decoys = len(chosen & neg)
    breaks = len(chosen & unstable)
    return {
        "selected": len(chosen),
        "true_positives": tp,
        # The metric that matters: a correlated-but-not-cointegrated pair that
        # the screen let through. These are untradeable by construction.
        "decoys_selected": decoys,
        "decoy_admission_rate": decoys / len(chosen) if chosen else float("nan"),
        # Regime-break pairs are genuinely cointegrated *before* the break, so a
        # window that predates it selecting them is not an error. Reported
        # separately and never folded into precision.
        "regime_break_selected": breaks,
        "unclassified_selected": len(chosen - pos - neg - unstable),
        "precision": tp / len(chosen) if chosen else float("nan"),
        "recall": tp / len(pos) if pos else float("nan"),
        "n_true_pairs": len(pos), "n_decoy_pairs": len(neg),
    }

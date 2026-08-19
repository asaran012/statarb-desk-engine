"""Pre-trade risk limits.

Checked *before* a fill, against the equity mark at the moment of the trade.
Breaches scale the order down rather than rejecting outright, because a desk
that flips between full size and no size is worse than one that trims; below
``min_fraction`` of intended size the trade is dropped and logged so the
rejection is visible in the report instead of vanishing.

Limits are also asserted after the fact: ``check_gross_exposure_limit`` in
:mod:`statarb.backtest.accounting` re-derives peak gross exposure from the
equity history, so a sizing bug cannot hide behind a passing pre-trade check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class RiskLimits:
    max_pair_gross: float = 0.10
    max_symbol_exposure: float = 0.15
    max_portfolio_gross: float = 1.00
    min_fraction: float = 0.10
    enabled: bool = True

    def scale_for_trade(
        self,
        *,
        equity: float,
        intended_notional: float,
        current_gross: float,
        symbol_exposures: Dict[str, float],
        leg_notionals: Dict[str, float],
    ) -> Tuple[float, Optional[str]]:
        """Return ``(scale in [0,1], binding_limit_name_or_None)``.

        ``leg_notionals`` is the *intended* absolute dollar exposure per symbol
        for this trade, used to check the post-trade symbol concentration.
        """
        if not self.enabled:
            return 1.0, None
        if equity <= 0 or intended_notional <= 0:
            return 0.0, "non_positive_equity"

        scale, binding = 1.0, None

        pair_cap = self.max_pair_gross * equity
        if intended_notional > pair_cap:
            s = pair_cap / intended_notional
            if s < scale:
                scale, binding = s, "max_pair_gross"

        room = self.max_portfolio_gross * equity - current_gross
        if room <= 0:
            return 0.0, "max_portfolio_gross"
        if intended_notional * scale > room:
            s = room / intended_notional
            if s < scale:
                scale, binding = s, "max_portfolio_gross"

        sym_cap = self.max_symbol_exposure * equity
        for sym, leg in leg_notionals.items():
            post = symbol_exposures.get(sym, 0.0) + leg * scale
            if post > sym_cap:
                headroom = sym_cap - symbol_exposures.get(sym, 0.0)
                s = max(headroom, 0.0) / leg if leg > 0 else 0.0
                if s < scale:
                    scale, binding = s, "max_symbol_exposure"

        if scale < self.min_fraction:
            return 0.0, binding or "below_min_fraction"
        return float(min(max(scale, 0.0), 1.0)), binding

"""Transaction-cost model.

Costs are charged as a proportional haircut on *traded notional*, split into
commission and slippage:

    cost = notional_traded * (commission_bps + slippage_bps) / 10_000

Modelling slippage as a symmetric proportional charge (rather than simulating a
bid/ask crossing) is a deliberate limitation, not an oversight.  The dataset has
no quote data, so any spread-crossing simulation would be an invented number
dressed up as a measurement.  To first order the two are equivalent: buying at
``P*(1+s)`` and selling at ``P*(1-s)`` costs ``s`` of notional either way.

What this model does NOT capture, and the README repeats it:
  * market impact that grows with size (we are linear in notional at any size)
  * borrow cost and short availability on the short leg
  * financing on the long leg
  * the fact that a spread blowing out is exactly when liquidity is worst,
    so realised slippage is correlated with the stop-loss exits
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 1.0
    slippage_bps: float = 2.0

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.slippage_bps

    def charge(self, notional_traded: float) -> float:
        """Non-negative cash cost for turning over ``|notional_traded|``."""
        return abs(float(notional_traded)) * self.total_bps / 10_000.0

    def breakdown(self, notional_traded: float) -> dict:
        n = abs(float(notional_traded))
        return {"commission": n * self.commission_bps / 10_000.0,
                "slippage": n * self.slippage_bps / 10_000.0,
                "total": self.charge(n)}

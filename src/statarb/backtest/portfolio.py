"""Portfolio bookkeeping.

Cash-and-positions accounting, deliberately boring:

    equity_t = cash_t + sum_over_symbols(qty_s * price_s,t)

Opening a pair leg moves cash by ``-qty * fill_price`` (so a short leg *adds*
cash) and charges the cost separately.  Because every state change goes through
:meth:`Portfolio.trade`, the identity
``ending_equity == starting_equity + sum(net_pnl)`` holds exactly once all
positions are flat, and :mod:`statarb.backtest.accounting` asserts it after
every run rather than trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .costs import CostModel


@dataclass
class OpenTrade:
    pair: str
    symbol_a: str
    symbol_b: str
    direction: int              # +1 long spread, -1 short spread
    qty_a: float
    qty_b: float
    entry_date: pd.Timestamp
    entry_signal_date: pd.Timestamp
    entry_price_a: float
    entry_price_b: float
    entry_z: float
    entry_cost: float
    beta: float
    notional: float

    @property
    def gross_notional(self) -> float:
        return abs(self.qty_a * self.entry_price_a) + abs(self.qty_b * self.entry_price_b)


class Portfolio:
    def __init__(self, initial_capital: float, cost_model: CostModel):
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.cost_model = cost_model
        self.positions: Dict[str, float] = {}
        self.open_trades: Dict[str, OpenTrade] = {}
        self.trades: List[dict] = []
        self.equity_history: List[dict] = []
        self.rejected_trades: List[dict] = []
        self.position_events: int = 0
        self.total_traded_notional: float = 0.0
        self.total_costs: float = 0.0
        self.post_trade_gross_ratios: List[float] = []

    # ---- primitives -------------------------------------------------------
    def _apply_leg(self, symbol: str, qty: float, price: float) -> None:
        self.cash -= qty * price
        new = self.positions.get(symbol, 0.0) + qty
        if abs(new) < 1e-12:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = new

    def market_value(self, prices: pd.Series) -> float:
        if not self.positions:
            return 0.0
        return float(sum(q * float(prices.get(s, np.nan))
                         for s, q in self.positions.items()))

    def equity(self, prices: pd.Series) -> float:
        return self.cash + self.market_value(prices)

    def gross_exposure(self, prices: pd.Series) -> float:
        return float(sum(abs(q * float(prices.get(s, np.nan)))
                         for s, q in self.positions.items()))

    def net_exposure(self, prices: pd.Series) -> float:
        return self.market_value(prices)

    def symbol_exposure(self, prices: pd.Series) -> Dict[str, float]:
        return {s: abs(q * float(prices.get(s, np.nan)))
                for s, q in self.positions.items()}

    # ---- trade lifecycle --------------------------------------------------
    def open_pair(self, *, pair: str, symbol_a: str, symbol_b: str, direction: int,
                  qty_a: float, qty_b: float, price_a: float, price_b: float,
                  date: pd.Timestamp, signal_date: pd.Timestamp, zscore: float,
                  beta: float, notional: float) -> OpenTrade:
        gross = abs(qty_a * price_a) + abs(qty_b * price_b)
        cost = self.cost_model.charge(gross)
        self._apply_leg(symbol_a, qty_a, price_a)
        self._apply_leg(symbol_b, qty_b, price_b)
        self.cash -= cost
        self.total_traded_notional += gross
        self.total_costs += cost
        self.position_events += 1

        t = OpenTrade(pair=pair, symbol_a=symbol_a, symbol_b=symbol_b,
                      direction=direction, qty_a=qty_a, qty_b=qty_b,
                      entry_date=date, entry_signal_date=signal_date,
                      entry_price_a=price_a, entry_price_b=price_b,
                      entry_z=zscore, entry_cost=cost, beta=beta, notional=notional)
        self.open_trades[pair] = t
        return t

    def record_post_trade_exposure(self, prices: pd.Series) -> None:
        eq = self.equity(prices)
        if eq > 0:
            self.post_trade_gross_ratios.append(self.gross_exposure(prices) / eq)

    def close_pair(self, pair: str, *, price_a: float, price_b: float,
                   date: pd.Timestamp, zscore: float, reason: str) -> dict:
        t = self.open_trades.pop(pair)
        gross_notional = abs(t.qty_a * price_a) + abs(t.qty_b * price_b)
        cost = self.cost_model.charge(gross_notional)
        self._apply_leg(t.symbol_a, -t.qty_a, price_a)
        self._apply_leg(t.symbol_b, -t.qty_b, price_b)
        self.cash -= cost
        self.total_traded_notional += gross_notional
        self.total_costs += cost
        self.position_events += 1

        gross_pnl = (t.qty_a * (price_a - t.entry_price_a)
                     + t.qty_b * (price_b - t.entry_price_b))
        total_cost = t.entry_cost + cost
        net_pnl = gross_pnl - total_cost
        holding = int(np.busday_count(t.entry_date.date(), pd.Timestamp(date).date()))

        rec = {
            "pair": pair, "symbol_a": t.symbol_a, "symbol_b": t.symbol_b,
            "direction": "long_spread" if t.direction > 0 else "short_spread",
            "entry_signal_date": t.entry_signal_date,
            "entry_date": t.entry_date, "exit_date": date,
            "entry_price_a": t.entry_price_a, "entry_price_b": t.entry_price_b,
            "exit_price_a": price_a, "exit_price_b": price_b,
            "qty_a": t.qty_a, "qty_b": t.qty_b, "beta": t.beta,
            "entry_z": t.entry_z, "exit_z": zscore,
            "holding_period": holding,
            "entry_notional": t.gross_notional,
            "gross_pnl": gross_pnl, "cost": total_cost, "net_pnl": net_pnl,
            "return": net_pnl / t.gross_notional if t.gross_notional > 0 else np.nan,
            "exit_reason": reason,
        }
        self.trades.append(rec)
        return rec

    def reject(self, pair: str, date: pd.Timestamp, reason: str, detail: str = "") -> None:
        self.rejected_trades.append({"pair": pair, "date": date, "reason": reason,
                                     "detail": detail})

    def mark(self, date: pd.Timestamp, prices: pd.Series) -> None:
        eq = self.equity(prices)
        self.equity_history.append({
            "date": date, "equity": eq, "cash": self.cash,
            "gross_exposure": self.gross_exposure(prices),
            "net_exposure": self.net_exposure(prices),
            "n_positions": len(self.positions),
            "n_open_pairs": len(self.open_trades),
        })

    # ---- outputs ----------------------------------------------------------
    def trades_frame(self) -> pd.DataFrame:
        cols = ["pair", "symbol_a", "symbol_b", "direction", "entry_signal_date",
                "entry_date", "exit_date", "entry_z", "exit_z", "holding_period",
                "entry_notional", "gross_pnl", "cost", "net_pnl", "return",
                "exit_reason", "beta", "qty_a", "qty_b", "entry_price_a",
                "entry_price_b", "exit_price_a", "exit_price_b"]
        if not self.trades:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(self.trades)[cols]

    def equity_frame(self) -> pd.DataFrame:
        if not self.equity_history:
            return pd.DataFrame(columns=["date", "equity", "cash", "gross_exposure",
                                         "net_exposure", "n_positions",
                                         "n_open_pairs"]).set_index("date")
        return pd.DataFrame(self.equity_history).set_index("date")

    def rejected_frame(self) -> pd.DataFrame:
        return (pd.DataFrame(self.rejected_trades)
                if self.rejected_trades
                else pd.DataFrame(columns=["pair", "date", "reason", "detail"]))

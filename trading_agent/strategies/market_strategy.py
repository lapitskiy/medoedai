from __future__ import annotations

from typing import Optional
import time

from trading_agent.domain.interfaces import ExecutionStrategy, StateStore, ExchangeGateway
from trading_agent.domain.models import Intent, IntentState, LimitConfig, Side


class MarketStrategy(ExecutionStrategy):
    """Заглушка: пока просто создаёт мгновенный intent FILLED (интеграция позже)."""

    def __init__(self, store: StateStore, gateway: ExchangeGateway):
        self.store = store
        self.gateway = gateway

    def place_intent(self, symbol: str, side: Side, qty: float, cfg: Optional[LimitConfig] = None) -> Intent:
        intent = Intent(
            intent_id=f"market-{symbol}-{int(time.time()*1000)}",
            symbol=symbol,
            side=side,
            qty_total=qty,
            qty_remaining=0.0,
            state=IntentState.FILLED,
        )
        self.store.save_intent(intent)
        self.store.set_state(intent.intent_id, IntentState.FILLED)
        return intent

    def run_until_done(self, intent: Intent) -> Intent:
        return self.store.load_intent(intent.intent_id) or intent

    def cancel_intent(self, intent_id: str) -> None:
        self.store.set_state(intent_id, IntentState.CANCELLED)



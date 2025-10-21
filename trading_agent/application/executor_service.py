from __future__ import annotations

from typing import Optional, Dict
import time

from trading_agent.domain.models import Intent, LimitConfig, Side, IntentState
from trading_agent.domain.interfaces import ExecutionStrategy, StateStore


class ExecutionService:
    """
    Сервис верхнего уровня: выбор стратегии исполнения, guard от параллельных intent'ов,
    соблюдение бизнес-ограничений (max_lifetime_sec ≤ 300 и т.п.).
    """

    def __init__(self, store: StateStore, strategies: Dict[str, ExecutionStrategy]):
        self.store = store
        self.strategies = strategies

    def start(self, execution_mode: str, symbol: str, side: Side, qty: float, cfg: Optional[LimitConfig] = None) -> Intent:
        # Guard: один активный intent на символ
        if self.store.has_active_intent(symbol):
            raise RuntimeError(f"Active intent exists for {symbol}; new signals are ignored until completion")

        strategy = self._get_strategy(execution_mode)
        intent = strategy.place_intent(symbol, side, qty, cfg)

        # Жёсткая защита на max_lifetime_sec из доменной конфигурации
        max_sec = min(300, (cfg.max_lifetime_sec if cfg else 300))
        deadline = time.time() + max_sec

        intent = strategy.run_until_done(intent)

        # Если стратегия по какой-то причине не завершила (защёлка)
        if intent.state not in (IntentState.FILLED, IntentState.CANCELLED, IntentState.FAILED, IntentState.EXPIRED):
            if time.time() >= deadline:
                try:
                    strategy.cancel_intent(intent.intent_id)
                finally:
                    self.store.set_state(intent.intent_id, IntentState.EXPIRED, last_error="deadline exceeded")
        return self.store.load_intent(intent.intent_id) or intent

    def _get_strategy(self, mode: str) -> ExecutionStrategy:
        if mode not in self.strategies:
            raise ValueError(f"Unknown execution_mode: {mode}")
        return self.strategies[mode]



from __future__ import annotations

import json
import time
from typing import Optional

import redis

from trading_agent.domain.interfaces import StateStore
from trading_agent.domain.models import Intent, IntentState, Side, LimitConfig


class RedisStateStore(StateStore):
    def __init__(self, host: str = 'redis', port: int = 6379, db: int = 0):
        self.r = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    def _k_intent(self, intent_id: str) -> str:
        return f"exec:intent:{intent_id}"

    def _k_pending(self, symbol: str) -> str:
        return f"exec:pending:{symbol}"

    def _k_symbol_active(self, symbol: str) -> str:
        return f"exec:active_intent:{symbol}"

    def save_intent(self, intent: Intent) -> None:
        self.r.set(self._k_intent(intent.intent_id), json.dumps(intent, default=lambda o: o.__dict__))
        self.r.set(self._k_symbol_active(intent.symbol), intent.intent_id)

    def load_intent(self, intent_id: str) -> Optional[Intent]:
        raw = self.r.get(self._k_intent(intent_id))
        if not raw:
            return None
        data = json.loads(raw)
        # Реконструкция Side (enum) из сериализованного значения
        def _restore_side(val):
            try:
                if isinstance(val, str):
                    return Side(val.lower())
                if isinstance(val, dict):
                    v = val.get('value') or val.get('_value_') or val.get('name') or ''
                    return Side(str(v).lower())
                return Side(val)
            except Exception:
                return Side.BUY

        side = _restore_side(data.get('side'))

        # лёгкая реконструкция (достаточно для хранения и UI)
        intent = Intent(
            intent_id=data['intent_id'],
            symbol=data['symbol'],
            side=side,
            qty_total=data['qty_total'],
            qty_remaining=data['qty_remaining'],
        )
        # перенос остальных полей
        intent.state = IntentState(data.get('state', 'pending'))
        intent.created_at = data.get('created_at', time.time())
        intent.updated_at = data.get('updated_at', time.time())
        intent.post_only = bool(data.get('post_only', True))
        intent.tick_size = data.get('tick_size')
        intent.price = data.get('price')
        intent.offset_ticks = int(data.get('offset_ticks', 1))
        intent.attempts = int(data.get('attempts', 0))
        intent.last_error = data.get('last_error')
        intent.logs = list(data.get('logs', []))
        intent.exchange_order_id = data.get('exchange_order_id')
        # Восстановление cfg (важно для SLA-реквота и TTL)
        try:
            cfg = data.get('cfg') or {}
            intent.cfg = LimitConfig(
                requote_interval_sec=int((cfg or {}).get('requote_interval_sec', intent.cfg.requote_interval_sec)),
                max_lifetime_sec=int((cfg or {}).get('max_lifetime_sec', intent.cfg.max_lifetime_sec)),
                offset_max_ticks=int((cfg or {}).get('offset_max_ticks', intent.cfg.offset_max_ticks)),
            )
        except Exception:
            pass

        return intent

    def update_intent(self, intent: Intent) -> None:
        intent.updated_at = time.time()
        self.save_intent(intent)

    def set_state(self, intent_id: str, state: IntentState, last_error: Optional[str] = None) -> None:
        it = self.load_intent(intent_id)
        if not it:
            return
        it.state = state
        it.last_error = last_error
        it.updated_at = time.time()
        self.update_intent(it)
        if state in (IntentState.FILLED, IntentState.CANCELLED, IntentState.FAILED, IntentState.EXPIRED):
            # снять active
            self.r.delete(self._k_symbol_active(it.symbol))
            self.r.srem(self._k_pending(it.symbol), intent_id)

    def append_log(self, intent_id: str, price: Optional[float], event: str, reason: Optional[str] = None) -> None:
        it = self.load_intent(intent_id)
        if not it:
            return
        it.logs.append({"ts": time.time(), "price": price, "event": event, "reason": reason})
        self.update_intent(it)

    def bump_attempts(self, intent_id: str, delta: int = 1) -> None:
        it = self.load_intent(intent_id)
        if not it:
            return
        it.attempts += delta
        self.update_intent(it)

    def add_pending(self, symbol: str, intent_id: str) -> None:
        self.r.sadd(self._k_pending(symbol), intent_id)

    def remove_pending(self, symbol: str, intent_id: str) -> None:
        self.r.srem(self._k_pending(symbol), intent_id)

    def has_active_intent(self, symbol: str) -> bool:
        return bool(self.r.get(self._k_symbol_active(symbol)))



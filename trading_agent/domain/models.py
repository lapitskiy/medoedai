from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, List
import time


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class IntentState(str, Enum):
    PENDING = "pending"            # создан, ждёт размещения начального ордера
    WORKING = "working"            # ордер размещён и поддерживается
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class LimitConfig:
    requote_interval_sec: int = 15
    max_lifetime_sec: int = 300   # ≤ 5 минут
    # Сколько тиков максимум допускаем отходить от edge (best bid/ask) в post-only.
    # Дефолт 2, чтобы лимитка всегда была "рядом" (1-2 тика), и не улетала на 8-16 тиков при ретраях.
    offset_max_ticks: int = 2


@dataclass
class Intent:
    intent_id: str
    symbol: str
    side: Side
    qty_total: float
    qty_remaining: float
    state: IntentState = IntentState.PENDING
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    exchange_order_id: Optional[str] = None
    post_only: bool = True
    tick_size: Optional[float] = None
    price: Optional[float] = None
    offset_ticks: int = 1
    attempts: int = 0
    last_error: Optional[str] = None
    cfg: LimitConfig = field(default_factory=LimitConfig)
    logs: List[Dict[str, Any]] = field(default_factory=list)  # попытки: price, reason, ts

    def add_log(self, price: Optional[float], event: str, reason: Optional[str] = None) -> None:
        self.logs.append({
            "ts": time.time(),
            "price": price,
            "event": event,
            "reason": reason,
        })
        self.updated_at = time.time()



from __future__ import annotations

from typing import Tuple, Dict, Any

from trading_agent.domain.interfaces import ExchangeGateway
from trading_agent.domain.models import Side


class ExchangeGatewayStub(ExchangeGateway):
    """Заглушка для начальной интеграции. Реализуйте Bybit REST/WS отдельно."""

    def get_tick_size(self, symbol: str) -> float:
        return 0.1

    def get_best_bid_ask(self, symbol: str) -> Tuple[float, float]:
        return (100.0, 100.5)

    def place_limit_post_only(self, symbol: str, side: Side, qty: float, price: float) -> Dict[str, Any]:
        return {"id": "stub-order-id", "price": price, "qty": qty, "side": side.value}

    def amend_order(self, symbol: str, order_id: str, price: float) -> Dict[str, Any]:
        return {"id": order_id, "price": price}

    def cancel_order(self, symbol: str, order_id: str) -> None:
        return None

    def get_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
        return {"id": order_id, "status": "working"}

    def connect_ws(self) -> None:
        return None

    def subscribe_orders(self, symbol: str, on_event: callable) -> None:
        return None

    def subscribe_ticker(self, symbol: str, on_event: callable) -> None:
        return None



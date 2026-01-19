from __future__ import annotations

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def setup_trailing_stop_bybit(
    exchange,
    symbol: str,
    qty: float,
    entry_price: float,
    mode: Optional[str],
    activate_mode: Optional[str],
    activate_value: Optional[float],
    atr_trail_k: float,
    get_atr_1h_func=None,
) -> Dict[str, Any]:
    """
    Создаёт биржевой trailing-stop на Bybit, опираясь на ATR.

    Инкапсулирует доменную логику, используется и из HTTP-роутов, и из Celery-задач.
    """
    if exchange is None:
        raise RuntimeError("exchange object not available for trailing setup")

    exch_id = str(getattr(exchange, "id", "")).lower()
    if exch_id != "bybit":
        raise RuntimeError(f"Trailing stop currently supported only on Bybit (got {exch_id})")

    mode = (mode or "").strip().lower()
    if mode and mode != "atr_trailing":
        raise RuntimeError(f"Trailing mode '{mode}' not supported for exchange trailing")

    if atr_trail_k is None or atr_trail_k <= 0:
        raise RuntimeError("atr_trail_mult must be positive")

    # ATR
    if get_atr_1h_func is None:
        from utils.indicators import get_atr_1h as get_atr_1h_func  # lazy import to avoid cycles

    atr_abs, _, _ = get_atr_1h_func(symbol, length=21)
    if atr_abs <= 0 or entry_price <= 0:
        raise RuntimeError("Invalid ATR or entry price for trailing")

    # Bybit v5 trailing-stop задаётся как trailing distance (в цене), а не callbackRate через order/create.
    # Используем price-distance на базе ATR.
    trailing_dist = float(atr_trail_k) * float(atr_abs)
    if trailing_dist <= 0:
        raise RuntimeError("Invalid trailing distance computed from ATR")

    # Цена активации
    active_price = None
    act_mode = (activate_mode or "percent").strip().lower()
    act_value = activate_value if activate_value is not None else 0.5
    if act_mode == "atr":
        active_price = entry_price + float(act_value) * atr_abs
    else:  # percent
        active_price = entry_price * (1.0 + float(act_value) / 100.0)

    # Нормализуем цены под tickSize
    try:
        market = exchange.market(symbol)
        price_tick = None
        try:
            info = market.get("info", {})
            pf = info.get("priceFilter", {})
            price_tick = float(pf.get("tickSize")) if pf.get("tickSize") else None
        except Exception:
            price_tick = None
        if price_tick:
            if active_price:
                active_price = round(round(active_price / price_tick) * price_tick, 8)
            trailing_dist = round(round(trailing_dist / price_tick) * price_tick, 8)
    except Exception:
        pass

    request: Dict[str, Any] = {"category": "linear", "symbol": symbol}
    # Bybit v5 expects trailingStop as price distance
    request["trailingStop"] = f"{float(trailing_dist):.8f}"
    # Trigger activation price (optional but recommended)
    if active_price and float(active_price) > 0:
        request["activePrice"] = f"{float(active_price):.8f}"
    # One-way mode by default
    request["positionIdx"] = 0

    try:
        logger.info(
            f"[trailing_setup] symbol={symbol} qty={qty} entry={entry_price} atr_abs={atr_abs} "
            f"trailingStop={request.get('trailingStop')} activePrice={request.get('activePrice')}"
        )
    except Exception:
        pass

    # Bybit v5: set trading stop for position (supports trailingStop + activePrice)
    call = (
        getattr(exchange, "privatePostV5PositionTradingStop", None)
        or getattr(exchange, "privatePostV5PositionSetTradingStop", None)
    )
    if call is None:
        raise RuntimeError("ccxt bybit method for v5 position trading-stop not found on exchange object")
    response = call(request)
    try:
        # Bybit v5 обычно возвращает retCode/retMsg/result
        rc = response.get("retCode") if isinstance(response, dict) else None
        rm = response.get("retMsg") if isinstance(response, dict) else None
        logger.info(f"[trailing_setup] bybit_response retCode={rc} retMsg={rm}")
    except Exception:
        pass
    return {
        "exchange_id": exch_id,
        "request": request,
        "trailing_dist": trailing_dist,
        "active_price": active_price,
        "response": response,
    }



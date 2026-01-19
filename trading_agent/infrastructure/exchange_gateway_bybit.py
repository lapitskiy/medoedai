from __future__ import annotations

import os
import threading
import time
from typing import Tuple, Dict, Any, Optional, Callable

import ccxt
import redis

from trading_agent.domain.interfaces import ExchangeGateway
from trading_agent.domain.models import Side
from utils.settings_store import ensure_settings_table, get_setting_value


class BybitExchangeGateway(ExchangeGateway):
    """
    Реализация ExchangeGateway для Bybit на базе ccxt (REST) + простая подписка тикера.
    - Заказы: place/amend/cancel/status через ccxt unified API
    - Ticker: публичный поток реализован опросом (fallback) с коллбэком каждые ~1-2с.
    Приватный WS для ордеров планируется добавить; пока используется опрос статуса при необходимости.
    """

    def __init__(self, symbol_for_markets: Optional[str] = None):
        # Строгий режим: если выбран account_id в Redis — НИКАКИХ фолбэков.
        selected = None
        try:
            r = redis.Redis(host='redis', port=6379, db=0, decode_responses=True, socket_connect_timeout=2)
            selected = None
            try:
                if symbol_for_markets:
                    sym = str(symbol_for_markets).strip()
                    if sym:
                        selected = r.get(f'trading:account_id:{sym}')
            except Exception:
                selected = None
            if selected is None:
                selected = r.get('trading:account_id')
            selected = str(selected).strip() if selected else None
        except Exception:
            selected = None

        if selected:
            try:
                ensure_settings_table()
                api_key = get_setting_value('api', 'bybit', f'BYBIT_{selected}_API_KEY')
                secret = get_setting_value('api', 'bybit', f'BYBIT_{selected}_SECRET_KEY')
            except Exception:
                api_key = None
                secret = None
        else:
            try:
                ensure_settings_table()
                api_key = get_setting_value('api', 'bybit', 'BYBIT_1_API_KEY') or os.getenv('BYBIT_API_KEY')
                secret = get_setting_value('api', 'bybit', 'BYBIT_1_SECRET_KEY') or os.getenv('BYBIT_SECRET_KEY')
            except Exception:
                api_key = os.getenv('BYBIT_1_API_KEY') or os.getenv('BYBIT_API_KEY')
                secret = os.getenv('BYBIT_1_SECRET_KEY') or os.getenv('BYBIT_SECRET_KEY')
        if not api_key or not secret:
            if selected:
                raise RuntimeError(
                    f'Bybit API keys not set for selected account id={selected} '
                    f'(need BYBIT_{selected}_API_KEY and BYBIT_{selected}_SECRET_KEY)'
                )
            raise RuntimeError('Bybit API keys not set in environment')
        self.ex = ccxt.bybit({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {
                'defaultType': 'swap',
                'defaultMarginMode': 'isolated',
                'recv_window': 20000,
                'recvWindow': 20000,
                'adjustForTimeDifference': True,
                'timeDifference': True,
            }
        })
        self.ex.load_markets()
        self._ticker_thread: Optional[threading.Thread] = None
        self._ticker_stop = threading.Event()
        self._ticker_cb: Optional[Callable[[dict], None]] = None
        self._ticker_symbol: Optional[str] = None

    # === REST helpers ===
    def get_tick_size(self, symbol: str) -> float:
        try:
            m = self.ex.market(symbol)
            # попытка извлечь шаг цены из info или precision
            tick = None
            try:
                info = m.get('info', {})
                prc = info.get('priceFilter', {})
                if 'tickSize' in prc:
                    tick = float(prc['tickSize'])
            except Exception:
                tick = None
            if tick is None:
                precision = m.get('precision', {}).get('price', 2)
                tick = 10 ** (-precision)
            return float(tick)
        except Exception:
            return 0.01

    def get_best_bid_ask(self, symbol: str) -> Tuple[float, float]:
        ob = self.ex.fetch_order_book(symbol, limit=5)
        bid = float(ob['bids'][0][0]) if ob.get('bids') else 0.0
        ask = float(ob['asks'][0][0]) if ob.get('asks') else 0.0
        return bid, ask

    def _normalize_qty(self, symbol: str, qty: float) -> float:
        """
        Нормализует количество под требования биржи:
        - не меньше min amount
        - не больше max amount (если задан)
        - округление/квантование по правилам ccxt (amount_to_precision)
        """
        try:
            q = float(qty or 0.0)
        except Exception:
            q = 0.0
        try:
            m = self.ex.market(symbol) or {}
            limits = (m.get('limits', {}) or {}).get('amount', {}) or {}
            min_q = float(limits.get('min') or 0.0)
            max_raw = limits.get('max')
            max_q = float(max_raw) if max_raw not in (None, '', 0) else None

            if min_q > 0:
                q = max(q, min_q)
            if max_q is not None:
                q = min(q, max_q)

            # ccxt обычно возвращает строку, уже квантованную по шагу/precision
            q = float(self.ex.amount_to_precision(symbol, q))

            # На всякий случай: если округление "вниз" уронило ниже минимума
            if min_q > 0 and q < min_q:
                q = float(self.ex.amount_to_precision(symbol, min_q))
            if max_q is not None and q > max_q:
                q = float(self.ex.amount_to_precision(symbol, max_q))

            return float(q)
        except Exception:
            return float(q)

    def place_limit_post_only(self, symbol: str, side: Side, qty: float, price: float, extra_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        qty = self._normalize_qty(symbol, qty)
        params = {
            'postOnly': True,
            'timeInForce': 'GTC',
            'reduceOnly': False,
        }
        try:
            if isinstance(extra_params, dict) and extra_params:
                # Поверх дефолтов — чтобы postOnly/timeInForce не были затёрты
                params.update(extra_params)
        except Exception:
            pass
        if side == Side.BUY:
            order = self.ex.create_limit_buy_order(symbol, qty, price, params)
        else:
            order = self.ex.create_limit_sell_order(symbol, qty, price, params)
        return order

    def amend_order(self, symbol: str, order_id: str, price: float) -> Dict[str, Any]:
        try:
            # ccxt unified edit_order
            return self.ex.edit_order(order_id, symbol, None, None, price)
        except Exception:
            # Fallback: cancel + place new at same qty unknown -> status check first
            st = self.get_order_status(symbol, order_id)
            amt = st.get('amount') or st.get('remaining') or st.get('filled')
            side = (st.get('side') or 'buy').lower()
            if st:
                try:
                    self.cancel_order(symbol, order_id)
                except Exception:
                    pass
                try:
                    amt = self._normalize_qty(symbol, float(amt or 0.0))
                except Exception:
                    amt = amt
                if side == 'buy':
                    return self.ex.create_limit_buy_order(symbol, amt, price, {'postOnly': True, 'timeInForce': 'GTC'})
                else:
                    return self.ex.create_limit_sell_order(symbol, amt, price, {'postOnly': True, 'timeInForce': 'GTC'})
            raise

    def cancel_order(self, symbol: str, order_id: str) -> None:
        self.ex.cancel_order(order_id, symbol)

    def get_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
        try:
            return self.ex.fetch_order(order_id, symbol) or {}
        except Exception:
            return {"id": order_id, "status": "unknown"}

    # === WS lifecycle (ticker polling fallback) ===
    def connect_ws(self) -> None:
        # polling fallback — ничего делать не нужно
        return None

    def _ticker_worker(self, symbol: str) -> None:
        while not self._ticker_stop.is_set():
            try:
                bid, ask = self.get_best_bid_ask(symbol)
                if self._ticker_cb:
                    self._ticker_cb({'best_bid': bid, 'best_ask': ask})
            except Exception:
                pass
            time.sleep(1.5)

    def subscribe_orders(self, symbol: str, on_event: callable) -> None:
        # TODO: реализовать приватный WS; пока опрос статуса выполняется стратегией при amend/place
        return None

    def subscribe_ticker(self, symbol: str, on_event: callable) -> None:
        self._ticker_cb = on_event
        self._ticker_symbol = symbol
        self._ticker_stop.clear()
        if self._ticker_thread and self._ticker_thread.is_alive():
            return
        self._ticker_thread = threading.Thread(target=self._ticker_worker, args=(symbol,), daemon=True)
        self._ticker_thread.start()

    def stop(self) -> None:
        self._ticker_stop.set()
        if self._ticker_thread and self._ticker_thread.is_alive():
            try:
                self._ticker_thread.join(timeout=1)
            except Exception:
                pass



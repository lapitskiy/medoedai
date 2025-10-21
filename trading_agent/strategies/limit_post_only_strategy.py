from __future__ import annotations

import time
from typing import Optional, Callable

from trading_agent.domain.interfaces import ExecutionStrategy, StateStore, ExchangeGateway
from trading_agent.domain.models import Intent, IntentState, LimitConfig, Side


class LimitPostOnlyStrategy(ExecutionStrategy):
    """
    Limit post-only стратегия с WS: держится у ближайшего тика, пере-котирует по событиям,
    соблюдает max_lifetime_sec (≤5 минут). Лог каждой попытки: цена и причина отказа.
    """

    def __init__(self, store: StateStore, gateway: ExchangeGateway):
        self.store = store
        self.gateway = gateway

    def place_intent(self, symbol: str, side: Side, qty: float, cfg: Optional[LimitConfig] = None) -> Intent:
        cfg = cfg or LimitConfig()
        tick = self.gateway.get_tick_size(symbol)
        best_bid, best_ask = self.gateway.get_best_bid_ask(symbol)
        price = self._edge_price(side, best_bid, best_ask, tick, offset_ticks=1)

        intent = Intent(
            intent_id=f"limitpo-{symbol}-{int(time.time()*1000)}",
            symbol=symbol,
            side=side,
            qty_total=qty,
            qty_remaining=qty,
            tick_size=tick,
            price=price,
            offset_ticks=1,
            cfg=cfg,
        )
        self.store.save_intent(intent)
        self.store.add_pending(symbol, intent.intent_id)

        # Размещаем стартовый лимит post-only
        try:
            resp = self.gateway.place_limit_post_only(symbol, side, qty, price)
            intent.exchange_order_id = resp.get('id')
            intent.state = IntentState.WORKING
            intent.add_log(price, event='place', reason=None)
            self.store.update_intent(intent)
        except Exception as e:
            intent.add_log(price, event='place', reason=str(e))
            # отступ x2 и повтор на следующем цикле run_until_done
            intent.offset_ticks = min(intent.cfg.offset_max_ticks, max(2, intent.offset_ticks * 2))
            self.store.update_intent(intent)
        return intent

    def run_until_done(self, intent: Intent) -> Intent:
        start_ts = time.time()
        deadline = start_ts + min(300, intent.cfg.max_lifetime_sec)

        # WS подписки
        order_cb = self._make_order_cb(intent.intent_id)
        ticker_cb = self._make_ticker_cb(intent.intent_id)
        try:
            self.gateway.connect_ws()
            self.gateway.subscribe_orders(intent.symbol, order_cb)
            self.gateway.subscribe_ticker(intent.symbol, ticker_cb)
        except Exception as _:
            # fallback: периодический опрос только по времени
            pass

        # failsafe цикл (периодическая пере-проверка)
        while True:
            cur = self.store.load_intent(intent.intent_id) or intent
            if cur.state in (IntentState.FILLED, IntentState.CANCELLED, IntentState.FAILED, IntentState.EXPIRED):
                break
            if time.time() >= deadline:
                self._expire_and_cleanup(cur)
                break
            # Requote по SLA, если долго без событий
            time.sleep(max(1, int(cur.cfg.requote_interval_sec)))
            self._requote_if_needed(cur)
        return self.store.load_intent(intent.intent_id) or intent

    def cancel_intent(self, intent_id: str) -> None:
        cur = self.store.load_intent(intent_id)
        if not cur:
            return
        try:
            if cur.exchange_order_id:
                self.gateway.cancel_order(cur.symbol, cur.exchange_order_id)
        finally:
            self.store.set_state(intent_id, IntentState.CANCELLED)
            self.store.remove_pending(cur.symbol, intent_id)

    # === WS callbacks ===
    def _make_order_cb(self, intent_id: str) -> Callable[[dict], None]:
        def on_order(ev: dict) -> None:
            cur = self.store.load_intent(intent_id)
            if not cur:
                return
            status = (ev.get('status') or '').lower()
            if status in ('filled', 'partially_filled'):
                filled = float(ev.get('filled', 0) or 0)
                cur.qty_remaining = max(0.0, cur.qty_total - filled)
                if cur.qty_remaining <= 0:
                    self.store.set_state(intent_id, IntentState.FILLED)
                    self.store.remove_pending(cur.symbol, intent_id)
                else:
                    cur.state = IntentState.PARTIALLY_FILLED
                    self.store.update_intent(cur)
            elif status in ('canceled', 'rejected'):
                reason = ev.get('reason') or status
                self.store.append_log(intent_id, cur.price, event='order_' + status, reason=reason)
                # попробуем увеличить offset и перевыставить позже
                cur.offset_ticks = min(cur.cfg.offset_max_ticks, max(2, cur.offset_ticks * 2))
                cur.state = IntentState.WORKING
                self.store.update_intent(cur)
        return on_order

    def _make_ticker_cb(self, intent_id: str) -> Callable[[dict], None]:
        def on_tick(ev: dict) -> None:
            cur = self.store.load_intent(intent_id)
            if not cur:
                return
            best_bid = float(ev.get('best_bid')) if ev.get('best_bid') is not None else None
            best_ask = float(ev.get('best_ask')) if ev.get('best_ask') is not None else None
            if best_bid is None or best_ask is None:
                return
            self._requote_edge(cur, best_bid, best_ask)
        return on_tick

    # === helpers ===
    def _edge_price(self, side: Side, best_bid: float, best_ask: float, tick: float, offset_ticks: int) -> float:
        if side == Side.BUY:
            return round(best_bid - tick * offset_ticks, 8)
        else:
            return round(best_ask + tick * offset_ticks, 8)

    def _requote_edge(self, cur: Intent, best_bid: float, best_ask: float) -> None:
        if not cur.tick_size:
            return
        target = self._edge_price(cur.side, best_bid, best_ask, cur.tick_size, cur.offset_ticks)
        if cur.price == target:
            return
        try:
            if cur.exchange_order_id:
                self.gateway.amend_order(cur.symbol, cur.exchange_order_id, target)
                self.store.append_log(cur.intent_id, target, event='amend', reason=None)
            else:
                resp = self.gateway.place_limit_post_only(cur.symbol, cur.side, cur.qty_remaining, target)
                cur.exchange_order_id = resp.get('id')
                self.store.append_log(cur.intent_id, target, event='place', reason=None)
            cur.price = target
            self.store.bump_attempts(cur.intent_id, 1)
            self.store.update_intent(cur)
        except Exception as e:
            self.store.append_log(cur.intent_id, target, event='amend_failed', reason=str(e))
            # отступ x2 при отказе
            cur.offset_ticks = min(cur.cfg.offset_max_ticks, max(2, cur.offset_ticks * 2))
            self.store.update_intent(cur)

    def _requote_if_needed(self, cur: Intent) -> None:
        # периодический SLA-ребаланс на край книги
        try:
            best_bid, best_ask = self.gateway.get_best_bid_ask(cur.symbol)
            self._requote_edge(cur, best_bid, best_ask)
        except Exception as e:
            self.store.append_log(cur.intent_id, cur.price, event='poll_failed', reason=str(e))

    def _expire_and_cleanup(self, cur: Intent) -> None:
        try:
            if cur.exchange_order_id:
                try:
                    self.gateway.cancel_order(cur.symbol, cur.exchange_order_id)
                except Exception:
                    pass
        finally:
            self.store.set_state(cur.intent_id, IntentState.EXPIRED, last_error='deadline exceeded')
            self.store.remove_pending(cur.symbol, cur.intent_id)



import os
import sys
import time
import math
import json
import argparse
from datetime import datetime, timezone, timedelta

import pandas as pd
import ccxt
from sqlalchemy.orm import Session as SASession
from sqlalchemy import select, func

# Локальные импорты
from utils.cctx_utils import normalize_symbol, normalize_to_db
from utils.db_utils import engine
from orm.models import Symbol, OHLCV


def iso_from_ms(ts_ms: int) -> str:
    try:
        return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return str(ts_ms)


def create_exchange(exchange_id: str) -> ccxt.Exchange:
    exchange_id = (exchange_id or 'bybit').lower().strip()
    ex_class = getattr(ccxt, exchange_id)
    opts = {
        'enableRateLimit': True,
        'timeout': 30000,
        'options': {
            'warnOnFetchOHLCVLimitArgument': False, # отключаем предупреждения об ограничении лимита
        }
    }
    # Для Bybit теперь полагаемся исключительно на category='linear' в запросах
    ex = ex_class(opts)
    ex.load_markets()
    return ex


def map_symbol_on_exchange(ex: ccxt.Exchange, symbol: str) -> str:
    """Возвращает корректный unified-символ для биржи.
    Для Bybit linear swap unified-символ будет сконструирован в reconcile_symbol_timeframe.
    """
    return normalize_symbol(symbol)


def fetch_range_ohlcv(
    ex: ccxt.Exchange,
    uni_symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    limit_total: int = 1_000_000,
    params: dict | None = None,
) -> list[list]:
    """Загружает OHLCV пачками по 1000, двигая курсор по времени."""
    rows: list[list] = []
    cursor = since_ms
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    effective_params = params
    if effective_params is None and getattr(ex, "id", "") == "bybit":
        # Для Bybit linear опираемся на category='linear'
        effective_params = {"category": "linear"}
    while cursor <= until_ms and len(rows) < limit_total:
        if effective_params:
            batch = ex.fetch_ohlcv(uni_symbol, timeframe, since=cursor, limit=1000, params=effective_params)
        else:
            batch = ex.fetch_ohlcv(uni_symbol, timeframe, since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        next_cursor = last_ts + tf_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        # мягкий троттлинг
        ex.sleep(300)
    return rows


def get_or_create_symbol_id(db: SASession, symbol_db: str) -> int:
    sym = db.execute(select(Symbol).where(Symbol.name == symbol_db)).scalar_one_or_none()
    if not sym:
        sym = Symbol(name=symbol_db)
        db.add(sym)
        db.commit()
        db.refresh(sym)
    return sym.id


def load_existing_batch(db: SASession, symbol_id: int, timeframe: str, ts_list: list[int]) -> dict[int, OHLCV]:
    if not ts_list:
        return {}
    q = db.query(OHLCV).filter(
        OHLCV.symbol_id == symbol_id,
        OHLCV.timeframe == timeframe,
        OHLCV.timestamp.in_(ts_list),
    )
    res = {}
    for row in q.all():
        res[int(row.timestamp)] = row
    return res


def floats_differs(a: float, b: float, rel: float = 1e-8, abs_tol: float = 1e-6) -> bool:
    try:
        return not math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_tol)
    except Exception:
        return True


def reconcile_symbol_timeframe(symbol: str, timeframe: str, exchange_id: str, from_days: int | None, verbose: bool) -> None:
    """
    Если from_days задан — берём now-from_days..now. Иначе — от MIN(timestamp в БД)..now.
    """
    # Жёстко работаем в Bybit linear, без fallback
    if (exchange_id or 'bybit').lower().strip() == 'bybit':
        ex = create_exchange(exchange_id)
        # Для Bybit linear мы явно конструируем unified символ
        uni_base = normalize_symbol(symbol).split('/')[0]
        uni_quote = normalize_symbol(symbol).split('/')[1]
        uni = f"{uni_base}/{uni_quote}:USDT" # Предполагаем USDT perpetual
        if uni not in getattr(ex, 'symbols', []):
            raise ValueError(f"{uni} not found in Bybit linear. Please check the symbol or market.")
    else:
        ex = create_exchange(exchange_id)
        uni = map_symbol_on_exchange(ex, symbol)
    symbol_db = normalize_to_db(symbol)

    with SASession(bind=engine) as db:
        symbol_id = get_or_create_symbol_id(db, symbol_db)

        # Диапазон: если from_days задан — от now-Х дней до now; иначе от MIN(timestamp) в БД до now
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        if from_days is not None:
            since_ms = int((datetime.now(tz=timezone.utc) - timedelta(days=int(from_days))).timestamp() * 1000)
        else:
            min_ts = db.query(func.min(OHLCV.timestamp)).filter(OHLCV.symbol_id == symbol_id, OHLCV.timeframe == timeframe).scalar()
            since_ms = int(min_ts) if min_ts else now_ms - 30 * 24 * 3600 * 1000
        until_ms = now_ms

        if verbose:
            print(f"[RECONCILE] {symbol_db} {timeframe} range {iso_from_ms(since_ms)} .. {iso_from_ms(until_ms)} via {exchange_id} as {uni}")

        rows = fetch_range_ohlcv(ex, uni, timeframe, since_ms, until_ms)
        if not rows:
            print("[RECONCILE] No rows fetched from exchange. Nothing to do.")
            return

        # Обрабатываем пакетами для снижения нагрузки на БД
        batch_size = 2000
        updates = inserts = equals = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            ts_list = [int(r[0]) for r in batch]
            existing = load_existing_batch(db, symbol_id, timeframe, ts_list)
            for ts, o, h, l, c, v in ((int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in batch):
                row = existing.get(ts)
                if row is None:
                    # Вставка
                    new_row = OHLCV(symbol_id=symbol_id, timeframe=timeframe, timestamp=ts, open=o, high=h, low=l, close=c, volume=v)
                    db.add(new_row)
                    inserts += 1
                    if verbose:
                        print(f"[RECONCILE][INSERT] {iso_from_ms(ts)} O:{o} H:{h} L:{l} C:{c} V:{v}")
                else:
                    # Сравнение
                    changed_fields = []
                    if floats_differs(row.open, o):
                        changed_fields.append(("open", row.open, o)); row.open = o
                    if floats_differs(row.high, h):
                        changed_fields.append(("high", row.high, h)); row.high = h
                    if floats_differs(row.low, l):
                        changed_fields.append(("low", row.low, l)); row.low = l
                    if floats_differs(row.close, c):
                        changed_fields.append(("close", row.close, c)); row.close = c
                    if floats_differs(row.volume, v):
                        changed_fields.append(("volume", row.volume, v)); row.volume = v
                    if changed_fields:
                        updates += 1
                        if verbose:
                            diff_str = ", ".join([f"{f}:{old}->{new}" for f, old, new in changed_fields])
                            print(f"[RECONCILE][UPDATE] {iso_from_ms(ts)} {diff_str}")
                    else:
                        equals += 1
            db.commit()

        # Убеждаемся, что получаем информацию для linear рынка
        market_info = None
        if ex.id == 'bybit':
            # Ищем по market id среди linear swap/future
            sym_db = normalize_to_db(symbol)
            for k, m in getattr(ex, 'markets', {}).items():
                if not isinstance(m, dict):
                    continue
                mid = str(m.get('id') or '').upper()
                m_type = m.get('type')
                is_linear = bool(m.get('linear'))
                if mid == sym_db.upper() and is_linear and (m_type in ('swap', 'future')):
                    market_info = m
                    break
        if market_info is None:
            # Фоллбэк: если по каким-то причинам не нашли через перебор, пробуем напрямую
            try:
                market_info = ex.market(uni)
            except Exception:
                pass

        if verbose and market_info:
            print(f"[RECONCILE] Market info: type={market_info.get('type')} linear={market_info.get('linear')} inverse={market_info.get('inverse')} contractSize={market_info.get('contractSize')}")
        print(f"[RECONCILE] Done: inserted={inserts}, updated={updates}, unchanged={equals}, total_fetched={len(rows)}")


def reconcile_range_ohlcv(symbol: str, timeframe: str, exchange_id: str, since_ms: int, until_ms: int, verbose: bool = False) -> None:
    """Сверяет и обновляет OHLCV в БД с биржей в заданном диапазоне [since_ms..until_ms]."""
    if (exchange_id or 'bybit').lower().strip() == 'bybit':
        ex = create_exchange(exchange_id)
        # Для Bybit linear мы явно конструируем unified символ
        uni_base = normalize_symbol(symbol).split('/')[0]
        uni_quote = normalize_symbol(symbol).split('/')[1]
        uni = f"{uni_base}/{uni_quote}:USDT" # Предполагаем USDT perpetual
        if uni not in getattr(ex, 'symbols', []):
            raise ValueError(f"[RANGE] {uni} not found in Bybit linear. Please check the symbol or market.")
    else:
        ex = create_exchange(exchange_id)
        uni = map_symbol_on_exchange(ex, symbol)
    symbol_db = normalize_to_db(symbol)

    with SASession(bind=engine) as db:
        symbol_id = get_or_create_symbol_id(db, symbol_db)

        if verbose:
            print(f"[RECONCILE][RANGE] {symbol_db} {timeframe} range {iso_from_ms(since_ms)} .. {iso_from_ms(until_ms)} via {exchange_id} as {uni}")

        rows = fetch_range_ohlcv(ex, uni, timeframe, since_ms, until_ms, params={'category': 'linear'})
        if not rows:
            if verbose:
                print("[RECONCILE][RANGE] No rows fetched from exchange. Nothing to do.")
            return

        updates = inserts = equals = 0
        ts_list = [int(r[0]) for r in rows]
        existing = load_existing_batch(db, symbol_id, timeframe, ts_list)

        for ts, o, h, l, c, v in ((int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in rows):
            row = existing.get(ts)
            if row is None:
                # Вставка
                new_row = OHLCV(symbol_id=symbol_id, timeframe=timeframe, timestamp=ts, open=o, high=h, low=l, close=c, volume=v)
                db.add(new_row)
                inserts += 1
                if verbose:
                    print(f"[RECONCILE][RANGE][INSERT] {iso_from_ms(ts)} O:{o} H:{h} L:{l} C:{c} V:{v}")
            else:
                # Сравнение
                changed_fields = []
                if floats_differs(row.open, o):
                    changed_fields.append(("open", row.open, o)); row.open = o
                if floats_differs(row.high, h):
                    changed_fields.append(("high", row.high, h)); row.high = h
                if floats_differs(row.low, l):
                    changed_fields.append(("low", row.low, l)); row.low = l
                if floats_differs(row.close, c):
                    changed_fields.append(("close", row.close, c)); row.close = c
                if floats_differs(row.volume, v):
                    changed_fields.append(("volume", row.volume, v)); row.volume = v
                if changed_fields:
                    updates += 1
                    if verbose:
                        diff_str = ", ".join([f"{f}:{old}->{new}" for f, old, new in changed_fields])
                        print(f"[RECONCILE][RANGE][UPDATE] {iso_from_ms(ts)} {diff_str}")
                else:
                    equals += 1
        db.commit()

        # Убеждаемся, что получаем информацию для linear рынка
        market_info = None
        if ex.id == 'bybit':
            # Ищем по market id среди linear swap/future
            sym_db = normalize_to_db(symbol)
            for k, m in getattr(ex, 'markets', {}).items():
                if not isinstance(m, dict):
                    continue
                mid = str(m.get('id') or '').upper()
                m_type = m.get('type')
                is_linear = bool(m.get('linear'))
                if mid == sym_db.upper() and is_linear and (m_type in ('swap', 'future')):
                    market_info = m
                    break
        if market_info is None:
            # Фоллбэк: если по каким-то причинам не нашли через перебор, пробуем напрямую
            try:
                market_info = ex.market(uni)
            except Exception:
                pass

        if verbose and market_info:
            print(f"[RECONCILE][RANGE] Market info: type={market_info.get('type')} linear={market_info.get('linear')} inverse={market_info.get('inverse')} contractSize={market_info.get('contractSize')}")
        if verbose:
            print(f"[RECONCILE][RANGE] Done: inserted={inserts}, updated={updates}, unchanged={equals}, total_fetched={len(rows)}")

def main():
    ap = argparse.ArgumentParser(description="Синхронизация OHLCV в PostgreSQL с Bybit")
    ap.add_argument('--symbol', default='BTCUSDT', help='Символ, например BTCUSDT или BTC/USDT')
    ap.add_argument('--timeframe', default='5m', help='Таймфрейм (по умолчанию 5m)')
    ap.add_argument('--exchange', default='bybit', help='Биржа (bybit|binance)')
    ap.add_argument('--from-days', type=int, default=None, help='Сколько дней назад начинать (если не задано — от MIN(timestamp) в БД)')
    ap.add_argument('--use-db-days', action='store_true', help='Определить число дней по БД и использовать такой же диапазон (MIN(timestamp)..now)')
    ap.add_argument('--verbose', action='store_true', help='Подробный лог отличий')
    args = ap.parse_args()

    # Если задан use-db-days, игнорируем from-days и берём MIN(timestamp)..now
    if args.use_db_days:
        args.from_days = None

    reconcile_symbol_timeframe(symbol=args.symbol, timeframe=args.timeframe, exchange_id=args.exchange, from_days=args.from_days, verbose=args.verbose)


if __name__ == '__main__':
    main()

"""
Загрузка OHLCV данных через T-Invest API (Tinkoff / T-Bank).

Использует REST-обёртку SDK `tinkoff-investments`.
Для работы нужен токен: Settings → API → TINKOFF_TOKEN.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Маппинг интервалов SDK → timedelta для пагинации
_INTERVAL_DELTA = {
    "1min":  timedelta(days=1),
    "5min":  timedelta(days=1),
    "15min": timedelta(days=1),
    "hour":  timedelta(days=7),
    "day":   timedelta(days=365),
}


def _get_token() -> str:
    """Получить токен из env или из Postgres settings_store."""
    token = os.getenv("TINKOFF_TOKEN", "").strip()
    if token:
        return token
    try:
        from utils.settings_store import get_setting_value
        token = get_setting_value("api", "tinkoff", "TINKOFF_TOKEN") or ""
    except Exception:
        pass
    if not token:
        raise RuntimeError("TINKOFF_TOKEN не задан (env или Settings → API)")
    return token


def _resolve_figi(token: str, ticker: str) -> str:
    """По тикеру (SBER, GAZP, …) получить FIGI через T-Invest API."""
    from tinkoff.invest import Client, InstrumentIdType

    with Client(token) as client:
        resp = client.instruments.find_instrument(query=ticker)
        for inst in resp.instruments:
            if inst.ticker.upper() == ticker.upper() and inst.instrument_kind.name == "INSTRUMENT_KIND_SHARE":
                return inst.figi
        # fallback — первый результат
        if resp.instruments:
            return resp.instruments[0].figi
    raise ValueError(f"Инструмент '{ticker}' не найден в T-Invest API")


def _interval_enum(tf: str):
    """Преобразовать строковый timeframe в enum CandleInterval."""
    from tinkoff.invest import CandleInterval

    mapping = {
        "1m": CandleInterval.CANDLE_INTERVAL_1_MIN,
        "1min": CandleInterval.CANDLE_INTERVAL_1_MIN,
        "5m": CandleInterval.CANDLE_INTERVAL_5_MIN,
        "5min": CandleInterval.CANDLE_INTERVAL_5_MIN,
        "15m": CandleInterval.CANDLE_INTERVAL_15_MIN,
        "15min": CandleInterval.CANDLE_INTERVAL_15_MIN,
        "1h": CandleInterval.CANDLE_INTERVAL_HOUR,
        "hour": CandleInterval.CANDLE_INTERVAL_HOUR,
        "1d": CandleInterval.CANDLE_INTERVAL_DAY,
        "day": CandleInterval.CANDLE_INTERVAL_DAY,
    }
    ci = mapping.get(tf.lower())
    if ci is None:
        raise ValueError(f"Неизвестный timeframe: {tf}. Доступны: {list(mapping.keys())}")
    return ci


def _quotation_to_float(q) -> float:
    """tinkoff.invest.Quotation → float."""
    return q.units + q.nano / 1e9


def fetch_tinkoff_ohlcv(
    ticker: str,
    timeframe: str = "5m",
    limit_candles: int = 10_000,
    figi: str | None = None,
) -> pd.DataFrame:
    """
    Загрузить OHLCV через T-Invest API.

    Args:
        ticker: Тикер (SBER, GAZP, LKOH, …)
        timeframe: '1m','5m','15m','1h','1d'
        limit_candles: сколько свечей получить (максимум)
        figi: если известен FIGI, можно передать напрямую

    Returns:
        pd.DataFrame с колонками [timestamp, open, high, low, close, volume]
    """
    from tinkoff.invest import Client

    token = _get_token()
    if not figi:
        figi = _resolve_figi(token, ticker)
        logger.info(f"[TINKOFF] {ticker} → FIGI={figi}")

    interval = _interval_enum(timeframe)

    # Определяем шаг пагинации (API ограничивает окно запроса)
    tf_key = timeframe.lower().replace("1m", "1min").replace("5m", "5min").replace("15m", "15min").replace("1h", "hour").replace("1d", "day")
    page_delta = _INTERVAL_DELTA.get(tf_key, timedelta(days=1))

    now = datetime.now(timezone.utc)
    all_candles: list[dict] = []

    with Client(token) as client:
        cursor = now
        while len(all_candles) < limit_candles:
            from_ = cursor - page_delta
            resp = client.market_data.get_candles(
                figi=figi,
                from_=from_,
                to=cursor,
                interval=interval,
            )
            if not resp.candles:
                break

            batch = []
            for c in resp.candles:
                batch.append({
                    "timestamp": int(c.time.timestamp() * 1000),
                    "open": _quotation_to_float(c.open),
                    "high": _quotation_to_float(c.high),
                    "low": _quotation_to_float(c.low),
                    "close": _quotation_to_float(c.close),
                    "volume": int(c.volume),
                })
            all_candles = batch + all_candles  # prepend (идём назад)
            cursor = from_

            if len(batch) == 0:
                break

    if not all_candles:
        logger.warning(f"[TINKOFF] Нет свечей для {ticker} ({timeframe})")
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_candles)
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Обрезаем до limit
    if len(df) > limit_candles:
        df = df.tail(limit_candles).reset_index(drop=True)

    logger.info(f"[TINKOFF] {ticker} {timeframe}: загружено {len(df)} свечей")
    return df

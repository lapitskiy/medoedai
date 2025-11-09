from __future__ import annotations

from typing import Tuple
import time

import pandas as pd

from utils.db_utils import db_get_or_fetch_ohlcv
from utils.redis_utils import get_redis_client


def get_atr_1h(symbol: str, length: int = 21, cache_ttl_sec: int = 90) -> Tuple[float, float, float]:
    """
    Возвращает кортеж (atr_abs, atr_norm, last_close) по 1h свечам.
    ATR считается по классической формуле Wilder (EMA с alpha=1/length).
    Результат кэшируется в Redis на короткое время для снижения нагрузки.
    """
    # Попробуем вернуть из кэша
    rc = None
    try:
        rc = get_redis_client()
        cache_key = f"atr:1h:{symbol}:{length}"
        raw = rc.get(cache_key) if rc else None
        if raw:
            import json as _json
            d = _json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
            if (time.time() - float(d.get("ts", 0))) <= cache_ttl_sec:
                return float(d["atr_abs"]), float(d["atr_norm"]), float(d["close"])
    except Exception:
        pass

    # Берём нужный объём свечей
    df_1h = db_get_or_fetch_ohlcv(
        symbol_name=symbol,
        timeframe="1h",
        limit_candles=max(60, int(length) + 5),
        exchange_id="bybit",
    )
    if not isinstance(df_1h, pd.DataFrame) or df_1h.empty or len(df_1h) < (length + 1):
        raise RuntimeError("Not enough 1h candles for ATR calculation")

    df = df_1h.copy().sort_values("timestamp")
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    # True Range
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder ATR через EMA; fallback на SMA при ошибках
    try:
        atr = tr.ewm(alpha=1.0 / float(length), adjust=False).mean()
    except Exception:
        atr = tr.rolling(window=length, min_periods=length).mean()

    atr_abs = float(atr.iloc[-1])
    last_close = float(close.iloc[-1])
    atr_norm = (atr_abs / last_close) if last_close > 0 else 0.0

    # Сохраняем в кэш
    try:
        if rc:
            import json as _json
            rc.set(
                f"atr:1h:{symbol}:{length}",
                _json.dumps(
                    {
                        "atr_abs": atr_abs,
                        "atr_norm": atr_norm,
                        "close": last_close,
                        "ts": time.time(),
                    },
                    ensure_ascii=False,
                ),
                ex=cache_ttl_sec,
            )
    except Exception:
        pass

    return atr_abs, atr_norm, last_close



"""
Centralized Bybit API rate limiter + load_markets() cache.
All containers share limits via Redis.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import ccxt
import redis

logger = logging.getLogger(__name__)

_REDIS_PARAMS = dict(host='redis', port=6379, db=0, decode_responses=True, socket_connect_timeout=2)

_MARKETS_CACHE_KEY = "bybit:markets:cache"
_MARKETS_LOCK_KEY = "bybit:markets:load_lock"
_RATE_KEY = "bybit:rate:sliding"
_COOLDOWN_KEY = "bybit:cooldown_until"

_MARKETS_TTL = 1800  # 30 min
_RATE_WINDOW = 60    # 1 min sliding window
_RATE_MAX = 10       # max requests per window across all containers


def _is_access_too_frequent(err: Exception | str) -> bool:
    try:
        s = str(err)
    except Exception:
        return False
    return ("Access too frequent" in s) or ("try again in 5 minutes" in s)


def _get_redis() -> redis.Redis:
    return redis.Redis(**_REDIS_PARAMS)


# ── Rate limiter ──────────────────────────────────────────────

def acquire_slot(caller: str = "", timeout_sec: float = 120) -> bool:
    """Wait for a free slot in the global Bybit rate limiter. Returns True when OK."""
    try:
        r = _get_redis()
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            # Global cooldown (e.g. Bybit temporary ban message).
            try:
                until_raw = r.get(_COOLDOWN_KEY)
                until_ts = float(until_raw) if until_raw else 0.0
                now_ts = time.time()
                if until_ts and now_ts < until_ts:
                    wait = min(max(0.5, until_ts - now_ts + 0.2), deadline - time.monotonic())
                    if wait > 0:
                        logger.warning(f"[BYBIT_RATE] cooldown active, waiting {wait:.1f}s… caller={caller}")
                        time.sleep(wait)
                        continue
            except Exception:
                pass
            now = time.time()
            pipe = r.pipeline()
            pipe.zremrangebyscore(_RATE_KEY, 0, now - _RATE_WINDOW)
            pipe.zcard(_RATE_KEY)
            _, count = pipe.execute()
            if count < _RATE_MAX:
                r.zadd(_RATE_KEY, {f"{caller}:{uuid.uuid4().hex[:8]}": now})
                r.expire(_RATE_KEY, _RATE_WINDOW + 10)
                return True
            oldest = r.zrange(_RATE_KEY, 0, 0, withscores=True)
            if oldest:
                wait = max(0.5, oldest[0][1] + _RATE_WINDOW - now + 0.1)
            else:
                wait = 1.0
            wait = min(wait, deadline - time.monotonic())
            if wait <= 0:
                break
            logger.info(f"[BYBIT_RATE] slot busy ({count}/{_RATE_MAX}), waiting {wait:.1f}s… caller={caller}")
            time.sleep(wait)
        logger.warning(f"[BYBIT_RATE] timeout after {timeout_sec}s, caller={caller}")
        return True  # не блокируем навсегда
    except Exception as e:
        logger.warning(f"[BYBIT_RATE] redis error: {e}, proceeding without limit")
        return True


# ── Cached load_markets ──────────────────────────────────────

def load_markets_cached(exchange: ccxt.Exchange, force: bool = False) -> dict:
    """
    Load markets from Redis cache or from exchange (with distributed lock).
    Populates exchange.markets / exchange.symbols in-place.
    """
    try:
        r = _get_redis()
        if not force:
            raw = r.get(_MARKETS_CACHE_KEY)
            if raw:
                data = json.loads(raw)
                exchange.markets = data.get("markets", {})
                exchange.symbols = list(exchange.markets.keys())
                exchange.markets_by_id = data.get("markets_by_id", {})
                logger.debug("[BYBIT_CACHE] markets loaded from Redis cache")
                return exchange.markets
    except Exception as e:
        logger.warning(f"[BYBIT_CACHE] redis read error: {e}")

    acquire_slot(caller="load_markets")
    try:
        r = _get_redis()
        lock = r.lock(_MARKETS_LOCK_KEY, timeout=60, blocking_timeout=90)
        if lock.acquire():
            try:
                if not force:
                    raw = r.get(_MARKETS_CACHE_KEY)
                    if raw:
                        data = json.loads(raw)
                        exchange.markets = data.get("markets", {})
                        exchange.symbols = list(exchange.markets.keys())
                        exchange.markets_by_id = data.get("markets_by_id", {})
                        return exchange.markets
                exchange.load_markets()
                cache_data = {
                    "markets": exchange.markets,
                    "markets_by_id": getattr(exchange, "markets_by_id", {}),
                    "ts": time.time(),
                }
                r.setex(_MARKETS_CACHE_KEY, _MARKETS_TTL, json.dumps(cache_data, default=str))
                logger.info(f"[BYBIT_CACHE] markets fetched from API & cached ({len(exchange.markets)} symbols, TTL={_MARKETS_TTL}s)")
                return exchange.markets
            finally:
                try:
                    lock.release()
                except Exception:
                    pass
    except Exception as e:
        hard_cooldown = _is_access_too_frequent(e)
        if hard_cooldown:
            try:
                r = _get_redis()
                r.setex(_COOLDOWN_KEY, 360, str(time.time() + 300.0))
            except Exception:
                pass
            # Do NOT double-hit API during Bybit cooldown.
            raise
        logger.warning(f"[BYBIT_CACHE] lock/fetch error: {e}, falling back to direct load_markets()")

    exchange.load_markets()
    return exchange.markets

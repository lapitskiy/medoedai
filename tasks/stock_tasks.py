"""
Celery задачи для обучения DQN на акциях (РФ рынок / T-Invest API).
"""
from __future__ import annotations

import os
import logging
import traceback
from datetime import datetime

import pandas as pd

from tasks import celery
from utils.seed import set_global_seed

logger = logging.getLogger(__name__)


@celery.task(bind=True, name="tasks.stock_tasks.train_stock_dqn")
def train_stock_dqn(
    self,
    ticker: str,
    episodes: int | None = None,
    seed: int | None = None,
    episode_length: int = 2000,
    direction: str = "long",
    figi: str | None = None,
):
    """Обучение DQN для одной акции через T-Invest API.

    Загружает OHLCV из Tinkoff, строит 5m/15m/1h, запускает train_model_optimized.
    """
    self.update_state(state="IN_PROGRESS", meta={"progress": 0, "ticker": ticker})

    # Дедупликация
    try:
        from utils.redis_utils import get_redis_client
        _rc = get_redis_client()
        _task_id = getattr(getattr(self, "request", None), "id", None)
        _running_key = f"celery:train:stock:task:{ticker.upper()}"
        if _task_id:
            _done_key = f"celery:train:stock:done:{_task_id}"
            if _rc.get(_done_key):
                return {"message": f"⏭️ {ticker}: уже завершена (дедупликация)", "skipped": True}
            _rc.setex(_running_key, 48 * 3600, _task_id)
    except Exception:
        pass

    try:
        if seed is not None:
            set_global_seed(int(seed))
            print(f"🔒 Seed установлен: {seed}")

        print(f"\n🚀 [STOCK] Старт обучения для {ticker} [{datetime.now()}]")

        # --- Загрузка данных через T-Invest API ---
        from utils.tinkoff_data import fetch_tinkoff_ohlcv

        df_5min = fetch_tinkoff_ohlcv(
            ticker=ticker,
            timeframe="5m",
            limit_candles=100_000,
            figi=figi,
        )

        if df_5min is None or df_5min.empty:
            return {"message": f"❌ Данные для {ticker} не найдены (T-Invest API)"}

        df_5min["datetime"] = pd.to_datetime(df_5min["timestamp"], unit="ms")
        df_5min.set_index("datetime", inplace=True)

        df_15min = (
            df_5min.resample("15min")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
            .reset_index()
        )
        df_1h = (
            df_5min.resample("1h")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
            .reset_index()
        )

        dfs = {
            "df_5min": df_5min,
            "df_15min": df_15min,
            "df_1h": df_1h,
            "symbol": ticker.upper(),
        }
        print(f"📈 [STOCK] {ticker}: 5m={len(df_5min)}, 15m={len(df_15min)}, 1h={len(df_1h)}")

        if episodes is None:
            episodes = int(os.getenv("DEFAULT_EPISODES", 5))
        print(f"🎯 Эпизодов: {episodes}, длина: {episode_length}")

        # --- Обучение через train_model_optimized с StockTradingEnv ---
        from agents.vdqn.v_train_model_optimized import train_model_optimized

        result = train_model_optimized(
            dfs=dfs,
            episodes=episodes,
            seed=seed,
            episode_length=episode_length,
            direction=direction,
            env_class_override="stock",  # флаг для выбора env
        )
        return {"message": f"✅ [STOCK] Обучение {ticker} завершено: {result}"}

    except Exception as e:
        traceback.print_exc()
        return {"message": f"❌ [STOCK] Ошибка обучения {ticker}: {e}"}
    finally:
        try:
            from utils.redis_utils import get_redis_client
            rc = get_redis_client()
            rc.delete(f"celery:train:stock:task:{ticker.upper()}")
            _tid = getattr(getattr(self, "request", None), "id", None)
            if _tid:
                rc.setex(f"celery:train:stock:done:{_tid}", 24 * 3600, "1")
        except Exception:
            pass

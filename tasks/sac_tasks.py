"""Celery задачи для обучения SAC агентов."""

from __future__ import annotations

import os
import random
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from tasks import celery
from utils.db_utils import db_get_or_fetch_ohlcv, load_latest_candles_from_csv_to_db
from utils.parser import parser_download_and_combine_with_library
from utils.redis_utils import get_redis_client
from utils.seed import set_global_seed

from agents.sac.agents.config import SacConfig
from agents.sac.agents.trainer import SacTrainer
from envs.dqn_model.gym.gconfig import GymConfig


@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0}, queue='train')
def train_sac_symbol(
    self,
    symbol: str,
    episodes: int | None = None,
    seed: int | None = None,
    episode_length: int | None = None,
) -> Dict[str, Any]:
    """Фоновая задача обучения SAC агента для одного символа."""

    symbol = (symbol or "BTCUSDT").upper()
    redis_client = get_redis_client()
    running_key = f"celery:train:sac:task:{symbol}"
    logs: List[str] = []

    def push_log(message: str) -> None:
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        logs.append(entry)
        self.update_state(state="PROGRESS", meta={"logs": list(logs), "symbol": symbol})

    try:
        push_log(f"🚀 Запуск обучения SAC для {symbol}")

        # Проверяем, не завершилось ли обучение недавно успешно (чтобы избежать дубликатов)
        try:
            last_run_key = f"sac:last_run:{symbol}"
            last_run_time = redis_client.get(last_run_key)
            if last_run_time:
                current_time = time.time()
                # Если прошло менее 5 минут с момента успешного завершения, пропускаем
                if current_time - float(last_run_time) < 300:  # 5 минут
                    push_log(f"⏭️ Обучение {symbol} завершилось недавно ({current_time - float(last_run_time):.1f} сек назад), пропускаем повторный запуск")
                    # Возвращаем информацию о предыдущем запуске
                    result_dir = f"result/sac/{symbol.lower()}/runs/sac-{str(uuid.uuid4())[:4].lower()}"
                    info = {
                        'success': True,
                        'symbol': symbol,
                        'run_name': f"sac-{str(uuid.uuid4())[:4].lower()}",
                        'result_dir': result_dir,
                        'model_path': f"{result_dir}/model.pth",
                        'metrics_path': f"{result_dir}/metrics.json",
                        'skipped': True,
                        'reason': 'recently_completed'
                    }
                    self.update_state(state="SUCCESS", meta={"logs": list(logs), "symbol": symbol, "result": info})
                    return info
        except Exception as e:
            push_log(f"⚠️ Не удалось проверить время последнего запуска: {e}")

        # Устанавливаем глобальную блокировку перед запуском
        try:
            redis_client.setex("sac:training:global_lock", 3600, "1")  # Блокируем на 1 час
        except Exception:
            pass

        # Сид: случайный, если не передан
        if seed is None:
            seed = random.randint(1, 2**31 - 1)
            push_log(f"🎲 Сгенерирован сид {seed}")
        set_global_seed(int(seed))

        # Загружаем данные из БД
        push_log("📥 Загрузка 5m свечей из базы данных...")
        df_5min = db_get_or_fetch_ohlcv(
            symbol_name=symbol,
            timeframe='5m',
            limit_candles=100_000,
            exchange_id='bybit'
        )

        if df_5min is None or df_5min.empty:
            push_log("⚠️ Данные не найдены в БД, выполняю скачивание...")
            try:
                csv_path = parser_download_and_combine_with_library(
                    symbol=symbol,
                    interval='5m',
                    months_to_fetch=12,
                    desired_candles=100_000
                )
                if csv_path:
                    loaded = load_latest_candles_from_csv_to_db(
                        file_path=csv_path,
                        symbol_name=symbol,
                        timeframe='5m'
                    )
                    push_log(f"✅ В БД загружено {loaded} свечей")
                df_5min = db_get_or_fetch_ohlcv(
                    symbol_name=symbol,
                    timeframe='5m',
                    limit_candles=100_000,
                    exchange_id='bybit'
                )
            except Exception as exc:  # noqa: BLE001
                push_log(f"❌ Не удалось загрузить данные автоматически: {exc}")
                df_5min = None

        if df_5min is None or df_5min.empty:
            error_msg = f"Данные для {symbol} не найдены"
            push_log(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

        df_5min = df_5min.copy()
        df_5min['datetime'] = pd.to_datetime(df_5min['timestamp'], unit='ms')
        df_5min.set_index('datetime', inplace=True)

        df_15min = df_5min.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna().reset_index()

        df_1h = df_5min.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna().reset_index()

        push_log(
            f"📊 Подготовлено свечей: 5m={len(df_5min)}, 15m={len(df_15min)}, 1h={len(df_1h)}"
        )

        cfg = SacConfig()
        # Используем короткий UUID как в DQN: sac-{4_chars}
        short_id = str(uuid.uuid4())[:4].lower()
        cfg.run_name = f"sac-{short_id}"
        cfg.train_episodes = int(episodes) if episodes else cfg.train_episodes
        cfg.seed = int(seed)
        if episode_length:
            cfg.max_episode_steps = int(episode_length)

        gym_cfg = GymConfig()
        if episode_length:
            gym_cfg.episode_length = int(episode_length)

        # Создаем тренер (это вызовет __post_init__ и _ensure_result_paths)
        trainer = SacTrainer(
            cfg=cfg,
            progress_callback=push_log,
        )

        # Устанавливаем символ после создания тренера, чтобы пути создались правильно
        cfg.extra = cfg.extra.copy() if cfg.extra else {}
        cfg.extra['symbol'] = symbol
        # Пересчитываем пути с новым символом
        cfg.update_result_paths()

        push_log(
            f"🎯 Старт тренировки: эпизодов={cfg.train_episodes}, длина эпизода={gym_cfg.episode_length}"
        )

        dfs = {
            'df_5min': df_5min.reset_index(),
            'df_15min': df_15min,
            'df_1h': df_1h,
        }

        trainer.train(dfs=dfs, gym_cfg=gym_cfg)

        push_log("💾 Сохранение результатов обучения")

        metrics_path = os.path.join(cfg.result_dir, 'metrics.json')
        info: Dict[str, Any] = {
            'success': True,
            'symbol': symbol,
            'run_name': cfg.run_name,
            'result_dir': cfg.result_dir,
            'model_path': cfg.model_path,
            'metrics_path': metrics_path if os.path.exists(metrics_path) else None,
        }

        push_log("✅ Обучение SAC завершено")

        # Сохраняем время завершения для блокировки повторных запусков
        try:
            current_time = time.time()
            redis_client.setex(f"sac:last_run:{symbol}", 3600, current_time)  # Храним 1 час
            # Снимаем глобальную блокировку после завершения
            redis_client.delete("sac:training:global_lock")

            # Очищаем очередь задач для предотвращения повторных запусков
            # Удаляем задачу из очереди ui:tasks после успешного завершения
            try:
                # Ищем и удаляем текущую задачу из списка UI задач
                ui_tasks_key = "ui:tasks"
                # Получаем список задач и удаляем текущую
                task_ids = redis_client.lrange(ui_tasks_key, 0, -1)
                current_task_id = self.request.id
                for i, task_id_bytes in enumerate(task_ids):
                    try:
                        task_id = task_id_bytes.decode('utf-8')
                        if task_id == current_task_id:
                            redis_client.lrem(ui_tasks_key, 0, task_id)
                            push_log(f"🗑️ Задача {current_task_id} удалена из очереди UI задач")
                            break
                    except:
                        continue
            except Exception as e:
                push_log(f"⚠️ Не удалось очистить очередь UI задач: {e}")

        except Exception as e:
            push_log(f"⚠️ Не удалось сохранить время завершения: {e}")

        self.update_state(state="SUCCESS", meta={"logs": list(logs), "symbol": symbol, "result": info})
        return info

    except Exception:
        push_log("❌ Ошибка обучения SAC, подробнее см. traceback в Celery")
        raise
    finally:
        try:
            redis_client.delete(running_key)
            # Снимаем глобальную блокировку в случае ошибки
            redis_client.delete("sac:training:global_lock")
        except Exception:
            pass



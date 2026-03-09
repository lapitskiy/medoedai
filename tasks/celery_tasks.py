import os
import logging
import time
import json
from datetime import datetime
from redis import Redis  # type: ignore
import pandas as pd  # type: ignore

from dotenv import load_dotenv, find_dotenv  # type: ignore

from agents.vdqn.v_train_model import train_model
from agents.vdqn.v_train_model_optimized import train_model_optimized

from utils.db_utils import db_get_or_fetch_ohlcv
from utils.db_utils import load_latest_candles_from_csv_to_db
from utils.parser import parser_download_and_combine_with_library
from utils.seed import set_global_seed
from utils.config_loader import get_config_value
from tasks import celery
from agents.vdqn.tianshou_train import train_tianshou_dqn

logger = logging.getLogger(__name__)

# API ключи Bybit (поддержка новых имён BYBIT_<N>_*)
def _discover_bybit_api_keys() -> tuple[str | None, str | None]:
    try:
        ak = os.getenv('BYBIT_1_API_KEY')
        sk = os.getenv('BYBIT_1_SECRET_KEY')
        if ak and sk:
            return ak, sk
        # Автоскан: BYBIT_<ID>_API_KEY
        candidates = []
        for k, v in os.environ.items():
            if not k.startswith('BYBIT_') or not k.endswith('_API_KEY'):
                continue
            idx = k[len('BYBIT_'):-len('_API_KEY')]
            sec_name = f'BYBIT_{idx}_SECRET_KEY'
            sec_val = os.getenv(sec_name)
            if v and sec_val:
                candidates.append((k, v, sec_name, sec_val))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1], candidates[0][3]
        return None, None
    except Exception:
        return None, None

BYBIT_API_KEY, BYBIT_SECRET_KEY = _discover_bybit_api_keys()

def are_bybit_keys_configured() -> bool:
    try:
        return bool(BYBIT_API_KEY) and bool(BYBIT_SECRET_KEY)
    except Exception:
        return False

# Настраиваем Celery с Redis как брокером и бекендом
# Настройки для автоматической очистки результатов задач

# Определяем очереди и маршрутизацию задач:
# По умолчанию все задачи идут в очередь 'celery',
# а тренировочные задачи направляем в отдельную очередь 'train'.

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0})
def search_lstm_task(self, query):
    """Фоновая задача, которая выполняется долго"""
    self.update_state(state="IN_PROGRESS", meta={"progress": 0})

    for i in range(5):  # Имитация долгого вычисления
        time.sleep(2)
        self.update_state(state="IN_PROGRESS", meta={"progress": (i + 1) * 20})

    return {"message": "Task completed!", "query": query}

@celery.task(bind=True, acks_late=False, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0}, queue='train')  # добавлено acks_late=False
def train_dqn(self, seed: int | None = None):
    
    self.update_state(state="IN_PROGRESS", meta={"progress": 0})
    
    # Устанавливаем сид, если задан аргументом
    seed = int(seed) if seed is not None else None
    if seed is not None:
        set_global_seed(seed)
        print(f"🔒 Seed установлен: {seed}")
        # ENV больше не используем для сидов

    print("🚀 Начинаю загрузку данных для мультивалютного обучения...")
    
    # Список всех криптовалют для обучения
    crypto_symbols = [
        'BTCUSDT',  # Биткоин
        'TONUSDT',  # TON
        'ETHUSDT',  # Эфириум
        'SOLUSDT',  # Solana
        'ADAUSDT',  # Cardano
        'BNBUSDT',  # Binance Coin
        'XMRUSDT',  # Monero
        'XRPUSDT'   # Ripple
    ]
    
    all_dfs = {}
    
    for symbol in crypto_symbols:
        try:
            print(f"📥 Загружаю {symbol}...")
            
            # Загружаем данные из базы
            df_5min = db_get_or_fetch_ohlcv(
                symbol_name=symbol, 
                timeframe='5m', 
                limit_candles=100000,
                exchange_id='bybit'
            )
            
            if df_5min is not None and not df_5min.empty:
                print(f"  ✅ {symbol}: {len(df_5min)} свечей загружено")
                
                # Подготавливаем данные для этого символа
                df_5min['datetime'] = pd.to_datetime(df_5min['timestamp'], unit='ms')
                df_5min.set_index('datetime', inplace=True)
                
                # Создаем 15-минутные и 1-часовые данные
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
                
                # Сохраняем в общий словарь
                all_dfs[symbol] = {
                    'df_5min': df_5min,
                    'df_15min': df_15min,
                    'df_1h': df_1h,
                    'symbol': symbol,
                    'candle_count': len(df_5min)
                }
                
            else:
                print(f"  ⚠️ {symbol}: данные не найдены, пропускаем")
                
        except Exception as e:
            print(f"  ❌ {symbol}: ошибка загрузки - {e}")
            continue
    
    if not all_dfs:
        print("❌ Не удалось загрузить данные ни для одной криптовалюты")
        return {"message": "Ошибка: данные не загружены"}
    
    print(f"\n📈 Успешно загружено {len(all_dfs)} криптовалют")
    
    # Проверяем количество свечей
    for symbol, data in all_dfs.items():
        print(f"  • {symbol}: {data['candle_count']} свечей")
    
    # Используем первую криптовалюту для совместимости с текущим кодом
    # В будущем можно будет переключиться на мультивалютное обучение
    first_symbol = list(all_dfs.keys())[0]
    df = {
        'df_5min': all_dfs[first_symbol]['df_5min'],
        'df_15min': all_dfs[first_symbol]['df_15min'],
        'df_1h': all_dfs[first_symbol]['df_1h']
    }
    
    # Выводим первые значения каждого df в формате JSON
    for key, value in df.items():
        records = value[:2].copy()
        if 'timestamp' in records.columns:
            records['timestamp'] = records['timestamp'].astype(str)
        else:
            for col in records.columns:
                if records[col].dtype.name == 'datetime64[ns]':
                    records[col] = records[col].astype(str)
        print(f"{key}: {json.dumps(records.to_dict(orient='records'), ensure_ascii=False, indent=2)}")
    
    print(f"\n🎯 Запуск обучения на {first_symbol}...")
    
    # Получаем количество эпизодов из переменной окружения
    episodes = int(os.getenv('DEFAULT_EPISODES', 10000))
    print(f"🎯 Количество эпизодов: {episodes}")
    
    result = train_model_optimized(dfs=df, episodes=episodes, seed=seed)
    return {"message": result}

@celery.task(bind=True, acks_late=False, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0}, queue='train')  # добавлено acks_late=False
def train_dqn_symbol(
    self,
    symbol: str,
    episodes: int = None,
    seed: int | None = None,
    episode_length: int = 2000,
    engine: str = 'optimized',
    encoder_id: str | None = None,
    train_encoder: bool = False,
    direction: str = 'long',
    env_overrides: dict | None = None,
    cfg_overrides: dict | None = None,
    limit_candles: int | None = None,
    cutoff_days: int | None = None,
):
    """Обучение DQN для одного символа (BTCUSDT/ETHUSDT/...)

    Загружает данные из БД, готовит 5m/15m/1h, запускает train_model_optimized.
    """
    self.update_state(state="IN_PROGRESS", meta={"progress": 0, "symbol": symbol})

    # Идемпотентность: если этот task_id уже успешно завершался, пропустим повторную доставку
    try:
        from utils.redis_utils import get_redis_client  # локальный импорт
        _rc = get_redis_client()
        _task_id = getattr(getattr(self, 'request', None), 'id', None)
        if _task_id:
            _done_key = f"celery:train:done:{_task_id}"
            if _rc.get(_done_key):
                return {"message": f"⏭️ Повторная доставка задачи {symbol} проигнорирована (уже завершена)", "skipped": True}
        # Бизнес-дедупликация: тот же набор параметров недавно завершался — пропускаем
        def _fmt_f(v):
            try:
                if v is None:
                    return "na"
                if isinstance(v, bool):
                    return "1" if v else "0"
                if isinstance(v, int):
                    return str(v)
                x = float(v)
                s = f"{x:.6f}".rstrip('0').rstrip('.')
                return s if s != '' else '0'
            except Exception:
                return str(v)

        def _rm_tag(overrides: dict | None) -> str:
            try:
                if not isinstance(overrides, dict):
                    return "rm:-"
                rm = overrides.get("risk_management")
                if not isinstance(rm, dict):
                    return "rm:-"
                sl = _fmt_f(rm.get("STOP_LOSS_PCT"))
                tp = _fmt_f(rm.get("TAKE_PROFIT_PCT"))
                mh = _fmt_f(rm.get("min_hold_steps"))
                vt = _fmt_f(rm.get("volume_threshold"))
                return f"rm:{sl}:{tp}:{mh}:{vt}"
            except Exception:
                return "rm:-"

        def _cfg_tag(overrides: dict | None) -> str:
            try:
                if not isinstance(overrides, dict) or not overrides:
                    return "cfg:-"
                # Keep tag short but deterministic for dedup
                bs = overrides.get("batch_size")
                ms = overrides.get("memory_size")
                hs = overrides.get("hidden_sizes")
                dec = overrides.get("eps_decay_steps")
                lr = overrides.get("lr")
                tr = overrides.get("train_repeats")
                return f"cfg:bs={_fmt_f(bs)}:ms={_fmt_f(ms)}:hs={str(hs)}:dec={_fmt_f(dec)}:lr={_fmt_f(lr)}:tr={_fmt_f(tr)}"
            except Exception:
                return "cfg:-"

        _engine = (engine or '').lower()
        _enc = (encoder_id or '-')
        _train_enc = 1 if train_encoder else 0
        _eps = episodes if (episodes is not None) else 'env'
        _ep_len = episode_length if (episode_length is not None) else 'cfg'
        _cut = int(cutoff_days or 0)
        _rm = _rm_tag(env_overrides)
        _cfg = _cfg_tag(cfg_overrides)
        _biz_key = f"{(symbol or '').upper()}|{_engine}|{_enc}|{_train_enc}|{_eps}|{_ep_len}|cut={_cut}|{(direction or 'long')}|{_rm}|{_cfg}"
        _finished_key = f"celery:train:finished:{_biz_key}"
        _queued_key = f"celery:train:queued:{_biz_key}"
        _running_key_biz = f"celery:train:running:{_biz_key}"
        if _rc.get(_finished_key):
            return {"message": f"⏭️ {symbol}: такой запуск уже недавно завершался (дедупликация)", "skipped": True}
        try:
            # Помечаем бизнес-‘running’ и очищаем ‘queued’ (если был двойной клик)
            _rc.setex(_running_key_biz, 48 * 3600, _task_id or '1')
            _rc.delete(_queued_key)
        except Exception:
            pass
    except Exception:
        pass

    try:
        # Сид до любых инициализаций
        seed = int(seed) if seed is not None else None
        if seed is not None:
            set_global_seed(seed)
            print(f"🔒 Seed установлен: {seed}")
            # ENV больше не используем для сидов

        print(f"\n🚀 Старт обучения для {symbol} [{datetime.now()}]")
        if encoder_id:
            print(f"🧩 Выбран энкодер: {encoder_id} | режим: {'unfrozen' if train_encoder else 'frozen'}")
        train_window = int(limit_candles) if limit_candles is not None else 100000
        cd = int(cutoff_days or 0)
        fetch_limit = int(train_window + max(0, cd) * 288 + 2000)
        df_5min = db_get_or_fetch_ohlcv(
            symbol_name=symbol,
            timeframe='5m',
            limit_candles=fetch_limit,
            exchange_id='bybit'
        )

        if df_5min is None or df_5min.empty:
            # Пытаемся автоматически скачать и загрузить свечи в БД
            try:
                print(f"📥 Данные не найдены для {symbol}. Пытаюсь скачать и загрузить в БД...")
                csv_file_path = parser_download_and_combine_with_library(
                    symbol=symbol,
                    interval='5m',
                    months_to_fetch=12,
                    desired_candles=fetch_limit
                )
                if csv_file_path:
                    loaded_count = load_latest_candles_from_csv_to_db(
                        file_path=csv_file_path,
                        symbol_name=symbol,
                        timeframe='5m'
                    )
                    print(f"✅ Загрузка в БД завершена: {loaded_count} свечей")
                # Повторно пробуем получить из БД
                df_5min = db_get_or_fetch_ohlcv(
                    symbol_name=symbol,
                    timeframe='5m',
                    limit_candles=fetch_limit,
                    exchange_id='bybit'
                )
            except Exception as fetch_err:
                print(f"❌ Не удалось автоматически загрузить данные для {symbol}: {fetch_err}")
                df_5min = None
        
        if df_5min is None or df_5min.empty:
            return {"message": f"❌ Данные для {symbol} не найдены"}

        # Cutoff by time: exclude last N days, then take last train_window candles
        try:
            if cd > 0:
                if "timestamp" not in df_5min.columns:
                    return {"message": f"❌ {symbol}: df_5min has no timestamp for cutoff_days={cd}"}
                max_ts = int(float(df_5min["timestamp"].max()))
                cutoff_ts = int(max_ts - cd * 24 * 3600 * 1000)
                df_cut = df_5min[df_5min["timestamp"] <= cutoff_ts].copy()
                if len(df_cut) < train_window:
                    return {"message": f"❌ {symbol}: not enough candles for cutoff_days={cd} (have={len(df_cut)} need={train_window})"}
                df_5min = df_cut.tail(train_window).copy()
                print(f"🗓 cutoff_days={cd} cutoff_ts_ms={cutoff_ts} train_window={train_window} fetched={fetch_limit}")
            else:
                df_5min = df_5min.tail(train_window).copy()
        except Exception as _ce:
            return {"message": f"❌ {symbol}: cutoff failed: {_ce}"}

        import pandas as pd
        df_5min['datetime'] = pd.to_datetime(df_5min['timestamp'], unit='ms')
        df_5min.set_index('datetime', inplace=True)

        df_15min = df_5min.resample('15min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
        }).dropna().reset_index()

        df_1h = df_5min.resample('1h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
        }).dropna().reset_index()

        dfs = {
            'df_5min': df_5min,
            'df_15min': df_15min,
            'df_1h': df_1h,
            'symbol': symbol,
            'cutoff_days': cd,
            'encoder': {
                'id': encoder_id,
                'train_encoder': bool(train_encoder),
            },
        }

        print(f"📈 {symbol}: 5m={len(df_5min)}, 15m={len(df_15min)}, 1h={len(df_1h)}")

        # Получаем количество эпизодов из аргумента или переменной окружения
        if episodes is None:
            episodes = int(os.getenv('DEFAULT_EPISODES', 5))
        print(f"🎯 Количество эпизодов: {episodes}")

        # Получаем длину эпизода из аргумента или GConfig
        if episode_length is None:
            # Берем из GConfig по умолчанию
            from envs.dqn_model.gym.gconfig import GConfig
            episode_length = GConfig.episode_length
        print(f"📏 Длина эпизода: {episode_length}")

        # Прокидываем пути для продолжения обучения из ENV/Redis если заданы
        load_model_path = os.environ.get('CONTINUE_MODEL_PATH')
        load_buffer_path = os.environ.get('CONTINUE_BUFFER_PATH')
        # Определим родителя/корень для цепочки run'ов при дообучении
        parent_run_id = None
        root_run_id = None
        try:
            # Попробуем Redis как приоритетный источник
            r = Redis(host='redis', port=6379, db=0, decode_responses=True)
            v_model = r.get('continue:model_path')
            v_buffer = r.get('continue:buffer_path')
            if v_model:
                load_model_path = v_model
            if v_buffer:
                load_buffer_path = v_buffer
            # Чистим ключи, чтобы не повлиять на другие задачи
            if v_model:
                r.delete('continue:model_path')
            if v_buffer:
                r.delete('continue:buffer_path')
        except Exception:
            pass

        # Если дообучаем из структурированного пути runs/.../model.pth — проставим связи
        try:
            if isinstance(load_model_path, str):
                norm_path = load_model_path.replace('\\', '/')
                parts = norm_path.split('/')
                if len(parts) >= 4 and parts[-1] == 'model.pth' and 'runs' in parts:
                    runs_idx = parts.index('runs')
                    if runs_idx + 1 < len(parts):
                        parent_run_id = parts[runs_idx + 1]
                        # Попытаемся прочитать root_id из манифеста родителя
                        try:
                            parent_dir = os.path.dirname(load_model_path)
                            manifest_path = os.path.join(parent_dir, 'manifest.json')
                            if os.path.exists(manifest_path):
                                import json as _json
                                with open(manifest_path, 'r', encoding='utf-8') as mf:
                                    mf_data = _json.load(mf)
                                root_run_id = mf_data.get('root_id') or parent_run_id
                            else:
                                root_run_id = parent_run_id
                        except Exception:
                            root_run_id = parent_run_id
        except Exception:
            parent_run_id = parent_run_id or None
            root_run_id = root_run_id or None

        if (engine or '').lower() in ('ts','tianshou'):
            # Запуск через Tianshou тренер
            run_dir = train_tianshou_dqn(
                dfs=dfs,
                episodes=episodes or int(os.getenv('DEFAULT_EPISODES', 5)),
                n_envs=int(get_config_value('TS_NUM_ENVS', '4')),
                batch_size=int(os.getenv('TS_BATCH_SIZE', '256')),
                lr=float(os.getenv('TS_LR', '0.001')),
                gamma=float(os.getenv('TS_GAMMA', '0.99')),
                n_step=int(os.getenv('TS_N_STEP', '1')),
                target_update_freq=int(os.getenv('TS_TARGET_UPDATE', '500')),
                memory_size=int(os.getenv('TS_MEMORY', '500000')),
                episode_length=episode_length,
                run_id=None,
                symbol_hint=symbol,
                parent_run_id=parent_run_id,
                root_id=root_run_id,
                load_model_path=load_model_path,
                load_buffer_path=load_buffer_path,
                save_frequency=int(os.getenv('TS_SAVE_FREQ', '50')),
                buffer_save_frequency=None,
                save_replay_on_improvement=True,
                seed=seed,
            )
            return {"message": f"Tianshou run saved to {run_dir}"}
        else:
            cfg_obj = None
            if isinstance(cfg_overrides, dict) and cfg_overrides:
                try:
                    from agents.vdqn.cfg.vconfig import vDqnConfig  # local import
                    cfg_obj = vDqnConfig()
                    for k, v in cfg_overrides.items():
                        if not hasattr(cfg_obj, k):
                            continue
                        if k in ("batch_size", "memory_size", "eps_decay_steps", "target_update_freq", "train_repeats"):
                            setattr(cfg_obj, k, int(v))
                        elif k in ("lr", "dropout_rate"):
                            setattr(cfg_obj, k, float(v))
                        elif k in ("use_amp", "use_gpu_storage", "use_torch_compile"):
                            setattr(cfg_obj, k, bool(v))
                        elif k == "hidden_sizes":
                            if isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
                                setattr(cfg_obj, k, tuple(int(x) for x in v))
                            else:
                                raise ValueError("hidden_sizes must be list/tuple of ints")
                        else:
                            # allow setting other scalar attrs if type is safe
                            if isinstance(v, (int, float, bool, str, type(None))):
                                setattr(cfg_obj, k, v)
                except Exception as _cfg_err:
                    raise ValueError(f"bad cfg_overrides: {_cfg_err}")

            result = train_model_optimized(
                dfs=dfs,
                cfg=cfg_obj,
                episodes=episodes,
                load_model_path=load_model_path,
                load_buffer_path=load_buffer_path,
                seed=seed,
                parent_run_id=parent_run_id,
                root_id=root_run_id,
                episode_length=episode_length,
                direction=(direction or 'long'),
                env_overrides=env_overrides,
            )
        return {"message": f"✅ Обучение {symbol} завершено: {result}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"message": f"❌ Ошибка обучения {symbol}: {str(e)}"}
    finally:
        # Снимаем per-symbol running_key и удаляем текущую задачу из UI-списка
        try:
            from utils.redis_utils import get_redis_client  # локальный импорт чтобы не тянуть при инициализации
            redis_client = get_redis_client()
            running_key = f"celery:train:task:{(symbol or '').upper()}"
            try:
                redis_client.delete(running_key)
            except Exception:
                pass
            # Удаляем текущий task_id из ui:tasks
            try:
                ui_tasks_key = "ui:tasks"
                current_task_id = getattr(getattr(self, 'request', None), 'id', None)
                if current_task_id:
                    redis_client.lrem(ui_tasks_key, 0, current_task_id)
                    # Помечаем задачу как завершённую для идемпотентности на 24 часа
                    try:
                        redis_client.setex(f"celery:train:done:{current_task_id}", 24 * 3600, "1")
                    except Exception:
                        pass
                # Снимаем бизнес-‘running’ и отмечаем ‘finished’ по набору параметров
                try:
                    _engine = (engine or '').lower()
                    _enc = (encoder_id or '-')
                    _train_enc = 1 if train_encoder else 0
                    _eps = episodes if (episodes is not None) else 'env'
                    _ep_len = episode_length if (episode_length is not None) else 'cfg'
                    _rm = _rm_tag(env_overrides)
                    _biz_key = f"{(symbol or '').upper()}|{_engine}|{_enc}|{_train_enc}|{_eps}|{_ep_len}|{(direction or 'long')}|{_rm}"
                    _running_key_biz = f"celery:train:running:{_biz_key}"
                    _finished_key = f"celery:train:finished:{_biz_key}"
                    redis_client.delete(_running_key_biz)
                    # TTL «недавно завершено»: по умолчанию 1 час (можно переопределить ENV TRAIN_FINISHED_TTL)
                    try:
                        _ttl = int(os.environ.get('TRAIN_FINISHED_TTL', '3600'))
                    except Exception:
                        _ttl = 3600
                    redis_client.setex(_finished_key, max(60, _ttl), '1')
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0}, queue='train')
def train_dqn_multi_crypto(self, episodes: int | None = None, seed: int | None = None, episode_length: int = 2000):
    """Задача для мультивалютного обучения DQN"""
    self.update_state(state="IN_PROGRESS", meta={"progress": 0})
    # Сид из аргумента/ENV
    seed = int(seed) if seed is not None else None
    if seed is not None:
        set_global_seed(seed)
        print(f"🔒 Seed установлен: {seed}")
        # ENV больше не используем для сидов

    print("🚀 Начинаю мультивалютное обучение DQN...")
    try:
        # Новый модуль для мульти-обучения
        from agents.multi.v_train_multi import train_multi
        
        # Получаем количество эпизодов из переменной окружения
        episodes = int(os.getenv('DEFAULT_EPISODES', 10001))
        print(f"🎯 Количество эпизодов для мульти-обучения: {episodes}")

        # Получаем длину эпизода из аргумента или GConfig
        if episode_length is None:
            # Берем из GConfig по умолчанию
            from envs.dqn_model.gym.gconfig import GConfig
            episode_length = GConfig.episode_length
        print(f"📏 Длина эпизода для мульти-обучения: {episode_length}")
        
        result = train_multi(symbols=[
            'BTCUSDT','TONUSDT','ETHUSDT','SOLUSDT','ADAUSDT','BNBUSDT'
        ], episodes=episodes, episode_length=episode_length)
        return {"message": f"Мультивалютное обучение завершено: {result}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"message": f"Ошибка мульти-обучения: {str(e)}"}

# --- CNN Training Task ---
@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0}, queue='train')
def train_cnn_model(self, symbol: str, model_type: str = "multiframe", 
                   seed: int = None):
    """Обучение CNN модели для анализа паттернов криптовалют"""
    
    self.update_state(state="IN_PROGRESS", meta={"progress": 0, "symbol": symbol})
    
    try:
        print(f"🧠 Начинаю обучение CNN модели для {symbol}")
        
        # Импортируем CNN модули
        try:
            from cnn_training.config import CNNTrainingConfig
            from cnn_training.trainer import CNNTrainer
        except ImportError as ie:
            print(f"❌ Ошибка импорта CNN модулей: {ie}")
            raise Exception(f"CNN модули не найдены: {ie}")
        
        # Создаем конфигурацию (все параметры из config.py)
        config = CNNTrainingConfig(
            symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"],  # Используем все символы из конфига
            timeframes=["5m", "15m", "1h"],
            device="auto"
        )
        
        # Создаем тренер
        trainer = CNNTrainer(config)
        
        self.update_state(state="IN_PROGRESS", meta={"progress": 20, "message": "CNN тренер создан"})
        
        # Обучаем модель prediction типа (рекомендуется)
        print(f"🎯 Обучение CNN модели {model_type} для всех символов: {config.symbols}")
        
        if model_type == "multiframe":
            # Обучаем мультифреймовую модель на всех символах
            result = trainer.train_multiframe_model(config.symbols)
        else:
            # Обучаем отдельную модель для 5m фрейма (основной для DQN)
            result = trainer.train_single_model(symbol, "5m", model_type)

        # Подготовим сериализуемый ответ (без PyTorch-моделей и прочих несериализуемых объектов)
        safe_result = None
        try:
            if isinstance(result, dict):
                best_val_accuracy = result.get('best_val_accuracy')
                safe_result = {
                    "best_val_accuracy": float(best_val_accuracy) if best_val_accuracy is not None else None,
                    "train_steps": int(len(result.get('train_losses', []) or [])),
                    "val_steps": int(len(result.get('val_losses', []) or [])),
                }
        except Exception:
            safe_result = {"best_val_accuracy": None, "train_steps": 0, "val_steps": 0}

        self.update_state(state="SUCCESS", meta={
            "progress": 100,
            "message": f"CNN обучение завершено для {symbol}",
            "result": safe_result
        })

        return {
            "success": True,
            "message": f"CNN обучение завершено для {symbol}",
            "symbol": symbol,
            "model_type": model_type,
            "result": safe_result
        }
        
    except Exception as e:
        error_msg = f"❌ Ошибка обучения CNN для {symbol}: {str(e)}"
        print(error_msg)
        import traceback
        print(traceback.format_exc())
        
        self.update_state(state="FAILURE", meta={
            "progress": 0,
            "error": error_msg
        })
        
        return {
            "success": False,
            "error": error_msg,
            "symbol": symbol
        }


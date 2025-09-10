from agents.vdqn.v_train_model import train_model
from agents.vdqn.v_train_model_optimized import train_model_optimized
from celery import Celery
from kombu import Queue
import time
import os

import pandas as pd

import json
import requests
from redis import Redis
import numpy as np

from utils.db_utils import db_get_or_fetch_ohlcv  # Импортируем функцию загрузки данных
from utils.db_utils import load_latest_candles_from_csv_to_db
from utils.parser import parser_download_and_combine_with_library
from utils.trade_utils import create_model_prediction
# Загружаем переменные окружения из .env (если есть), чтобы Celery видел ключи
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass
from datetime import datetime
from celery.schedules import crontab

# API ключи Bybit (без шумного вывода при импорте)
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_SECRET_KEY = os.getenv('BYBIT_SECRET_KEY')

def are_bybit_keys_configured() -> bool:
    try:
        return bool(BYBIT_API_KEY) and bool(BYBIT_SECRET_KEY)
    except Exception:
        return False

# Настраиваем Celery с Redis как брокером и бекендом
celery = Celery(
    "tasks",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0"
)

# Определяем очереди и маршрутизацию задач:
# По умолчанию все задачи идут в очередь 'celery',
# а тренировочные задачи направляем в отдельную очередь 'train'.
celery.conf.task_queues = (
    Queue('celery'),
    Queue('train'),
    Queue('trade'),
)
celery.conf.task_default_queue = 'celery'
celery.conf.task_routes = {
    'tasks.celery_tasks.train_dqn': {'queue': 'train'},
    'tasks.celery_tasks.train_dqn_symbol': {'queue': 'train'},
    'tasks.celery_tasks.train_dqn_multi_crypto': {'queue': 'train'},
    'tasks.celery_tasks.execute_trade': {'queue': 'trade'},
}

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0}, queue='trade')
def execute_trade(self, symbols: list, model_path: str | None = None, model_paths: list | None = None):
    """Исполнение торгового шага: предсказание через serving, торговля через TradingAgent."""
    try:
        from trading_agent.trading_agent import TradingAgent
        from utils.db_utils import db_get_or_fetch_ohlcv

        # 1) Читаем параметры из Redis при необходимости
        try:
            rc = Redis(host='redis', port=6379, db=0, decode_responses=True)
        except Exception:
            rc = None

        if (not symbols) and rc is not None:
            try:
                _sym_raw = rc.get('trading:symbols')
                if _sym_raw:
                    _sym = json.loads(_sym_raw)
                    if isinstance(_sym, list) and _sym:
                        symbols = _sym
            except Exception:
                pass
        if not symbols:
            symbols = ['BTCUSDT']

        # Если не передали model_paths аргументом — читаем из Redis
        if model_paths is None and rc is not None:
            try:
                _mps = rc.get('trading:model_paths')
                if _mps:
                    parsed = json.loads(_mps)
                    if isinstance(parsed, list) and parsed:
                        model_paths = parsed
            except Exception:
                model_paths = None
        if (model_paths is None or not model_paths) and model_path:
            model_paths = [model_path]
        if not model_paths:
            return {"success": False, "error": "model_paths not provided"}

        # 2) Готовим состояние для serving (как в агенте: закрытые 5m свечи -> плотный вектор)
        symbol = symbols[0]
        # Последняя закрытая метка времени (5m)
        def _last_closed_ts_ms():
            try:
                now_utc = datetime.utcnow().timestamp()
                last_closed = (int(now_utc) // 300) * 300 - 300
                return last_closed * 1000
            except Exception:
                return 0

        df_5m = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe='5m', limit_candles=120, exchange_id='bybit')
        if df_5m is None or df_5m.empty:
            return {"success": False, "error": f"no candles in DB for {symbol}"}
        cutoff = _last_closed_ts_ms()
        df_5m = df_5m[df_5m['timestamp'] <= cutoff]
        if df_5m is None or df_5m.empty:
            return {"success": False, "error": "no closed candles available"}
        # Простая нормализация: последние 100 строк OHLCV
        ohlcv_cols = ['open','high','low','close','volume']
        arr = df_5m[ohlcv_cols].tail(100).values.astype('float32')
        if arr.shape[0] < 20:
            return {"success": False, "error": "insufficient data for state"}
        max_vals = np.maximum(arr.max(axis=0), 1e-9)
        norm = (arr / max_vals).flatten()
        # Ограничим/дополняем до 100*5=500 признаков
        if norm.size < 500:
            norm = np.pad(norm, (0, 500 - norm.size))
        elif norm.size > 500:
            norm = norm[:500]
        state = norm.tolist()

        # 3) Вызов serving
        serving_url = os.environ.get('SERVING_URL', 'http://serving:8000/predict_ensemble')
        payload = {
            "state": state,
            "model_paths": model_paths,
            "symbol": symbol
        }
        try:
            resp = requests.post(serving_url, json=payload, timeout=30)
            # Пытаемся извлечь тело при ошибке
            if not resp.ok:
                body = None
                try:
                    body = resp.text
                except Exception:
                    body = None
                return {"success": False, "error": f"serving error: {resp.status_code} {resp.reason}", "body": body}
            pred_json = resp.json()
        except Exception as e:
            return {"success": False, "error": f"serving error: {e}"}

        if not pred_json.get('success'):
            return {"success": False, "error": pred_json.get('error', 'serving failed')}

        decision = pred_json.get('decision', 'hold')
        # --- Server-side Q-gate ---
        try:
            # Ищем агрегированные пороги в ответе или используем дефолт
            qgate = pred_json.get('qgate') or {}
            T1 = float(qgate.get('T1', pred_json.get('qgate_T1', 0.35)))
            T2 = float(qgate.get('T2', pred_json.get('qgate_T2', 0.25)))
            # Берём q_values из решения ансамбля (если есть) либо из первого предикта
            q_values = pred_json.get('q_values')
            if not isinstance(q_values, list):
                preds_list = pred_json.get('predictions') or []
                if preds_list:
                    q_values = preds_list[0].get('q_values')
            if isinstance(q_values, list) and len(q_values) >= 2:
                q_sorted = sorted([float(x) for x in q_values], reverse=True)
                maxQ = q_sorted[0]
                gapQ = q_sorted[0] - q_sorted[1]
                passed = (maxQ >= T1) and (gapQ >= T2)
                # Если не прошёл фильтр — принудительно HOLD
                if not passed:
                    decision = 'hold'
                # Лог для диагностики
                try:
                    print(f"Q‑gate: {'PASS' if passed else 'BLOCK'} (maxQ={maxQ:.3f}, gapQ={gapQ:.3f}, T1={T1:.3f}, T2={T2:.3f})")
                except Exception:
                    pass
        except Exception:
            pass

        # 4) Торговля через TradingAgent (без docker exec)
        agent = TradingAgent(model_path=(model_paths[0] if model_paths else None))
        agent.symbols = symbols
        agent.symbol = symbol
        agent.base_symbol = symbol
        try:
            agent.trade_amount = agent._calculate_trade_amount()
        except Exception:
            agent.trade_amount = getattr(agent, 'trade_amount', 0.0)

        # Отметим, что торговый цикл активен (для UI)
        try:
            agent.is_trading = True
        except Exception:
            pass

        # Проставим последнее предсказание для UI
        try:
            agent.last_model_prediction = decision
        except Exception:
            pass

        current_status_before = agent.get_trading_status()

        # 4.1) Сохраняем предсказания в БД (по каждому пути модели)
        try:
            # Текущая цена: возьмём close последней закрытой свечи
            try:
                current_price = float(df_5m['close'].iloc[-1]) if (df_5m is not None and not df_5m.empty) else None
            except Exception:
                current_price = None
            position_status = 'open' if getattr(agent, 'current_position', None) else 'none'
            preds_list = pred_json.get('predictions') or []
            for p in preds_list:
                try:
                    mp = p.get('model_path')
                    act = p.get('action')
                    qv = p.get('q_values') or []
                    # сохранение без market_conditions (можно расширить позже)
                    create_model_prediction(
                        symbol=symbol,
                        action=str(act or 'hold'),
                        q_values=list(qv) if isinstance(qv, (list, tuple)) else [],
                        current_price=current_price,
                        position_status=position_status,
                        model_path=str(mp) if mp is not None else '' ,
                        market_conditions=None
                    )
                except Exception:
                    # Не ломаем торговый цикл из-за БД
                    pass
        except Exception:
            pass

        trade_result = None
        if decision == 'buy' and not agent.current_position:
            trade_result = agent._execute_buy()
        elif decision == 'sell' and agent.current_position:
            sell_strategy = agent._determine_sell_amount(agent._get_current_price())
            trade_result = agent._execute_sell() if sell_strategy.get('sell_all') else agent._execute_partial_sell(sell_strategy.get('sell_amount', 0))
        else:
            trade_result = {"success": True, "action": "hold"}

        status_after = agent.get_trading_status()

        # 5) Сохранение результата в Redis (как раньше)
        try:
            if rc is not None:
                result_data = {
                    'timestamp': datetime.now().isoformat(),
                    'symbols': symbols,
                    'model_paths': model_paths,
                    'decision': decision,
                    'serving_url': serving_url,
                    'predictions_count': len(pred_json.get('predictions', []) or []),
                    'trade_result': trade_result,
                }
                rc.setex(f'trading:latest_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}', 3600, json.dumps(result_data, default=str))
                rc.set('trading:current_status', json.dumps(status_after, default=str))
                rc.set('trading:current_status_ts', datetime.utcnow().isoformat())
        except Exception:
            pass

        return {
            "success": True,
            "decision": decision,
            "status_before": current_status_before,
            "status_after": status_after,
            "trade_result": trade_result,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0})
def search_lstm_task(self, query):
    """Фоновая задача, которая выполняется долго"""
    self.update_state(state="IN_PROGRESS", meta={"progress": 0})

    for i in range(5):  # Имитация долгого вычисления
        time.sleep(2)
        self.update_state(state="IN_PROGRESS", meta={"progress": (i + 1) * 20})

    return {"message": "Task completed!", "query": query}

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0}, queue='train')
def train_dqn(self):
    
    self.update_state(state="IN_PROGRESS", meta={"progress": 0})
    
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
    
    result = train_model_optimized(dfs=df, episodes=episodes)
    return {"message": result}

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0}, queue='train')
def train_dqn_symbol(self, symbol: str, episodes: int = None):
    """Обучение DQN для одного символа (BTCUSDT/ETHUSDT/...)

    Загружает данные из БД, готовит 5m/15m/1h, запускает train_model_optimized.
    """
    self.update_state(state="IN_PROGRESS", meta={"progress": 0, "symbol": symbol})

    try:
        print(f"\n🚀 Старт обучения для {symbol} [{datetime.now()}]")
        df_5min = db_get_or_fetch_ohlcv(
            symbol_name=symbol,
            timeframe='5m',
            limit_candles=100000,
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
                    desired_candles=100000
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
                    limit_candles=100000,
                    exchange_id='bybit'
                )
            except Exception as fetch_err:
                print(f"❌ Не удалось автоматически загрузить данные для {symbol}: {fetch_err}")
                df_5min = None
        
        if df_5min is None or df_5min.empty:
            return {"message": f"❌ Данные для {symbol} не найдены"}

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
        }

        print(f"📈 {symbol}: 5m={len(df_5min)}, 15m={len(df_15min)}, 1h={len(df_1h)}")

        # Получаем количество эпизодов из аргумента или переменной окружения
        if episodes is None:
            episodes = int(os.getenv('DEFAULT_EPISODES', 5))
        print(f"🎯 Количество эпизодов: {episodes}")

        # Прокидываем пути для продолжения обучения из ENV/Redis если заданы
        load_model_path = os.environ.get('CONTINUE_MODEL_PATH')
        load_buffer_path = os.environ.get('CONTINUE_BUFFER_PATH')
        try:
            # Попробуем Redis как приоритетный источник
            from redis import Redis
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

        result = train_model_optimized(
            dfs=dfs,
            episodes=episodes,
            load_model_path=load_model_path,
            load_buffer_path=load_buffer_path
        )
        return {"message": f"✅ Обучение {symbol} завершено: {result}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"message": f"❌ Ошибка обучения {symbol}: {str(e)}"}

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0}, queue='train')
def train_dqn_multi_crypto(self):
    """Задача для мультивалютного обучения DQN"""
    self.update_state(state="IN_PROGRESS", meta={"progress": 0})
    print("🚀 Начинаю мультивалютное обучение DQN...")
    try:
        # Новый модуль для мульти-обучения
        from agents.multi.v_train_multi import train_multi
        
        # Получаем количество эпизодов из переменной окружения
        episodes = int(os.getenv('DEFAULT_EPISODES', 10001))
        print(f"🎯 Количество эпизодов для мульти-обучения: {episodes}")
        
        result = train_multi(symbols=[
            'BTCUSDT','TONUSDT','ETHUSDT','SOLUSDT','ADAUSDT','BNBUSDT'
        ], episodes=episodes)
        return {"message": f"Мультивалютное обучение завершено: {result}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"message": f"Ошибка мульти-обучения: {str(e)}"}

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0})
def trade_step():
    """
    Выполняет один торговый шаг с использованием API Bybit
    """
    try:
        # Проверяем наличие API ключей
        if BYBIT_API_KEY == 'your_bybit_api_key_here' or BYBIT_SECRET_KEY == 'your_bybit_secret_key_here':
            return {"error": "API ключи Bybit не настроены"}
        
        # Получаем текущее состояние рынка (замени на получение реальных данных)
        state = get_current_market_state()  # реализуй функцию получения состояния

        action = trade_once(state)

        # Здесь ты можешь сделать реальный ордер через API биржи
        # Используем API ключи для подключения к Bybit
        import ccxt
        exchange = ccxt.bybit({
            'apiKey': BYBIT_API_KEY,
            'secret': BYBIT_SECRET_KEY,
            'sandbox': False,  # True для тестового режима
            'enableRateLimit': True
        })

        return f"Торговое действие: {action} (API подключен)"
        
    except Exception as e:
        return {"error": f"Ошибка в trade_step: {str(e)}"}

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0})
def start_trading_task(self, symbols, model_path=None):
    """
    Оркестратор: получает параметры, делает Redis-лок, публикует provisional-статус
    и кладёт торговую задачу в очередь trade (без docker exec).
    """
    import os
    from celery import chain

    # Проверяем, должна ли работать торговля
    trading_enabled = os.environ.get('ENABLE_TRADING_BEAT', '1') in ('1', 'true', 'True')
    if not trading_enabled:
        return {"success": False, "skipped": True, "reason": "ENABLE_TRADING_BEAT=0"}

    # Redis-лок: предотвращаем параллельные запуски в пределах ~5 минут (per-symbol)
    try:
        from redis import Redis as _Redis
        _rc_lock = _Redis(host='redis', port=6379, db=0, decode_responses=True)
        # Определяем символ для ключа
        lock_symbol = None
        try:
            lock_symbol = (symbols[0] if (symbols and len(symbols) > 0) else None)
        except Exception:
            lock_symbol = None
        if not lock_symbol:
            lock_symbol = 'ALL'
        lock_key = f'trading:agent_lock:{lock_symbol}'
        # TTL 240с (4 минуты) — чтобы следующий тик в 5 минут не срезался из-за ручного запуска
        got_lock = _rc_lock.set(lock_key, self.request.id, nx=True, ex=240)
        if not got_lock:
            return {"success": False, "skipped": True, "reason": "agent_lock_active"}
    except Exception:
        pass

    # Если параметры не передали — пробуем взять их из Redis
    try:
        if (not symbols) or model_path is None:
            from redis import Redis
            _r = Redis(host='redis', port=6379, db=0, decode_responses=True)
            if (not symbols):
                try:
                    _sym_raw = _r.get('trading:symbols')
                    if _sym_raw:
                        import json as _json
                        _sym = _json.loads(_sym_raw)
                        if isinstance(_sym, list) and _sym:
                            symbols = _sym
                except Exception:
                    pass
            if model_path is None:
                try:
                    _mp = _r.get('trading:model_path')
                    if _mp:
                        model_path = _mp
                except Exception:
                    pass
    except Exception:
        pass

    # Дефолты
    if not symbols:
        symbols = ['BTCUSDT']

    print(f"🚀 Оркестрация торговли: symbols={symbols} | model_path={model_path if model_path else 'default'}")
    self.update_state(state="IN_PROGRESS", meta={"progress": 0})

    # Получаем список путей моделей (предпочтительно из Redis), иначе из model_path
    model_paths = None
    try:
        from redis import Redis as _Redis
        _r2 = _Redis(host='redis', port=6379, db=0, decode_responses=True)
        _mps = _r2.get('trading:model_paths')
        if _mps:
            import json as _json
            parsed = _json.loads(_mps)
            if isinstance(parsed, list) and parsed:
                model_paths = parsed
    except Exception:
        model_paths = None
    if (model_paths is None or not model_paths) and model_path:
        model_paths = [model_path]

    # Если моделей нет — снимаем лок и выходим без постановки задачи в trade
    if not model_paths:
        try:
            # Снять лок, чтобы не блокировать следующий запуск
            if '_rc_lock' in locals():
                lock_symbol = (symbols[0] if (symbols and len(symbols) > 0) else 'ALL')
                lock_key = f'trading:agent_lock:{lock_symbol}'
                _rc_lock.delete(lock_key)
        except Exception:
            pass
        return {"success": False, "skipped": True, "reason": "no_model_paths"}

    # Промежуточный статус в Redis для UI
    try:
        from redis import Redis as _Redis
        import json as _json
        provisional = {
            'success': True,
            'is_trading': True,
            'trading_status': 'Активна',
            'trading_status_emoji': '🟢',
            'trading_status_full': '🟢 Активна',
            'symbol': (symbols[0] if symbols else None),
            'symbol_display': (symbols[0] if symbols else 'Не указана'),
            'amount': None,
            'amount_display': 'Не указано',
            'amount_usdt': 0.0,
            'position': None,
            'trades_count': 0,
            'balance': {},
            'current_price': 0.0,
            'last_model_prediction': None,
        }
        _rc = _Redis(host='redis', port=6379, db=0, decode_responses=True)
        _rc.set('trading:current_status', _json.dumps(provisional, ensure_ascii=False))
        from datetime import datetime as _dt
        _rc.set('trading:current_status_ts', _dt.utcnow().isoformat())
    except Exception:
        pass

    # Кладём торговую задачу в очередь trade
    try:
        res = execute_trade.apply_async(kwargs={
            'symbols': symbols,
            'model_path': model_path,
            'model_paths': model_paths,
        }, queue='trade')
        return {"success": True, "enqueued": True, "task_id": res.id}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Включаем периодический запуск торговли
import os
# Настройка расписания Celery Beat по флагу окружения (не перетираем значение)
if os.environ.get('ENABLE_TRADING_BEAT', '0').lower() in ('1', 'true', 'yes', 'on'):
    celery.conf.beat_schedule = {
        'start-trading-every-5-minutes': {
            'task': 'tasks.celery_tasks.start_trading_task',
            'schedule': crontab(minute='*/5'),
            'args': ([], None)  # Символы и путь к модели будут передаваться из веб-интерфейса
        },
    }
    celery.conf.timezone = 'UTC'
    print("✅ Периодическая торговля включена (каждые 5 минут)")
else:
    print("⚠️ Периодическая торговля отключена (ENABLE_TRADING_BEAT=0)")

# --- Периодический апдейтер статуса в Redis ---
@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0})
def refresh_trading_status(self):
    """Обновляет trading:current_status в Redis, если он отсутствует или устарел.

    Лёгкий хелпер для UI: не лезет в биржу, не вызывает модель.
    Помечает is_trading исходя из наличия активного lock ключа.
    """
    try:
        from redis import Redis as _Redis
        import json as _json
        from datetime import datetime as _dt, timedelta as _td

        rc = _Redis(host='redis', port=6379, db=0, decode_responses=True)

        # Текущие параметры
        try:
            symbols_raw = rc.get('trading:symbols')
            symbols = _json.loads(symbols_raw) if symbols_raw else ['BTCUSDT']
            if not isinstance(symbols, list) or not symbols:
                symbols = ['BTCUSDT']
        except Exception:
            symbols = ['BTCUSDT']
        sym = symbols[0]

        # Текущий статус
        cached = rc.get('trading:current_status')
        cached_ts = rc.get('trading:current_status_ts')

        # Проверяем свежесть (6 минут)
        is_fresh = False
        try:
            if cached_ts:
                ts = _dt.fromisoformat(cached_ts)
                is_fresh = _dt.utcnow() <= (ts + _td(minutes=6))
        except Exception:
            is_fresh = False

        if cached and is_fresh:
            return {"success": True, "updated": False, "reason": "fresh"}

        # Активность оцениваем по наличию lock ключа с TTL > 0
        is_active = False
        try:
            lock_key = f'trading:agent_lock:{sym}'
            ttl = rc.ttl(lock_key)
            if ttl is not None and int(ttl) > 0:
                is_active = True
        except Exception:
            is_active = False

        # Базовый статус
        status = {
            'success': True,
            'is_trading': bool(is_active),
            'trading_status': 'Активна' if is_active else 'Остановлена',
            'trading_status_emoji': '🟢' if is_active else '🔴',
            'trading_status_full': ('🟢 Активна' if is_active else '🔴 Остановлена'),
            'symbol': sym,
            'symbol_display': sym,
            'amount': None,
            'amount_display': 'Не указано',
            'amount_usdt': 0.0,
            'position': None,
            'trades_count': 0,
            'balance': {},
            'current_price': 0.0,
            'last_model_prediction': None,
        }

        # Не перетираем имеющиеся поля, если cached есть
        try:
            if cached:
                prev = _json.loads(cached)
                if isinstance(prev, dict):
                    prev.update({k: v for k, v in status.items() if k not in prev or prev.get(k) is None})
                    status = prev
        except Exception:
            pass

        rc.set('trading:current_status', _json.dumps(status, ensure_ascii=False))
        rc.set('trading:current_status_ts', _dt.utcnow().isoformat())
        return {"success": True, "updated": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

# keep beat schedule extension together with trading beat
if os.environ.get('ENABLE_TRADING_BEAT', '0').lower() in ('1', 'true', 'yes', 'on'):
    try:
        # Расширяем уже созданный beat_schedule
        celery.conf.beat_schedule.update({
            'refresh-trading-status-every-minute': {
                'task': 'tasks.celery_tasks.refresh_trading_status',
                'schedule': crontab(minute='*'),
                'args': (),
            },
        })
    except Exception:
        celery.conf.beat_schedule = {
            'refresh-trading-status-every-minute': {
                'task': 'tasks.celery_tasks.refresh_trading_status',
                'schedule': crontab(minute='*'),
                'args': (),
            },
        }

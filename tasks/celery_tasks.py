from agents.vdqn.v_train_model import train_model
from agents.vdqn.v_train_model_optimized import train_model_optimized
from celery import Celery
from kombu import Queue
import time
import os

import pandas as pd

import json

from utils.db_utils import db_get_or_fetch_ohlcv  # Импортируем функцию загрузки данных
from utils.db_utils import load_latest_candles_from_csv_to_db
from utils.parser import parser_download_and_combine_with_library
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
)
celery.conf.task_default_queue = 'celery'
celery.conf.task_routes = {
    'tasks.celery_tasks.train_dqn': {'queue': 'train'},
    'tasks.celery_tasks.train_dqn_symbol': {'queue': 'train'},
    'tasks.celery_tasks.train_dqn_multi_crypto': {'queue': 'train'},
}

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
    Task to start trading in the trading_agent container every 5 minutes.
    """
    import docker
    import os

    # Проверяем, должна ли работать торговля
    trading_enabled = os.environ.get('ENABLE_TRADING_BEAT', '1') in ('1', 'true', 'True')
    if not trading_enabled:
        return {"success": False, "skipped": True, "reason": "ENABLE_TRADING_BEAT=0"}
    
    # Redis-лок: предотвращаем параллельные запуски в пределах 5 минут
    try:
        from redis import Redis as _Redis
        _rc_lock = _Redis(host='redis', port=6379, db=0, decode_responses=True)
        lock_key = 'trading:agent_lock'
        # TTL 300с, set if not exists
        got_lock = _rc_lock.set(lock_key, datetime.utcnow().isoformat(), nx=True, ex=300)
        if not got_lock:
            return {"success": False, "skipped": True, "reason": "agent_lock_active"}
    except Exception as _e:
        # Если Redis недоступен — продолжаем без лока (мягкая деградация)
        pass
    
    # Если параметры не передали в расписании — пробуем взять их из Redis (последние заданные из веб‑интерфейса)
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

    # Дефолты на всякий случай
    if not symbols:
        symbols = ['BTCUSDT']

    print(f"🚀 Запуск торговой задачи для символов: {symbols} | model_path={model_path if model_path else 'default'}")
    
    self.update_state(state="IN_PROGRESS", meta={"progress": 0})
    
    # Перед запуском зафиксируем предварительный статус в Redis,
    # чтобы UI мгновенно видел "Активна" даже до первых RESULT
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

    # Connect to Docker
    client = docker.from_env()
    
    try:
        # Get the medoedai container
        container = client.containers.get('medoedai')
        
        # Check if the container is running
        if container.status != 'running':
            return {"success": False, "error": f'Container medoedai is not running. Status: {container.status}'}
        
        # Start trading via exec with API keys
        if model_path:
            cmd = f'python -c "import json; import os; os.environ[\'BYBIT_API_KEY\'] = \'{BYBIT_API_KEY}\'; os.environ[\'BYBIT_SECRET_KEY\'] = \'{BYBIT_SECRET_KEY}\'; from trading_agent.trading_agent import TradingAgent; agent = TradingAgent(model_path=\\"{model_path}\\"); start_result = agent.start_trading(symbols={symbols}); status_result = agent.get_trading_status(); print(\\"RESULT: \\" + json.dumps({{**start_result, **status_result}}, default=str))"'
        else:
            cmd = f'python -c "import json; import os; os.environ[\'BYBIT_API_KEY\'] = \'{BYBIT_API_KEY}\'; os.environ[\'BYBIT_SECRET_KEY\'] = \'{BYBIT_SECRET_KEY}\'; from trading_agent.trading_agent import TradingAgent; agent = TradingAgent(); start_result = agent.start_trading(symbols={symbols}); status_result = agent.get_trading_status(); print(\\"RESULT: \\" + json.dumps({{**start_result, **status_result}}, default=str))"'
        
        exec_result = container.exec_run(cmd, tty=True)
        
        # Log the execution result
        print(f"🚀 Start trading - Command: {cmd}")
        print(f"📊 Start trading - Exit code: {exec_result.exit_code}")
        
        # Инициализируем output_str
        output_str = ""
        if exec_result.output:
            output_str = exec_result.output.decode('utf-8')
            print(f"📝 Start trading - Output: {output_str}")
    
        
            # Сохраняем результат в Redis для веб-интерфейса
            try:
                from redis import Redis
                
                # Подключение к Redis
                redis_client = Redis(host='redis', port=6379, db=0, decode_responses=True)
                
                # Создаем результат для сохранения
                result_data = {
                    'timestamp': datetime.now().isoformat(),
                    'symbols': symbols,
                    'model_path': model_path,
                    'command': cmd,
                    'exit_code': exec_result.exit_code,
                    'output': output_str
                }
                
                # Парсим результат из вывода команды
                parsed_result = None
                if 'RESULT:' in output_str:
                    try:
                        result_str = output_str.split('RESULT:')[1].strip()
                        parsed_result = json.loads(result_str)
                        result_data['parsed_result'] = parsed_result
                        
                        # Определяем, была ли реальная торговая операция
                        trade_executed = parsed_result.get('trade_executed', 'hold')
                        if trade_executed in ['buy', 'sell', 'sell_all', 'sell_partial']:
                            # Реальная торговая операция
                            result_data['trade_executed'] = True
                            result_data['trade_type'] = trade_executed
                        else:
                            # Просто HOLD или ожидание
                            result_data['trade_executed'] = False
                            result_data['trade_type'] = 'hold'
                            
                    except Exception as parse_error:
                        print(f"Ошибка парсинга результата: {parse_error}")
                        result_data['parse_error'] = str(parse_error)
                        result_data['trade_executed'] = False
                        result_data['trade_type'] = 'unknown'
                
                # Сохраняем в Redis (последние 10 результатов)
                redis_key = f'trading:latest_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                redis_client.setex(redis_key, 3600, json.dumps(result_data, default=str))  # Храним 1 час
                
                # Единый текущий статус для быстрого чтения UI
                if parsed_result:
                    redis_client.set('trading:current_status', json.dumps(parsed_result, default=str))
                    redis_client.set('trading:current_status_ts', datetime.now().isoformat())
                
                # Очищаем старые результаты (оставляем только последние 10)
                all_keys = redis_client.keys('trading:latest_result_*')
                if len(all_keys) > 20:
                    # Сортируем по времени и удаляем старые
                    sorted_keys = sorted(all_keys)
                    for old_key in sorted_keys[:-10]:
                        redis_client.delete(old_key)
                        
            except Exception as redis_error:
                print(f"Ошибка сохранения в Redis: {redis_error}")
        
        if exec_result.exit_code == 0:
            if exec_result.output:
                output = exec_result.output.decode('utf-8')
                # Log the result
                if 'RESULT:' in output:
                    result_str = output.split('RESULT:')[1].strip()
                    try:
                        result = json.loads(result_str)
                        return result
                    except:                    
                        return {"success": True, "message": f'Trading started for {symbols}', "output": output}
                else:
                    return {"success": True, "message": f'Trading started for {symbols}', "output": output}
            else:
                return {"success": True, "message": f'Trading started for {symbols}', "output": "No output"}
        else:
            error_output = exec_result.output.decode("utf-8") if exec_result.output else "No output"
            return {"success": False, "error": f'Command execution error: {error_output}'}
        
    except docker.errors.NotFound:
        return {"success": False, "error": 'Container medoedai not found. Start it with docker-compose up medoedai'}
    except Exception as e:
        return {"success": False, "error": f'Docker error: {str(e)}'}

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

   

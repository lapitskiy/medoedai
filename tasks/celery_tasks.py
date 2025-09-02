from agents.vdqn.v_train_model import train_model
from agents.vdqn.v_train_model_optimized import train_model_optimized
from celery import Celery
import time
import os

import pandas as pd

import json

from utils.db_utils import db_get_or_fetch_ohlcv  # Импортируем функцию загрузки данных
from datetime import datetime
from celery.schedules import crontab

# API ключи Bybit
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY', 'your_bybit_api_key_here')
BYBIT_SECRET_KEY = os.getenv('BYBIT_SECRET_KEY', 'your_bybit_secret_key_here')

# Проверяем наличие API ключей
if BYBIT_API_KEY == 'your_bybit_api_key_here' or BYBIT_SECRET_KEY == 'your_bybit_secret_key_here':
    print("⚠️ ВНИМАНИЕ: API ключи Bybit не настроены!")
    print("Установите переменные окружения BYBIT_API_KEY и BYBIT_SECRET_KEY")
else:
    print("✅ API ключи Bybit настроены")

# Настраиваем Celery с Redis как брокером и бекендом
celery = Celery(
    "tasks",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0"
)

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0})
def search_lstm_task(self, query):
    """Фоновая задача, которая выполняется долго"""
    self.update_state(state="IN_PROGRESS", meta={"progress": 0})

    for i in range(5):  # Имитация долгого вычисления
        time.sleep(2)
        self.update_state(state="IN_PROGRESS", meta={"progress": (i + 1) * 20})

    return {"message": "Task completed!", "query": query}

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0})
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
        'BNBUSDT'   # Binance Coin
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

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0})
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

        result = train_model_optimized(dfs=dfs, episodes=episodes)
        return {"message": f"✅ Обучение {symbol} завершено: {result}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"message": f"❌ Ошибка обучения {symbol}: {str(e)}"}

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0})
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
    
    print(f"🚀 Запуск торговой задачи для символов: {symbols}")
    
    self.update_state(state="IN_PROGRESS", meta={"progress": 0})
    
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
            
            # Парсим результат
            if 'RESULT:' in output_str:
                try:
                    result_str = output_str.split('RESULT:')[1].strip()
                    result = json.loads(result_str)
                    print(f"✅ Parsed result: {result}")
                except Exception as parse_error:
                    print(f"❌ Error parsing result: {parse_error}")
                    print(f"Raw result string: {result_str}")
        
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
            if 'RESULT:' in output_str:
                try:
                    result_str = output_str.split('RESULT:')[1].strip()
                    parsed_result = json.loads(result_str)
                    result_data['parsed_result'] = parsed_result
                except Exception as parse_error:
                    print(f"Ошибка парсинга результата: {parse_error}")
                    result_data['parse_error'] = str(parse_error)
            
            # Сохраняем в Redis (последние 10 результатов)
            redis_key = f'trading:latest_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            redis_client.setex(redis_key, 3600, json.dumps(result_data, default=str))  # Храним 1 час
            
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
# Устанавливаем ENABLE_TRADING_BEAT=0 для отключения автоматической торговли
if os.environ.get('ENABLE_TRADING_BEAT') is None:
    os.environ['ENABLE_TRADING_BEAT'] = '0'
    print("✅ Автоматически отключен ENABLE_TRADING_BEAT=0")

if os.environ.get('ENABLE_TRADING_BEAT', '0') in ('1', 'true', 'True'):
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

   

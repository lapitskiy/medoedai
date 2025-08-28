from agents.vdqn.v_train_model import train_model
from agents.vdqn.v_train_model_optimized import train_model_optimized
from celery import Celery
import time

import pandas as pd

import json

from utils.db_utils import db_get_or_fetch_ohlcv  # Импортируем функцию загрузки данных

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
                limit_candles=100000
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
    result = train_model_optimized(dfs=df, episodes=10000)
    return {"message": result}

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0})
def train_dqn_multi_crypto(self):
    """Задача для мультивалютного обучения DQN"""
    
    self.update_state(state="IN_PROGRESS", meta={"progress": 0})
    
    print("🚀 Начинаю мультивалютное обучение DQN...")
    
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
                limit_candles=100000
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
    
    # Запускаем мультивалютное обучение
    print(f"\n🎯 Запуск мультивалютного обучения...")
    
    # Проверяем структуру данных перед передачей
    print("\n🔍 Проверка структуры данных:")
    for symbol, data in all_dfs.items():
        print(f"  {symbol}:")
        print(f"    df_5min: {type(data['df_5min'])} - {len(data['df_5min'])} строк")
        print(f"    df_15min: {type(data['df_15min'])} - {len(data['df_15min'])} строк")
        print(f"    df_1h: {type(data['df_1h'])} - {len(data['df_1h'])} строк")
        print(f"    symbol: {data['symbol']}")
        print(f"    candle_count: {data['candle_count']}")
    
    try:
        # Импортируем функцию мультивалютного обучения
        from agents.vdqn.v_train_model_optimized import train_model_optimized
        
        # ИСПРАВЛЕНИЕ: Передаем ВСЕ данные криптовалют для мультивалютного обучения
        print(f"🎯 Передаем данные всех {len(all_dfs)} криптовалют для мультивалютного обучения")
        print(f"📊 Структура данных для мультивалютного обучения:")
        for symbol, data in all_dfs.items():
            print(f"  • {symbol}: {data['candle_count']} свечей")
        
        print(f"🚀 Запуск МУЛЬТИВАЛЮТНОГО обучения:")
        print(f"  • Автоматическое переключение между криптовалютами для каждого эпизода")
        print(f"  • Случайный выбор криптовалюты при каждом reset()")
        print(f"  • Обучение на разнообразных рыночных условиях")
        print(f"  • Более стабильная модель благодаря разнообразию данных")
        
        # Передаем все данные криптовалют в мультивалютное окружение
        result = train_model_optimized(dfs=all_dfs, episodes=10001, use_wandb=False)
        return {"message": f"Мультивалютное обучение завершено: {result}"}
    except Exception as e:
        error_msg = f"Ошибка при мультивалютном обучении: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        print(f"🔍 Полный traceback:")
        traceback.print_exc()
        return {"message": error_msg}

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0})
def trade_step():
    # Получаем текущее состояние рынка (замени на получение реальных данных)
    state = get_current_market_state()  # реализуй функцию получения состояния

    action = trade_once(state)

    # Здесь ты можешь сделать реальный ордер через API биржи

    return f"Торговое действие: {action}"

   

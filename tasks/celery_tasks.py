from celery import Celery
import time

import math

from dqn import train_model
from model.dqn_model.gym.crypto_trading_env import CryptoTradingEnv
import json

from utils.cctx import fetch_ohlcv  # Импортируем функцию загрузки данных

# Настраиваем Celery с Redis как брокером и бекендом
celery = Celery(
    "tasks",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0"
)

@celery.task(bind=True)
def search_lstm_task(self, query):
    """Фоновая задача, которая выполняется долго"""
    self.update_state(state="IN_PROGRESS", meta={"progress": 0})

    for i in range(5):  # Имитация долгого вычисления
        time.sleep(2)
        self.update_state(state="IN_PROGRESS", meta={"progress": (i + 1) * 20})

    return {"message": "Task completed!", "query": query}

@celery.task(bind=True)
def train_dqn(self):

    self.update_state(state="IN_PROGRESS", meta={"progress": 0})
    
    symbol = 'BTC/USDT'
    exchange_name = 'bybit'         
    
    episode_length = 1000
    lookback_window = 20
    
    total_steps_needed = episode_length + lookback_window + 50  # запас 50
    
    # Для 15min и 1h считаем сколько свечей нужно, округляем вверх
    total_15min = math.ceil(total_steps_needed / 3) + 10  # запас 10 свечей
    total_1h = math.ceil(total_steps_needed / 12) + 5     # запас 5 свечей
    
    df = {}
    df['df_5min'] = fetch_ohlcv(exchange_name, symbol, '5m', limit=total_steps_needed)
    df['df_15min'] = fetch_ohlcv(exchange_name, symbol, '15m', limit=total_15min)
    df['df_1h'] = fetch_ohlcv(exchange_name, symbol, '1h', limit=total_1h)

    # Выводим первые 3 значения каждого df в формате JSON
    for key, value in df.items():
        records = value[:3].copy()
        if 'timestamp' in records.columns:
            records['timestamp'] = records['timestamp'].astype(str)
        else:
            for col in records.columns:
                if records[col].dtype.name == 'datetime64[ns]':
                    records[col] = records[col].astype(str)
        print(f"{key}: {json.dumps(records.to_dict(orient='records'), ensure_ascii=False, indent=2)}")
    result = train_model(dfs=df)

    return {"message": result}

@celery.task(bind=True)
def trade_step():
    # Получаем текущее состояние рынка (замени на получение реальных данных)
    state = get_current_market_state()  # реализуй функцию получения состояния

    action = trade_once(state)

    # Здесь ты можешь сделать реальный ордер через API биржи

    return f"Торговое действие: {action}"

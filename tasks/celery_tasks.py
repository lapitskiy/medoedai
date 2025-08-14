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
           
    df = {}
    
    df['df_5min'] = db_get_or_fetch_ohlcv(symbol_name='BTCUSDT', timeframe='5m', limit_candles=100000)

    df_5min = df['df_5min']
    
    if df_5min is not None and not df_5min.empty:
        print(f"Загружено свечей: {len(df_5min)}")
    else:
        print("Данные не загружены или пусты")

    df_5min['datetime'] = pd.to_datetime(df_5min['timestamp'], unit='ms')

    df_5min.set_index('datetime', inplace=True)

    df_15min = df_5min.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna().reset_index()

    df_1h = df_5min.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna().reset_index()

    df['df_15min'] = df_15min
    df['df_1h'] = df_1h

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
    result = train_model_optimized(dfs=df, episodes=100)
    return {"message": result}

@celery.task(bind=True)
def trade_step():
    # Получаем текущее состояние рынка (замени на получение реальных данных)
    state = get_current_market_state()  # реализуй функцию получения состояния

    action = trade_once(state)

    # Здесь ты можешь сделать реальный ордер через API биржи

    return f"Торговое действие: {action}"

   

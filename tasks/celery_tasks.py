from celery import Celery
import time

from dqn import train_model
from model.dqn_model.gym.crypto_trading_env import CryptoTradingEnv
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
    df['df_5min'] = db_get_or_fetch_ohlcv('BTC/USDT', '5m', 10000)
    df['df_15min'] = db_get_or_fetch_ohlcv('BTC/USDT', '15m', 10000)
    df['df_1h'] = db_get_or_fetch_ohlcv('BTC/USDT', '1h', 10000)

    # Выводим первые значения каждого df в формате JSON
    for key, value in df.items():
        records = value[:1].copy()
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

   

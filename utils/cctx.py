import ccxt
import pandas as pd
from datetime import datetime

def fetch_ohlcv(exchange_name, symbol, timeframe, limit=100):
    """
    Загружает исторические свечи OHLCV с биржи через ccxt.

    :param exchange_name: str, имя биржи (например, 'bybit')
    :param symbol: str, торговая пара (например, 'BTC/USDT')
    :param timeframe: str, таймфрейм (например, '5m', '15m', '1h')
    :param limit: int, сколько свечей загрузить (по умолчанию 100)
    :return: pandas.DataFrame с колонками ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    """
    # Создаём объект биржи
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({
        'enableRateLimit': True,
    })

    # Загружаем свечи
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    # Преобразуем в DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Конвертируем timestamp в читаемый формат
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    return df

# Пример вызова
if __name__ == "__main__":
    df_5min = fetch_ohlcv('bybit', 'BTC/USDT', '5m', limit=12)  # последние 12 свечей по 5 минутам (~1 час)
    print(df_5min)
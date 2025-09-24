import pandas as pd
from utils.db_utils import db_get_or_fetch_ohlcv
from agents.vdqn.v_train_model_optimized import train_model_optimized

def _build_dfs_for_symbol(symbol: str, limit: int = 100000):
    df_5min = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe='5m', limit_candles=limit)
    if df_5min is None or df_5min.empty:
        return None
    df_5min['datetime'] = pd.to_datetime(df_5min['timestamp'], unit='ms')
    df_5min.set_index('datetime', inplace=True)
    df_15min = df_5min.resample('15min').agg({
        'open': 'first','high': 'max','low': 'min','close': 'last','volume': 'sum',
    }).dropna().reset_index()
    df_1h = df_5min.resample('1h').agg({
        'open': 'first','high': 'max','low': 'min','close': 'last','volume': 'sum',
    }).dropna().reset_index()
    return {
        'df_5min': df_5min,
        'df_15min': df_15min,
        'df_1h': df_1h,
        'symbol': symbol,
        'candle_count': len(df_5min)
    }

def train_multi(symbols: list[str], episodes: int = 10001, episode_length: int = 2000):
    all_dfs = {}
    for s in symbols:
        dfs = _build_dfs_for_symbol(s)
        if dfs is not None:
            all_dfs[s] = dfs
    if not all_dfs:
        return "Нет данных ни для одного символа"
    # Передаём сразу словарь all_dfs как в оптимизированной функции
    return train_model_optimized(dfs=all_dfs, episodes=episodes, use_wandb=False, episode_length=episode_length)



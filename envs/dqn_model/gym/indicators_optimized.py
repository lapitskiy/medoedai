import logging
import numpy as np
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

def calculate_rsi(prices: np.ndarray, length: int = 14) -> np.ndarray:
    """
    Рассчитывает RSI без использования pandas
    """
    if len(prices) < length + 1:
        return np.full(len(prices), np.nan)
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Первые значения
    avg_gain = np.mean(gains[:length])
    avg_loss = np.mean(losses[:length])
    
    rsi = np.full(len(prices), np.nan)
    rsi[length] = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss != 0 else 100
    
    # Остальные значения
    for i in range(length + 1, len(prices)):
        avg_gain = (avg_gain * (length - 1) + gains[i-1]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i-1]) / length
        rsi[i] = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss != 0 else 100
    
    return rsi

def calculate_ema(prices: np.ndarray, length: int) -> np.ndarray:
    """
    Рассчитывает EMA без использования pandas
    """
    if len(prices) < length:
        return np.full(len(prices), np.nan)
    
    alpha = 2.0 / (length + 1)
    ema = np.full(len(prices), np.nan)
    
    # Первое значение - SMA
    ema[length-1] = np.mean(prices[:length])
    
    # Остальные значения
    for i in range(length, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    
    return ema

def calculate_sma(prices: np.ndarray, length: int) -> np.ndarray:
    """
    Рассчитывает SMA без использования pandas
    """
    if len(prices) < length:
        return np.full(len(prices), np.nan)
    
    sma = np.full(len(prices), np.nan)
    
    for i in range(length - 1, len(prices)):
        sma[i] = np.mean(prices[i-length+1:i+1])
    
    return sma

def calculate_ema_cross_features(short_ema: np.ndarray, long_ema: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Рассчитывает признаки пересечения EMA
    """
    # Короткая EMA выше длинной EMA
    above = np.where(short_ema > long_ema, 1.0, 0.0)
    
    # Пересечение снизу вверх (bullish crossover)
    cross_up = np.zeros_like(above)
    for i in range(1, len(above)):
        if above[i] == 1.0 and above[i-1] == 0.0:
            cross_up[i] = 1.0
    
    # Пересечение сверху вниз (bearish crossover)
    cross_down = np.zeros_like(above)
    for i in range(1, len(above)):
        if above[i] == 0.0 and above[i-1] == 1.0:
            cross_down[i] = 1.0
    
    return above, cross_up, cross_down

def precalculate_all_indicators(
    df_5min: np.ndarray, 
    indicators_config: Dict
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Предварительно рассчитывает все индикаторы и возвращает numpy массивы
    
    Args:
        df_5min: numpy array с OHLCV данными (без timestamp)
        indicators_config: конфигурация индикаторов
    
    Returns:
        Tuple[np.ndarray, Dict]: основной массив индикаторов и словарь с отдельными индикаторами
    """
    close_prices = df_5min[:, 3]  # Close prices
    features = []
    feature_names = []
    individual_indicators = {}
    
    # RSI
    if 'rsi' in indicators_config:
        length_rsi = indicators_config['rsi'].get('length', 14)
        rsi_values = calculate_rsi(close_prices, length_rsi)
        features.append(rsi_values)
        feature_names.append(f'RSI_{length_rsi}')
        individual_indicators[f'RSI_{length_rsi}'] = rsi_values
    
    # EMA
    if 'ema' in indicators_config and 'lengths' in indicators_config['ema']:
        for length in indicators_config['ema']['lengths']:
            ema_values = calculate_ema(close_prices, length)
            features.append(ema_values)
            feature_names.append(f'EMA_{length}')
            individual_indicators[f'EMA_{length}'] = ema_values
    
    # SMA
    if 'sma' in indicators_config:
        length_sma = indicators_config['sma'].get('length', 14)
        sma_values = calculate_sma(close_prices, length_sma)
        features.append(sma_values)
        feature_names.append(f'SMA_{length_sma}')
        individual_indicators[f'SMA_{length_sma}'] = sma_values
    
    # EMA Cross features
    if 'ema_cross' in indicators_config and 'pairs' in indicators_config['ema_cross']:
        for short_len, long_len in indicators_config['ema_cross']['pairs']:
            short_ema_key = f'EMA_{short_len}'
            long_ema_key = f'EMA_{long_len}'
            
            if short_ema_key in individual_indicators and long_ema_key in individual_indicators:
                short_ema = individual_indicators[short_ema_key]
                long_ema = individual_indicators[long_ema_key]
                
                above, cross_up, cross_down = calculate_ema_cross_features(short_ema, long_ema)
                
                features.extend([above, cross_up, cross_down])
                feature_names.extend([
                    f'EMA_{short_len}_above_{long_len}',
                    f'EMA_{short_len}_cross_up_{long_len}',
                    f'EMA_{short_len}_cross_down_{long_len}'
                ])
                
                individual_indicators[f'EMA_{short_len}_above_{long_len}'] = above
                individual_indicators[f'EMA_{short_len}_cross_up_{long_len}'] = cross_up
                individual_indicators[f'EMA_{short_len}_cross_down_{long_len}'] = cross_down
    
    # Объединяем все признаки в один массив
    if features:
        indicators_array = np.column_stack(features)
        # Заполняем NaN нулями (warmup-строки EMA-200 и т.д.)
        nan_count = int(np.isnan(indicators_array).sum())
        if nan_count > 0:
            logger.debug("indicators: %d NaN values replaced with 0 (warmup rows)", nan_count)
        indicators_array = np.nan_to_num(indicators_array, nan=0.0).astype(np.float32)
    else:
        indicators_array = np.zeros((len(close_prices), 0), dtype=np.float32)
    
    return indicators_array, individual_indicators

def preprocess_dataframes(
    df_5min: np.ndarray,
    df_15min: np.ndarray, 
    df_1h: np.ndarray,
    indicators_config: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Предварительно обрабатывает все DataFrame и рассчитывает индикаторы
    
    Args:
        df_5min, df_15min, df_1h: numpy arrays с OHLCV данными
        indicators_config: конфигурация индикаторов
    
    Returns:
        Tuple: обработанные данные и индикаторы
    """
    # Убираем timestamp колонку если есть (предполагаем, что она первая)
    # Проверяем, есть ли timestamp колонка по типу данных
    if df_5min.dtype.names is not None or df_5min.shape[1] > 5:
        # Если есть timestamp, убираем его
        df_5min_clean = df_5min[:, 1:6] if df_5min.shape[1] > 5 else df_5min
        df_15min_clean = df_15min[:, 1:6] if df_15min.shape[1] > 5 else df_15min
        df_1h_clean = df_1h[:, 1:6] if df_1h.shape[1] > 5 else df_1h
    else:
        df_5min_clean = df_5min
        df_15min_clean = df_15min
        df_1h_clean = df_1h
    
    # Рассчитываем индикаторы только для 5-минутных данных
    indicators_array, individual_indicators = precalculate_all_indicators(df_5min_clean, indicators_config)
    
    return df_5min_clean, df_15min_clean, df_1h_clean, indicators_array, individual_indicators

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

def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> np.ndarray:
    """ATR (Average True Range) — волатильность"""
    n = len(close)
    if n < length + 1:
        return np.full(n, np.nan)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.full(n, np.nan)
    atr[length] = np.mean(tr[1:length+1])
    for i in range(length + 1, n):
        atr[i] = (atr[i-1] * (length - 1) + tr[i]) / length
    return atr


def calculate_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On-Balance Volume"""
    n = len(close)
    obv = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    return obv


def calculate_returns(close: np.ndarray, periods: list) -> dict:
    """Скользящие доходности за periods баров"""
    n = len(close)
    result = {}
    for p in periods:
        ret = np.full(n, np.nan)
        for i in range(p, n):
            if close[i - p] != 0:
                ret[i] = (close[i] - close[i - p]) / close[i - p]
        result[p] = ret
    return result


def calculate_zscore(close: np.ndarray, ema: np.ndarray, window: int = 20) -> np.ndarray:
    """Z-score цены относительно EMA: (close - ema) / rolling_std(close - ema)"""
    n = len(close)
    diff = close - ema
    zs = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = diff[i - window + 1:i + 1]
        std = np.std(seg)
        zs[i] = diff[i] / std if std > 1e-12 else 0.0
    return zs


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
    high_prices = df_5min[:, 1]   # High
    low_prices = df_5min[:, 2]    # Low
    volume = df_5min[:, 4]        # Volume
    features = []
    feature_names = []
    individual_indicators = {}
    
    # RSI (все ключи вида rsi, rsi_7, rsi_21 и т.д.)
    for key, cfg in indicators_config.items():
        if key == 'rsi' or key.startswith('rsi_'):
            length_rsi = cfg.get('length', 14)
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
    
    # ATR
    if 'atr' in indicators_config:
        length_atr = indicators_config['atr'].get('length', 14)
        atr_values = calculate_atr(high_prices, low_prices, close_prices, length_atr)
        features.append(atr_values)
        feature_names.append(f'ATR_{length_atr}')
        individual_indicators[f'ATR_{length_atr}'] = atr_values
    
    # OBV
    if 'obv' in indicators_config:
        obv_values = calculate_obv(close_prices, volume)
        features.append(obv_values)
        feature_names.append('OBV')
        individual_indicators['OBV'] = obv_values
    
    # Returns
    if 'returns' in indicators_config:
        periods = indicators_config['returns'].get('periods', [1, 3, 12])
        rets = calculate_returns(close_prices, periods)
        for p, r in rets.items():
            features.append(r)
            feature_names.append(f'RET_{p}')
            individual_indicators[f'RET_{p}'] = r
    
    # Z-score
    if 'zscore' in indicators_config:
        ema_len = indicators_config['zscore'].get('ema_length', 50)
        window = indicators_config['zscore'].get('window', 20)
        ema_key = f'EMA_{ema_len}'
        if ema_key in individual_indicators:
            zs = calculate_zscore(close_prices, individual_indicators[ema_key], window)
        else:
            ema_for_zs = calculate_ema(close_prices, ema_len)
            zs = calculate_zscore(close_prices, ema_for_zs, window)
        features.append(zs)
        feature_names.append(f'ZSCORE_{ema_len}_{window}')
        individual_indicators[f'ZSCORE_{ema_len}_{window}'] = zs
    
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

    # === ATR 1H normalized (for ATR-based stops in env) ===
    # Env ожидает ключ 'ATR1H_21_NORM' длиной как df_5min (5m), апсемпленный из 1h.
    # atr_norm = atr_abs_1h / close_1h  -> далее env делает atr_abs = atr_norm * current_price_5m.
    try:
        if df_1h_clean is not None and hasattr(df_1h_clean, "shape") and df_1h_clean.shape[0] >= 25 and df_5min_clean is not None:
            high_1h = df_1h_clean[:, 1].astype(np.float64)
            low_1h = df_1h_clean[:, 2].astype(np.float64)
            close_1h = df_1h_clean[:, 3].astype(np.float64)
            atr_1h_abs = calculate_atr(high_1h, low_1h, close_1h, length=21)
            atr_1h_abs = np.nan_to_num(atr_1h_abs, nan=0.0)
            atr_1h_norm = (atr_1h_abs / np.maximum(close_1h, 1e-12)).astype(np.float32)

            # Upsample 1h -> 5m by repeating each 1h value for 12 bars
            T5 = int(df_5min_clean.shape[0])
            T1 = int(df_1h_clean.shape[0])
            atr_1h_norm_5m = np.zeros(T5, dtype=np.float32)
            for i in range(T5):
                idx_1h = i // 12
                if idx_1h < 0:
                    idx_1h = 0
                elif idx_1h >= T1:
                    idx_1h = T1 - 1
                atr_1h_norm_5m[i] = atr_1h_norm[idx_1h]
            individual_indicators["ATR1H_21_NORM"] = atr_1h_norm_5m
    except Exception:
        # Если что-то пошло не так — просто не добавляем ключ (env корректно фоллбечит на фиксированный thr)
        pass
    
    return df_5min_clean, df_15min_clean, df_1h_clean, indicators_array, individual_indicators

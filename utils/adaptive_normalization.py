import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AdaptiveNormalizer:
    """
    Адаптивная нормализация для разных криптовалют
    Учитывает волатильность, объем и рыночные условия
    """
    
    def __init__(self):
        # Характеристики разных криптовалют
        self.crypto_profiles = {
            'BTCUSDT': {
                'volatility_multiplier': 1.0,      # Базовая волатильность
                'volume_threshold': 0.002,         # Порог объема
                'price_sensitivity': 1.0,          # Чувствительность к цене
                'trend_strength': 1.0,             # Сила тренда
                'min_hold_time': 24,               # Минимальное время удержания (шаги)
                'stop_loss': -0.05,                # Stop Loss
                'take_profit': 0.08,               # Take Profit
            },
            'ETHUSDT': {
                'volatility_multiplier': 1.2,      # ETH более волатилен
                'volume_threshold': 0.0015,        # Меньший порог объема
                'price_sensitivity': 1.1,          # Более чувствителен
                'trend_strength': 1.2,             # Сильнее тренды
                'min_hold_time': 20,               # Меньше времени удержания
                'stop_loss': -0.06,                # Больше терпения
                'take_profit': 0.10,               # Больше прибыли
            },
            'TONUSDT': {
                'volatility_multiplier': 0.8,      # TON менее волатилен
                'volume_threshold': 0.003,         # Больший порог объема
                'price_sensitivity': 0.9,          # Менее чувствителен
                'trend_strength': 0.8,             # Слабые тренды
                'min_hold_time': 30,               # Больше времени удержания
                'stop_loss': -0.04,                # Меньше терпения
                'take_profit': 0.06,               # Меньше прибыли
            },
            'SOLUSDT': {
                'volatility_multiplier': 1.5,      # SOL очень волатилен
                'volume_threshold': 0.001,         # Очень низкий порог
                'price_sensitivity': 1.3,          # Очень чувствителен
                'trend_strength': 1.5,             # Очень сильные тренды
                'min_hold_time': 16,               # Очень быстрое реагирование
                'stop_loss': -0.07,                # Много терпения
                'take_profit': 0.12,               # Много прибыли
            },
            'ADAUSDT': {
                'volatility_multiplier': 0.9,      # ADA умеренно волатилен
                'volume_threshold': 0.0025,        # Средний порог
                'price_sensitivity': 0.95,         # Средняя чувствительность
                'trend_strength': 0.9,             # Средние тренды
                'min_hold_time': 26,               # Среднее время
                'stop_loss': -0.045,               # Среднее терпение
                'take_profit': 0.07,               # Средняя прибыль
            }
        }
        
        # Динамические параметры
        self.dynamic_params = {}
        
    def get_crypto_profile(self, symbol: str) -> Dict:
        """Получает профиль криптовалюты"""
        # Убираем временные метки и получаем базовый символ
        base_symbol = symbol.split('_')[0] if '_' in symbol else symbol
        
        if base_symbol in self.crypto_profiles:
            return self.crypto_profiles[base_symbol].copy()
        else:
            # Дефолтный профиль для неизвестных криптовалют
            logger.warning(f"Неизвестная криптовалюта: {symbol}, используем дефолтный профиль")
            return {
                'volatility_multiplier': 1.0,
                'volume_threshold': 0.002,
                'price_sensitivity': 1.0,
                'trend_strength': 1.0,
                'min_hold_time': 24,
                'stop_loss': -0.05,
                'take_profit': 0.08,
            }
    
    def analyze_market_conditions(self, df: pd.DataFrame) -> Dict:
        """
        Анализирует рыночные условия для адаптации параметров
        """
        try:
            # Волатильность (стандартное отклонение returns)
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Объем (средний объем за последние N баров)
            volume_ma = df['volume'].rolling(20).mean()
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / volume_ma.iloc[-1] if not pd.isna(volume_ma.iloc[-1]) else 1.0
            
            # Тренд (наклон линии тренда)
            price_trend = np.polyfit(range(len(df)), df['close'], 1)[0]
            trend_strength = abs(price_trend) / df['close'].mean()
            
            # Рыночные условия
            market_conditions = {
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'trend_strength': trend_strength,
                'is_high_volatility': volatility > 0.02,  # >2% в день
                'is_high_volume': volume_ratio > 1.5,     # >150% от среднего
                'is_strong_trend': trend_strength > 0.001, # Сильный тренд
            }
            
            return market_conditions
            
        except Exception as e:
            logger.error(f"Ошибка анализа рыночных условий: {e}")
            return {
                'volatility': 0.02,
                'volume_ratio': 1.0,
                'trend_strength': 0.001,
                'is_high_volatility': False,
                'is_high_volume': False,
                'is_strong_trend': False,
            }
    
    def adapt_parameters(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        Адаптирует параметры под конкретную криптовалюту и рыночные условия
        """
        # Базовый профиль
        profile = self.get_crypto_profile(symbol)
        
        # Анализ рыночных условий
        market_conditions = self.analyze_market_conditions(df)
        
        # Адаптация параметров
        adapted_params = profile.copy()
        
        # 1. Адаптация под волатильность
        if market_conditions['is_high_volatility']:
            adapted_params['volatility_multiplier'] *= 1.2
            adapted_params['stop_loss'] *= 1.2  # Больше терпения
            adapted_params['take_profit'] *= 1.2  # Больше прибыли
            adapted_params['min_hold_time'] = max(16, adapted_params['min_hold_time'] - 4)
        
        # 2. Адаптация под объем
        if market_conditions['is_high_volume']:
            adapted_params['volume_threshold'] *= 0.8  # Снижаем порог
            adapted_params['price_sensitivity'] *= 1.1  # Повышаем чувствительность
        
        # 3. Адаптация под тренд
        if market_conditions['is_strong_trend']:
            adapted_params['trend_strength'] *= 1.3
            adapted_params['min_hold_time'] = max(12, adapted_params['min_hold_time'] - 8)
            adapted_params['take_profit'] *= 1.1  # Больше прибыли в тренде
        
        # 4. Специфичные адаптации для разных криптовалют
        if 'BTC' in symbol:
            # BTC: более консервативно в нестабильные периоды
            if market_conditions['volatility'] > 0.03:
                adapted_params['min_hold_time'] += 8
                adapted_params['stop_loss'] *= 0.9
        elif 'ETH' in symbol:
            # ETH: более агрессивно в тренде
            if market_conditions['is_strong_trend']:
                adapted_params['take_profit'] *= 1.2
        elif 'TON' in symbol:
            # TON: стабильная монета, меньше адаптации
            adapted_params['volatility_multiplier'] *= 0.9
        
        # Ограничиваем значения
        adapted_params['min_hold_time'] = max(12, min(48, adapted_params['min_hold_time']))
        adapted_params['stop_loss'] = max(-0.10, min(-0.02, adapted_params['stop_loss']))
        adapted_params['take_profit'] = max(0.04, min(0.20, adapted_params['take_profit']))
        
        return adapted_params
    
    def normalize_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Нормализует признаки с учетом профиля криптовалюты
        """
        try:
            # Получаем адаптированные параметры
            params = self.adapt_parameters(symbol, df)
            
            # Копируем DataFrame
            normalized_df = df.copy()
            
            # 1. Нормализация цены с учетом волатильности
            volatility_mult = params['volatility_multiplier']
            normalized_df['price_normalized'] = df['close'].pct_change() * volatility_mult
            
            # 2. Нормализация объема с учетом порога
            volume_threshold = params['volume_threshold']
            normalized_df['volume_normalized'] = np.where(
                df['volume'] > df['volume'].rolling(20).mean() * volume_threshold,
                1.0,  # Высокий объем
                0.5   # Низкий объем
            )
            
            # 3. Нормализация тренда с учетом силы
            trend_strength = params['trend_strength']
            price_ma_short = df['close'].rolling(5).mean()
            price_ma_long = df['close'].rolling(20).mean()
            normalized_df['trend_normalized'] = (
                (price_ma_short - price_ma_long) / price_ma_long * trend_strength
            )
            
            # 4. Адаптивная нормализация технических индикаторов
            if 'rsi' in df.columns:
                # RSI: более чувствительный для волатильных монет
                sensitivity = params['price_sensitivity']
                normalized_df['rsi_normalized'] = (df['rsi'] - 50) / 50 * sensitivity
            
            if 'macd' in df.columns:
                # MACD: учитываем силу тренда
                trend_mult = params['trend_strength']
                normalized_df['macd_normalized'] = df['macd'] * trend_mult
            
            # 5. Добавляем адаптивные параметры в DataFrame
            for key, value in params.items():
                normalized_df[f'param_{key}'] = value
            
            logger.info(f"Адаптивная нормализация для {symbol}: volatility_mult={params['volatility_multiplier']:.2f}, "
                       f"min_hold={params['min_hold_time']}, stop_loss={params['stop_loss']:.3f}")
            
            return normalized_df
            
        except Exception as e:
            logger.error(f"Ошибка нормализации для {symbol}: {e}")
            return df
    
    def get_trading_params(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        Возвращает адаптированные торговые параметры
        """
        params = self.adapt_parameters(symbol, df)
        
        return {
            'min_hold_steps': int(params['min_hold_time']),
            'stop_loss_pct': params['stop_loss'],
            'take_profit_pct': params['take_profit'],
            'volume_threshold': params['volume_threshold'],
            'volatility_multiplier': params['volatility_multiplier'],
            'price_sensitivity': params['price_sensitivity'],
            'trend_strength': params['trend_strength'],
        }

# Глобальный экземпляр
adaptive_normalizer = AdaptiveNormalizer()

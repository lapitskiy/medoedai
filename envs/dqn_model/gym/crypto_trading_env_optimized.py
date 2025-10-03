from envs.dqn_model.gym.gutils_optimized import calc_relative_vol_numpy, commission_penalty, update_roi_stats, update_vol_stats
from envs.dqn_model.gym.indicators_optimized import preprocess_dataframes
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import torch
import time
import pandas as pd  # ИСПРАВЛЕНИЕ: Добавляем pandas для datetime операций
from sklearn.preprocessing import StandardScaler
from envs.dqn_model.gym.gconfig import GymConfig
from typing import Optional, Dict
from collections import deque

# Импортируем адаптивную нормализацию
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'utils'))
try:
    from adaptive_normalization import adaptive_normalizer
    ADAPTIVE_NORMALIZATION_AVAILABLE = True
except ImportError:
    print("⚠️ Адаптивная нормализация недоступна, используем стандартные параметры")
    ADAPTIVE_NORMALIZATION_AVAILABLE = False

class CryptoTradingEnvOptimized(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, dfs: Dict, cfg: Optional[GymConfig] = None, lookback_window: int = 20, indicators_config=None, episode_length: Optional[int] = None, normalization_stats: Optional[Dict] = None):        
        super(CryptoTradingEnvOptimized, self).__init__() 
        self.cfg = cfg or GymConfig()
        
        # Устанавливаем длину эпизода
        self.episode_length = episode_length or getattr(self.cfg, 'episode_length', 10000) # Fallback на 10000 шагов
        if self.episode_length < 100:
            print(f"⚠️ Длина эпизода {self.episode_length} слишком мала, устанавливаю 10000")
            self.episode_length = 10000
        
        self.vol_scaled = 0
        self.epsilon = 1.0
        
        # PRINT LOG DOCKER
        self._episode_idx = -1
        self._log_interval = 20
        self._can_log = False

        # фильтры «душат» покупки?
        self.buy_attempts = 0
        self.buy_rejected_vol = 0
        self.buy_rejected_roi = 0
        
        # Получаем размер окна с значением по умолчанию
        window_size = getattr(self.cfg, 'window288', 288)  # По умолчанию 288 (24 часа * 12 пятиминуток)
        self.vol_buf = deque(maxlen=window_size)
        self.roi_buf = deque(maxlen=window_size)
        
        # ДИНАМИЧЕСКИЙ размер позиции на основе рыночных условий
        self.base_position_fraction = 0.30  # Базовый размер позиции
        self.position_fraction = self.base_position_fraction  # Текущий размер позиции
        self.position_confidence_threshold = 0.7  # Порог уверенности для увеличения позиции

        # Константы окружения
        self.trade_fee_percent = 0.00055 # Комиссия 0.055%
        
        # Адаптивные параметры риск-менеджмента
        # Корректно извлекаем символ из dfs (dict) или объекта
        if isinstance(dfs, dict):
            self.symbol = dfs.get('symbol', 'BTCUSDT')
        else:
            self.symbol = getattr(dfs, 'symbol', 'BTCUSDT')
        
        if ADAPTIVE_NORMALIZATION_AVAILABLE:
            # Получаем адаптивные параметры для конкретной криптовалюты
            trading_params = adaptive_normalizer.get_trading_params(self.symbol, dfs['df_5min'])
            self.STOP_LOSS_PCT = trading_params['stop_loss_pct']
            self.TAKE_PROFIT_PCT = trading_params['take_profit_pct']
            self.min_hold_steps = trading_params['min_hold_steps']
            self.volume_threshold = trading_params['volume_threshold']
            print(f"🔧 Адаптивные параметры для {self.symbol}:")
            print(f"  • Stop Loss: {self.STOP_LOSS_PCT:.3f}")
            print(f"  • Take Profit: {self.TAKE_PROFIT_PCT:.3f}")
            print(f"  • Min Hold: {self.min_hold_steps} шагов")
            print(f"  • Volume Threshold: {self.volume_threshold:.4f}")
        else:
            # УЛУЧШЕНО: Динамические параметры риск-менеджмента
            self.base_stop_loss = -0.03      # Базовый stop-loss
            self.base_take_profit = +0.06    # Базовый take-profit
            self.base_min_hold = 8           # Базовое минимальное время удержания
            self.volume_threshold = 0.0001   # Базовый порог объема
            
            # Динамические параметры (будут обновляться во время торговли)
            self.STOP_LOSS_PCT = self.base_stop_loss
            self.TAKE_PROFIT_PCT = self.base_take_profit
            self.min_hold_steps = self.base_min_hold
            
            print(f"🔧 ДИНАМИЧЕСКИЕ параметры для {self.symbol}:")
            print(f"  • Базовый Stop Loss: {self.base_stop_loss:.3f}")
            print(f"  • Базовый Take Profit: {self.base_take_profit:.3f}")
            print(f"  • Базовый Min Hold: {self.base_min_hold} шагов")
            print(f"  • Volume Threshold: {self.volume_threshold:.4f}")
            print(f"  • Параметры будут адаптироваться к рыночным условиям")
        
        # Multi-step Learning параметры
        self.n_step = getattr(self.cfg, 'n_step', 3)  # Количество шагов для n-step learning
        self.n_step_buffer = deque(maxlen=self.n_step)  # Буфер для n-step transitions
        self.gamma = getattr(self.cfg, 'gamma', 0.99)  # Discount factor
        
        # Конвертируем pandas DataFrames в numpy arrays и предварительно обрабатываем
        df_5min_raw = dfs['df_5min'].values if hasattr(dfs['df_5min'], 'values') else dfs['df_5min']
        df_15min_raw = dfs['df_15min'].values if hasattr(dfs['df_15min'], 'values') else dfs['df_15min']
        df_1h_raw = dfs['df_1h'].values if hasattr(dfs['df_1h'], 'values') else dfs['df_1h']
        
        if indicators_config is None:
            self.indicators_config = {
                'rsi': {'length': 14},
                'ema': {'lengths': [100, 200]},
                'ema_cross': {
                    'pairs': [(100, 200)],
                    'include_cross_signal': True
                },
                'sma': {'length': 14},
            }
        else:
            self.indicators_config = indicators_config
        
        # Предварительно обрабатываем все данные
        (self.df_5min, self.df_15min, self.df_1h, 
         self.indicators, self.individual_indicators) = preprocess_dataframes(
            df_5min_raw, df_15min_raw, df_1h_raw, self.indicators_config
        )
        
        # Добавляем funding-фичи как дополнительные индикаторы, если есть в DataFrame
        try:
            df5_src = dfs['df_5min'] if isinstance(dfs, dict) else None
            funding_cols = ['funding_rate_bp', 'funding_rate_ema', 'funding_rate_change', 'funding_sign']
            present_cols = [c for c in funding_cols if (df5_src is not None and hasattr(df5_src, 'columns') and c in df5_src.columns)]
            if present_cols and len(self.df_5min) == len(df5_src):
                F = []
                for c in present_cols:
                    v = df5_src[c].astype(float).values
                    # Приводим к безопасному диапазону и типу
                    if c in ('funding_rate_ema', 'funding_rate_change'):
                        v = v * 10000.0  # в бипсы
                    # клип и нормализация в [-1,1] на 50 bp
                    v = np.clip(v / 50.0, -1.0, 1.0).astype(np.float32)
                    F.append(v.reshape(-1, 1))
                if F:
                    F_arr = np.concatenate(F, axis=1).astype(np.float32)
                    if self.indicators.size > 0:
                        self.indicators = np.concatenate([self.indicators, F_arr], axis=1).astype(np.float32)
                    else:
                        self.indicators = F_arr
                    print(f"🧠 Funding признаки добавлены: {present_cols} (нормализованы)")
        except Exception as e:
            print(f"⚠️ Не удалось добавить funding признаки: {e}")
        
        # РЕЖИМ РЫНКА: добавляем per-window непрерывные фичи (drift, vol, slope, r2) для каждого окна
        try:
            def _compute_regime_metrics_per_window(closes: np.ndarray, windows=(60,180,300)) -> np.ndarray:
                T = closes.shape[0]
                if T < 10:
                    return np.zeros((T, 4 * len(windows)), dtype=np.float32)
                y = closes.astype(np.float64)
                # предвычислим returns
                r = np.zeros_like(y)
                r[1:] = (y[1:] - y[:-1]) / np.maximum(y[:-1], 1e-12)
                all_feats = []
                for w in windows:
                    if w < 2 or T <= w:
                        all_feats.append(np.zeros((T, 4), dtype=np.float32))
                        continue
                    N = float(w)
                    # Суммы по x (0..w-1)
                    sum_x = N*(N-1.0)/2.0
                    sum_x2 = (N*(N-1.0)*(2.0*N-1.0))/6.0
                    xbar = sum_x / N
                    S_xx = sum_x2 - N*(xbar**2)
                    # Скольжения по y
                    ones_w = np.ones(w, dtype=np.float64)
                    sum_y = np.convolve(y, ones_w, mode='valid')  # len T-w+1
                    sum_y2 = np.convolve(y*y, ones_w, mode='valid')
                    # sum(x*y) со свёрткой ядра x[::-1]
                    kernel = np.arange(w, dtype=np.float64)[::-1]
                    sum_xy = np.convolve(y, kernel, mode='valid')
                    # Drift: (y_t - y_{t-w})/y_{t-w}
                    drift = (y[w-1:] - y[:-w+1]) / np.maximum(y[:-w+1], 1e-12)
                    # Volatility: std returns за (w-1)
                    try:
                        win_r = np.lib.stride_tricks.sliding_window_view(r[1:], w-1)
                        vol = win_r.std(axis=1)
                    except Exception:
                        vol = np.array([r[i-w+2:i+1].std() if i+1 >= (w-1) else 0.0 for i in range(T-1)], dtype=np.float64)
                        vol = vol[w-2:]
                    # Регрессия: slope и R^2
                    S_xy = sum_xy - xbar*sum_y
                    slope = S_xy / np.maximum(S_xx, 1e-12)
                    ybar = sum_y / N
                    SST = sum_y2 - N*(ybar**2)
                    SSR = (slope**2) * S_xx
                    r2 = SSR / np.maximum(SST, 1e-12)
                    # Приводим к длине T: заполняем нулями первые w-1
                    zeros_head = np.zeros((w-1, 4), dtype=np.float32)
                    M = min(drift.shape[0], vol.shape[0], slope.shape[0], r2.shape[0])
                    tail = np.stack([
                        drift[:M].astype(np.float32),
                        vol[:M].astype(np.float32),
                        slope[:M].astype(np.float32),
                        np.clip(r2[:M], 0.0, 1.0).astype(np.float32)
                    ], axis=1)
                    feats_w = np.concatenate([zeros_head, tail], axis=0)
                    if feats_w.shape[0] < T:
                        pad = np.zeros((T - feats_w.shape[0], 4), dtype=np.float32)
                        feats_w = np.vstack([feats_w, pad])
                    all_feats.append(feats_w)
                if all_feats:
                    return np.concatenate(all_feats, axis=1).astype(np.float32)
                return np.zeros((T, 4 * len(windows)), dtype=np.float32)

            closes = self.df_5min[:, 3].astype(np.float32)
            regime_feats = _compute_regime_metrics_per_window(closes)
            if regime_feats is not None and regime_feats.shape[0] == self.df_5min.shape[0]:
                if self.indicators.size > 0:
                    self.indicators = np.concatenate([self.indicators, regime_feats], axis=1).astype(np.float32)
                else:
                    self.indicators = regime_feats.astype(np.float32)
                print("🧭 Regime per-window metrics added: for each window [drift, vol, slope, r2]")
        except Exception as e:
            print(f"⚠️ Не удалось добавить regime признаки: {e}")
        
        # ИСПРАВЛЕНИЕ: Создаем datetime информацию для фильтра времени
        try:
            if hasattr(dfs, 'df_5min') and hasattr(dfs['df_5min'], 'index'):
                # Если у нас есть pandas DataFrame с datetime индексом
                self._candle_datetimes = dfs['df_5min'].index.to_pydatetime()
            elif hasattr(dfs, 'df_5min') and hasattr(dfs['df_5min'], 'columns'):
                # Если у нас есть pandas DataFrame с колонками
                if 'datetime' in dfs['df_5min'].columns:
                    self._candle_datetimes = pd.to_datetime(dfs['df_5min']['datetime']).dt.to_pydatetime()
                elif 'timestamp' in dfs['df_5min'].columns:
                    self._candle_datetimes = pd.to_datetime(dfs['df_5min']['timestamp'], unit='ms').dt.to_pydatetime()
                else:
                    # Создаем фиктивные datetime для совместимости
                    self._candle_datetimes = [pd.Timestamp.now() + pd.Timedelta(minutes=i*5) for i in range(len(self.df_5min))]
            else:
                # Fallback: создаем фиктивные datetime
                self._candle_datetimes = [pd.Timestamp.now() + pd.Timedelta(minutes=i*5) for i in range(len(self.df_5min))]
        except Exception as e:
            print(f"⚠️ Не удалось создать datetime информацию: {e}")
            # Создаем фиктивные datetime для совместимости
            self._candle_datetimes = [pd.Timestamp.now() + pd.Timedelta(minutes=i*5) for i in range(len(self.df_5min))]
        
        self.total_steps = len(self.df_5min)
        self.lookback_window = lookback_window 
        
        self.action_space = spaces.Discrete(3) # 0: HOLD, 1: BUY, 2: SELL
        
        num_features_per_candle = 5 # Open, High, Low, Close, Volume        
        num_indicator_features = self.indicators.shape[1] if self.indicators.size > 0 else 0
        
        # Рассчитываем размер состояния для предвычисленных состояний
        # 5min OHLCV + 15min OHLCV + 1h OHLCV + индикаторы + баланс/криптовалюта
        total_features_per_step = (
            num_features_per_candle +  # 5min OHLCV
            num_features_per_candle +  # 15min OHLCV (интерполированный)
            num_features_per_candle +  # 1h OHLCV (интерполированный)
            (num_indicator_features if self.indicators.size > 0 else 0)  # индикаторы на шаг
        )
        
        # Упрощенный расчет для отладки
        print(f"🔍 Расчет размера состояния:")
        print(f"  • num_features_per_candle: {num_features_per_candle}")
        print(f"  • num_indicator_features: {num_indicator_features}")
        print(f"  • total_features_per_step: {total_features_per_step}")
        print(f"  • lookback_window: {self.lookback_window}")
        
        # Проверяем, что размеры корректны
        if total_features_per_step <= 0:
            print(f"⚠️ Ошибка: total_features_per_step = {total_features_per_step}")
            total_features_per_step = 15  # Минимальный размер
        
        self.observation_space_shape = (
            self.lookback_window * total_features_per_step +
            2  # normalized_balance и normalized_crypto_held
        )
        
        print(f"  • observation_space_shape: {self.observation_space_shape}")

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(self.observation_space_shape,), 
                                            dtype=np.float32)
        
        self.min_valid_start_step = self.lookback_window 
        self.current_train_start_idx = self.min_valid_start_step 
        
        # Эти будут инициализированы в reset
        self.start_step = None
        self.current_step = None
        self.balance = None
        self.crypto_held = None
        self.last_buy_price = None
        self.last_buy_step = None    
        self.trailing_stop_counter = 0
        self.max_price_during_hold = None
        
        # Предварительно рассчитываем/устанавливаем статистики нормализации
        if normalization_stats is not None:
            try:
                self._apply_normalization_stats(normalization_stats)
                print("✅ Приняты внешние статистики нормализации (train) — единый препроцессинг train/val/serving")
            except Exception as e:
                print(f"⚠️ Не удалось применить внешние статистики нормализации: {e}. Пересчитываю по train split")
                self._calculate_normalization_stats()
        else:
            self._calculate_normalization_stats()
        
        # Инициализируем скалеры для баланса и криптовалюты
        self.balance_scaler = StandardScaler()
        self.crypto_held_scaler = StandardScaler()
        
        self._precompute_all_states()
                
        # Получаем начальный баланс с значением по умолчанию
        initial_balance = getattr(self.cfg, 'initial_balance', 10000.0)  # По умолчанию 10000
        
        # Подготавливаем данные для скалеров
        max_balance = initial_balance * 10  # Предполагаем максимальный баланс
        max_crypto = initial_balance / 100  # Предполагаем максимальное количество криптовалюты
        
        balance_samples = np.array([[0], [initial_balance], [max_balance]])
        crypto_samples = np.array([[0], [max_crypto/2], [max_crypto]])
        
        self.balance_scaler.fit(balance_samples)
        self.crypto_held_scaler.fit(crypto_samples)
        
        # Список сделок
        self.trades = []
        
        # ИСПРАВЛЕНИЕ: Добавляем общий список сделок для правильного расчета winrate
        self.all_trades = []
        
        # Статистики для фильтров
        self.vol_stats = {'mean': 0, 'std': 1}
        self.roi_stats = {'mean': 0, 'std': 1}
        
        # Счетчик действий для статистики
        self.action_counts = {0: 0, 1: 0, 2: 0}
        
        # Время для отслеживания производительности
        self.episode_start_time = None
        self.episode_step_count = 0
        
        print(f"✅ Оптимизированное окружение инициализировано")
        print(f"📊 Размер данных: 5min={len(self.df_5min)}, 15min={len(self.df_15min)}, 1h={len(self.df_1h)}")
        print(f"📈 Количество индикаторов: {num_indicator_features}")
        print(f"🔄 Размер окна: {window_size}")
        if hasattr(self, 'indicators') and self.indicators.size > 0:
            print(f"🧠 Итоговая матрица индикаторов: {self.indicators.shape}")

    def _calculate_normalization_stats(self):
        """
        Рассчитывает статистики нормализации ТОЛЬКО по обучающим данным
        для предотвращения look-ahead bias
        """
        print("Начинаю расчет статистик для нормализации...")
        
        # ИСПРАВЛЕНИЕ: Используем параметр train_split_ratio для разделения данных
        train_split_ratio = getattr(self.cfg, 'train_split_ratio', 0.8)  # По умолчанию 80% для обучения
        
        # Определяем точки разделения для каждого таймфрейма
        split_5min = int(self.df_5min.shape[0] * train_split_ratio)
        split_15min = int(self.df_15min.shape[0] * train_split_ratio)
        split_1h = int(self.df_1h.shape[0] * train_split_ratio)
        
        # Сохраняем точки разделения для использования в других методах
        self.train_split_points = {
            '5min': split_5min,
            '15min': split_15min,
            '1h': split_1h
        }
        
        print(f"Использую {train_split_ratio*100:.0f}% данных для расчета статистик (train split)")
        print(f"Точки разделения: 5min={split_5min}, 15min={split_15min}, 1h={split_1h}")
        
        # 1. Собираем цены OHLC ТОЛЬКО из обучающих данных
        all_prices = np.concatenate([
            self.df_5min[:split_5min, :4].flatten(),
            self.df_15min[:split_15min, :4].flatten(),
            self.df_1h[:split_1h, :4].flatten()
        ]).astype(np.float32)
        
        self.price_mean = np.mean(all_prices)
        self.price_std = np.std(all_prices) + 1e-8

        # 2. Собираем объемы ТОЛЬКО из обучающих данных
        all_volumes = np.concatenate([
            self.df_5min[:split_5min, 4].flatten(),
            self.df_15min[:split_15min, 4].flatten(),
            self.df_1h[:split_1h, 4].flatten()
        ]).astype(np.float32)
        
        self.volume_mean = np.mean(all_volumes)
        self.volume_std = np.std(all_volumes) + 1e-8

        # 3. Статистики для индикаторов ТОЛЬКО из обучающих данных
        if self.indicators.size > 0:
            train_indicators = self.indicators[:split_5min]  # Индикаторы синхронизированы с 5min данными
            self.indicator_means = np.mean(train_indicators, axis=0)
            self.indicator_stds = np.std(train_indicators, axis=0) + 1e-8
            self.indicator_stds[self.indicator_stds == 0] = 1e-8
        else:
            self.indicator_means = np.array([])
            self.indicator_stds = np.array([])
        
        print(f"✅ Статистики нормализации рассчитаны (БЕЗ look-ahead bias)")
        print(f"💰 Price (train): mean={self.price_mean:.2f}, std={self.price_std:.2f}")
        print(f"📊 Volume (train): mean={self.volume_mean:.2f}, std={self.volume_std:.2f}")

    def _apply_normalization_stats(self, stats: Dict):
        """
        Устанавливает статистики нормализации, полученные при обучении.
        Ожидаемые ключи: price_mean, price_std, volume_mean, volume_std, indicator_means, indicator_stds
        """
        self.price_mean = float(stats.get('price_mean'))
        self.price_std = float(stats.get('price_std')) + 1e-8
        self.volume_mean = float(stats.get('volume_mean'))
        self.volume_std = float(stats.get('volume_std')) + 1e-8
        # Индикаторы могут отсутствовать
        im = stats.get('indicator_means')
        istd = stats.get('indicator_stds')
        if im is None or istd is None:
            self.indicator_means = np.array([])
            self.indicator_stds = np.array([])
        else:
            self.indicator_means = np.array(im, dtype=np.float32)
            self.indicator_stds = np.array(istd, dtype=np.float32)
            # защита от нулевых std
            self.indicator_stds[self.indicator_stds == 0] = 1e-8

    def export_normalization_stats(self) -> Dict:
        """Возвращает текущие статистики нормализации для сохранения в чекпойнт."""
        return {
            'price_mean': float(getattr(self, 'price_mean', 0.0)),
            'price_std': float(getattr(self, 'price_std', 1.0)),
            'volume_mean': float(getattr(self, 'volume_mean', 0.0)),
            'volume_std': float(getattr(self, 'volume_std', 1.0)),
            'indicator_means': (getattr(self, 'indicator_means', np.array([])).astype(float).tolist() if hasattr(self, 'indicator_means') else []),
            'indicator_stds': (getattr(self, 'indicator_stds', np.array([])).astype(float).tolist() if hasattr(self, 'indicator_stds') else []),
        }

    def _precompute_all_states(self):
        """
        Предвычисляет все возможные состояния для максимальной производительности
        """
        # Создаем объединенный массив всех признаков
        total_features = []
        
        # 1. 5-минутные данные (OHLCV)
        total_features.append(self.df_5min)
        
        # 2. 15-минутные данные (OHLCV) - интерполируем до 5-минутных
        df_15min_interpolated = self._interpolate_15min_to_5min()
        total_features.append(df_15min_interpolated)
        
        # 3. 1-часовые данные (OHLCV) - интерполируем до 5-минутных
        df_1h_interpolated = self._interpolate_1h_to_5min()
        total_features.append(df_1h_interpolated)
        
        # 4. Индикаторы (добавляем как есть, они уже соответствуют 5-минутным данным)
        if self.indicators.size > 0:
            total_features.append(self.indicators)
        
        # Объединяем все признаки
        X = np.concatenate(total_features, axis=1).astype(np.float32)
        
        print(f"🔍 Размеры данных для предвычисления:")
        print(f"  • df_5min: {self.df_5min.shape}")
        print(f"  • df_15min: {self.df_15min.shape}")
        print(f"  • df_1h: {self.df_1h.shape}")
        print(f"  • indicators: {self.indicators.shape if self.indicators.size > 0 else 'пустой'}")
        print(f"  • Объединенный X: {X.shape}")
        print(f"  • indicator_means: {self.indicator_means.shape if len(self.indicator_means) > 0 else 'пустой'}")
        
        # Нормализуем данные
        X_normalized = self._normalize_features(X)
        
        # Создаем скользящие окна
        W = self.lookback_window
        if len(X_normalized) >= W:
            # Используем sliding_window_view для эффективности
            sw = np.lib.stride_tricks.sliding_window_view(X_normalized, (W, X_normalized.shape[1]))[:, 0]
            # Преобразуем в [T-W+1, W*features]
            self.precomputed_states = sw.reshape(sw.shape[0], -1).astype(np.float32)
        else:
            # Если данных меньше окна, создаем нулевые состояния
            self.precomputed_states = np.zeros((1, self.observation_space_shape), dtype=np.float32)
        
        # Конвертируем в torch тензор для быстрого доступа
        self.states_tensor = torch.from_numpy(self.precomputed_states)
        
        print(f"📊 Предвычислено {len(self.precomputed_states)} состояний размером {self.precomputed_states.shape[1]}")
        print(f"🔍 Проверка размеров:")
        print(f"  • precomputed_states.shape: {self.precomputed_states.shape}")
        print(f"  • observation_space_shape: {self.observation_space_shape}")
        print(f"  • Соответствие: {'✅' if self.precomputed_states.shape[1] == self.observation_space_shape - 2 else '❌'}")
        
        # Если размеры не совпадают, исправляем
        if self.precomputed_states.shape[1] != self.observation_space_shape - 2:
            print(f"⚠️ Исправляю размер observation_space_shape")
            self.observation_space_shape = self.precomputed_states.shape[1] + 2
            print(f"  • Новый observation_space_shape: {self.observation_space_shape}")
            
            # Обновляем observation_space
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                                shape=(self.observation_space_shape,), 
                                                dtype=np.float32)

    def _interpolate_15min_to_5min(self):
        """
        Интерполирует 15-минутные данные до 5-минутных
        """
        if len(self.df_15min) == 0:
            return np.zeros((len(self.df_5min), 5), dtype=np.float32)
        
        # Каждые 3 пятиминутки = 1 пятнадцатиминутка
        interpolated = np.zeros((len(self.df_5min), 5), dtype=np.float32)
        
        for i in range(len(self.df_5min)):
            # Находим соответствующую 15-минутную свечу
            idx_15min = i // 3
            if idx_15min < len(self.df_15min):
                interpolated[i] = self.df_15min[idx_15min]
            else:
                # Если выходим за границы, используем последнюю доступную
                interpolated[i] = self.df_15min[-1] if len(self.df_15min) > 0 else 0
        
        return interpolated

    def _interpolate_1h_to_5min(self):
        """
        Интерполирует 1-часовые данные до 5-минутных
        """
        if len(self.df_1h) == 0:
            return np.zeros((len(self.df_5min), 5), dtype=np.float32)
        
        # Каждые 12 пятиминуток = 1 час
        interpolated = np.zeros((len(self.df_5min), 5), dtype=np.float32)
        
        for i in range(len(self.df_5min)):
            # Находим соответствующую 1-часовую свечу
            idx_1h = i // 12
            if idx_1h < len(self.df_1h):
                interpolated[i] = self.df_1h[idx_1h]
            else:
                # Если выходим за границы, используем последнюю доступную
                interpolated[i] = self.df_1h[-1] if len(self.df_1h) > 0 else 0
        
        return interpolated

    def _normalize_features(self, X):
        """
        Нормализует признаки используя предвычисленные статистики
        """
        normalized = X.copy()
        
        # Нормализация цен (первые 4 колонки каждого блока OHLCV)
        price_cols = []
        for i in range(0, X.shape[1], 5):
            if i + 4 <= X.shape[1]:
                price_cols.extend(range(i, i + 4))
        
        if price_cols:
            normalized[:, price_cols] = (X[:, price_cols] - self.price_mean) / self.price_std
        
        # Нормализация объемов (5-я колонка каждого блока OHLCV)
        volume_cols = list(range(4, X.shape[1], 5))
        if volume_cols:
            normalized[:, volume_cols] = (X[:, volume_cols] - self.volume_mean) / self.volume_std
        
        # Нормализация индикаторов (если есть)
        if self.indicators.size > 0:
            # Индикаторы начинаются после всех OHLCV данных
            # У нас 3 блока OHLCV (5min, 15min, 1h) по 5 колонок каждый = 15 колонок
            indicator_start = 15  # 3 * 5 = 15 колонок OHLCV
            if indicator_start < X.shape[1]:
                indicator_cols = range(indicator_start, X.shape[1])
                print(f"🔍 Нормализация индикаторов:")
                print(f"  • indicator_start: {indicator_start}")
                print(f"  • indicator_cols: {len(indicator_cols)} колонок")
                print(f"  • indicator_means: {len(self.indicator_means)} значений")
                print(f"  • X.shape: {X.shape}")
                # Включаем нормализацию индикаторов
                if hasattr(self, 'indicator_stds') and self.indicator_stds is not None:
                    normalized[:, indicator_cols] = (X[:, indicator_cols] - self.indicator_means) / np.where(self.indicator_stds==0, 1.0, self.indicator_stds)
        
        # Очищаем от NaN
        normalized = np.nan_to_num(normalized, nan=0.0)
        
        return normalized

    def _get_state(self) -> np.ndarray:
        """
        УЛЬТРА-БЫСТРАЯ версия _get_state с предвычисленными состояниями
        """
        current_5min_candle_idx = self.current_step - 1 
        
        # Проверяем границы
        if current_5min_candle_idx < self.lookback_window - 1 or current_5min_candle_idx >= len(self.precomputed_states):
            return np.zeros(self.observation_space_shape, dtype=np.float32)
        
        # Получаем предвычисленное состояние
        state_idx = current_5min_candle_idx - (self.lookback_window - 1)
        if state_idx < 0 or state_idx >= len(self.precomputed_states):
            return np.zeros(self.observation_space_shape, dtype=np.float32)
        
        # Берем предвычисленное состояние
        precomputed_state = self.precomputed_states[state_idx].copy()
        
        # Добавляем нормализованные баланс и криптовалюту
        try:
            normalized_balance = self.balance_scaler.transform(np.array([[self.balance]]))[0][0]
            normalized_crypto_held = self.crypto_held_scaler.transform(np.array([[self.crypto_held]]))[0][0]
        except Exception:
            normalized_balance = 0.0
            normalized_crypto_held = 0.0
        
        # Добавляем баланс и криптовалюту в конец состояния
        final_state = np.concatenate([
            precomputed_state,
            np.array([normalized_balance, normalized_crypto_held], dtype=np.float32)
        ])
        
        return final_state

    def get_state_tensor(self, step_idx: int) -> torch.Tensor:
        """
        УЛЬТРА-БЫСТРЫЙ доступ к состоянию через torch тензор
        """
        current_5min_candle_idx = step_idx - 1 
        
        # Проверяем границы
        if current_5min_candle_idx < self.lookback_window - 1 or current_5min_candle_idx >= len(self.states_tensor):
            return torch.zeros(self.observation_space_shape, dtype=torch.float32)
        
        # Получаем предвычисленное состояние
        state_idx = current_5min_candle_idx - (self.lookback_window - 1)
        if state_idx < 0 or state_idx >= len(self.states_tensor):
            return torch.zeros(self.observation_space_shape, dtype=torch.float32)
        
        # Берем предвычисленное состояние из тензора
        precomputed_state = self.states_tensor[state_idx].clone()
        
        # Добавляем нормализованные баланс и криптовалюту
        try:
            normalized_balance = self.balance_scaler.transform(np.array([[self.balance]]))[0][0]
            normalized_crypto_held = self.crypto_held_scaler.transform(np.array([[self.crypto_held]]))[0][0]
        except Exception:
            normalized_balance = 0.0
            normalized_crypto_held = 0.0
        
        # Добавляем баланс и криптовалюту в конец состояния
        final_state = torch.cat([
            precomputed_state,
            torch.tensor([normalized_balance, normalized_crypto_held], dtype=torch.float32)
        ])
        
        return final_state

    def reset(self):
        """
        Сброс окружения для нового эпизода
        """
        self._episode_idx += 1
        self._can_log = (self._episode_idx % self._log_interval == 0)
        
        # Сброс состояния
        initial_balance = getattr(self.cfg, 'initial_balance', 10000.0)  # По умолчанию 10000
        self.balance = initial_balance
        self.crypto_held = 0.0
        self.fee_entry = 0.0  # Инициализируем fee_entry
        self.last_buy_price = None
        self.last_buy_step = None
        self.trailing_stop_counter = 0
        self.max_price_during_hold = None
        self.balance_history = [self.balance]
        
        # ИСПРАВЛЯЕМ: НЕ очищаем сделки между эпизодами для правильного расчета winrate
        # self.trades = []  # ЗАКОММЕНТИРОВАНО
        if not hasattr(self, 'all_trades'):
            self.all_trades = []  # Создаем общий список сделок если его нет

        
        # Сброс статистик
        self.vol_buf.clear()
        self.roi_buf.clear()
        self.buy_attempts = 0
        self.buy_rejected_vol = 0
        self.buy_rejected_roi = 0
        
        # Сброс счетчиков действий
        self.action_counts = {0: 0, 1: 0, 2: 0}
        
        # Выбор случайной начальной точки с учетом длины эпизода
        if self._can_log:
            print(f"🌀 episode_length = {self.episode_length} шагов (≈ {self.episode_length*5/60:.1f} часов)")
        max_start = self.total_steps - self.episode_length
        min_start = self.min_valid_start_step
        
        if max_start <= min_start:
            self.start_step = min_start
        else:
            self.start_step = random.randint(min_start, max_start)
        
        self.current_step = self.start_step
        
        # Начинаем отсчет времени для эпизода
        self.episode_start_time = time.time()
        self.episode_step_count = 0
        
        # Используем быстрый доступ к предвычисленному состоянию
        return self._get_state()

    def step(self, action):
        """
        Выполнение действия в окружении
        """
        # Коэффициент масштабирования награды
        reward_scale = float(getattr(self.cfg, 'reward_scale', 1.0))
        # Получаем текущую цену
        current_price = self.df_5min[self.current_step - 1, 3]  # Close price
        
        # Базовая награда
        reward = 0.0
        done = False
        info = {}
        
        # Подсчитываем действие
        self.action_counts[action] += 1
        self.episode_step_count += 1
        
        # Выполняем действие
        if action == 1:  # BUY
            if self.crypto_held == 0:  # Только если не держим криптовалюту
                # Проверяем фильтры
                if self._check_buy_filters():
                    # ДИНАМИЧЕСКИЙ размер позиции на основе уверенности
                    entry_confidence = self._calculate_entry_confidence()
                    
                    # Адаптируем размер позиции: высокая уверенность = большая позиция
                    if entry_confidence > self.position_confidence_threshold:
                        # Высокая уверенность - увеличиваем позицию
                        self.position_fraction = min(self.base_position_fraction * 1.5, 0.5)  # Максимум 50%
                        if self._can_log:
                            print(f"🎯 Высокая уверенность ({entry_confidence:.2f}): увеличиваем позицию до {self.position_fraction:.1%}")
                    elif entry_confidence > 0.5:
                        # Средняя уверенность - стандартная позиция
                        self.position_fraction = self.base_position_fraction
                    else:
                        # Низкая уверенность - уменьшаем позицию
                        self.position_fraction = max(self.base_position_fraction * 0.7, 0.15)  # Минимум 15%
                        if self._can_log:
                            print(f"🎯 Низкая уверенность ({entry_confidence:.2f}): уменьшаем позицию до {self.position_fraction:.1%}")
                    
                    # Покупаем с учётом комиссии
                    buy_amount = self.balance * self.position_fraction
                    self.fee_entry = buy_amount * self.trade_fee_percent
                    crypto_to_buy = (buy_amount - self.fee_entry) / current_price
                    self.crypto_held = crypto_to_buy
                    self.balance -= buy_amount  # списываем всю сумму, комиссия учтена в купленной крипте
                    reward -= self.fee_entry / max(self.balance, 1e-9)  # мелкий штраф за комиссию
                    self.last_buy_price = current_price
                    self.last_buy_step = self.current_step
                    
                    # Награда зависит от уверенности входа
                    base_reward = 0.03
                    confidence_bonus = entry_confidence * 0.02  # Дополнительная награда за высокую уверенность
                    reward = base_reward + confidence_bonus
                    
                    if self._can_log:
                        print(f"🔵 BUY: {crypto_to_buy:.4f} at {current_price:.2f}, уверенность: {entry_confidence:.2f}, награда: {reward:.4f}")
                else:
                    reward = -0.002  # Уменьшил штраф за отклонение фильтрами
            else:
                reward = -0.01  # Уменьшил штраф за попытку купить при наличии позиции
                
        elif action == 2:  # SELL
            if self.crypto_held > 0:  # Только если держим криптовалюту
                # Проверяем минимальное время удержания
                if hasattr(self, 'last_buy_step') and self.last_buy_step is not None:
                    hold_time = self.current_step - self.last_buy_step
                    if hold_time < self.min_hold_steps:
                        # Штраф за слишком раннюю продажу
                        reward = -0.02
                        #self._log(f"[{self.current_step}] ⚠️ Слишком ранняя продажа: {hold_time} шагов < {self.min_hold_steps}")
                        # Масштабируем награду перед возвратом
                        reward = reward * reward_scale
                        return self._get_state(), reward, False, {}
                
                # Продаем
                sell_amount = self.crypto_held * current_price
                fee_exit = sell_amount * self.trade_fee_percent
                self.balance += (sell_amount - fee_exit)
                
                # Рассчитываем прибыль/убыток
                pnl = ((current_price - self.last_buy_price) / self.last_buy_price) - (self.fee_entry + fee_exit)/max(self.last_buy_price * self.crypto_held,1e-9)
                net_profit_loss = sell_amount - (self.crypto_held * self.last_buy_price)
                
                # Награда за продажу (как в оригинале)
                reward += np.tanh(pnl * 25) * 2  # За результат сделки
                
                # Дополнительные награды за качество сделки (УЛУЧШЕНО)
                if pnl > 0.05:  # Прибыль > 5%
                    reward += 0.5  # Увеличил с 0.3 до 0.5 для больших прибылей
                elif pnl > 0.02:  # Прибыль > 2%
                    reward += 0.2  # Новая награда за среднюю прибыль
                elif pnl < -0.03:  # Убыток > 3%
                    reward -= 0.005  # Уменьшил штраф с -0.01 до -0.005
                elif pnl < -0.01:  # Убыток > 1%
                    reward -= 0.001  # Небольшой штраф за малые убытки
                
                # ИСПРАВЛЯЕМ: Записываем сделку в оба списка для правильного расчета winrate
                trade_data = {
                    "roi": pnl,
                    "net": net_profit_loss,
                    "reward": reward,
                    "duration": (self.current_step - self.last_buy_step) * 5 if self.last_buy_step else 0
                }
                self.trades.append(trade_data)
                
                # Также добавляем в общий список сделок
                if not hasattr(self, 'all_trades'):
                    self.all_trades = []                    
                self.all_trades.append(trade_data)
                                
                
                self.crypto_held = 0
                self.last_buy_price = None
                self.last_buy_step = None
                self.trailing_stop_counter = 0
                self.max_price_during_hold = None
                
                #self._log(f"[{self.current_step}] 🔴 SELL: {sell_amount:.2f}, PnL: {pnl:.2%}")
            else:
                reward = -0.01  # Уменьшил штраф за попытку продать без позиции
        
        # Добавляем переход в n-step buffer только если не terminal
        if not done:
            transition = {
                'state': self._get_state(),
                'action': action,
                'reward': reward,
                'next_state': None,  # Будет заполнено позже
                'done': done
            }
            self.n_step_buffer.append(transition)
        
        # Обработка HOLD действия (как в оригинале)
        if action == 0:
            if self.crypto_held > 0 and self.last_buy_price is not None:
                # Награда за длительное удержание позиции
                if hasattr(self, 'last_buy_step') and self.last_buy_step is not None:
                    hold_time = self.current_step - self.last_buy_step
                    if hold_time >= self.min_hold_steps * 2:  # Двойное минимальное время
                        reward += 0.001  # Небольшая награда за терпение
                        if self.current_step % 100 == 0:
                            self._log(f"[{self.current_step}] 🕐 Награда за терпение: {hold_time} шагов")
                # --- Трейлинг-стоп (как в оригинале) ---
                if self.epsilon <= 0.2:  # фаза exploitation
                    # 1. обновляем максимум
                    if (not hasattr(self, "max_price_during_hold") 
                        or self.max_price_during_hold is None 
                        or self.last_buy_step == self.current_step):
                        self.max_price_during_hold = current_price
                    
                    if current_price > self.max_price_during_hold:  # новый пик
                        self.max_price_during_hold = current_price
                        self.trailing_stop_counter = 0
                    
                    # 2. считаем просадку от пика
                    drawdown = (self.max_price_during_hold - current_price) / self.max_price_during_hold
                    if drawdown > 0.02:
                        self.trailing_stop_counter += 1
                    
                    # 3. три подряд бара с drawdown > 2% → принудительный SELL
                    # Но только если позиция держится достаточно долго
                    if (self.trailing_stop_counter >= 3 and 
                        hasattr(self, 'last_buy_step') and 
                        self.last_buy_step is not None and
                        (self.current_step - self.last_buy_step) >= self.min_hold_steps):
                        
                        reward -= 0.03
                        self._log(f"[{self.current_step}] 🔻 TRAILING STOP — SELL by drawdown: {drawdown:.2%}")
                        self._force_sell(current_price, 'TRAILING STOP')
                        
                        # Обновляем статистики
                        self._update_stats(current_price)
                        self.current_step += 1
                        
                        # Проверяем завершение эпизода
                        done = (
                            self.current_step >= self.start_step + self.episode_length or
                            self.current_step >= self.total_steps
                        )
                        
                        info.update({
                            "current_balance": self.balance,
                            "current_price": current_price,
                            "total_profit": (self.balance + self.crypto_held * current_price) - getattr(self.cfg, 'initial_balance', 10000.0),
                        })
                        # Масштабируем награду перед возвратом
                        reward = reward * reward_scale
                        info["reward"] = reward
                        return self._get_state(), reward, done, info
                
                # --- Take Profit / Stop Loss (как в оригинале) ---
                unrealized_pnl_percent = (current_price - self.last_buy_price) / self.last_buy_price
                
                if unrealized_pnl_percent <= self.STOP_LOSS_PCT:
                    reward -= 0.05  # штраф за стоп-лосс
                    self._force_sell(current_price, "❌ - STOP-LOSS triggered")
                    
                elif unrealized_pnl_percent >= self.TAKE_PROFIT_PCT:
                    reward += 0.05  # поощрение за фиксацию профита
                    self._force_sell(current_price, "🎯 - TAKE-PROFIT hit")
                
                # --- Награды за удержание позиции (УЛУЧШЕНО) ---
                if unrealized_pnl_percent > 0:
                    # Чем выше нереализованная прибыль, тем больше награда за удержание
                    reward += unrealized_pnl_percent * 3  # Увеличил с 2 до 3 для лучшего удержания прибыли
                else:
                    # Чем больше нереализованный убыток, тем больше штраф за удержание
                    reward += unrealized_pnl_percent * 2  # Уменьшил с 3 до 2 для меньшего давления
            else:
                # Если action == HOLD и нет открытой позиции
                reward = 0.001  # Небольшое поощрение за разумное бездействие вместо штрафа
        
        # Обновляем статистики
        self._update_stats(current_price)
        
        # АДАПТИВНЫЕ НАГРАДЫ для разных рыночных условий
        if action != 0:  # Если действие не HOLD
            base_activity_reward = 0.001
            
            # Адаптируем награду к времени дня (НЕ блокируем, а обучаем)
            if hasattr(self, '_candle_datetimes') and self.current_step - 1 < len(self._candle_datetimes):
                current_hour = self._candle_datetimes[self.current_step - 1].hour
                
                if 6 <= current_hour <= 22:
                    # Активные часы - стандартная награда
                    reward += base_activity_reward
                else:
                    # Низкая ликвидность - повышенная награда за смелость
                    # Агент учится торговать в сложных условиях
                    reward += base_activity_reward * 1.5
                    
            else:
                # Если нет информации о времени - стандартная награда
                reward += base_activity_reward
        
        # Переходим к следующему шагу
        self.current_step += 1
        
        # Проверяем завершение эпизода
        episode_length = getattr(self.cfg, 'episode_length')
        
        done = (
            self.current_step >= self.start_step + episode_length or  # Ограничение длины эпизода
            self.current_step >= self.total_steps
        )
        
        if done:
            # Принудительно продаем если есть позиция
            if self.crypto_held > 0:
                final_price = self.df_5min[self.current_step - 1, 3]
                final_sell_amount = self.crypto_held * final_price
                self.balance += final_sell_amount
                pnl = (final_price - self.last_buy_price) / self.last_buy_price
                net_profit_loss = final_sell_amount - (self.crypto_held * self.last_buy_price)
                
                trade_data = {
                    "roi": pnl,
                    "net": net_profit_loss,
                    "reward": 0,
                    "duration": (self.current_step - self.last_buy_step) * 5 if self.last_buy_step else 0
                }
                self.trades.append(trade_data)
                if self._can_log:
                    print(f"    📝 Финальная сделка: ROI={pnl:.4f}, прибыль={trade_data['roi'] > 0}, всего сделок={len(self.trades)}")
            
            # Статистика времени эпизода (теперь выводится в train_model_optimized.py)
            # if self.episode_start_time is not None:
            #     episode_duration = time.time() - self.episode_start_time
            #     steps_per_second = self.episode_step_count / episode_duration if episode_duration > 0 else 0
            #     print(f"⏱️ Эпизод завершен: {episode_duration:.2f}с, {self.episode_step_count} шагов, {steps_per_second:.1f} шаг/с")
        
        # Информация для отладки
        initial_balance = getattr(self.cfg, 'initial_balance', 10000.0)  # По умолчанию 10000
        info.update({
            "current_balance": self.balance,
            "current_price": current_price,
            "total_profit": (self.balance + self.crypto_held * current_price) - initial_balance,
            "reward": reward
        })
        self.balance_history.append(self.balance + self.crypto_held * current_price)
        
        # Обновляем next_state для всех переходов в n-step buffer
        for transition in self.n_step_buffer:
            if transition['next_state'] is None:
                transition['next_state'] = self._get_state()
        
        # Отладочная информация        
        # Масштабируем награду перед возвратом
        reward = reward * reward_scale
        info["reward"] = reward
        return self._get_state(), reward, done, info

    def _check_buy_filters(self) -> bool:
        """
        Проверяет фильтры для покупки (УЛУЧШЕНО)
        """
        self.buy_attempts += 1
        
        # АДАПТИВНАЯ СТРОГОСТЬ ФИЛЬТРОВ ПО ЭПСИЛОНУ
        # eps > 0.8 → свободное исследование (почти без фильтров)
        # eps 0.8..0.2 → плавное ужесточение
        # eps <= 0.2 → строгие пороги
        eps = 1.0
        try:
            eps = float(getattr(self, 'epsilon', 1.0))
        except Exception:
            eps = 1.0
        
        if eps > 0.8:
            return True
        # Степень строгости [0..1]
        strictness = np.clip((0.8 - eps) / (0.8 - 0.2), 0.0, 1.0)
        
        # 1. Проверка объема - АДАПТИВНЫЙ порог
        current_volume = self.df_5min[self.current_step - 1, 4]
        vol_relative = calc_relative_vol_numpy(self.df_5min, self.current_step - 1, 12)
        
        # Порог объема: от мягкого 0.0010 к строгому 0.0030 (или конфигурационному, если выше)
        cfg_thr = float(getattr(self, 'volume_threshold', 0.0005))
        base_lenient_vol = max(cfg_thr, 0.0010)
        base_strict_vol  = max(cfg_thr, 0.0030)
        vol_thr = base_lenient_vol + strictness * (base_strict_vol - base_lenient_vol)
        if vol_relative < vol_thr:
            self.buy_rejected_vol += 1
            if self.current_step % 100 == 0:
                print(f"🔍 Фильтр объема: vol_relative={vol_relative:.4f} < {vol_thr:.4f}, отклонено")
            return False
        
        # 2. Проверка ROI - УЛУЧШЕНО: Более умный фильтр
        if len(self.roi_buf) > 0:
            recent_roi_mean = np.mean(list(self.roi_buf))
            # Порог по среднему ROI: от -6% (мягко) к -1% (строго)
            roi_thr = -0.06 + strictness * ( -0.01 + 0.06 )  # -0.06 → -0.01
            if recent_roi_mean < roi_thr:
                self.buy_rejected_roi += 1
                if self.current_step % 100 == 0:
                    print(f"🔍 Фильтр ROI: recent_roi_mean={recent_roi_mean:.4f} < {roi_thr:.4f}, отклонено")
                return False
        
        # 3. НОВЫЙ: Фильтр по тренду цены
        if self.current_step >= 20:
            recent_prices = self.df_5min[self.current_step-20:self.current_step, 3]  # Close prices
            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            # Порог тренда: от -2.0% (мягко) к -0.5% (строго)
            trend_thr = -0.02 + strictness * ( -0.005 + 0.02 )  # -0.02 → -0.005
            if price_trend < trend_thr:
                if self.current_step % 100 == 0:
                    print(f"🔍 Фильтр тренда: price_trend={price_trend:.4f} < {trend_thr:.4f}, отклонено")
                return False
        
        # 4. НОВЫЙ: Фильтр по волатильности
        if self.current_step >= 12:
            recent_highs = self.df_5min[self.current_step-12:self.current_step, 1]  # High prices
            recent_lows = self.df_5min[self.current_step-12:self.current_step, 2]   # Low prices
            volatility = np.mean((recent_highs - recent_lows) / recent_lows)
            # Порог волатильности: от 0.002 (мягко) к 0.0045 (строго)
            volat_thr = 0.002 + strictness * (0.0045 - 0.002)
            if volatility < volat_thr:
                if self.current_step % 100 == 0:
                    print(f"🔍 Фильтр волатильности: volatility={volatility:.4f} < {volat_thr:.4f}, отклонено")
                return False
        
        # 5. НОВЫЙ: Фильтр по силе тренда (ADX-подобный)
        if self.current_step >= 20:
            # Рассчитываем силу тренда на основе изменения цены
            recent_prices = self.df_5min[self.current_step-20:self.current_step, 3]  # Close prices
            price_changes = np.diff(recent_prices)
            trend_strength = np.abs(np.mean(price_changes)) / (np.std(price_changes) + 1e-8)
            # Порог силы тренда: от 0.15 (мягко) к 0.35 (строго)
            ts_thr = 0.15 + strictness * (0.35 - 0.15)
            if trend_strength < ts_thr:
                if self.current_step % 100 == 0:
                    print(f"🔍 Фильтр тренда: trend_strength={trend_strength:.4f} < {ts_thr:.4f}, отклонено")
                return False
        
        # Все фильтры пройдены - разрешаем покупку
        return True
    
    def _calculate_entry_confidence(self) -> float:
        """
        Рассчитывает уверенность входа в позицию на основе множественных факторов
        """
        confidence = 0.0
        
        try:
            # 1. Уверенность по объему (0-25%)
            if self.current_step >= 12:
                vol_relative = calc_relative_vol_numpy(self.df_5min, self.current_step - 1, 12)
                vol_confidence = min(vol_relative / (self.volume_threshold * 2), 1.0) * 0.25
                confidence += vol_confidence
            
            # 2. Уверенность по тренду (0-25%)
            if self.current_step >= 20:
                recent_prices = self.df_5min[self.current_step-20:self.current_step, 3]
                price_changes = np.diff(recent_prices)
                trend_strength = np.abs(np.mean(price_changes)) / (np.std(price_changes) + 1e-8)
                trend_confidence = min(trend_strength / 0.5, 1.0) * 0.25
                confidence += trend_confidence
            
            # 3. Уверенность по историческому ROI (0-25%)
            if len(self.roi_buf) >= 10:
                recent_roi_mean = np.mean(list(self.roi_buf)[-10:])
                roi_confidence = max(0, (recent_roi_mean + 0.1) / 0.2) * 0.25  # Нормализуем от -10% до +10%
                confidence += roi_confidence
            
            # 4. Уверенность по времени дня (0-25%)
            if hasattr(self, '_candle_datetimes') and self.current_step - 1 < len(self._candle_datetimes):
                current_hour = self._candle_datetimes[self.current_step - 1].hour
                if 6 <= current_hour <= 22:  # Активные часы
                    time_confidence = 0.25
                else:  # Низкая ликвидность
                    time_confidence = 0.15  # Немного снижаем уверенность
                confidence += time_confidence
            else:
                confidence += 0.20  # Средняя уверенность если нет информации о времени
            
        except Exception as e:
            if self._can_log:
                print(f"⚠️ Ошибка при расчете уверенности: {e}")
            confidence = 0.5  # Средняя уверенность при ошибке
        
        return min(max(confidence, 0.0), 1.0)  # Ограничиваем от 0 до 1

    def _update_stats(self, current_price: float):
        """
        Обновляет статистики и динамические параметры
        """
        # Обновляем статистики объема
        current_volume = self.df_5min[self.current_step - 1, 4]
        self.vol_buf.append(current_volume)
        update_vol_stats(self.vol_buf, self.vol_stats)
        
        # Обновляем статистики ROI если есть позиция
        if self.crypto_held > 0 and self.last_buy_price is not None:
            unrealized_roi = (current_price - self.last_buy_price) / self.last_buy_price
            self.roi_buf.append(unrealized_roi)
            update_roi_stats(self.roi_buf, self.roi_stats)
        
        # ДИНАМИЧЕСКОЕ ОБНОВЛЕНИЕ ПАРАМЕТРОВ РИСК-МЕНЕДЖМЕНТА
        self._update_dynamic_parameters(current_price)

    def _force_sell(self, current_price: float, reason: str):
        """
        Принудительная продажа (как в оригинале)
        """
        if self.crypto_held > 0:
            sell_amount = self.crypto_held * current_price
            self.balance += sell_amount
            
            pnl = (current_price - self.last_buy_price) / self.last_buy_price
            net_profit_loss = sell_amount - (self.crypto_held * self.last_buy_price)
            
            # ИСПРАВЛЯЕМ: Записываем сделку в оба списка
            trade_data = {
                "roi": pnl,
                "net": net_profit_loss,
                "reward": 0,
                "duration": (self.current_step - self.last_buy_step) * 5 if self.last_buy_step else 0
            }
            self.trades.append(trade_data)
            
            # Также добавляем в общий список сделок
            if not hasattr(self, 'all_trades'):
                self.all_trades = []
            self.all_trades.append(trade_data)
            
            self.crypto_held = 0
            self.last_buy_price = None
            self.last_buy_step = None
            self.trailing_stop_counter = 0
            self.max_price_during_hold = None
            
            self._log(f"[{self.current_step}] 🔴 FORCE SELL ({reason}): {sell_amount:.2f}, PnL: {pnl:.2%}")

    def _log(self, *args, **kwargs):
        if self._can_log:
            print(*args, **kwargs)
    
    def _update_dynamic_parameters(self, current_price: float):
        """
        Динамически обновляет параметры риск-менеджмента на основе рыночных условий
        """
        if not hasattr(self, 'base_stop_loss'):
            return  # Если не используем динамические параметры
        
        # Обновляем параметры каждые 100 шагов для стабильности
        if self.current_step % 100 != 0:
            return
        
        try:
            # 1. Адаптация к волатильности
            if len(self.vol_buf) >= 20:
                recent_volatility = np.std(list(self.vol_buf)[-20:]) / (np.mean(list(self.vol_buf)[-20:]) + 1e-8)
                
                # Если волатильность высокая - ужесточаем stop-loss
                if recent_volatility > 0.5:  # Высокая волатильность
                    self.STOP_LOSS_PCT = self.base_stop_loss * 1.5  # Увеличиваем stop-loss
                    self.TAKE_PROFIT_PCT = self.base_take_profit * 1.2  # Увеличиваем take-profit
                    if self._can_log:
                        print(f"🔧 Высокая волатильность: SL={self.STOP_LOSS_PCT:.3f}, TP={self.TAKE_PROFIT_PCT:.3f}")
                elif recent_volatility < 0.1:  # Низкая волатильность
                    self.STOP_LOSS_PCT = self.base_stop_loss * 0.8  # Уменьшаем stop-loss
                    self.TAKE_PROFIT_PCT = self.base_take_profit * 0.9  # Уменьшаем take-profit
                    if self._can_log:
                        print(f"🔧 Низкая волатильность: SL={self.STOP_LOSS_PCT:.3f}, TP={self.TAKE_PROFIT_PCT:.3f}")
                else:
                    # Возвращаем к базовым значениям
                    self.STOP_LOSS_PCT = self.base_stop_loss
                    self.TAKE_PROFIT_PCT = self.base_take_profit
            
            # 2. Адаптация к тренду
            if len(self.roi_buf) >= 30:
                recent_trend = np.mean(list(self.roi_buf)[-30:])
                
                # Если тренд положительный - увеличиваем take-profit
                if recent_trend > 0.02:  # Хороший тренд
                    self.TAKE_PROFIT_PCT = min(self.base_take_profit * 1.3, 0.15)  # Максимум 15%
                    self.min_hold_steps = max(self.base_min_hold - 2, 4)  # Уменьшаем время удержания
                    if self._can_log:
                        print(f"🔧 Хороший тренд: TP={self.TAKE_PROFIT_PCT:.3f}, MinHold={self.min_hold_steps}")
                elif recent_trend < -0.02:  # Плохой тренд
                    self.STOP_LOSS_PCT = self.base_stop_loss * 0.7  # Ужесточаем stop-loss
                    self.min_hold_steps = self.base_min_hold + 4  # Увеличиваем время удержания
                    if self._can_log:
                        print(f"🔧 Плохой тренд: SL={self.STOP_LOSS_PCT:.3f}, MinHold={self.min_hold_steps}")
                else:
                    # Возвращаем к базовым значениям
                    self.TAKE_PROFIT_PCT = self.base_take_profit
                    self.min_hold_steps = self.base_min_hold
            
            # 3. Адаптация к времени дня (НЕ блокируем, а адаптируем параметры)
            if hasattr(self, '_candle_datetimes') and self.current_step - 1 < len(self._candle_datetimes):
                current_hour = self._candle_datetimes[self.current_step - 1].hour
                
                # Адаптируем параметры к времени дня, но НЕ блокируем торговлю
                # Агент сам научится торговать в разных условиях
                if 6 <= current_hour <= 22:
                    # Активные часы - более агрессивные параметры
                    self.TAKE_PROFIT_PCT = min(self.TAKE_PROFIT_PCT * 1.1, 0.15)
                    self.min_hold_steps = max(self.min_hold_steps - 1, 4)
                    if self._can_log:
                        print(f"🔧 Активные часы ({current_hour}:00 UTC): агрессивные параметры")
                else:
                    # Низкая ликвидность - более консервативные параметры, но торговля разрешена
                    self.STOP_LOSS_PCT = self.STOP_LOSS_PCT * 0.9
                    self.min_hold_steps = self.min_hold_steps + 2
                    if self._can_log:
                        print(f"🔧 Низкая ликвидность ({current_hour}:00 UTC): консервативные параметры (торговля разрешена)")
            
            # 4. Ограничиваем параметры разумными пределами
            self.STOP_LOSS_PCT = max(self.STOP_LOSS_PCT, -0.08)  # Не более -8%
            self.STOP_LOSS_PCT = min(self.STOP_LOSS_PCT, -0.01)  # Не менее -1%
            self.TAKE_PROFIT_PCT = max(self.TAKE_PROFIT_PCT, 0.03)   # Не менее 3%
            self.TAKE_PROFIT_PCT = min(self.TAKE_PROFIT_PCT, 0.20)   # Не более 20%
            self.min_hold_steps = max(self.min_hold_steps, 4)     # Не менее 4 шагов
            self.min_hold_steps = min(self.min_hold_steps, 20)    # Не более 20 шагов
            
            # 5. НОВЫЙ: Динамическое обновление размера позиции на основе рыночных условий
            if hasattr(self, 'base_position_fraction'):
                # Адаптируем базовый размер позиции к текущим рыночным условиям
                market_conditions_score = 0.0
                
                # Оценка рыночных условий по волатильности
                if len(self.vol_buf) >= 20:
                    recent_volatility = np.std(list(self.vol_buf)[-20:]) / (np.mean(list(self.vol_buf)[-20:]) + 1e-8)
                    if 0.1 <= recent_volatility <= 0.5:  # Оптимальная волатильность
                        market_conditions_score += 0.3
                    elif recent_volatility > 0.5:  # Высокая волатильность
                        market_conditions_score += 0.1  # Снижаем размер позиции
                    else:  # Низкая волатильность
                        market_conditions_score += 0.2
                
                # Оценка по тренду
                if len(self.roi_buf) >= 30:
                    recent_trend = np.mean(list(self.roi_buf)[-30:])
                    if recent_trend > 0.02:  # Хороший тренд
                        market_conditions_score += 0.4
                    elif recent_trend > -0.01:  # Нейтральный тренд
                        market_conditions_score += 0.3
                    else:  # Плохой тренд
                        market_conditions_score += 0.1
                
                # Оценка по времени дня
                if hasattr(self, '_candle_datetimes') and self.current_step - 1 < len(self._candle_datetimes):
                    current_hour = self._candle_datetimes[self.current_step - 1].hour
                    if 6 <= current_hour <= 22:  # Активные часы
                        market_conditions_score += 0.3
                    else:  # Низкая ликвидность
                        market_conditions_score += 0.1
                else:
                    market_conditions_score += 0.2
                
                # Нормализуем оценку рыночных условий (0-1)
                market_conditions_score = min(max(market_conditions_score, 0.0), 1.0)
                
                # Адаптируем базовый размер позиции
                if market_conditions_score > 0.7:  # Отличные условия
                    self.base_position_fraction = min(0.4, self.base_position_fraction * 1.1)  # Увеличиваем до 40%
                elif market_conditions_score > 0.5:  # Хорошие условия
                    self.base_position_fraction = min(0.35, self.base_position_fraction * 1.05)  # Увеличиваем до 35%
                elif market_conditions_score > 0.3:  # Средние условия
                    self.base_position_fraction = max(0.25, self.base_position_fraction * 0.95)  # Уменьшаем до 25%
                else:  # Плохие условия
                    self.base_position_fraction = max(0.2, self.base_position_fraction * 0.9)   # Уменьшаем до 20%
                
                if self._can_log:
                    print(f"🔧 Рыночные условия: {market_conditions_score:.2f}, базовый размер позиции: {self.base_position_fraction:.1%}")
            
        except Exception as e:
            if self._can_log:
                print(f"⚠️ Ошибка при обновлении динамических параметров: {e}")
            # Возвращаем к базовым значениям при ошибке
            self.STOP_LOSS_PCT = self.base_stop_loss
            self.TAKE_PROFIT_PCT = self.base_take_profit
            self.min_hold_steps = self.base_min_hold
    
    def get_n_step_return(self, n_steps: int = None) -> list:
        """
        Возвращает n-step transitions для обучения
        
        Args:
            n_steps: количество шагов (по умолчанию использует self.n_step)
            
        Returns:
            list: список n-step transitions
        """
        if n_steps is None:
            n_steps = self.n_step
            
        if len(self.n_step_buffer) < n_steps:
            return []
            
        transitions = []
        for i in range(len(self.n_step_buffer) - n_steps + 1):
            # Берем n последовательных переходов
            n_step_transitions = list(self.n_step_buffer)[i:i+n_steps]
            
            # Проверяем, что все next_state заполнены
            if any(t['next_state'] is None for t in n_step_transitions):
                continue  # Пропускаем неполные transitions
            
            # Рассчитываем n-step return
            total_reward = 0
            for j, transition in enumerate(n_step_transitions):
                total_reward += transition['reward'] * (self.gamma ** j)
            
            # Создаем n-step transition
            first_transition = n_step_transitions[0]
            last_transition = n_step_transitions[-1]
            
            n_step_transition = {
                'state': first_transition['state'],
                'action': first_transition['action'],
                'reward': total_reward,
                'next_state': last_transition['next_state'],
                'done': last_transition['done']
            }
            
            transitions.append(n_step_transition)
            
        return transitions

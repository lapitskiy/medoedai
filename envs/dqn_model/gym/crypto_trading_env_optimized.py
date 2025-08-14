from envs.dqn_model.gym.gutils_optimized import calc_relative_vol_numpy, commission_penalty, update_roi_stats, update_vol_stats
from envs.dqn_model.gym.indicators_optimized import preprocess_dataframes
import gym
from gym import spaces
import numpy as np
import random
import torch
import time
from sklearn.preprocessing import StandardScaler
from envs.dqn_model.gym.gconfig import GymConfig
from typing import Optional, Dict
from collections import deque

class CryptoTradingEnvOptimized(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, dfs: Dict, cfg: Optional[GymConfig] = None, lookback_window: int = 20, indicators_config=None):        
        super(CryptoTradingEnvOptimized, self).__init__() 
        self.cfg = cfg or GymConfig()
        
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
        
        # вход в позицию на 30%
        self.position_fraction = 0.30

        # Константы окружения
        self.trade_fee_percent = 0.00075 # Комиссия 0.075%
        
        # Константы риск-менеджмента (как в оригинале)
        self.STOP_LOSS_PCT = -0.03    # -3%
        self.TAKE_PROFIT_PCT = +0.05  # +5%
        
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
        
        # Предварительно рассчитываем статистики нормализации
        self._calculate_normalization_stats()
        
        # Инициализируем скалеры для баланса и криптовалюты
        self.balance_scaler = StandardScaler()
        self.crypto_held_scaler = StandardScaler()
        
        # Предвычисляем все состояния для максимальной производительности
        print("🚀 Предвычисляю все состояния...")
        self._precompute_all_states()
        print("✅ Все состояния предвычислены!")
                
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
        print(f"💰 Начальный баланс: {initial_balance}")
        print(f"🔄 Размер окна: {window_size}")
        print(f"✅ ФИЛЬТРЫ ПОКУПКИ ВКЛЮЧЕНЫ: объем (0.5%) + ROI (-3%)")
        print(f"🚀 ПРЕДВЫЧИСЛЕНИЕ СОСТОЯНИЙ: env.step() = просто сдвиг индекса!")

    def _calculate_normalization_stats(self):
        """
        Рассчитывает статистики нормализации для всех данных
        """
        print("Начинаю расчет статистик для нормализации...")
        
        # 1. Собираем все цены OHLC из всех массивов
        all_prices = np.concatenate([
            self.df_5min[:, :4].flatten(),
            self.df_15min[:, :4].flatten(),
            self.df_1h[:, :4].flatten()
        ]).astype(np.float32)
        
        self.price_mean = np.mean(all_prices)
        self.price_std = np.std(all_prices) + 1e-8

        # 2. Собираем все объемы
        all_volumes = np.concatenate([
            self.df_5min[:, 4].flatten(),
            self.df_15min[:, 4].flatten(),
            self.df_1h[:, 4].flatten()
        ]).astype(np.float32)
        
        self.volume_mean = np.mean(all_volumes)
        self.volume_std = np.std(all_volumes) + 1e-8

        # 3. Статистики для индикаторов
        if self.indicators.size > 0:
            self.indicator_means = np.mean(self.indicators, axis=0)
            self.indicator_stds = np.std(self.indicators, axis=0) + 1e-8
            self.indicator_stds[self.indicator_stds == 0] = 1e-8
        else:
            self.indicator_means = np.array([])
            self.indicator_stds = np.array([])
        
        print(f"✅ Статистики нормализации рассчитаны")
        print(f"💰 Price: mean={self.price_mean:.2f}, std={self.price_std:.2f}")
        print(f"📊 Volume: mean={self.volume_mean:.2f}, std={self.volume_std:.2f}")

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
                
                # Временно отключаем нормализацию индикаторов для отладки
                print(f"⚠️ Временно отключаю нормализацию индикаторов")
                # normalized[:, indicator_cols] = (X[:, indicator_cols] - self.indicator_means) / self.indicator_stds
        
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
        self.last_buy_price = None
        self.last_buy_step = None
        self.trailing_stop_counter = 0
        self.max_price_during_hold = None
        
        # Очистка списка сделок
        self.trades = []
        
        # Сброс статистик
        self.vol_buf.clear()
        self.roi_buf.clear()
        self.buy_attempts = 0
        self.buy_rejected_vol = 0
        self.buy_rejected_roi = 0
        
        # Сброс счетчиков действий
        self.action_counts = {0: 0, 1: 0, 2: 0}
        
        # Выбор случайной начальной точки с учетом длины эпизода
        episode_length = getattr(self.cfg, 'episode_length', 1000)  # По умолчанию 1000 шагов
        max_start = self.total_steps - episode_length
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
                    # Покупаем
                    buy_amount = self.balance * self.position_fraction
                    crypto_to_buy = buy_amount / current_price
                    self.crypto_held = crypto_to_buy
                    self.balance -= buy_amount
                    self.last_buy_price = current_price
                    self.last_buy_step = self.current_step
                    reward = 0.03  # Увеличил награду за успешную покупку с фильтрами
                    self._log(f"[{self.current_step}] 🔵 BUY: {crypto_to_buy:.4f} at {current_price:.2f}")
                else:
                    reward = -0.002  # Уменьшил штраф за отклонение фильтрами
            else:
                reward = -0.01  # Уменьшил штраф за попытку купить при наличии позиции
                
        elif action == 2:  # SELL
            if self.crypto_held > 0:  # Только если держим криптовалюту
                # Продаем
                sell_amount = self.crypto_held * current_price
                self.balance += sell_amount
                
                # Рассчитываем прибыль/убыток
                pnl = (current_price - self.last_buy_price) / self.last_buy_price
                net_profit_loss = sell_amount - (self.crypto_held * self.last_buy_price)
                
                # Награда за продажу (как в оригинале)
                reward += np.tanh(pnl * 25) * 2  # За результат сделки
                
                # Дополнительные награды за качество сделки
                if pnl > 0.05:  # Прибыль > 5%
                    reward += 0.3  # Увеличил бонус за хорошую сделку с фильтрами
                elif pnl < -0.03:  # Убыток > 3%
                    reward -= 0.01  # Уменьшил штраф за большой убыток
                
                # Записываем сделку
                self.trades.append({
                    "roi": pnl,
                    "net": net_profit_loss,
                    "reward": reward,
                    "duration": (self.current_step - self.last_buy_step) * 5 if self.last_buy_step else 0
                })
                
                self.crypto_held = 0
                self.last_buy_price = None
                self.last_buy_step = None
                self.trailing_stop_counter = 0
                self.max_price_during_hold = None
                
                self._log(f"[{self.current_step}] 🔴 SELL: {sell_amount:.2f}, PnL: {pnl:.2%}")
            else:
                reward = -0.01  # Уменьшил штраф за попытку продать без позиции
        
        # Обработка HOLD действия (как в оригинале)
        if action == 0:
            if self.crypto_held > 0 and self.last_buy_price is not None:
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
                    if self.trailing_stop_counter >= 3:
                        reward -= 0.03
                        self._log(f"[{self.current_step}] 🔻 TRAILING STOP — SELL by drawdown: {drawdown:.2%}")
                        self._force_sell(current_price, 'TRAILING STOP')
                        
                        # Обновляем статистики
                        self._update_stats(current_price)
                        self.current_step += 1
                        
                        # Проверяем завершение эпизода
                        episode_length = getattr(self.cfg, 'episode_length', 1000)
                        done = (
                            self.current_step >= self.start_step + episode_length or
                            self.current_step >= self.total_steps
                        )
                        
                        info.update({
                            "current_balance": self.balance,
                            "current_price": current_price,
                            "total_profit": (self.balance + self.crypto_held * current_price) - getattr(self.cfg, 'initial_balance', 10000.0),
                            "reward": reward
                        })
                        return self._get_state(), reward, done, info
                
                # --- Take Profit / Stop Loss (как в оригинале) ---
                unrealized_pnl_percent = (current_price - self.last_buy_price) / self.last_buy_price
                
                if unrealized_pnl_percent <= self.STOP_LOSS_PCT:
                    reward -= 0.05  # штраф за стоп-лосс
                    self._force_sell(current_price, "❌ - STOP-LOSS triggered")
                    
                elif unrealized_pnl_percent >= self.TAKE_PROFIT_PCT:
                    reward += 0.05  # поощрение за фиксацию профита
                    self._force_sell(current_price, "🎯 - TAKE-PROFIT hit")
                
                # --- Награды за удержание позиции (как в оригинале) ---
                if unrealized_pnl_percent > 0:
                    # Чем выше нереализованная прибыль, тем больше награда за удержание
                    reward += unrealized_pnl_percent * 2  # Уменьшил множитель с 5 до 2
                else:
                    # Чем больше нереализованный убыток, тем больше штраф за удержание
                    reward += unrealized_pnl_percent * 3  # Уменьшил множитель с 10 до 3
            else:
                # Если action == HOLD и нет открытой позиции
                reward = 0.001  # Небольшое поощрение за разумное бездействие вместо штрафа
        
        # Обновляем статистики
        self._update_stats(current_price)
        
        # Небольшая награда за активность торговли (поощряем исследование)
        if action != 0:  # Если действие не HOLD
            reward += 0.001  # Небольшое поощрение за активность
        
        # Переходим к следующему шагу
        self.current_step += 1
        
        # Проверяем завершение эпизода
        episode_length = getattr(self.cfg, 'episode_length', 1000)  # По умолчанию 1000 шагов
        
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
                
                self.trades.append({
                    "roi": pnl,
                    "net": net_profit_loss,
                    "reward": 0,
                    "duration": (self.current_step - self.last_buy_step) * 5 if self.last_buy_step else 0
                })
            
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
        
        return self._get_state(), reward, done, info

    def _check_buy_filters(self) -> bool:
        """
        Проверяет фильтры для покупки
        """
        self.buy_attempts += 1
        
        # ВКЛЮЧАЕМ ФИЛЬТРЫ ДЛЯ УЛУЧШЕНИЯ КАЧЕСТВА СДЕЛОК
        
        # 1. Проверка объема - ВКЛЮЧЕН с мягким порогом
        current_volume = self.df_5min[self.current_step - 1, 4]
        vol_relative = calc_relative_vol_numpy(self.df_5min, self.current_step - 1, 12)
        
        if vol_relative < 0.002:  # Мягкий порог 0.5% вместо 1%
            self.buy_rejected_vol += 1
            if self.current_step % 100 == 0:
                print(f"🔍 Фильтр объема: vol_relative={vol_relative:.4f} < 0.005, отклонено")
            return False
        
        # 2. Проверка ROI - ВКЛЮЧЕН
        if len(self.roi_buf) > 0:
            recent_roi_mean = np.mean(list(self.roi_buf))
            if recent_roi_mean < -0.03:  # Мягкий порог -3% вместо -2%
                self.buy_rejected_roi += 1
                if self.current_step % 100 == 0:
                    print(f"🔍 Фильтр ROI: recent_roi_mean={recent_roi_mean:.4f} < -0.03, отклонено")
                return False
        
        # Все фильтры пройдены - разрешаем покупку
        return True

    def _update_stats(self, current_price: float):
        """
        Обновляет статистики
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

    def _force_sell(self, current_price: float, reason: str):
        """
        Принудительная продажа (как в оригинале)
        """
        if self.crypto_held > 0:
            sell_amount = self.crypto_held * current_price
            self.balance += sell_amount
            
            pnl = (current_price - self.last_buy_price) / self.last_buy_price
            net_profit_loss = sell_amount - (self.crypto_held * self.last_buy_price)
            
            self.trades.append({
                "roi": pnl,
                "net": net_profit_loss,
                "reward": 0,
                "duration": (self.current_step - self.last_buy_step) * 5 if self.last_buy_step else 0
            })
            
            self.crypto_held = 0
            self.last_buy_price = None
            self.last_buy_step = None
            self.trailing_stop_counter = 0
            self.max_price_during_hold = None
            
            self._log(f"[{self.current_step}] 🔴 FORCE SELL ({reason}): {sell_amount:.2f}, PnL: {pnl:.2%}")

    def _log(self, *args, **kwargs):
        if self._can_log:
            print(*args, **kwargs)

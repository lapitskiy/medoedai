import gym
from gym import spaces
import numpy as np
import pandas as pd
import pandas_ta as ta
import random
from sklearn.preprocessing import StandardScaler # Добавлено для нормализации

class CryptoTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, dfs: dict, lookback_window: int = 20, indicators_config=None):
        super(CryptoTradingEnv, self).__init__()
 
         # Константы окружения
        self.episode_length = 500 # Длина каждого эпизода в 5-минутных свечах
        self.initial_balance = 1000 # Начальный баланс в USDT
        self.trade_fee_percent = 0.00075 # Комиссия 0.075%        
        self.df_5min = dfs['df_5min']
        self.df_15min = dfs['df_15min']
        self.df_1h = dfs['df_1h']                        
        self.total_steps = len(self.df_5min)        
 
        if indicators_config is None:
            self.indicators_config = {
            'rsi': {'length': 14},
            'ema': {'lengths': [100, 200]},   # несколько EMA
            'ema_cross': {                       # индикаторы пересечения EMA
                'pairs': [(100, 200)], # пары EMA для проверки пересечения
                'include_cross_signal': True    # включить кроссоверы (пересечения)
                        },
            'sma': {'length': 14},
            }
        else:
            self.indicators_config = indicators_config                
        
                       # Рассчитываем индикаторы при инициализации окружения
        self.indicators = self._calculate_indicators()        
        
        
        self.lookback_window = lookback_window 
        
        self.action_space = spaces.Discrete(3) # 0: HOLD, 1: BUY, 2: SELL
        
        num_features_per_candle = 5 # Open, High, Low, Close, Volume        
        num_indicator_features = self.indicators.shape[1]  # динамически
        
        max_15min_candles = self.lookback_window // 3
        max_1h_candles = self.lookback_window // 12
        
        
        self.observation_space_shape = (
            self.lookback_window * num_features_per_candle +
            max_15min_candles * num_features_per_candle +
            max_1h_candles * num_features_per_candle +
            num_indicator_features +
            2 # Добавляем для normalized_balance и normalized_crypto_held
        )

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
        self.history = None

        # Рассчитываем статистики нормализации
        self._calculate_normalization_stats() 
        
        
        # Добавляем скалеры для баланса и crypto_held
        # Их нужно инициализировать после того, как initial_balance определен
        # Эти скалеры будут использоваться для нормализации текущего баланса/крипты в _get_state
        
    
        
        
        self.balance_scaler = StandardScaler()
        self.crypto_held_scaler = StandardScaler()
                        
        min_price, max_price = self.calculate_price_ranges(self.df_5min, self.df_15min, self.df_1h)

        # Теперь считаем, какое количество крипты ты мог купить на initial_balance
        min_crypto = self.initial_balance / max_price  # по самой высокой цене — наименьшее количество
        max_crypto = self.initial_balance / min_price  # по самой низкой цене — максимум, что можно купить

        # Баланс: просто берем статистику движений, допустим, от 0 до 2 * initial_balance
        balance_range = np.linspace(0, 2 * self.initial_balance, 100).reshape(-1, 1)
        self.balance_scaler.fit(balance_range)

        crypto_range = np.linspace(min_crypto, max_crypto, 100).reshape(-1, 1)
        self.crypto_held_scaler.fit(crypto_range)
        print(f"[Scaler Init] min_crypto: {min_crypto:.6f}, max_crypto: {max_crypto:.6f}")
        print(f"[Scaler Init] min_price: {min_price:.2f}, max_price: {max_price:.2f}")


    def _calculate_indicators(self):
        df = self.df_5min.copy()
        features = []

        # RSI
        if 'rsi' in self.indicators_config:
            length_rsi = self.indicators_config['rsi'].get('length', 14)
            df[f'RSI_{length_rsi}'] = ta.rsi(df['close'], length=length_rsi)
            features.append(f'RSI_{length_rsi}')

        # EMA
        if 'ema' in self.indicators_config and 'lengths' in self.indicators_config['ema']:
            for length in self.indicators_config['ema']['lengths']:
                col = f'EMA_{length}'
                df[col] = ta.ema(df['close'], length=length)
                features.append(col)

        # EMA Cross (если определено)
        if 'ema_cross' in self.indicators_config and 'pairs' in self.indicators_config['ema_cross']:
            for short_len, long_len in self.indicators_config['ema_cross']['pairs']:
                short_col = f'EMA_{short_len}'
                long_col = f'EMA_{long_len}'

                # Проверка на наличие колонок EMA
                if short_col not in df.columns or long_col not in df.columns:
                    print(f"ВНИМАНИЕ: Для EMA кроссовера {short_col} и/или {long_col} не найдены.")
                    continue

                # Признак: короткая EMA выше длинной EMA (булево)
                above_col = f'EMA_{short_len}_above_{long_len}'
                df[above_col] = (df[short_col] > df[long_col]).astype(float)
                features.append(above_col)

                # Пересечение снизу вверх (bullish crossover)
                cross_up_col = f'EMA_{short_len}_cross_up_{long_len}'
                df[cross_up_col] = (
                    (df[short_col] > df[long_col]) & 
                    (df[short_col].shift(1) <= df[long_col].shift(1))
                ).astype(float)
                features.append(cross_up_col)

                # Пересечение сверху вниз (bearish crossover)
                cross_down_col = f'EMA_{short_len}_cross_down_{long_len}'
                df[cross_down_col] = (
                    (df[short_col] < df[long_col]) & 
                    (df[short_col].shift(1) >= df[long_col].shift(1))
                ).astype(float)
                features.append(cross_down_col)

        # Заполняем NaN нулями и возвращаем как numpy array float32
        indicators = df[features].fillna(0).values.astype(np.float32)
        return indicators
    
    def _calculate_normalization_stats(self):
        """
        Рассчитывает статистики нормализации (среднее и стандартное отклонение) для каждого признака
        по всем историческим данным. Это критично для согласованного масштабирования.
        """
        print("Начинаю расчет статистик для нормализации...")


        for df_name, df in zip(['5min', '15min', '1h'], [self.df_5min, self.df_15min, self.df_1h]):
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col not in df.columns:
                    raise ValueError(f"Отсутствует колонка '{col}' в df_{df_name}")

        if self.indicators is None or len(self.indicators) == 0:
            raise ValueError("Индикаторы не инициализированы — self.indicators пустой или None")


        # 1. Собираем все цены OHLC (open, high, low, close) из всех DF
        all_prices = np.concatenate([
            self.df_5min[['open', 'high', 'low', 'close']].values.flatten(),
            self.df_15min[['open', 'high', 'low', 'close']].values.flatten(),
            self.df_1h[['open', 'high', 'low', 'close']].values.flatten()
        ]).astype(np.float32)
        self.price_mean = np.mean(all_prices)
        self.price_std = np.std(all_prices) + 1e-8 # Добавляем эпсилон для стабильности

        # 2. Собираем все объемы из всех DF
        all_volumes = np.concatenate([
            self.df_5min['volume'].values.flatten(),
            self.df_15min['volume'].values.flatten(),
            self.df_1h['volume'].values.flatten()
        ]).astype(np.float32)
        self.volume_mean = np.mean(all_volumes)
        self.volume_std = np.std(all_volumes) + 1e-8

        # 3. Собираем индикаторы (предполагая self.indicators имеет форму (количество_строк, 3))
        # Каждый столбец индикаторов нормализуется отдельно.
        self.indicator_means = np.mean(self.indicators, axis=0) # Среднее для каждого из 3 индикаторов
        self.indicator_stds = np.std(self.indicators, axis=0) + 1e-8
        # Обработка нулевого стандартного отклонения
        self.indicator_stds[self.indicator_stds == 0] = 1e-8 
        
        print("Расчет статистик для нормализации завершен.")
        print(f"Средняя цена: {self.price_mean:.4f}, std: {self.price_std:.4f}")
        print(f"Средний объем: {self.volume_mean:.4f}, std: {self.volume_std:.4f}")
        print(f"Индикаторы: среднее {self.indicator_means}, std {self.indicator_stds}")

    def _get_state(self) -> np.ndarray:
        """
        Формирует текущий вектор состояния для агента.
        """
        current_5min_candle_idx = self.current_step - 1 
        
        if current_5min_candle_idx < self.lookback_window - 1 or current_5min_candle_idx >= self.total_steps:
            # В этом случае возвращаем паддированное состояние
            # Это должно быть предотвращено правильной логикой в reset(),
            # но это запасной вариант для устойчивости.
            # print(f"Warning: Insufficient data for lookback_window at current_step {self.current_step}. Padding state.")
            return np.zeros(self.observation_space_shape, dtype=np.float32)
    
        
        start_5min_idx = current_5min_candle_idx - self.lookback_window + 1
        # Убедимся, что start_5min_idx не отрицательный (хотя reset() должен это гарантировать)
        start_5min_idx = max(0, start_5min_idx) 
        state_5min_raw = self.df_5min.iloc[start_5min_idx : current_5min_candle_idx + 1].values
        
        # --- 15-минутные данные ---
        # Количество 15-минутных свечей в окне lookback_window
        num_15min_candles_in_window = self.lookback_window // 3
        # Индекс последней завершенной 15-минутной свечи, соответствующей текущей 5-минутной
        last_completed_15min_candle_idx = current_5min_candle_idx // 3 
        
        start_15min_idx_for_window = last_completed_15min_candle_idx - num_15min_candles_in_window + 1
        end_15min_idx_for_window = last_completed_15min_candle_idx + 1 

        # Проверка и паддинг, если недостаточно данных для 15-минутного окна
        if start_15min_idx_for_window < 0 or end_15min_idx_for_window > len(self.df_15min):
            state_15min_raw = np.zeros((num_15min_candles_in_window, state_5min_raw.shape[1]), dtype=np.float32)
        else:
            state_15min_raw = self.df_15min.iloc[start_15min_idx_for_window : end_15min_idx_for_window].values
        
  
       # --- 1-часовые данные ---
        num_1h_candles_in_window = self.lookback_window // 12
        last_completed_1h_candle_idx = current_5min_candle_idx // 12
        start_1h_idx_for_window = last_completed_1h_candle_idx - num_1h_candles_in_window + 1
        end_1h_idx_for_window = last_completed_1h_candle_idx + 1 
        # Проверка и паддинг, если недостаточно данных для 1-часового окна
        if start_1h_idx_for_window < 0 or end_1h_idx_for_window > len(self.df_1h):
            state_1h_raw = np.zeros((num_1h_candles_in_window, state_5min_raw.shape[1]), dtype=np.float32)
        else:
            state_1h_raw = self.df_1h.iloc[start_1h_idx_for_window : end_1h_idx_for_window].values
      # --- Текущие индикаторы ---
        # Убедимся, что current_5min_candle_idx не выходит за пределы self.indicators
        if current_5min_candle_idx < 0 or current_5min_candle_idx >= len(self.indicators):
            current_indicators_raw = np.zeros(self.indicators.shape[1], dtype=np.float32)
        else:
            current_indicators_raw = self.indicators[current_5min_candle_idx] 

        # --- Применяем нормализацию к "сырым" данным ---
        # Вспомогательная функция для нормализации OHLCV данных
        def normalize_ohlcv(raw_data, price_mean, price_std, volume_mean, volume_std, has_timestamp_col):
            if raw_data.size == 0: # Если массив пустой (из-за паддинга нулями)
                # Возвращаем пустой массив, который будет обработан np.concatenate
                return np.array([], dtype=np.float32) 
            
            if has_timestamp_col:
                # Если первая колонка - временная метка, берем со 2 по 6
                numeric_data = raw_data[:, 1:6]
            else:
                # Иначе берем первые 5 колонок
                numeric_data = raw_data[:, :5]
            
            prices = (numeric_data[:, :4] - price_mean) / price_std
            volumes = (numeric_data[:, 4] - volume_mean) / volume_std
            return np.concatenate((prices, volumes[:, np.newaxis]), axis=1).flatten()

        # Определяем, есть ли колонка с временной меткой для каждого DataFrame
        # Это должно быть определено в __init__ и быть константой
        has_timestamp_5min = self.df_5min.columns[0].lower() in ['timestamp', 'date', 'datetime']
        has_timestamp_15min = self.df_15min.columns[0].lower() in ['timestamp', 'date', 'datetime']
        has_timestamp_1h = self.df_1h.columns[0].lower() in ['timestamp', 'date', 'datetime']

        state_5min = normalize_ohlcv(state_5min_raw, self.price_mean, self.price_std, self.volume_mean, self.volume_std, has_timestamp_5min)
        state_15min = normalize_ohlcv(state_15min_raw, self.price_mean, self.price_std, self.volume_mean, self.volume_std, has_timestamp_15min)
        state_1h = normalize_ohlcv(state_1h_raw, self.price_mean, self.price_std, self.volume_mean, self.volume_std, has_timestamp_1h)
        
        # Нормализация индикаторов
        current_indicators = (current_indicators_raw - self.indicator_means) / self.indicator_stds
        current_indicators = current_indicators.flatten() # Убеждаемся, что это 1D-массив

        # --- Добавляем баланс и количество криптовалюты в состояние ---
        # Важно: нормализовать эти значения!
        # Используем StandardScaler, который мы инициализировали в __init__
        normalized_balance = self.balance_scaler.transform(np.array([[self.balance]]))[0][0]
        # Если self.crypto_held - это количество монет, то его тоже можно нормализовать
        # Например, относительно максимально возможного количества или начального баланса
        # Сейчас у нас есть placeholder scaler
        normalized_crypto_held = self.crypto_held_scaler.transform(np.array([[self.crypto_held]]))[0][0]
        
        # Объединяем все нормализованные части в конечный вектор состояния
        state_parts = [
            state_5min,
            state_15min,
            state_1h,
            current_indicators,
            np.array([normalized_balance, normalized_crypto_held]) # Добавлены нормализованные баланс и крипта
        ]
        
        # Если какой-либо из массивов state_Xmin пуст (из-за паддинга в normalize_ohlcv),
        # np.concatenate корректно обработает np.array([])
        state = np.concatenate(state_parts).astype(np.float32)

        # --- Обработка несоответствия размера состояния (паддинг) ---
        # Если фактический размер состояния меньше ожидаемого (из-за отсутствия данных в начале),
        # дополняем нулями.
        if state.shape[0] < self.observation_space_shape:
            padded_state = np.zeros(self.observation_space_shape, dtype=np.float32)
            padded_state[:state.shape[0]] = state
            state = padded_state
        elif state.shape[0] > self.observation_space_shape:
            # Это должно произойти только при ошибке логики
            raise ValueError(f"Размер состояния ({state.shape[0]}) превышает ожидаемый ({self.observation_space_shape}). Проверьте логику _get_state().")

        return state

    def step(self, action: int):
        self.current_step += 1  # Переходим на следующую 5-минутную свечу

        # Проверяем конец эпизода по длине
        # Важно: текущий шаг должен быть строго меньше total_steps, иначе iloc может выдать ошибку
        # Также проверяем, что у нас есть данные для следующих таймфреймов
        done = self.current_step >= self.start_step + self.episode_length or \
               self.current_step >= self.total_steps or \
               (self.current_step // 3) >= len(self.df_15min) or \
               (self.current_step // 12) >= len(self.df_1h)

        reward = 0.0
        info = {}
        
        # Получаем текущую цену закрытия (для предыдущей свечи, так как решение принимается после ее закрытия)
        # current_step - 1, потому что current_step - это следующий шаг, а данные - по предыдущему
        current_price_idx = max(0, self.current_step - 1)
        if current_price_idx >= len(self.df_5min):
            # Если мы вышли за пределы данных, то цена недействительна.
            # Это должно быть обработано 'done' условием выше.
            # Но на всякий случай, чтобы избежать IndexError.
            current_price = self.df_5min.iloc[-1]['close'] # Берем последнюю известную цену
        else:
            current_price = self.df_5min.iloc[current_price_idx]['close']


        # Логика действий и вознаграждения
        if not done:
            if action == 1: # BUY
                if self.balance > 0 and self.crypto_held == 0: # Покупаем, только если нет крипты
                    amount_to_buy = self.balance / current_price 
                    fee = amount_to_buy * current_price * self.trade_fee_percent
                    self.crypto_held += amount_to_buy * (1 - self.trade_fee_percent) 
                    self.balance = 0 
                    self.last_buy_price = current_price # Запоминаем цену покупки
                    self.history.append({'step': self.current_step, 'action': 'BUY', 'price': current_price, 'amount': amount_to_buy})
                else:
                    self.history.append({'step': self.current_step, 'action': 'INVALID_BUY_ATTEMPT'})
                                            
            elif action == 2:  # SELL                                     
                if self.crypto_held > 0:
                    amount_to_sell = self.crypto_held                                                             
                    fee = amount_to_sell * current_price * self.trade_fee_percent                                        
                    
                    self.balance += amount_to_sell * current_price * (1 - self.trade_fee_percent) 
                    self.crypto_held = 0 

                    if self.last_buy_price is not None:        
                        profit_on_trade = (current_price - self.last_buy_price) * amount_to_sell  # ← добавлено
                        net_profit = profit_on_trade - fee  # ← добавлено                                                                   
                        if net_profit < 0:  # ← изменено: проверяем прибыль с учётом комиссии
                            reward += net_profit / self.initial_balance * 10  # ← изменено: штрафуем по чистому убытку
                            reward -= 100  # ← оставил дополнительный штраф, можно регулировать
                        else:
                            reward += net_profit / self.initial_balance * 100  # ← изменено: награда без штрафа за комиссию


                    self.last_buy_price = None
                    self.history.append({
                        'step': self.current_step,
                        'action': 'SELL',
                        'price': current_price,
                        'amount': amount_to_sell
                    })

                else:
                    reward -= 10  # Штраф за невалидную продажу

            # Награда за удержание (или штраф за бездействие)
            # Если не было покупки/продажи (action == 0), или если действие было невалидным
            # Можно добавить небольшую награду/штраф за "HOLD"
            if action == 0:
                #reward += 0.01 # Очень маленькая награда за "HOLD"
                reward -= 0.0001

            # Награда за нереализованную прибыль/убыток (можно использовать, но осторожно)
            # Это может сделать функцию награды слишком "шумной", но может помочь агенту видеть текущий PnL
            # if self.crypto_held > 0 and self.last_buy_price is not None:
            #     unrealized_profit_loss = (current_price - self.last_buy_price) * self.crypto_held
            #     reward += unrealized_profit_loss / self.initial_balance * 0.1 # Масштабируем и даем небольшой вес

            if self.crypto_held > 0 and self.last_buy_price is not None:
                # Награда за нереализованную прибыль (unrealized PnL)
                current_value_held = self.crypto_held * current_price
                # PnL относительно начального баланса
                unrealized_profit_loss_relative = (current_value_held - (self.crypto_held * self.last_buy_price)) / self.initial_balance

                # Добавьте это к reward. Масштабируйте так, чтобы оно было значимым, но не доминировало
                # над наградой за закрытую сделку
                reward += unrealized_profit_loss_relative * 1.0 # Например, множитель 1.0 или 2.0

            next_state = self._get_state()

            info['current_balance'] = self.balance
            info['crypto_held'] = self.crypto_held
            info['current_price'] = current_price
            info['total_profit'] = (self.balance + self.crypto_held * current_price) - self.initial_balance
            info['reward'] = reward

        else: # done is True (эпизод завершился)
            final_value = self.balance + (self.crypto_held * current_price)
            profit_loss = final_value - self.initial_balance
            reward = profit_loss / self.initial_balance * 1000  # нормализованная награда
            # Общая стоимость активов в конце эпизода
            
            # Обновляем информацию для логирования
            self.total_profit = profit_loss 
            self.current_balance = final_value
            
            # Состояние для конечного шага
            # Если done=True, обычно возвращают нулевое состояние или последнее валидное состояние.
            # Если next_state будет использоваться для обучения, он должен быть валидным.
            # В данном случае, _get_state() уже обрабатывает граничные условия.
            next_state = self._get_state()
            
            info['total_profit'] = profit_loss 
            info['current_balance'] = final_value
            info['crypto_held'] = self.crypto_held 
            info['current_price'] = current_price
            info['reward'] = reward

        return next_state, reward, done, info

    def reset(self) -> np.ndarray:
        # Управление стартовым шагом для последовательного прохода по данным
        # Мы хотим убедиться, что стартовый шаг позволяет извлечь полный lookback_window
        # и что есть достаточно данных для всего episode_length.
        
        # Если self.current_train_start_idx уже не позволяет завершить эпизод
        # (т.е. текущий индекс + длина эпизода превышает общее количество шагов)
        # или если мы приближаемся к концу df_5min
        if self.current_train_start_idx + self.episode_length >= self.total_steps:
            # Сброс к минимально возможному начальному индексу,
            # который позволяет получить полный lookback_window.
            self.current_train_start_idx = self.min_valid_start_step 
        
        self.start_step = self.current_train_start_idx
        self.current_step = self.start_step

        # Обновляем self.current_train_start_idx для следующего эпизода
        self.current_train_start_idx += self.episode_length
        # Если после увеличения мы вышли за пределы, сбросим на начало для следующего цикла
        if self.current_train_start_idx >= self.total_steps:
             self.current_train_start_idx = self.min_valid_start_step

        # Сброс состояния торгового агента
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.last_buy_price = None
        self.history = []

        return self._get_state()

    def render(self, mode='human'):
        # Убедимся, что current_step валиден для iloc
        current_price_idx = max(0, self.current_step - 1)
        if current_price_idx >= len(self.df_5min):
            current_price = self.df_5min.iloc[-1]['close'] # Берем последнюю известную цену
        else:
            current_price = self.df_5min.iloc[current_price_idx]['close']

        if len(self.history) == 0:
            print(f"Шаг: {self.current_step}, Баланс: {self.balance:.2f} USDT, Крипты: {self.crypto_held:.6f}, Цена: {current_price:.2f}, Действия пока нет")
        else:
            last_action = self.history[-1]
            print(f"Шаг: {self.current_step}, Баланс: {self.balance:.2f} USDT, Крипты: {self.crypto_held:.6f}, Цена: {current_price:.2f}, Последнее действие: {last_action['action']} по цене {last_action['price']:.2f} на количестве {last_action['amount']:.6f}")

    def close(self):
        pass
    
    @staticmethod
    def calculate_price_ranges(df_5min, df_15min, df_1h):
        """
        Рассчитывает минимальную и максимальную цену закрытия по объединённым данным разных таймфреймов.
        Убирает пропуски и конкатенирует все close цены.
        """
        all_close_prices = np.concatenate([
            df_5min['close'].dropna().values,
            df_15min['close'].dropna().values,
            df_1h['close'].dropna().values
        ])
        
        min_price = np.min(all_close_prices)
        max_price = np.max(all_close_prices)
        return min_price, max_price    

# Регистрация окружения
from gym.envs.registration import register 

register(
    id='CryptoTradingEnv-v0',
    entry_point='model.dqn_model.gym.crypto_trading_env:CryptoTradingEnv', 
)
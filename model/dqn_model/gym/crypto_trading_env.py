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
 
        self.episode_length = 200
 
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
        
        
        self.df_5min = dfs['df_5min']
        self.df_15min = dfs['df_15min']
        self.df_1h = dfs['df_1h']
        
        self.max_steps = len(self.df_5min) - 1
        
                       # Рассчитываем индикаторы при инициализации окружения
        self.indicators = self._calculate_indicators()        
        
        
        self.lookback_window = lookback_window 
        
        self.action_space = spaces.Discrete(3) # 0: HOLD, 1: BUY, 2: SELL
        
        num_features_per_candle = 5 # Open, High, Low, Close, Volume
        
        num_indicator_features = self.indicators.shape[1]  # динамически
        self.observation_space_shape = \
            self.lookback_window * num_features_per_candle + \
            (self.lookback_window // 3) * num_features_per_candle + \
            (self.lookback_window // 12) * num_features_per_candle + \
            num_indicator_features

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(self.observation_space_shape,), 
                                            dtype=np.float32)
        
        
             # Для последовательного обучения
        self.current_train_start_idx = self.lookback_window # Начинаем после lookback_window

        # Эти будут инициализированы в reset
        self.start_step = None
        self.current_step = None
        self.balance = None
        self.crypto_held = None
        self.last_buy_price = None
        self.history = None
        
        self.total_steps = len(self.df_5min)

        self.initial_balance = 1000 # Начальный баланс в USDT
        
        self.trade_fee_percent = 0.00075 # Комиссия 0.075%
        
        
        # Рассчитываем статистики нормализации
        self._calculate_normalization_stats() 

    def _calculate_indicators(self):
        df = self.df_5min.copy()
        features = []

        # RSI
        length_rsi = 14
        df['RSI'] = ta.rsi(df['close'], length=length_rsi)
        features.append('RSI')

        # EMA 100 и EMA 200
        ema_lengths = [100, 200]
        for length in ema_lengths:
            col = f'EMA_{length}'
            df[col] = ta.ema(df['close'], length=length)
            features.append(col)

        short_col = 'EMA_100'
        long_col = 'EMA_200'

        # Признак: EMA_100 выше EMA_200 (булево)
        above_col = 'EMA_100_above_200'
        df[above_col] = (df[short_col] > df[long_col]).astype(float)
        features.append(above_col)

        # Пересечение снизу вверх (bullish crossover)
        cross_up_col = 'EMA_100_cross_up_200'
        df[cross_up_col] = (
            (df[short_col] > df[long_col]) & 
            (df[short_col].shift(1) <= df[long_col].shift(1))
        ).astype(float)
        features.append(cross_up_col)

        # Пересечение сверху вниз (bearish crossover)
        cross_down_col = 'EMA_100_cross_down_200'
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
        
        print("Расчет статистик для нормализации завершен.")


    def _get_state(self) -> np.ndarray:
        """
        Формирует текущий вектор состояния для агента.
        """
        current_5min_candle_idx = self.current_step - 1 
        
        # --- 5-минутные данные ---
        state_5min_raw = self.df_5min.iloc[current_5min_candle_idx - self.lookback_window + 1 : current_5min_candle_idx + 1].values
        
        # --- 15-минутные данные ---
        num_15min_candles_in_window = self.lookback_window // 3
        last_completed_15min_candle_idx = current_5min_candle_idx // 3 
        start_15min_idx_for_window = max(0, last_completed_15min_candle_idx - num_15min_candles_in_window + 1)
        end_15min_idx_for_window = last_completed_15min_candle_idx + 1 
        state_15min_raw = self.df_15min.iloc[start_15min_idx_for_window : end_15min_idx_for_window].values
        
        # --- 1-часовые данные ---
        num_1h_candles_in_window = self.lookback_window // 12
        last_completed_1h_candle_idx = current_5min_candle_idx // 12
        start_1h_idx_for_window = max(0, last_completed_1h_candle_idx - num_1h_candles_in_window + 1)
        # ИСПРАВЛЕНИЕ ОПЕЧАТКИ: должно быть last_completed_1h_candle_idx, а не last_1h_candle_idx
        end_1h_idx_for_window = last_completed_1h_candle_idx + 1 
        state_1h_raw = self.df_1h.iloc[start_1h_idx_for_window : end_1h_idx_for_window].values
        
        # --- Текущие индикаторы ---
        # Индикаторы для 5-минутной свечи, которая только что завершилась
        current_indicators_raw = self.indicators[current_5min_candle_idx] 

        # --- Применяем нормализацию к "сырым" данным перед сглаживанием и конкатенацией ---
        # Это более точно, так как применяет правильные средние/стандартные отклонения для цен, объемов и индикаторов.
        
        # Убедитесь, что `state_Xmin_raw` имеет столбцы в порядке 'open', 'high', 'low', 'close', 'volume'
        
        # Нормализация 5-минутных данных
        # Предполагаем, что столбцы: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        # Если так, то берем только столбцы с 1 по 5 (open, high, low, close, volume)
        if self.df_5min.columns[0].lower() in ['timestamp', 'date', 'datetime']:
            state_5min_numeric = state_5min_raw[:, 1:6]
        else:
            state_5min_numeric = state_5min_raw[:, :5]
        state_5min_prices = (state_5min_numeric[:, :4] - self.price_mean) / self.price_std
        state_5min_volumes = (state_5min_numeric[:, 4] - self.volume_mean) / self.volume_std
        state_5min = np.concatenate((state_5min_prices, state_5min_volumes[:, np.newaxis]), axis=1).flatten()

        # Нормализация 15-минутных данных
        if self.df_15min.columns[0].lower() in ['timestamp', 'date', 'datetime']:
            state_15min_numeric = state_15min_raw[:, 1:6]
        else:
            state_15min_numeric = state_15min_raw[:, :5]
        state_15min_prices = (state_15min_numeric[:, :4] - self.price_mean) / self.price_std
        state_15min_volumes = (state_15min_numeric[:, 4] - self.volume_mean) / self.volume_std
        state_15min = np.concatenate((state_15min_prices, state_15min_volumes[:, np.newaxis]), axis=1).flatten()

        # Нормализация 1-часовых данных
        if self.df_1h.columns[0].lower() in ['timestamp', 'date', 'datetime']:
            state_1h_numeric = state_1h_raw[:, 1:6]
        else:
            state_1h_numeric = state_1h_raw[:, :5]
        state_1h_prices = (state_1h_numeric[:, :4] - self.price_mean) / self.price_std
        state_1h_volumes = (state_1h_numeric[:, 4] - self.volume_mean) / self.volume_std
        state_1h = np.concatenate((state_1h_prices, state_1h_volumes[:, np.newaxis]), axis=1).flatten()
        
        # Нормализация индикаторов (каждый столбец индикатора нормализуется своим средним/стандартным отклонением)
        current_indicators = (current_indicators_raw - self.indicator_means) / self.indicator_stds
        current_indicators = current_indicators.flatten() # Убеждаемся, что это 1D-массив

        # Объединяем все нормализованные части в конечный вектор состояния
        state = np.concatenate([
            state_5min,
            state_15min,
            state_1h,
            current_indicators
        ]).astype(np.float32)

        # --- Обработка несоответствия размера состояния (паддинг) ---
        if state.shape[0] < self.observation_space_shape:
            padded_state = np.zeros(self.observation_space_shape, dtype=np.float32)
            padded_state[:state.shape[0]] = state
            state = padded_state
        elif state.shape[0] > self.observation_space_shape:
            raise ValueError(f"Размер состояния ({state.shape[0]}) превышает ожидаемый ({self.observation_space_shape}). Проверьте логику _get_state().")

        return state

    def step(self, action: int):
        self.current_step += 1  # Переходим на следующую 5-минутную свечу

        # Проверяем конец эпизода по длине с учетом случайного старта
        done = self.current_step >= self.start_step + self.episode_length or \
            (self.current_step // 3) >= len(self.df_15min) or \
            (self.current_step // 12) >= len(self.df_1h)

        reward = 0.0
        info = {}
        
        # Получаем текущую цену закрытия (для предыдущей свечи, так как решение принимается после ее закрытия)
        current_price = self.df_5min.iloc[max(0, self.current_step - 1)]['close']

        if not done:
            if action == 1: # BUY
                if self.balance > 0 and self.crypto_held == 0: # Покупаем, только если нет крипты
                    amount_to_buy = self.balance / current_price 
                    fee = amount_to_buy * current_price * self.trade_fee_percent
                    self.crypto_held += amount_to_buy * (1 - self.trade_fee_percent) 
                    self.balance = 0 
                    self.last_buy_price = current_price # Запоминаем цену покупки
                    self.history.append({'step': self.current_step, 'action': 'BUY', 'price': current_price, 'amount': amount_to_buy})
                    reward -= fee # Штраф только за комиссию
                else:
                    reward -= 10 # Штраф за попытку купить без денег
            
            elif action == 2: # SELL                                                
                if self.crypto_held > 0:
                    amount_to_sell = self.crypto_held 
                    fee = amount_to_sell * current_price * self.trade_fee_percent
                    self.balance += amount_to_sell * current_price * (1 - self.trade_fee_percent) 
                    self.crypto_held = 0 
                    self.last_buy_price = None # Сбрасываем цену покупки после продажи
                    self.history.append({'step': self.current_step, 'action': 'SELL', 'price': current_price, 'amount': amount_to_sell})
                    reward -= fee # Небольшой штраф за комиссию
                    # Добавлено: Проверка, что last_buy_price не None
                    if self.last_buy_price is not None:
                        profit_on_trade = (current_price - self.last_buy_price) * amount_to_sell
                        # Используйте конкретное число вместо SOME_HIGH_MULTIPLIER, например 10
                        reward += profit_on_trade / self.initial_balance * 100
                    
                    self.last_buy_price = None # Сбрасываем цену покупки после продажи
                else:
                    reward -= 10 # Штраф за попытку продать без крипты                    

            # --- Расчет награды ---
            # Награда за нереализованную прибыль/убыток от открытых позиций
            #if self.crypto_held > 0 and self.last_buy_price is not None:
            #    # Награда основана на нереализованной прибыли/убытке с момента последней покупки
            #    unrealized_profit_loss = (current_price - self.last_buy_price) * self.crypto_held * 100
            #    reward += unrealized_profit_loss / self.initial_balance # Масштабируем по начальному балансу
            
            # Небольшой постоянный штраф за каждый шаг для поощрения более быстрого завершения
            #reward -= 0.00001 

            next_state = self._get_state()

            info['current_balance'] = self.balance
            info['crypto_held'] = self.crypto_held
            info['current_price'] = current_price
            info['total_profit'] = (self.balance + self.crypto_held * current_price) - self.initial_balance
            info['reward'] = reward

        else: # done is True
            
            # Общая стоимость активов в конце эпизода
            final_portfolio_value = self.balance + (self.crypto_held * current_price)
            # Чистая прибыль/убыток
            profit_loss = final_portfolio_value - self.initial_balance
            
            # Масштабируем и даем как финальное вознаграждение
            # Можно сделать его очень большим, чтобы оно доминировало
            reward += profit_loss * 1000 # Или 1000, 5000, в зависимости от масштаба ваших цен и баланса
                                       # Можно нормировать: profit_loss / self.initial_balance * 500
                                       # Начните с *500, если profit_loss - это уже абсолютная величина
                                       
            # Для логов:
            self.total_profit = profit_loss # Обновляем для вывода в консоль
            self.current_balance = final_portfolio_value # Обновляем для вывода в консоль
            
            # Расчет финального вознаграждения при завершении эпизода
            #if self.crypto_held > 0:
            #    self.balance += self.crypto_held * current_price * (1 - self.trade_fee_percent)
            #    self.crypto_held = 0
            
            #final_profit = self.balance - self.initial_balance
            
            # Финальное вознаграждение должно отражать общую производительность эпизода.
            # Вы можете масштабировать эту final_profit, чтобы она была значимым вознаграждением.
            #reward += final_profit / self.initial_balance * 1000 # Пример: масштабируем по начальному балансу и умножаем

            next_state = self._get_state()
            #next_state = np.zeros(self.observation_space_shape, dtype=np.float32)

            info['total_profit'] = profit_loss # Используйте profit_loss, а не final_profit
            info['current_balance'] = final_portfolio_value
            info['crypto_held'] = self.crypto_held # Должно быть 0, если агент сам продал. Если нет, то какая-то крипта.
            info['current_price'] = current_price
            info['reward'] = reward

        return next_state, reward, done, info

    def reset(self) -> np.ndarray:
        # Определяем начальный шаг для эпизода
        # Если оставшихся данных недостаточно для полного эпизода,
        # начинаем заново с начала доступного диапазона
        if self.current_train_start_idx + self.episode_length > self.total_steps:
            self.current_train_start_idx = self.lookback_window # Сброс к началу данных

        self.start_step = self.current_train_start_idx
        self.current_step = self.start_step

        # Обновляем self.current_train_start_idx для следующего эпизода
        self.current_train_start_idx += self.episode_length

        self.balance = self.initial_balance
        self.crypto_held = 0
        self.last_buy_price = None
        self.history = []

        return self._get_state()

    def render(self, mode='human'):
        if len(self.history) == 0:
            print(f"Step: {self.current_step}, Баланс: {self.balance:.2f} USDT, Крипты: {self.crypto_held:.6f}, Цена: {self.df_5min.iloc[self.current_step - 1]['close']:.2f}, Действия пока нет")
        else:
            last_action = self.history[-1]
            print(f"Step: {self.current_step}, Баланс: {self.balance:.2f} USDT, Крипты: {self.crypto_held:.6f}, Цена: {self.df_5min.iloc[self.current_step - 1]['close']:.2f}, Последнее действие: {last_action['action']} по цене {last_action['price']:.2f} на количестве {last_action['amount']:.6f}")

    def close(self):
        pass

# Регистрация окружения
from gym.envs.registration import register 

register(
    id='CryptoTradingEnv-v0',
    entry_point='model.dqn_model.gym.crypto_trading_env:CryptoTradingEnv', 
)
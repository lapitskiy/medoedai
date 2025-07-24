from envs.dqn_model.gym.gutils import calc_relative_vol, commission_penalty, update_roi_stats, update_vol_stats
import gym
from gym import spaces
import numpy as np
import pandas_ta as ta
import random
import torch
from sklearn.preprocessing import StandardScaler # Добавлено для нормализации
from envs.dqn_model.gym.gconfig import GymConfig
from typing import Optional
from collections import deque

class CryptoTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, dfs: dict, cfg: Optional[GymConfig] = None, lookback_window: int = 20, indicators_config=None):        
        super(CryptoTradingEnv, self).__init__() 
        self.cfg = cfg or GymConfig()
        
        self.epsilon = 1.0
        
        # PRINT LOG DOCKER
        self._episode_idx  = -1          # будет инкрементироваться в reset()
        self._log_interval = 20
        self._can_log      = False       # обновляется в reset()
 
        # фильтры «душат» покупки?
        self.buy_attempts      = 0
        self.buy_rejected_vol  = 0
        self.buy_rejected_roi  = 0
        
        
        self.vol_buf = deque(maxlen=self.cfg.window288)
        self.roi_buf = deque(maxlen=self.cfg.window288)
        
        # вход в позицию на 30%
        self.position_fraction = 0.30
 
         # Константы окружения
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
        self.last_buy_step = None    
        self.trailing_stop_counter = 0   

        # Рассчитываем статистики нормализации
        self._calculate_normalization_stats() 
        
        self.trades = []
        
        # Рассчитываем статистики нормализации
        self.low_volatility_warned = False
        self.low_roi_warned = False
        
        
        # Добавляем скалеры для баланса и crypto_held
        # Их нужно инициализировать после того, как initial_balance определен
        # Эти скалеры будут использоваться для нормализации текущего баланса/крипты в _get_state
        
    
        
        
        self.balance_scaler = StandardScaler()
        self.crypto_held_scaler = StandardScaler()
                        
        min_price, max_price = self.calculate_price_ranges(self.df_5min, self.df_15min, self.df_1h)

        # Теперь считаем, какое количество крипты ты мог купить на initial_balance
        min_crypto = self.cfg.initial_balance / max_price  # по самой высокой цене — наименьшее количество
        max_crypto = self.cfg.initial_balance / min_price  # по самой низкой цене — максимум, что можно купить

        # Баланс: просто берем статистику движений, допустим, от 0 до 2 * initial_balance
        balance_range = np.linspace(0, 2 * self.cfg.initial_balance, 100).reshape(-1, 1)
        self.balance_scaler.fit(balance_range)

        crypto_range = np.linspace(min_crypto, max_crypto, 100).reshape(-1, 1)
        self.crypto_held_scaler.fit(crypto_range)
        print(f"[Scaler Init] min_crypto: {min_crypto:.6f}, max_crypto: {max_crypto:.6f}")
        print(f"[Scaler Init] min_price: {min_price:.2f}, max_price: {max_price:.2f}")
 
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
        done = (
            self.current_step >= self.start_step + self.cfg.episode_length or
            self.current_step >= self.total_steps or
            (self.current_step // 3) + 1 > len(self.df_15min) or  # +1, т.к. end_15min_idx_for_window = last_completed_15min_candle_idx + 1
            (self.current_step // 12) + 1 > len(self.df_1h) or
            self.current_step >= len(self.df_5min)
        )

        reward = 0.0
        penalty = 0.0
        
        STOP_LOSS_PCT   = -0.03    # −3 %
        TAKE_PROFIT_PCT = +0.05    # +5 %
        
        info = {}
        
        # Получаем текущую цену закрытия (для предыдущей свечи, так как решение принимается после ее закрытия)
        # current_step - 1, потому что current_step - это следующий шаг, а данные - по предыдущему
        
        current_price_idx = min(self.start_step + self.cfg.episode_length - 1, max(0, self.current_step - 1))
        
        if current_price_idx >= len(self.df_5min):
            # Если мы вышли за пределы данных, то цена недействительна.
            # Это должно быть обработано 'done' условием выше.
            # Но на всякий случай, чтобы избежать IndexError.
            current_price = self.df_5min.iloc[-1]['close'] # Берем последнюю известную цену
        else:
            current_price = self.df_5min.iloc[current_price_idx]['close']


        if self.last_buy_step is not None:
            holding_period = self.current_step - self.last_buy_step  # в шагах
            holding_minutes = holding_period * 5  # если шаг — 5 минут                     
        else:            
            holding_period = 0
            holding_minutes = 0
                        
            
        net_profit_loss = 0.0
        if self.crypto_held > 0:
            amount_to_sell = self.crypto_held
            fee = amount_to_sell * current_price * self.trade_fee_percent            

            # Вычисляем чистую прибыль/убыток, если была предыдущая покупка
            if self.last_buy_price is not None:
                # Общая стоимость проданной крипты, если бы не было комиссии
                gross_sale_value = amount_to_sell * current_price
                # Изначальная стоимость покупки
                cost_of_purchase = amount_to_sell * self.last_buy_price # Предполагаем, что amount_to_buy был равен amount_to_sell
                
                # Чистая прибыль или убыток (стоимость продажи - стоимость покупки - комиссия)
                net_profit_loss = gross_sale_value - cost_of_purchase - fee            


        volatility = calc_relative_vol(self.df_5min, self.current_step, lookback=30)            
        median_vol, iqr_vol = update_vol_stats(volatility, self.vol_buf)
        
        k = 0.1 + 0.4 * self.epsilon          # линейная интерполяция
        volatility_threshold = median_vol + k * iqr_vol                         
    
                 

        # Логика действий и вознаграждения
        if not done:            
    
            if action == 1: # BUY                                
                                
                if self.balance > 0 and self.crypto_held == 0: # Покупаем, только если нет крипты                                        
                    
                    score  = 0.0                    

                    rsi_14 = self.df_5min['RSI_14'].iloc[self.current_step] 
                    ema_target = self.df_5min['EMA_100'].iloc[self.current_step]
                    expected_roi = max(0, ema_target - current_price) / current_price 
                    
                    self.buy_attempts += 1
                    
                     ## ---  плавный score для опредления позиции входа -----------------------------------------------------------     
                    # --- 1.  непрерывные факторы ---
                    delta_ema = (ema_target - current_price) / current_price          # ≈ ожидаемый ROI
                    norm_rsi  = (50 - rsi_14) / 50.0                                  # -1…1

                    # --- 2.  взвешивание ---
                    score = (
                        1.5 * delta_ema           +      # тренд / потенциал
                        0.8 * norm_rsi            +      # перепроданность
                        0.5 * (volatility / 0.01)        # чуть поощряем «живой» рынок
                    )

                    # --- 3.  squash в [0,1] ---
                    fraction = 0.1 + 0.4 * torch.sigmoid(torch.tensor(score)).item()   # 10‑50 % баланса                                                
                    # ------------------------------------------------------------------------                 
                        
                    
                    amount_to_buy = fraction * self.balance / current_price
                    cost = amount_to_buy * current_price                    
                    fee = cost * self.trade_fee_percent   
                    
                    if self.epsilon < 0.10:                        
                        if volatility < volatility_threshold:
                            self.buy_rejected_vol += 1
                            if not self.low_volatility_warned:                                                                                    
                                self._log(f"[{self.current_step}] 🚫 - LOW VOLATILITY — no BUY")
                                self.low_volatility_warned = True
                            return self._get_state(), reward, False, info
                        else:
                            self.low_volatility_warned = False
                        
                        # --- ROI gate -----------------------------------------------------------
                        #min_roi = 0.002 + 0.8 * volatility                                       
                        q75_roi = update_roi_stats(expected_roi, self.roi_buf)
                        min_roi = 0.5 * q75_roi 
                        
                        if expected_roi < min_roi:
                            reward += commission_penalty(fee, self.cfg.initial_balance)                      
                            self.buy_rejected_roi += 1
                            #if not self.low_roi_warned:                            
                            #    self._log(f"[{self.current_step}] 🚫 - LOW ROI {expected_roi:.3%} < {min_roi:.3%}")
                            #    self.low_roi_warned = True
                            #return self._get_state(), reward, False, info
                        #else:
                        #    self.low_roi_warned = False
                        
                        
                    # ------------------------------------------------------------------------
                    
                    
                    if self.df_5min['EMA_100_cross_up_200'].iloc[self.current_step] == 1.0:  # BUY
                        reward += 0.1  # стимулируем вход в потенциальный тренд
                    
                       
                    if rsi_14 > 80:
                        reward -= 0.03  # штраф за вход в перегретый рынок
                        
                    if rsi_14 < 30:
                        reward += 0.03  
                    
                    if expected_roi > 0.01:
                        reward += 0.03
                         
                    reward += self.combined_signal_reward(action=1, step=self.current_step)                                   
                    
                    self.last_buy_step = self.current_step                    
                    
                    if cost + fee > self.balance:
                        amount_to_buy = self.balance / (1 + self.trade_fee_percent) / current_price
                        cost  = amount_to_buy * current_price
                        fee   = cost * self.trade_fee_percent      
                        
                    self.balance -= cost + fee
                    self.crypto_held += amount_to_buy 
                    self.last_buy_price = current_price # Запоминаем цену покупки     
                                    
                    rsi_msg = self._log_rsi_signal(self.current_step, label="BUY")  # пусть возвращает строку, а не печатает
                    ema_msg = self._log_ema_signals(self.current_step, label="BUY")  # тоже возвращает строку

                    self._log(f"[{self.current_step}] 🔼  BUY: amount: {cost + fee:.2f}, price: {current_price:.2f}, reward: {reward:.2f}, {rsi_msg}, {ema_msg}")                                                                   

                    reward -= fee / self.cfg.initial_balance * k
                    
                    #reward -= fee * 10 # Штраф за комиссию, если она существенна
                else:
                    reward -= 0.05 # Это очень важно, чтобы агент учился избегать таких действий
            
                                            
            elif action == 2:  # SELL
                if self.crypto_held > 0:                                                            
                    
                    # Обновляем баланс и количество крипты
                    gross = amount_to_sell * current_price
                    fee   = gross * self.trade_fee_percent
                    self.balance += gross - fee
                    self.crypto_held = 0 

                    # Вычисляем чистую прибыль/убыток, если была предыдущая покупка
                    if self.last_buy_price is not None:                                                
                        
                                    # --- ROI / P&L -----------------------------------------------------------------
                        pnl = (current_price - self.last_buy_price) / self.last_buy_price  # ROI
                        net_profit_loss = pnl * amount_to_sell * self.last_buy_price
                        
                        
                        # --- reward ----------------------------------------------------
                        reward += np.tanh(pnl * 25) * 2             # за результат сделки
                        penalty = commission_penalty(fee, self.cfg.initial_balance)
                        reward += penalty             
                        # ---------------------------------------------------------------
                        
                        result = "✅ - PROFIT" if pnl > 0 else "LOSS"
                        
                        self.max_price_during_hold = None                        
                        self.trailing_stop_counter = 0

                        if self.last_buy_price < current_price * 0.1:
                            self._log(f"⚠️ Возможно старая цена покупки: last_buy_price={self.last_buy_price:.2f}, current_price={current_price:.2f}")                                                                  
                        
                        ema_msg = self._log_ema_signals(self.current_step, label="SELL")
                        rsi_msg = self._log_rsi_signal(self.current_step, label="SELL")                                            
                    
                        # награда за продажу по тренду вниз
                        if self.df_5min['EMA_100_cross_down_200'].iloc[self.current_step] == 1.0:  # SELL на пересечении вниз
                            reward += 0.05  # стимулируем выход из падающего тренда
                            
                        rsi_14 = self.df_5min['RSI_14'].iloc[self.current_step]
                        if rsi_14 > 70:
                            reward += 0.03 

                        if rsi_14 > 90:
                            reward += 0.05                        
                            
                        if rsi_14 < 40 and net_profit_loss < 0:
                            reward += 0.01  # поощри немного, что он не усугубил убыток    

                        # Продаёт при RSI < 20
                        if rsi_14 < 20:
                            if pnl < 0.01:
                                reward -= 0.03  # продал слишком рано, даже не заработал 1%
                            else:
                                reward += 0.01  # молодец, взял своё
                            
                        reward += self.combined_signal_reward(action=2, step=self.current_step)                                                                                                                                                
                        
                        info['net_profit'] = net_profit_loss                                                                    
                        self._log(f"[{self.current_step}] 🔒  SELL {result}: {net_profit_loss:.2f}, price: {current_price:.2f}, reward: {reward:.2f}, held {holding_minutes} min,  {rsi_msg}, {ema_msg}")                                                                
                    
                        self.trades.append({
                            "roi": pnl,
                            "net": net_profit_loss,
                            "reward": reward,
                            "duration": holding_minutes
                        })

                    else:
                        reward -= 0.2  # Продаем без покупки
                        self._log(f"[{self.current_step}] SELL (INVALID): reward: {reward:.2f}")
                    self.last_buy_price = None 
                else:
                    reward -= 0.05 # Штраф за невалидную продажу (нет крипты для продажи)

            # Награда за удержание (или штраф за бездействие)
            # Если не было покупки/продажи (action == 0), или если действие было невалидным
            # Можно добавить небольшую награду/штраф за "HOLD"
            if action == 0:
                # Этот блок вознаграждает (или штрафует) агента за "HOLD" при открытой позиции                                
                
                if self.crypto_held > 0 and self.last_buy_price is not None:
                    
                            # --- трейлинг-стоп ---
                    if self.epsilon <= 0.2:                  # фаза exploitation
                        # 1. обновляем максимум
                        if (not hasattr(self, "max_price_during_hold")
                            or self.max_price_during_hold is None
                            or self.last_buy_step == self.current_step):
                            self.max_price_during_hold = current_price

                        if current_price > self.max_price_during_hold:      # новый пик
                            self.max_price_during_hold = current_price
                            self.trailing_stop_counter = 0                  

                        # 2. считаем просадку от пика
                        drawdown = (self.max_price_during_hold - current_price) / self.max_price_during_hold
                        if drawdown > 0.02:
                            self.trailing_stop_counter += 1

                        # 3. три подряд бара с drawdown > 2 %  → принудительный SELL
                        if self.trailing_stop_counter >= 3:
                            reward -= 0.03
                            self._log(f"[{self.current_step}] 🔻 TRAILING STOP — SELL by drawdown: {drawdown:.2%}")

                            self._force_sell(current_price, 'TRAILING STOP')

                            next_state = self._get_state()
                            info.update({
                                "current_balance": self.balance,
                                "current_price":   current_price,
                                "total_profit":   (self.balance + self.crypto_held * current_price) - self.cfg.initial_balance,
                                "reward":          reward
                            })
                            return next_state, reward, done, info
                                                            
                    if holding_minutes > 180 and net_profit_loss < 0:                        
                        reward -= 0.03  # он держит минусовую сделку слишком долго
                                            
                    max_hold_steps = int(max(
                            self.cfg.window288,                # 24 ч как нижний лимит
                            (6*60/self.cfg.step_minutes) *     # 6 ч в шагах
                            max(1, volatility / median_vol)      # чем выше вола, тем длиннее разрешаем
                            ))
                                            
                    if holding_minutes >= max_hold_steps:
                        self._force_sell(current_price, 'MAX-HOLD 24h')
                        next_state = self._get_state()
                        info.update({
                            "current_balance": self.balance,
                            "current_price":   current_price,
                            "total_profit":    self.balance - self.cfg.initial_balance,
                            "reward":          reward
                        })
                        return next_state, reward, done, info
                    
                    unrealized_pnl_percent = (current_price - self.last_buy_price) / self.last_buy_price
                                        
                    # --- TP / SL ------------------------------------------------------
                    if unrealized_pnl_percent <= STOP_LOSS_PCT:
                        reward -= 0.05                       # небольшой штраф за стоп‑лосс
                        self._force_sell(current_price, label="❌ - STOP‑LOSS triggered")
                        ...

                    elif unrealized_pnl_percent >= TAKE_PROFIT_PCT:
                        reward += 0.05                       # поощрение за фиксацию профита
                        self._force_sell(current_price, label="🎯 - TAKE‑PROFIT hit")
                    # -----------------------------------------------------------------
                    
                    # Если позиция прибыльна
                    if unrealized_pnl_percent > 0: # Если есть хоть какая-то нереализованная прибыль
                        # Чем выше нереализованная прибыль, тем больше награда за удержание
                        reward += unrealized_pnl_percent * 5 # Умеренный множитель, чтобы стимулировать удержание прибыльных позиций
                    # Если позиция убыточна
                    else: # unrealized_pnl_percent <= 0 (включая ноль)
                        # Чем больше нереализованный убыток, тем больше штраф за удержание
                        # Этот штраф должен быть достаточно чувствительным, чтобы побудить агента закрыть убыточную позицию
                        reward += unrealized_pnl_percent * 10 # Множитель для штрафа, пусть будет в 2 раза сильнее, чем награда за прибыль
                else: # Если action == HOLD и нет открытой позиции (просто бездействие)
                    # Очень маленький, едва заметный штраф за полное бездействие (когда нет открытых сделок)
                    # Это побудит агента искать точки входа, а не просто сидеть на балансе.
                    reward -= 0.005 # Очень маленький штраф

            # Награда за нереализованную прибыль/убыток (можно использовать, но осторожно)
            # Это может сделать функцию награды слишком "шумной", но может помочь агенту видеть текущий PnL
            # if self.crypto_held > 0 and self.last_buy_price is not None:
            #     unrealized_profit_loss = (current_price - self.last_buy_price) * self.crypto_held
            #     reward += unrealized_profit_loss / self.initial_balance * 0.1 # Масштабируем и даем небольшой вес

            self.cumulative_reward += reward

            next_state = self._get_state()

            info['roi_block'] = self.buy_rejected_roi / max(1, self.buy_attempts)                       
            info['vol_block'] = self.buy_rejected_vol / max(1, self.buy_attempts)
            info['volatility'] = volatility            
            info['volatility_threshold'] = volatility_threshold
            info['crypto_held'] = self.crypto_held
            info['penalty'] = penalty 
            info['current_balance'] = self.balance
            info['crypto_held'] = self.crypto_held
            info['current_price'] = current_price
            info['total_profit'] = (self.balance + self.crypto_held * current_price) - self.cfg.initial_balance            

        else: # done is True (эпизод завершился)
            
            if self.crypto_held > 0:
                pnl = (current_price - self.last_buy_price) * self.crypto_held
                self._force_sell(current_price, 'EPISODE DONE HELD')     # <— ваш метод SELL без награды
                reward += np.tanh(pnl / self.cfg.initial_balance * 10) * 20 
            
            final_value = self.balance + (self.crypto_held * current_price)
            profit_loss = final_value - self.cfg.initial_balance
            reward = profit_loss / self.cfg.initial_balance * 1000  # нормализованная награда
            # Общая стоимость активов в конце эпизода
            
            # Обновляем информацию для логирования
            self.total_profit = profit_loss 
            self.current_balance = final_value
            
            # Состояние для конечного шага
            # Если done=True, обычно возвращают нулевое состояние или последнее валидное состояние.
            # Если next_state будет использоваться для обучения, он должен быть валидным.
            # В данном случае, _get_state() уже обрабатывает граничные условия.
            next_state = self._get_state()
            
            self.cumulative_reward += reward
            
            info['cumulative_reward'] = self.cumulative_reward            
            info['total_profit'] = profit_loss 
            info['current_balance'] = final_value
            info['crypto_held'] = self.crypto_held 
            info['current_price'] = current_price
            info['buy_attempts'] = self.buy_attempts
            


        info['raw_reward'] = reward                
        abs_cap       = 3.0             # «эффективный» диапазон; подберите по статистике
        reward = np.tanh(reward / abs_cap)
        info['reward'] = reward
        # ----------------------------------------------------------------------
        return next_state, reward, done, info

    def reset(self) -> np.ndarray:
        # Случайно выбираем стартовый шаг так, чтобы хватило данных для lookback_window и episode_length
        max_start = self.total_steps - self.cfg.episode_length
        min_start = self.min_valid_start_step
        if max_start <= min_start:
            self.start_step = min_start
        else:
            self.start_step = random.randint(min_start, max_start)
        self.current_step = self.start_step

        # Сброс состояния торгового агента
        self.balance = self.cfg.initial_balance
        self.crypto_held = 0
        self.last_buy_price = None
        self.cumulative_reward = 0.0
        self.trades = []        
        self.max_price_during_hold = None
        
        # ---------- учёт эпизода ----------
        self._episode_idx += 1
        self._can_log = (self._episode_idx % self._log_interval == 0)
        # ----------------------------------
        
        
        self.buy_attempts = self.buy_rejected_vol = self.buy_rejected_roi = 0
        
        self.low_volatility_warned = False
        self.low_roi_warned        = False

        return self._get_state()
               
    def _log_ema_signals(self, step, label="INFO"):
        req = ['EMA_100', 'EMA_200', 'EMA_100_cross_up_200', 'EMA_100_cross_down_200']
        if not all(col in self.df_5min.columns for col in req):
            return "EMA ❌ not found"

        row = self.df_5min.iloc[step]
        if row['EMA_100_cross_up_200'] or row['EMA_100_cross_down_200']:
            return "EMA - ✅ - cross"
        return ""      
            
    def _log_rsi_signal(self, step: int, label: str = ""):
        rsi_col = 'RSI_14'  # или другой, если ты используешь другую длину
        if rsi_col in self.df_5min.columns:
            value = self.df_5min[rsi_col].iloc[step]
            return f"{rsi_col} = {value:.2f}"
        else:
            return f"{rsi_col} = not found"  
            
    def combined_signal_reward(self, action: int, step: int) -> float:
        """
        Вычисляет дополнительную награду (reward) на основе комбинации технических индикаторов
        для конкретного действия (покупка или продажа) на заданном шаге (свечи).

        Логика:
        - Для покупки (action == 1):
        - EMA_100 пересекает EMA_200 снизу вверх (bullish crossover) — сигнал потенциального начала восходящего тренда.
        - RSI ниже 30 — рынок считается перепроданным, возможен разворот вверх.
        - Объем торгов выше среднего за последние 20 свечей — подтверждение активности рынка.
        Если все три условия выполняются, агент получает дополнительную награду +5, чтобы стимулировать вход именно в такие благоприятные моменты.

        - Для продажи (action == 2):
        - EMA_100 пересекает EMA_200 сверху вниз (bearish crossover) — сигнал потенциального начала нисходящего тренда.
        - RSI выше 80 — рынок считается перекупленным, возможен разворот вниз.
        - Объем торгов выше среднего за последние 20 свечей — подтверждение активности рынка.
        Если все три условия выполняются, агент получает дополнительную награду +5, чтобы стимулировать выход именно в такие моменты.

        Параметры:
        - action (int): номер действия — 1 для покупки, 2 для продажи.
        - step (int): текущий индекс (номер свечи) в датафрейме self.df_5min.

        Возвращаемое значение:
        - float: дополнительная награда (0.0, если условия не выполнены, или положительное число).

        Использование:
        В основном коде награды добавляется так:
            reward += self.combined_signal_reward(action, self.current_step)

        Это помогает агенту лучше ориентироваться в важных сигналах технического анализа, 
        улучшая качество принимаемых торговых решений.
        """
        required_cols = [
            'EMA_100_cross_up_200', 'EMA_100_cross_down_200', 
            'RSI_14', 'volume'
        ]

        if not all(col in self.df_5min.columns for col in required_cols):
            print(f"❌ Отсутствуют необходимые колонки для combined_signal_reward")
            return 0.0

        row = self.df_5min[required_cols].iloc[step]
        mean_volume = self.df_5min['volume'].rolling(window=20).mean().iloc[step]

        reward = 0.0

        if action == 1:  # BUY
            buy_signal = (
                (row['EMA_100_cross_up_200'] == 1.0) and
                (row['RSI_14'] < 30) and
                (row['volume'] > mean_volume)
            )
            if buy_signal:
                reward = 0.05

        elif action == 2:  # SELL
            sell_signal = (
                (row['EMA_100_cross_down_200'] == 1.0) and
                (row['RSI_14'] > 80) and
                (row['volume'] > mean_volume)
            )
            if sell_signal:
                reward = 0.05

        return reward                        
        
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

    def _force_sell(self, sell_price, label="FORCED"):
        if self.crypto_held > 0 and self.last_buy_price is not None:
            amount_to_sell = self.crypto_held
            gross_sale_value = amount_to_sell * sell_price
            fee = gross_sale_value * self.trade_fee_percent
            net_proceeds = gross_sale_value - fee

            cost_of_purchase = amount_to_sell * self.last_buy_price
            net_profit_loss = net_proceeds - cost_of_purchase
            roi = net_profit_loss / cost_of_purchase if cost_of_purchase != 0 else 0

            self.balance += net_proceeds
            self.crypto_held = 0
            self.last_buy_price = None
            self.last_buy_step = None
            self.trailing_stop_counter = 0
            self.max_price_during_hold = None

            # Запись сделки
            trade = {
                "roi": roi,
                "net": net_profit_loss,
                "reward": 0,  # Можно добавить логику подсчёта награды при принудительной продаже, если нужно
                "duration": (self.current_step - self.last_buy_step) * 5 if self.last_buy_step else 0
            }
            self.trades.append(trade)

            self._log(f"[{self.current_step}] {label} SELL at {sell_price:.2f}, Profit: {net_profit_loss:.2f} (ROI: {roi:.2%})")
        else:
            self._log(f"[{self.current_step}] {label} SELL called, but no position held.")        
    
    def _log(self, *args, **kwargs):
        if self._can_log:
            print(*args, **kwargs)    

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
        self.df_5min = df
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

    @property
    def can_log(self) -> bool:
        """True — этот эпизод разрешено выводить в консоль."""
        return self._can_log     

    def render(self, mode='human'):
        pass
            
    def close(self):
        pass               
            
# Регистрация окружения
from gym.envs.registration import register 

register(
    id='CryptoTradingEnv-v0',
    entry_point='model.dqn_model.gym.crypto_trading_env:CryptoTradingEnv', 
)
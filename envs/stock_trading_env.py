"""
Gymnasium env для обучения DQN на акциях (РФ рынок / Tinkoff).

Отличия от CryptoTradingEnvOptimized:
  - нет funding rate
  - нет perpetual swap / маржинальной торговли
  - комиссия брокерская (0.05%)
  - торговля лотами (целые единицы)
  - нет 24/7 (но env этого не контролирует — просто работает по OHLCV)
"""
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from collections import deque
from typing import Dict, Optional
from dataclasses import dataclass, field
from envs.dqn_model.gym.indicators_optimized import preprocess_dataframes


@dataclass
class StockGymConfig:
    episode_length: int = 2_000
    initial_balance: float = 100_000.0   # рубли
    trade_fee_percent: float = 0.0005    # 0.05% брокерская комиссия
    lookback_window: int = 144
    max_hold_steps: int = 288
    position_fraction: float = 0.30
    # risk
    stop_loss_pct: float = -0.03
    take_profit_pct: float = 0.02
    min_hold_steps: int = 8
    volume_threshold: float = 0.0001
    # indicators
    indicators_config: Dict = field(default_factory=lambda: {
        'rsi': {'length': 14},
        'ema': {'lengths': [100, 200]},
        'ema_cross': {'pairs': [(100, 200)], 'include_cross_signal': True},
        'sma': {'length': 14},
    })
    reward_scale: float = 1.0
    train_split_ratio: float = 0.8


class StockTradingEnv(gym.Env):
    """DQN env для торговли акциями."""

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        dfs: Dict,
        cfg: Optional[StockGymConfig] = None,
        lookback_window: int = 144,
        indicators_config=None,
        episode_length: Optional[int] = None,
    ):
        super().__init__()
        self.cfg = cfg or StockGymConfig()
        self.episode_length = episode_length or self.cfg.episode_length
        if self.episode_length < 100:
            self.episode_length = 2000

        self.symbol = dfs.get('symbol', 'SBER') if isinstance(dfs, dict) else 'SBER'

        # Risk parameters
        self.STOP_LOSS_PCT = self.cfg.stop_loss_pct
        self.TAKE_PROFIT_PCT = self.cfg.take_profit_pct
        self.min_hold_steps = self.cfg.min_hold_steps
        self.volume_threshold = self.cfg.volume_threshold
        self.trade_fee_percent = self.cfg.trade_fee_percent
        self.initial_balance = self.cfg.initial_balance
        self.position_fraction = self.cfg.position_fraction
        self.max_hold_steps = self.cfg.max_hold_steps
        self.reward_scale = self.cfg.reward_scale

        # Prepare data
        df_5min_raw = dfs['df_5min'].values if hasattr(dfs['df_5min'], 'values') else dfs['df_5min']
        df_15min_raw = dfs['df_15min'].values if hasattr(dfs['df_15min'], 'values') else dfs['df_15min']
        df_1h_raw = dfs['df_1h'].values if hasattr(dfs['df_1h'], 'values') else dfs['df_1h']

        ic = indicators_config or self.cfg.indicators_config
        (self.df_5min, self.df_15min, self.df_1h,
         self.indicators, self.individual_indicators) = preprocess_dataframes(
            df_5min_raw, df_15min_raw, df_1h_raw, ic
        )

        self.total_steps = len(self.df_5min)
        self.lookback_window = lookback_window

        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        num_features_per_candle = 5  # OHLCV
        num_indicator_features = self.indicators.shape[1] if self.indicators.size > 0 else 0
        total_features_per_step = num_features_per_candle + num_indicator_features
        if total_features_per_step < 5:
            total_features_per_step = 15

        self.observation_space_shape = self.lookback_window * total_features_per_step + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.observation_space_shape,),
            dtype=np.float32,
        )

        self.min_valid_start_step = self.lookback_window
        self._precompute_states()
        self.reset()

    # ------------------------------------------------------------------ precompute
    def _precompute_states(self):
        candles = self.df_5min[:, :5].astype(np.float32)
        if self.indicators.size > 0:
            data = np.concatenate([candles, self.indicators.astype(np.float32)], axis=1)
        else:
            data = candles

        n, f = data.shape
        if n >= self.lookback_window:
            idx = np.arange(self.lookback_window)[None, :] + np.arange(n - self.lookback_window + 1)[:, None]
            sw = data[idx]
            self.precomputed_states = sw.reshape(sw.shape[0], -1).astype(np.float32)
        else:
            self.precomputed_states = np.zeros((1, self.observation_space_shape), dtype=np.float32)

        self.states_tensor = torch.from_numpy(self.precomputed_states)

        # Fix observation space if needed
        if self.precomputed_states.shape[1] != self.observation_space_shape - 2:
            self.observation_space_shape = self.precomputed_states.shape[1] + 2
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.observation_space_shape,),
                dtype=np.float32,
            )
        print(f"📊 [Stock] Предвычислено {len(self.precomputed_states)} состояний, shape={self.precomputed_states.shape}")

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.entry_price = 0.0
        self.current_step = 0
        self.total_reward = 0.0
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.hold_steps = 0
        self.episode_pnl = 0.0

        max_start = max(0, self.total_steps - self.episode_length - 1)
        if max_start > self.min_valid_start_step:
            self.start_step = self.np_random.integers(self.min_valid_start_step, max_start)
        else:
            self.start_step = self.min_valid_start_step
        self.current_step = self.start_step
        return self._get_observation(), {}

    # ------------------------------------------------------------------ observation
    def _get_observation(self):
        idx = self.current_step
        if idx < self.lookback_window - 1 or idx >= len(self.precomputed_states) + self.lookback_window - 1:
            return np.zeros(self.observation_space_shape, dtype=np.float32)
        state_idx = idx - (self.lookback_window - 1)
        if state_idx < 0 or state_idx >= len(self.precomputed_states):
            return np.zeros(self.observation_space_shape, dtype=np.float32)
        precomputed = self.precomputed_states[state_idx].copy()
        norm_balance = self.balance / self.initial_balance - 1.0
        norm_held = 1.0 if self.shares_held > 0 else 0.0
        return np.concatenate([precomputed, [norm_balance, norm_held]]).astype(np.float32)

    # ------------------------------------------------------------------ step
    def step(self, action):
        reward = 0.0
        done = False
        close_price = float(self.df_5min[self.current_step, 3])  # close

        if action == 1 and self.shares_held == 0:
            # BUY
            invest = self.balance * self.position_fraction
            fee = invest * self.trade_fee_percent
            shares = int((invest - fee) / close_price) if close_price > 0 else 0
            if shares > 0:
                cost = shares * close_price + fee
                self.balance -= cost
                self.shares_held = shares
                self.entry_price = close_price
                self.hold_steps = 0
                reward = -0.001 * self.reward_scale  # small penalty for entry

        elif action == 2 and self.shares_held > 0:
            # SELL
            revenue = self.shares_held * close_price
            fee = revenue * self.trade_fee_percent
            pnl_pct = (close_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0
            self.balance += revenue - fee
            self.episode_pnl += (revenue - fee) - (self.shares_held * self.entry_price)
            self.trades += 1
            if pnl_pct > 0:
                self.wins += 1
            else:
                self.losses += 1
            reward = pnl_pct * 10.0 * self.reward_scale
            self.shares_held = 0
            self.entry_price = 0.0
            self.hold_steps = 0

        # If holding, check SL/TP and max hold
        if self.shares_held > 0:
            self.hold_steps += 1
            pnl_pct = (close_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0
            if pnl_pct <= self.STOP_LOSS_PCT:
                revenue = self.shares_held * close_price
                fee = revenue * self.trade_fee_percent
                self.balance += revenue - fee
                self.episode_pnl += (revenue - fee) - (self.shares_held * self.entry_price)
                self.trades += 1
                self.losses += 1
                reward = pnl_pct * 10.0 * self.reward_scale
                self.shares_held = 0
                self.entry_price = 0.0
                self.hold_steps = 0
            elif pnl_pct >= self.TAKE_PROFIT_PCT:
                revenue = self.shares_held * close_price
                fee = revenue * self.trade_fee_percent
                self.balance += revenue - fee
                self.episode_pnl += (revenue - fee) - (self.shares_held * self.entry_price)
                self.trades += 1
                self.wins += 1
                reward = pnl_pct * 10.0 * self.reward_scale
                self.shares_held = 0
                self.entry_price = 0.0
                self.hold_steps = 0
            elif self.hold_steps >= self.max_hold_steps:
                revenue = self.shares_held * close_price
                fee = revenue * self.trade_fee_percent
                self.balance += revenue - fee
                self.episode_pnl += (revenue - fee) - (self.shares_held * self.entry_price)
                self.trades += 1
                if pnl_pct > 0:
                    self.wins += 1
                else:
                    self.losses += 1
                reward = pnl_pct * 10.0 * self.reward_scale
                self.shares_held = 0
                self.entry_price = 0.0
                self.hold_steps = 0

        self.current_step += 1
        self.total_reward += reward

        if self.current_step >= self.start_step + self.episode_length or self.current_step >= self.total_steps - 1:
            done = True

        obs = self._get_observation()
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'trades': self.trades,
            'wins': self.wins,
            'losses': self.losses,
            'total_reward': self.total_reward,
            'episode_pnl': self.episode_pnl,
        }
        return obs, reward, done, False, info

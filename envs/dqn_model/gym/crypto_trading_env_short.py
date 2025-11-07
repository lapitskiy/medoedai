from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Tuple

try:
    import gymnasium as gym
except Exception:  # pragma: no cover
    import gym  # type: ignore

from .crypto_trading_env_optimized import CryptoTradingEnvOptimized


class CryptoTradingEnvShort(CryptoTradingEnvOptimized):
    """
    Зеркальная среда под шорт:
    - Действия: 0=HOLD, 1=ENTER_SHORT (SELL), 2=COVER (BUY reduceOnly)
    - Вознаграждение и PnL инвертированы относительно лонга (падение цены => плюс)

    Замечание: реализовано минимально инвазивно, без изменения long-логики базовой среды.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.short_held: float = 0.0
        self.short_entry_price: Optional[float] = None
        self.fee_entry_short: float = 0.0

    def reset(self):
        state = super().reset()
        self.short_held = 0.0
        self.short_entry_price = None
        self.fee_entry_short = 0.0
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        # 0=HOLD, 1=ENTER_SHORT, 2=COVER
        reward_scale = float(getattr(self.cfg, 'reward_scale', 1.0))
        current_price = float(self.df_5min[self.current_step - 1, 3])

        reward = 0.0
        done = False
        info: Dict = {}

        # учёт статистики действий
        try:
            self.action_counts[action] += 1
        except Exception:
            pass
        self.episode_step_count += 1

        if action == 1:  # ENTER_SHORT
            if self.short_held == 0.0:
                # Используем ту же динамику размера позиции, что и BUY, но для шорта
                entry_confidence = 0.5
                try:
                    entry_confidence = self._calculate_entry_confidence()
                except Exception:
                    entry_confidence = 0.5

                if entry_confidence > self.position_confidence_threshold:
                    self.position_fraction = min(self.base_position_fraction * 1.5, 0.5)
                elif entry_confidence > 0.5:
                    self.position_fraction = self.base_position_fraction
                else:
                    self.position_fraction = max(self.base_position_fraction * 0.7, 0.15)

                # Продаём «заёмный» актив: баланс растёт на выручку (минус комиссия)
                short_notional = self.balance * self.position_fraction
                self.fee_entry_short = short_notional * self.trade_fee_percent
                qty = (short_notional - self.fee_entry_short) / max(current_price, 1e-9)
                qty = float(max(0.0, qty))

                self.short_held = qty
                self.short_entry_price = current_price
                self.balance += (short_notional - self.fee_entry_short)

                # небольшой базовый бонус + бонус уверенности
                reward = 0.03 + entry_confidence * 0.02
            else:
                # уже в шорте — интерпретируем как HOLD с лёгким штрафом за избыточную активность
                reward = -0.001

        elif action == 2:  # COVER
            if self.short_held > 0.0 and self.short_entry_price is not None:
                cover_cost = self.short_held * current_price
                fee_exit = cover_cost * self.trade_fee_percent

                # закрываем шорт: баланс уменьшается на стоимость выкупа + комиссия
                self.balance -= (cover_cost + fee_exit)

                # PnL (инвертированный): прибыль, если текущая цена ниже цены открытия
                pnl = ((self.short_entry_price - current_price) / self.short_entry_price) \
                      - ((self.fee_entry_short + fee_exit) / max(self.short_entry_price * self.short_held, 1e-9))

                reward += float(np.tanh(pnl * 25.0) * 2.0)

                # сбрасываем позицию
                self.short_held = 0.0
                self.short_entry_price = None
                self.fee_entry_short = 0.0
            else:
                # нет шорт позиции — трактуем как HOLD с лёгким штрафом
                reward = -0.001

        else:  # HOLD
            if self.short_held > 0.0 and self.short_entry_price is not None:
                unrealized = (self.short_entry_price - current_price) / self.short_entry_price
                # поощряем удержание, когда в прибыли; штрафуем при убытке
                if unrealized > 0:
                    reward += unrealized * 3.0
                else:
                    reward += unrealized * 2.0
            else:
                reward += 0.001

        # продвигаем шаг
        self.current_step += 1
        episode_length = getattr(self.cfg, 'episode_length', getattr(self, 'episode_length', None))
        try:
            episode_length = int(episode_length) if episode_length is not None else None
        except Exception:
            episode_length = None

        done = (
            (episode_length is not None and self.episode_step_count >= episode_length) or
            self.current_step >= self.total_steps
        )

        # при завершении эпизода закрываем шорт по последней цене
        if done and self.short_held > 0.0 and self.short_entry_price is not None:
            final_price = float(self.df_5min[self.current_step - 1, 3])
            cover_cost = self.short_held * final_price
            fee_exit = cover_cost * self.trade_fee_percent
            self.balance -= (cover_cost + fee_exit)

            pnl = ((self.short_entry_price - final_price) / self.short_entry_price) \
                  - ((self.fee_entry_short + fee_exit) / max(self.short_entry_price * self.short_held, 1e-9))
            # не добавляем reward здесь (закрытие по таймауту)

            self.short_held = 0.0
            self.short_entry_price = None
            self.fee_entry_short = 0.0

        info.update({
            'current_balance': float(self.balance),
            'current_price': current_price,
            'direction': 'short',
            'reward': reward * reward_scale,
        })

        # масштабируем награду как и в базе
        reward = reward * reward_scale

        # состояние из предвычисленного буфера
        next_state = self._get_state()
        return next_state, float(reward), bool(done), info



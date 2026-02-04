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
        self._short_entry_step: Optional[int] = None

    def reset(self):
        state = super().reset()
        self.short_held = 0.0
        self.short_entry_price = None
        self.fee_entry_short = 0.0
        self._short_entry_step = None
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

        def _append_trade(exit_price: float, fee_exit: float, reason: str) -> None:
            """Записать сделку SHORT в trades/all_trades + обновить причины закрытия."""
            try:
                qty = float(self.short_held or 0.0)
                entry_price = float(self.short_entry_price or 0.0)
                if qty <= 0.0 or entry_price <= 0.0:
                    return

                fees = float((self.fee_entry_short or 0.0) + (fee_exit or 0.0))
                gross_pnl = float((entry_price - float(exit_price)) * qty)
                net_pnl = float(gross_pnl - fees)
                roi = float(((entry_price - float(exit_price)) / max(entry_price, 1e-9)) - (fees / max(entry_price * qty, 1e-9)))

                # timestamps (best-effort)
                exit_dt = None
                entry_dt = None
                try:
                    if hasattr(self, '_candle_datetimes') and (self.current_step - 1) < len(self._candle_datetimes):
                        exit_dt = self._candle_datetimes[self.current_step - 1]
                    es = getattr(self, '_short_entry_step', None)
                    if hasattr(self, '_candle_datetimes') and es is not None and (es - 1) < len(self._candle_datetimes):
                        entry_dt = self._candle_datetimes[es - 1]
                except Exception:
                    exit_dt = None
                    entry_dt = None

                trade_data = {
                    "symbol": getattr(self, 'symbol', None),
                    "side": "SHORT",
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "qty": float(qty),
                    "fees": float(fees),
                    "entry_step": int(getattr(self, '_short_entry_step', self.current_step) or self.current_step),
                    "exit_step": int(self.current_step),
                    "entry_time": (entry_dt.isoformat() if entry_dt is not None else None),
                    "exit_time": (exit_dt.isoformat() if exit_dt is not None else None),
                    "roi": float(roi),
                    "pnl": float(gross_pnl),
                    "net": float(net_pnl),
                    "reward": float(reward),
                    "duration": float((self.current_step - (getattr(self, '_short_entry_step', self.current_step) or self.current_step)) * 5),
                    "close_reason": str(reason or "other"),
                }
                try:
                    self.trades.append(trade_data)
                except Exception:
                    self.trades = [trade_data]
                try:
                    if not hasattr(self, 'all_trades'):
                        self.all_trades = []
                    self.all_trades.append(trade_data)
                except Exception:
                    pass

                # Причины закрытия (эпизодные и кумулятивные)
                key = str(reason or "other")
                if key not in getattr(self, 'sell_types', {}):
                    key = "other"
                try:
                    self.sell_types[key] += 1
                    self.cumulative_sell_types[key] += 1
                except Exception:
                    pass
            except Exception:
                # статистика не должна ломать env
                return

        def _close_short(exit_price: float, reason: str, *, apply_balance: bool = True) -> None:
            """Закрыть шорт по цене exit_price, записать сделку, обновить баланс/состояние."""
            if self.short_held <= 0.0 or self.short_entry_price is None:
                return
            fee_exit = 0.0
            try:
                if apply_balance:
                    cover_cost = float(self.short_held) * float(exit_price)
                    fee_exit = float(cover_cost) * float(self.trade_fee_percent)
                    self.balance -= (cover_cost + fee_exit)
            except Exception:
                fee_exit = 0.0
            _append_trade(float(exit_price), float(fee_exit), str(reason or "other"))
            # сбрасываем позицию + уровни
            self.short_held = 0.0
            self.short_entry_price = None
            self.fee_entry_short = 0.0
            self._short_entry_step = None
            try:
                self._short_sl_price_atr = None
                self._short_tp_price_atr = None
                self._short_entry_atr_abs = None
            except Exception:
                pass

        if action == 1:  # ENTER_SHORT
            try:
                self.buy_attempts += 1
            except Exception:
                pass
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
                self._short_entry_step = int(self.current_step)
                self.balance += (short_notional - self.fee_entry_short)

                # Reward на входе ≈ 0: оставляем только комиссию (минус)
                try:
                    reward += -float(self.fee_entry_short) / max(float(short_notional), 1e-9)
                except Exception:
                    pass
                # Статистика ENTER_SHORT (используем buy_stats_total как "входы")
                try:
                    self.buy_stats_total['executed'] += 1
                except Exception:
                    pass

                # ATR freeze на входе и уровни SL/TP по ATR (если включено и ATR доступен)
                try:
                    self._short_entry_atr_abs = None
                    self._short_sl_price_atr = None
                    self._short_tp_price_atr = None
                    if bool(getattr(self.cfg, 'use_atr_stop', True)) and getattr(self, 'atr1h_norm_5m', None) is not None:
                        idx_atr = self.current_step - 1
                        if 0 <= idx_atr < len(self.atr1h_norm_5m):
                            atr_norm = float(self.atr1h_norm_5m[idx_atr])
                            atr_abs = max(1e-8, atr_norm * current_price)
                            self._short_entry_atr_abs = atr_abs
                            k_sl = float(getattr(self.cfg, 'atr_sl_mult', 1.5))
                            self._short_sl_price_atr = float(self.short_entry_price + k_sl * atr_abs)
                            k_tp = getattr(self.cfg, 'atr_tp_mult', None)
                            if k_tp is not None:
                                self._short_tp_price_atr = float(self.short_entry_price - float(k_tp) * atr_abs)
                except Exception:
                    self._short_entry_atr_abs = None
                    self._short_sl_price_atr = None
                    self._short_tp_price_atr = None
            else:
                # уже в шорте — интерпретируем как HOLD с лёгким штрафом за избыточную активность
                reward = -0.001
                try:
                    self.buy_stats_total['already_holding'] += 1
                except Exception:
                    pass

        elif action == 2:  # COVER
            if self.short_held > 0.0 and self.short_entry_price is not None:
                # Закрытие по агенту
                try:
                    pnl = ((self.short_entry_price - current_price) / self.short_entry_price) \
                          - ((self.fee_entry_short + (self.short_held * current_price * self.trade_fee_percent)) / max(self.short_entry_price * self.short_held, 1e-9))
                    reward += float(np.tanh(pnl * 25.0) * 2.0)
                except Exception:
                    pass
                _close_short(current_price, reason="agent", apply_balance=True)
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
                try:
                    self.hold_stats_total['with_position'] += 1
                except Exception:
                    pass
            else:
                # Без бонуса за бездействие
                reward += 0.0
                try:
                    self.hold_stats_total['no_position'] += 1
                except Exception:
                    pass

        # --- Зеркальные SL/TP/Trailing для шорта + ATR-нормировка награды (опц.) ---
        try:
            if self.short_held > 0.0 and self.short_entry_price is not None:
                unrealized = (self.short_entry_price - current_price) / max(self.short_entry_price, 1e-9)  # pnl%
                closed = False
                # ATR-нормировка (опционально, мягко)
                try:
                    if getattr(self, 'atr1h_norm_5m', None) is not None and (self.current_step - 1) < len(self.atr1h_norm_5m):
                        atr_now = float(self.atr1h_norm_5m[self.current_step - 1])
                        if atr_now > 1e-6:
                            reward = reward / max(atr_now, 1e-4)
                except Exception:
                    pass

                # ATR‑стопы: проверяем до процентных SL/TP
                try:
                    if bool(getattr(self.cfg, 'use_atr_stop', True)):
                        if getattr(self, '_short_sl_price_atr', None) is not None and current_price >= self._short_sl_price_atr:
                            reward += -0.05
                            _close_short(current_price, reason="stop_loss", apply_balance=True)
                            closed = True
                        elif getattr(self, '_short_tp_price_atr', None) is not None and current_price <= self._short_tp_price_atr:
                            reward += 0.05
                            _close_short(current_price, reason="take_profit", apply_balance=True)
                            closed = True
                except Exception:
                    pass

                if closed:
                    # позиция уже закрыта — не продолжаем percent SL/TP/trailing
                    pass
                else:
                    # Процентные SL/TP (зеркально)
                    sl = float(getattr(self, 'STOP_LOSS_PCT', -0.03))   # отрицательная величина
                    tp = float(getattr(self, 'TAKE_PROFIT_PCT', 0.05))  # положительная величина
                    if unrealized <= sl:
                        reward += -0.05
                        _close_short(current_price, reason="stop_loss", apply_balance=True)
                        closed = True
                    elif unrealized >= tp:
                        reward += 0.05
                        _close_short(current_price, reason="take_profit", apply_balance=True)
                        closed = True

                    if closed:
                        pass
                    else:
                        # Trailing (для шорта — drawup от минимума/вверх)
                        if not hasattr(self, '_short_trailing_counter'):
                            self._short_trailing_counter = 0
                        drawup = (current_price - self.short_entry_price) / max(self.short_entry_price, 1e-9)
                        # Порог трейлинга: ATR‑базированный (freeze at entry) или фикс. 2%
                        thr_trail = 0.02
                        try:
                            if bool(getattr(self.cfg, 'use_atr_stop', True)) and getattr(self, '_short_entry_atr_abs', None) is not None:
                                k_tr = float(getattr(self.cfg, 'atr_trail_mult', 1.0))
                                thr_trail = float(np.clip(k_tr * (self._short_entry_atr_abs / max(self.short_entry_price, 1e-9)), 0.002, 0.08))
                        except Exception:
                            thr_trail = 0.02
                        if drawup > thr_trail:
                            self._short_trailing_counter += 1
                        else:
                            self._short_trailing_counter = 0
                        if self._short_trailing_counter >= 3:
                            reward += -0.03
                            _close_short(current_price, reason="trailing", apply_balance=True)
                            self._short_trailing_counter = 0
        except Exception:
            pass

        # обновим статистики как в базовой среде
        try:
            self._update_stats(current_price)
        except Exception:
            pass

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
            # не добавляем reward здесь (закрытие по таймауту), но сделку фиксируем
            _close_short(final_price, reason="timeout", apply_balance=True)

        info.update({
            'current_balance': float(self.balance),
            'current_price': current_price,
            'direction': 'short',
            'reward': reward * reward_scale,
        })

        # масштабируем награду как и в базе
        reward = reward * reward_scale

        # Обновляем market_state (ровно 1 раз на шаг) перед возвратом
        try:
            self._update_market_state_once()
        except Exception:
            pass

        # состояние из предвычисленного буфера
        next_state = self._get_state()
        return next_state, float(reward), bool(done), info



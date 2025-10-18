from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym

from train.domain.episode.extension_policy import EpisodeExtensionPolicy


class PositionAwareEpisodeWrapper(gym.Wrapper):
    """Обёртка, продлевающая эпизод, если позиция открыта на конце.

    Правила продления задаются через EpisodeExtensionPolicy (DDD-domain policy).
    Копит кумулятивные счётчики метрик внутри policy.
    """

    def __init__(self, env: gym.Env, policy: EpisodeExtensionPolicy | None = None, end_after_sell_during_extension: bool = True):
        super().__init__(env)
        self.policy = policy or EpisodeExtensionPolicy()
        self.end_after_sell_during_extension = bool(end_after_sell_during_extension)
        # Запоминаем исходную длину эпизода, если есть
        self._original_episode_length = getattr(self.env, "episode_length", None)
        # Подавляем принудительную продажу по таймауту в базовой среде — эту логику берёт на себя враппер
        try:
            setattr(self.env, '_suppress_timeout_force_sell', True)
        except Exception:
            pass
        # Флаг: находимся ли мы в фазе продления эпизода (после базового окна)
        self._in_extension: bool = False

    def reset(self, **kwargs) -> Any:
        # Сбрасываем политику на новый эпизод
        if self.policy:
            self.policy.reset_for_new_episode()
        # Восстанавливаем исходную длину эпизода
        if self._original_episode_length is not None:
            try:
                self.env.episode_length = int(self._original_episode_length)
            except Exception:
                pass
        # Выходим из режима продления
        self._in_extension = False
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)

        if done:
            # Определяем, открыта ли позиция
            position_open = False
            try:
                if hasattr(self.env, 'crypto_held') and getattr(self.env, 'crypto_held'):
                    position_open = True
                elif hasattr(self.env, 'current_position') and bool(getattr(self.env, 'current_position')):
                    position_open = True
                elif hasattr(self.env, 'last_buy_step') and getattr(self.env, 'last_buy_step') is not None:
                    # если был BUY и позиция не закрыта
                    position_open = bool(getattr(self.env, 'crypto_held', 0.0) > 0.0)
            except Exception:
                position_open = False

            if self.policy and self.policy.can_extend_now(position_open):
                # Продлеваем эпизод
                extend_by = int(self.policy.record_extension())
                try:
                    if hasattr(self.env, 'episode_length') and self.env.episode_length is not None:
                        self.env.episode_length = int(self.env.episode_length) + extend_by
                except Exception:
                    pass
                # Снимаем флаг завершения — продолжаем эпизод
                done = False
                # Входим в фазу продления
                self._in_extension = True
                # Добавим диагностическую метку в info
                try:
                    if isinstance(info, dict):
                        info['episode_extended_by'] = extend_by
                        info['episode_extensions_so_far'] = self.policy.total_episode_extensions
                except Exception:
                    pass

            else:
                # Если нельзя продлить, но позиция открыта — принудительно закрываем с причиной TIMEOUT (управляет враппер)
                if position_open:
                    try:
                        price = None
                        if isinstance(info, dict) and 'current_price' in info:
                            price = float(info['current_price'])
                        if price is None and hasattr(self.env, 'df_5min') and hasattr(self.env, 'current_step'):
                            idx = max(0, int(getattr(self.env, 'current_step', 1)) - 1)
                            price = float(self.env.df_5min[idx, 3])
                        if price is not None and hasattr(self.env, '_force_sell'):
                            self.env._force_sell(price, 'TIMEOUT')
                    except Exception:
                        pass

        # Если мы в фазе продления и только что успешно продали — завершаем эпизод, чтобы не было новых BUY в продлении
        if self._in_extension and self.end_after_sell_during_extension:
            try:
                # Признак успешной продажи: после действия SELL позиция стала закрыта
                position_now_open = False
                if hasattr(self.env, 'crypto_held'):
                    position_now_open = bool(getattr(self.env, 'crypto_held', 0.0) > 0.0)
                elif hasattr(self.env, 'current_position'):
                    position_now_open = bool(getattr(self.env, 'current_position'))
                # Если позиции нет — заканчиваем эпизод
                if not position_now_open:
                    done = True
                    if isinstance(info, dict):
                        info['episode_ended_after_sell_in_extension'] = True
                    # Выходим из фазы продления
                    self._in_extension = False
            except Exception:
                pass

        return obs, float(reward), bool(done), (info if isinstance(info, dict) else {})

    # Метрики для внешнего чтения
    @property
    def episode_extensions_total(self) -> int:
        return int(self.policy.total_episode_extensions if self.policy else 0)

    @property
    def episode_extension_steps_total(self) -> int:
        return int(self.policy.total_extension_steps if self.policy else 0)



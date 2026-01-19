import gymnasium as gym
from typing import Any, Dict, Tuple


class PositionAwareEpisodeWrapper(gym.Wrapper):
    """Расширяет эпизод, если на его границе остаётся открытая позиция.

    - Если done=True и позиция открыта, продлевает эпизод на extension_steps
      (не более max_extension раз) и возвращает done=False.
    - Добавляет в info ключи: 'episode_extended', 'extension_count',
      'episode_extension_limit_reached'.
    - Не изменяет внутреннюю динамику вознаграждений.
    """

    def __init__(self, env: gym.Env, max_extension: int = 20, extension_steps: int = 100):
        super().__init__(env)
        self.max_extension = int(max_extension)
        self.extension_steps = int(extension_steps)
        self.extension_count = 0
        # Кумулятивная статистика за всю сессию
        self.total_episode_extensions = 0
        self.total_extension_steps = 0
        # Сохраняем исходную длину эпизода, если доступна
        self._original_episode_length = getattr(self.env, "episode_length", None)

    def reset(self, *args, **kwargs):
        # Сбрасываем счётчик продлений и длину эпизода к исходной
        self.extension_count = 0
        if self._original_episode_length is not None:
            try:
                self.env.episode_length = int(self._original_episode_length)
            except Exception:
                pass
        # Возвращаем как есть: базовый TradingEnvWrapper уже нормализует (obs, info)
        return self.env.reset(*args, **kwargs)

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        result = self.env.step(action)

        # Совместимость со старым API (obs, reward, done, info)
        if isinstance(result, tuple) and len(result) == 4:
            obs, reward, done, info = result
            terminated = bool(done)
            truncated = False
        else:
            # Новый API (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)

        # На границе эпизода: если позиция открыта — пытаемся продлить
        position_open = False
        try:
            if hasattr(self.env, 'current_position'):
                position_open = bool(getattr(self.env, 'current_position'))
            else:
                # Фолбэк: проверим по признакам позиции
                position_open = bool(getattr(self.env, 'crypto_held', 0.0) > 0.0) or (getattr(self.env, 'last_buy_step', None) is not None)
        except Exception:
            position_open = False

        if done and position_open:
            if self.extension_count < self.max_extension:
                self.extension_count += 1
                # Накопительная статистика продлений
                try:
                    self.total_episode_extensions += 1
                    self.total_extension_steps += int(self.extension_steps)
                except Exception:
                    pass
                try:
                    # Увеличиваем целевую длину эпизода в базовом env
                    cur_len = int(getattr(self.env, 'episode_length'))
                    setattr(self.env, 'episode_length', cur_len + int(self.extension_steps))
                except Exception:
                    # Если не получилось — мягко фолбэк без изменения длины
                    pass
                # Снимаем флаги завершения эпизода
                done = False
                terminated = False
                truncated = False
            else:
                # Лимит продлений исчерпан — завершаем эпизод без добавления нестабильных ключей в info
                pass

        # Проксируем market_state и action_mask в info как СКАЛЯРЫ (для совместимости с sanitize)
        try:
            if not isinstance(info, dict):
                info = {}
            base_env = self.env
            for _ in range(10):
                if hasattr(base_env, 'env'):
                    base_env = getattr(base_env, 'env')
                else:
                    break
            ms = getattr(base_env, 'market_state', None)
            if ms is not None:
                try:
                    info['market_state'] = int(ms)
                except Exception:
                    pass
            if hasattr(base_env, 'get_action_mask'):
                m = base_env.get_action_mask()
                try:
                    m = list(m) if m is not None else [1, 1, 1]
                except Exception:
                    m = [1, 1, 1]
                if len(m) < 3:
                    m = m + [1] * (3 - len(m))
                info['mask_0'] = int(bool(m[0]))
                info['mask_1'] = int(bool(m[1]))
                info['mask_2'] = int(bool(m[2]))
        except Exception:
            pass

        return obs, float(reward), bool(terminated), bool(truncated), (info if isinstance(info, dict) else {})



from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeExtensionPolicy:
    """Политика продления эпизода.

    - max_extension: максимум продлений в одном эпизоде
    - extension_steps: сколько шагов добавлять на каждое продление
    Счётчики total_* ведутся кумулятивно по всем эпизодам для метрик.
    """
    max_extension: int = 5
    extension_steps: int = 100

    # Текущие счётчики внутри эпизода
    _episode_extension_count: int = 0

    # Кумулятивные счётчики по всем эпизодам
    total_episode_extensions: int = 0
    total_extension_steps: int = 0

    def reset_for_new_episode(self) -> None:
        self._episode_extension_count = 0

    def can_extend_now(self, position_open: bool) -> bool:
        return position_open and (self._episode_extension_count < int(self.max_extension))

    def record_extension(self) -> int:
        """Фиксирует продление и возвращает количество добавляемых шагов."""
        self._episode_extension_count += 1
        self.total_episode_extensions += 1
        self.total_extension_steps += int(self.extension_steps)
        return int(self.extension_steps)



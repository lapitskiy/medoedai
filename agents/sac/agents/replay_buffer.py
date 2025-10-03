"""Experience replay для SAC, переиспользующий реализацию DQN."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from agents.vdqn.dqnsolver import PrioritizedReplayBuffer
from agents.vdqn.cfg.vconfig import vDqnConfig


class SacReplayBuffer(PrioritizedReplayBuffer):
    """Переиспользуем приоритизированный буфер из DQN."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        device: torch.device,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-3,
        use_gpu_storage: bool | None = None,
    ) -> None:
        # SAC всегда использует CPU для буфера воспроизведения для экономии GPU памяти
        super().__init__(
            capacity=capacity,
            state_size=state_dim,
            alpha=alpha,
            beta=beta,
            beta_increment=beta_increment,
            use_gpu_storage=False,  # Принудительно отключаем GPU storage для SAC
        )
        self.device = device

    def add_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        gamma_n: torch.Tensor | float = 1.0,
    ) -> None:
        self.push(state, action, reward, next_state, done, gamma_n)

    def sample_batch(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, ...] | Tuple[None, ...]:
        return super().sample(batch_size)

    def clear(self) -> None:
        """Очищает буфер воспроизведения"""
        self.states.fill_(0)
        self.next_states.fill_(0)
        self.actions.fill_(0)
        self.rewards.fill_(0)
        self.dones.fill_(False)
        self.gamma_ns.fill_(1.0)
        self.priorities.fill_(1.0)
        self.position = 0
        self.size = 0
        print("🗑️ Буфер воспроизведения очищен")



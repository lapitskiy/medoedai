"""SAC нейросети, переиспользующие блоки из DQN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from agents.vdqn.dqnn import ResidualLinearBlock
from agents.sac.agents.config import SacConfig # Исправлен импорт SacConfig


# Удален класс MLPConfig, так как он не используется

class SacCategoricalActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg: SacConfig) -> None:
        super().__init__()
        layers = []
        in_dim = obs_dim
        print(f"[SacCategoricalActor] Building backbone with input_dim={in_dim}")
        for size in cfg.hidden_sizes:
            print(f"[SacCategoricalActor] Adding layer: in_dim={in_dim}, size={size}")
            layers.append(
                ResidualLinearBlock(
                    in_dim,
                    size,
                    activation=cfg.activation,
                    dropout=cfg.dropout_rate,
                    layer_norm=cfg.layer_norm,
                    residual=cfg.use_residual_blocks,
                    use_swiglu=cfg.use_swiglu_gate,
                )
            )
            in_dim = size
        self.backbone = nn.Sequential(*layers)
        self.logits_head = nn.Linear(cfg.hidden_sizes[-1], action_dim)

        # Инициализация весов
        for layer in self.backbone:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.xavier_uniform_(self.logits_head.weight, gain=1.0)
        nn.init.constant_(self.logits_head.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        # Применяем nan_to_num к входным данным
        if torch.isnan(obs).any():
            print("⚠️ [SacCategoricalActor] Обнаружены NaN во входных данных! Заменяем на 0.")
            obs = torch.nan_to_num(obs, nan=0.0, posinf=1e5, neginf=-1e5)
            
        x = self.backbone(obs)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5)
        x = torch.clamp(x, min=-100.0, max=100.0)  # Клиппинг больших значений
        
        # Генерируем логгиты
        logits = self.logits_head(x)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e5, neginf=-1e5)
        logits = torch.clamp(logits, min=-100.0, max=100.0) # Клиппинг логгитов
        
        return torch.distributions.Categorical(logits=logits)

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(obs)
        action = dist.sample().unsqueeze(-1)
        log_prob = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)
        return action.float(), log_prob


class SacCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg: SacConfig) -> None:
        super().__init__()
        layers = []
        in_dim = obs_dim  # Используем obs_dim как входной размер
        for size in cfg.hidden_sizes:
            layers.append(
                ResidualLinearBlock(
                    in_dim,
                    size,
                    activation=cfg.activation,
                    dropout=cfg.dropout_rate,
                    layer_norm=cfg.layer_norm,
                    residual=cfg.use_residual_blocks,
                    use_swiglu=cfg.use_swiglu_gate,
                )
            )
            in_dim = size
        layers.append(nn.Linear(in_dim, action_dim))
        self.q_net = nn.Sequential(*layers)

        # Инициализация весов
        for layer in self.q_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0)  # Для Q-сети используем gain=1.0
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_net(obs)


class SacDoubleCritic(nn.Module):
    """SAC-критик, использует две независимые сети для расчета Q-значений."""

    def __init__(self, obs_dim: int, action_dim: int, cfg: SacConfig):
        super().__init__()
        self.q1 = SacCritic(obs_dim, action_dim, cfg)
        self.q2 = SacCritic(obs_dim, action_dim, cfg)

        # Инициализация весов
        for network in [self.q1.q_net, self.q2.q_net]:  # Проходим по q_net внутри SacCritic
            for layer in network.children():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q1_values = self.q1(obs)
        q1_values = torch.clamp(q1_values, min=-1e5, max=1e5) # Клиппинг Q-значений

        q2_values = self.q2(obs)
        q2_values = torch.clamp(q2_values, min=-1e5, max=1e5) # Клиппинг Q-значений

        return q1_values, q2_values



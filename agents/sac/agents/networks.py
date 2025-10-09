"""SAC нейросети, переиспользующие блоки из DQN."""

from __future__ import annotations

import torch
import torch.nn as nn

from agents.vdqn.feature_extractor import FeatureExtractor
from .config import SacConfig


# Удален класс MLPConfig, так как он не используется

class SacCategoricalActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cfg: SacConfig,
        encoder: FeatureExtractor | None = None,
    ) -> None:
        super().__init__()
        self.backbone = encoder if encoder is not None else FeatureExtractor(
            obs_dim=obs_dim,
            hidden_sizes=cfg.hidden_sizes,
            dropout_rate=cfg.dropout_rate,
            layer_norm=cfg.layer_norm,
            activation=cfg.activation,
            use_residual=cfg.use_residual_blocks,
            use_swigлу=cfg.use_swigлу_gate,
        )
        self.logits_head = nn.Linear(self.backbone.feature_dim, action_dim)

        nn.init.xavier_uniform_(self.logits_head.weight, gain=1.0)
        nn.init.constant_(self.logits_head.bias, 0.0)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs)

    def forward(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        features = self.encode(obs)
        logits = self.logits_head(features)

        return torch.distributions.Categorical(logits=logits)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(obs)
        action = dist.sample().unsqueeze(-1)
        log_prob = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)
        return action.float(), log_prob


class SacCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cfg: SacConfig,
        encoder: FeatureExtractor | None = None,
    ) -> None:
        super().__init__()
        self.backbone = encoder if encoder is not None else FeatureExtractor(
            obs_dim=obs_dim,
            hidden_sizes=cfg.hidden_sizes,
            dropout_rate=cfg.dropout_rate,
            layer_norm=cfg.layer_norm,
            activation=cfg.activation,
            use_residual=cfg.use_residual_blocks,
            use_swigлу=cfg.use_swigлу_gate,
        )
        self.q_head = nn.Linear(self.backbone.feature_dim, action_dim)

        nn.init.xavier_uniform_(self.q_head.weight, gain=1.0)
        nn.init.constant_(self.q_head.bias, 0.0)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.encode(obs)
        return self.q_head(features)


class SacDoubleCritic(nn.Module):
    """SAC-критик, использует две независимые сети для расчета Q-значений."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cfg: SacConfig,
        encoder_q1: FeatureExtractor | None = None,
        encoder_q2: FeatureExtractor | None = None,
    ) -> None:
        super().__init__()
        self.q1 = SacCritic(obs_dim, action_dim, cfg, encoder=encoder_q1)
        self.q2 = SacCritic(obs_dim, action_dim, cfg, encoder=encoder_q2)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q1_values = self.q1(obs)
        q2_values = self.q2(obs)
        return q1_values, q2_values



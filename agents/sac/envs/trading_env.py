from __future__ import annotations

from typing import Dict, Optional

from envs.sac_model.gym.crypto_trading_env_sac import CryptoTradingEnvOptimized
from envs.dqn_model.gym.gconfig import GymConfig


def make_trading_env(
    dfs: Dict,
    gym_cfg: Optional[GymConfig] = None,
) -> CryptoTradingEnvOptimized:
    """Фабрика окружения для SAC, переиспользующая оптимизированный DQN env."""

    cfg = gym_cfg or GymConfig()

    episode_length = getattr(cfg, "episode_length", None)
    lookback_window = getattr(cfg, "lookback_window", getattr(cfg, "window288", 288))

    env = CryptoTradingEnvOptimized(
        dfs=dfs,
        cfg=cfg,
        lookback_window=lookback_window,
        episode_length=episode_length,
    )

    env.dfs = dfs
    return env

#!/usr/bin/env python3
"""
Тесты для market STATE + action masking (minimal, fast).
"""

import numpy as np
import torch

from envs.dqn_model.gym.gconfig import GymConfig
from envs.dqn_model.gym.gutils_optimized import MarketState, compute_market_state
from envs.dqn_model.gym.crypto_trading_env_optimized import CryptoTradingEnvOptimized
from agents.vdqn.dqnsolver import DQNSolver


def _make_df(closes: list[float]) -> np.ndarray:
    c = np.asarray(closes, dtype=np.float32)
    o = np.roll(c, 1); o[0] = c[0]
    h = np.maximum(o, c) * 1.001
    l = np.minimum(o, c) * 0.999
    v = np.ones_like(c) * 1000.0
    return np.stack([o, h, l, c, v], axis=1).astype(np.float32)


def test_market_state_masking():
    # NORMAL
    df = _make_df([100.0] * 50)
    st = compute_market_state(50, df, roi_buf=[0.0] * 10, vol_buf=[100.0] * 10, trend_regime=0, atr_rel=0.001)
    assert st == MarketState.NORMAL

    # HIGH_VOL
    df = _make_df([100.0, 103.0] * 25)
    st = compute_market_state(50, df, roi_buf=[0.0] * 10, vol_buf=[100.0] * 10, trend_regime=0, atr_rel=0.007)
    assert st == MarketState.HIGH_VOL

    # PANIC (high ATR + sharp last drop)
    df = _make_df([100.0] * 48 + [100.0, 97.0])
    st = compute_market_state(50, df, roi_buf=[0.0] * 10, vol_buf=[100.0] * 10, trend_regime=0, atr_rel=0.012)
    assert st == MarketState.PANIC

    # DRAWDOWN proxy (sustained negative ROI + non-uptrend)
    df = _make_df([100.0] * 50)
    st = compute_market_state(50, df, roi_buf=[-0.04] * 10, vol_buf=[100.0] * 10, trend_regime=0, atr_rel=0.001)
    assert st == MarketState.DRAWDOWN

    # env.get_action_mask uses only market_state + flag
    env = CryptoTradingEnvOptimized.__new__(CryptoTradingEnvOptimized)
    env.cfg = GymConfig()
    env.cfg.use_state_action_mask = True
    env.market_state = MarketState.HIGH_VOL
    assert env.get_action_mask() == [1, 0, 1]
    env.cfg.use_state_action_mask = False
    assert env.get_action_mask() == [1, 1, 1]

    # dqnsolver.act respects mask in exploit + explore
    solver = DQNSolver(observation_space=10, action_space=3, load=False)

    class _Fixed(torch.nn.Module):
        def forward(self, x):
            q = torch.tensor([[0.0, 10.0, 0.0]], device=x.device)
            return q

    solver.model = _Fixed().to(solver.cfg.device).eval()
    state = np.zeros(10, dtype=np.float32)

    solver.epsilon = 0.0
    a = solver.act(state, action_mask=[1, 0, 1])
    assert a in (0, 2)

    solver.epsilon = 1.0
    for _ in range(50):
        a = solver.act(state, action_mask=[1, 0, 1])
        assert a in (0, 2)

    return True, "market STATE + action masking tests passed"


if __name__ == "__main__":
    ok, msg = test_market_state_masking()
    print("✅" if ok else "❌", msg)

"""
Глобальные (общие) overrides для DQN, применяемые ко ВСЕМ символам.

Это не только риск-менеджмент — сюда можно класть любые общие настройки,
которые имеют смысл применять одинаково (и которые безопасно менять ПОСЛЕ
создания env через set_env_attr_safe).

Важно:
- Параметры из GPU профиля (agents/vdqn/cfg/gpu_configs.py) сюда не кладём.
- Параметры, влияющие на сборку наблюдений при init env (например lookback_window),
  лучше задавать в месте создания env, а не тут.
"""

from __future__ import annotations

from typing import Any, Dict


GLOBAL_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # Параметры риск-менеджмента, маппятся на атрибуты env:
    # STOP_LOSS_PCT, TAKE_PROFIT_PCT, min_hold_steps, volume_threshold
    "risk_management": {
        "STOP_LOSS_PCT": -0.02,
        "TAKE_PROFIT_PCT": 0.02,
        "min_hold_steps": 12,      # 12 шагов * 5 минут = ~60 минут
        "volume_threshold": 0.003,
    },

    # Параметры позиционирования (если нужно сделать общими):
    # base_position_fraction, position_fraction, position_confidence_threshold
    # "position_sizing": {
    #     "base_position_fraction": 0.25,
    #     "position_fraction": 0.25,
    #     "position_confidence_threshold": 0.75,
    # },
}


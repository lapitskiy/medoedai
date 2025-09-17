from __future__ import annotations

"""
Символьные оверрайды конфигурации: per-symbol настройки без изменения общих дефолтов.

Идея: хранить словарь символов -> словарь переопределений:
{
  'TONUSDT': {
     'training_params': {...},
     'indicators_config': {...},
     'risk_management': {...}
  },
  ...
}

Применение:
 - training_params прокидываются в vDqnConfig через setattr
 - indicators_config передаётся в env
 - risk_management переносится в поля env (STOP_LOSS_PCT/TAKE_PROFIT_PCT/min_hold_steps/volume_threshold)
"""

from typing import Dict, Any

from agents.vdqn.hyperparameter.ton_optimized_config import TON_OPTIMIZED_CONFIG
from agents.vdqn.hyperparameter.bnb_optimized_config import BNB_OPTIMIZED_CONFIG


SYMBOL_OVERRIDES: Dict[str, Dict[str, Any]] = {
    'TONUSDT': TON_OPTIMIZED_CONFIG,
    'BNBUSDT': BNB_OPTIMIZED_CONFIG,
}


def get_symbol_override(symbol: str) -> Dict[str, Any] | None:
    if not symbol:
        return None
    key = symbol.upper()
    return SYMBOL_OVERRIDES.get(key)



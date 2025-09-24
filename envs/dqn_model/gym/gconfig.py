# config.py
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class GymConfig:            
    # --- базовые параметры обучения / эпизода ---------------------------------
    episode_length: int = 10_000          # шагов (5‑мин свечей) в эпизоде
    initial_balance: float = 1_000.0     # USDT
    trade_fee_percent: float = 0.00075   # 0.075 %

    # --- окно состояния -------------------------------------------------------
    lookback_window: int = 100            # 5‑мин свечей в history‑stack

    # --- торговый риск‑менеджмент --------------------------------------------
    position_fraction: float = 0.30      # % баланса входа «по умолчанию»
    max_hold_steps: int = 288            # 24 ч в 5‑мин свечах
    trailing_stop_steps: int = 12        # пример: 1 ч «хвоста»

    # --- фильтры и динамические пороги ---------------------------------------
    volatility_threshold: float = 0.001  # < 0.1 % → LOW VOLATILITY
    base_min_roi: float = 0.0005         # 0.05 %
    roi_volatility_factor: float = 1.5   # min_roi = base + k × vol

    # --- индикаторы -----------------------------------------------------------
    indicators_config: Dict[str, Any] = field(default_factory=lambda: {
        'rsi': {'length': 14},
        'ema': {'lengths': [100, 200]},
        'ema_cross': {
            'pairs': [(100, 200)],
            'include_cross_signal': True,
        },
        'sma': {'length': 14},
    })

    # --- вывод/логирование ----------------------------------------------------
    log_interval: int = 20  # печатать каждые N шагов
    wandb_project: str = "crypto-rl"
    
    window288 = 288          # 24 часа при 5‑мин свечке
    step_minutes = 5
    
    # --- масштабирование награды ---------------------------------------------
    reward_scale: float = 1.0  # коэффициент масштабирования reward (например, 10.0)
    
    #
    comission_kappa = 2_000.0
    
    #
    vol_regime_beta: float = 1.0
    vol_regime_alpha: float      = 0.5   # порог: median + alpha * IQR
    
    low_vol_fraction: float      = 0.4   # во сколько раз режем позицию при low-vol
    vol_gate_min_eps: float      = 0.10  # включать режим, когда ε <= этого значения
    low_vol_hold_penalty: float  = 0.01  # лёгкий штраф к reward при low-vol входе
    
    # --- Multi-step Learning параметры ---
    n_step: int = 3              # Количество шагов для n-step learning
    gamma: float = 0.99          # Discount factor для n-step returns
    
    # --- Параметры разделения данных для предотвращения look-ahead bias ---
    train_split_ratio: float = 0.8  # Доля данных для обучения (80% по умолчанию)

 
     
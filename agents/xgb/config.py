from __future__ import annotations

from dataclasses import dataclass


@dataclass
class XgbConfig:
    # Task
    # directional: 3-class {hold,buy,sell} from future-return
    # entry_long/exit_long/entry_short/exit_short: binary {hold,enter/exit} from PnL-based labels
    task: str = "directional"

    # Dataset / labeling
    horizon_steps: int = 12  # 12*5m=60m ahead
    threshold: float = 0.002  # 0.2% future return band for HOLD
    direction: str = "long"  # long|short (affects buy/sell mapping)
    val_ratio: float = 0.2

    # Features (keep feature shape stable for old models)
    # NOTE: If disabled, ATR column is kept but zeroed (so models can't learn from it).
    use_atr_feature: bool = True
    use_1m_microvol: bool = False
    use_1m_momentum: bool = False
    use_1m_candle_structure: bool = False
    use_1m_volume: bool = False
    use_1d_regime: bool = False
    use_sr_features: bool = False
    sr_lookback_steps: int = 288  # 24h on 5m candles
    sr_min_window_steps: int = 48  # wait for enough past candles before level features
    sr_touch_tolerance_atr: float = 0.25
    sr_swing_window_steps: int = 12
    sr_consolidation_window_steps: int = 24
    sr_consolidation_atr_threshold: float = 4.0

    # Entry/Exit labeling (position-aware)
    fee_bps: float = 6.0  # round-trip fee/slippage in basis points (0.01% = 1 bps)
    max_hold_steps: int = 48  # window in 5m steps for entry/exit labels
    min_profit: float = 0.0  # require pnl >= this to label enter/exit=1
    label_delta: float = 0.0005  # for exit: treat as "near best" if pnl_now >= best_future - delta
    entry_stride: int = 20  # sample entry points every N steps for exit dataset generation
    max_trades: int = 2000  # cap number of sampled trades for exit datasets

    # Inference policy for entry_* (binary): use predict_proba + threshold
    p_enter_threshold: float = 0.5  # 0.5 = default XGBoost decision boundary

    # Simple exit policy for entry_* backtests (optional). If None → disabled.
    entry_tp_pct: float | None = None      # e.g. 0.015 = +1.5%
    entry_sl_pct: float | None = None      # e.g. -0.01 = -1.0%
    entry_trail_pct: float | None = None   # e.g. 0.02 = 2% trailing from peak

    # Model
    n_estimators: int = 600
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    min_child_weight: float = 1.0
    gamma: float = 0.0
    scale_pos_weight: float = 1.0   # auto=-1 → count_neg/count_pos; 1.0=off
    early_stopping_rounds: int = 0  # 0 = disabled

    # Runtime
    random_state: int = 42
    n_jobs: int = 8


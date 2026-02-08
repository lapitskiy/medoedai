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

    # Entry/Exit labeling (position-aware)
    fee_bps: float = 6.0  # round-trip fee/slippage in basis points (0.01% = 1 bps)
    max_hold_steps: int = 48  # window in 5m steps for entry/exit labels
    min_profit: float = 0.0  # require pnl >= this to label enter/exit=1
    label_delta: float = 0.0005  # for exit: treat as "near best" if pnl_now >= best_future - delta
    entry_stride: int = 20  # sample entry points every N steps for exit dataset generation
    max_trades: int = 2000  # cap number of sampled trades for exit datasets

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


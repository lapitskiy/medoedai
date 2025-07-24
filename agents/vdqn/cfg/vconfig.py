from dataclasses import dataclass, field
from typing import Dict, Any
import torch
   
@dataclass
class vDqnConfig:  
    # === ε‑greedy exploration ===
    eps_start: float       = 1.0     # начальное ε
    eps_final: float       = 0.05    # минимальное ε
    eps_decay_steps: int   = 10_000  # за сколько шагов экспоненциально спадаем

    # === replay‑buffer ===
    memory_size: int       = 200_000
    batch_size: int        = 64
    prioritized: bool      = False   # включишь позже, когда дойдёшь до PER

    # === сеть / обучение ===
    lr: float              = 1e-3
    gamma: float           = 0.99
    soft_tau: float        = 1e-2    # τ для soft‑update target‑net
    soft_update_every = 4
    hidden_sizes: tuple    = (128, 64)  # MLP‑слои
    target_update_freq = 5_000

    # === логирование / сервисное ===
    grad_clip: float | None = None
    device: str             = "cuda"      # или "cpu"
    run_name: str           = "vanilla-dqn"

    # можно заранее готовить «место» под будущие фичи
    sac_size_agent: bool    = False
    transformer_actor: bool = False

    # === внутренние счётчики (инициализируются в коде, не в __init__) ===
    global_step: int        = field(init=False, default=0)    
    
    # path
    model_path="dqn_model.pth"
    buffer_path="replay_buffer.pkl"
    
    # Определение устройства (GPU или CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    
    
    
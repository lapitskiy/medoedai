from dataclasses import dataclass, field
from typing import Dict, Any
import torch
from datetime import datetime
   
@dataclass
class vDqnConfig:  
    # === ε‑greedy exploration ===
    eps_start: float       = 1.0     # начальное ε
    eps_final: float       = 0.01    # минимальное ε
    eps_decay_steps: int   = 25_000  # уменьшил для более быстрого обучения

    # === replay‑buffer ===
    memory_size: int       = 200_000  # уменьшил для ускорения
    batch_size: int        = 1024     # увеличил batch size для лучшего GPU использования
    prioritized: bool      = True     # Prioritized Experience Replay
    alpha: float           = 0.6      # приоритет для PER
    beta: float            = 0.4      # importance sampling для PER
    beta_increment: float  = 0.001    # увеличение beta

    # === сеть / обучение ===
    lr: float              = 1e-3     # увеличил learning rate для ускорения
    gamma: float           = 0.99     # discount factor
    soft_tau: float        = 1e-2     # увеличил для более быстрого обновления
    soft_update_every: int = 1        # обновляем каждый шаг для ускорения
    hidden_sizes: tuple    = (256, 128, 64)  # уменьшил размеры для ускорения
    target_update_freq: int = 1_000   # уменьшил для более частого обновления
    train_repeats: int     = 2        # уменьшил количество тренировок
    
    # === улучшения сети ===
    dropout_rate: float    = 0.1      # уменьшил dropout для ускорения
    layer_norm: bool       = True     # Layer Normalization
    double_dqn: bool       = True     # Double DQN
    dueling_dqn: bool      = True     # Dueling DQN
    
    # === градиентный клиппинг ===
    grad_clip: float       = 1.0      # градиентный клиппинг
    
    # === GPU оптимизации ===
    device: str             = "cuda"      # GPU
    run_name: str           = "fast-dqn"
    use_gpu_storage: bool   = True        # Хранить replay buffer на GPU
    use_torch_compile: bool = True        # PyTorch 2.x compile для максимального ускорения
    
    # === оптимизации скорости ===
    use_mixed_precision: bool = True   # Mixed Precision Training
    use_amp: bool          = True      # Automatic Mixed Precision
    num_workers: int       = 4         # количество worker процессов
    pin_memory: bool       = True      # pin memory для GPU
    prefetch_factor: int   = 2         # prefetch factor для DataLoader
    
    # === внутренние счётчики ===
    global_step: int        = field(init=False, default=0)    
    
    # path
    model_path="dqn_model.pth"
    buffer_path="replay_buffer.pkl"
    
    # Определение устройства (GPU или CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    use_wandb: bool = False
    csv_metrics_path: str = f"./{datetime.now().strftime('%H-%M')}metrics.csv"
        
    tick_every: int = 1000      # уменьшил для более частого логирования
    tick_slow_ms: float = 10.0  # уменьшил порог для логирования медленных операций
    
    
    
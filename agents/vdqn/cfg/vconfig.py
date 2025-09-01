from dataclasses import dataclass, field
from typing import Dict, Any
import torch
from datetime import datetime
   
@dataclass
class vDqnConfig:  
    # === ε‑greedy exploration ===
    eps_start: float       = 1.0     # начальное ε
    eps_final: float       = 0.05    # увеличил минимальное ε для большего исследования
    eps_decay_steps: int   = 1_000_000 # увеличил для более медленного затухания исследования

    # === replay‑buffer ===
    memory_size: int       = 200_000  # уменьшил для ускорения
    batch_size: int        = 256      # уменьшил batch size для большего разнообразия
    prioritized: bool      = True     # Prioritized Experience Replay
    alpha: float           = 0.6      # приоритет для PER
    beta: float            = 0.4      # importance sampling для PER
    beta_increment: float  = 0.001    # увеличение beta

    # === сеть / обучение ===
    lr: float              = 0.001     # уменьшил learning rate для стабильности
    gamma: float           = 0.99     # discount factor
    soft_tau: float        = 5e-3     # уменьшил для более стабильного обновления
    soft_update_every: int = 4        # обновляем реже для стабильности
    hidden_sizes: tuple    = (512, 256, 128)  # увеличил размеры для лучшей способности обучения
    target_update_freq: int = 5_000   # увеличил для более стабильного обучения
    train_repeats: int     = 1        # уменьшил количество тренировок
    
    # === улучшения сети ===
    dropout_rate: float    = 0.2      # увеличил dropout для регуляризации
    layer_norm: bool       = True     # Layer Normalization
    double_dqn: bool       = True     # Double DQN
    dueling_dqn: bool      = True     # Dueling DQN
    
    # === градиентный клиппинг ===
    grad_clip: float       = 1.0      # градиентный клиппинг
    
    # === GPU оптимизации ===
    device: str             = "cuda"      # GPU
    run_name: str           = "stable-dqn"
    use_gpu_storage: bool   = True        # Хранить replay buffer на GPU
    use_torch_compile: bool = True        # PyTorch 2.x compile для максимального ускорения
    torch_compile_fallback: bool = True   # Автоматический fallback для старых GPU
    torch_compile_force_disable: bool = False  # Принудительно отключить torch.compile
    
    def __post_init__(self):
        """Проверяем переменные окружения после инициализации"""
        import os
        
        # Проверяем переменную окружения для отключения torch.compile
        if os.environ.get('DISABLE_TORCH_COMPILE', 'false').lower() == 'true':
            self.use_torch_compile = False
            self.torch_compile_force_disable = True
            print("⚠️ torch.compile отключен через переменную окружения DISABLE_TORCH_COMPILE=true")
    
    # === оптимизации скорости ===
    use_mixed_precision: bool = True   # Mixed Precision Training
    use_amp: bool          = True      # Automatic Mixed Precision
    num_workers: int       = 4         # количество worker процессов
    pin_memory: bool       = True      # pin memory для GPU
    prefetch_factor: int   = 2         # prefetch factor для DataLoader
    
    # === сохранение модели ===
    save_frequency: int    = 50        # Сохранять модель каждые N эпизодов
    save_only_on_improvement: bool = False  # Сохранять только при улучшении winrate
    
    # === Early Stopping параметры ===
    early_stopping_patience: int = 3000  # Базовый patience для early stopping
    min_episodes_before_stopping: int = 1000  # Увеличено с 500 до 1000
    early_stopping_trend_threshold: float = 0.03  # Увеличено с 0.02 до 0.03 для более мягкого stopping
    long_term_patience_multiplier: float = 2.5  # Увеличено с 2.0 до 2.5
    
    # === Rainbow DQN параметры ===
    use_n_step_learning: bool = True      # Включить n-step learning
    n_step: int = 3                       # Количество шагов для n-step learning
    use_distributional_rl: bool = False   # Включить Distributional RL (пока отключено)
    n_atoms: int = 51                     # Количество атомов для Distributional RL
    v_min: float = -10.0                  # Минимальное значение для Distributional RL
    v_max: float = 10.0                   # Максимальное значение для Distributional RL
    use_noisy_networks: bool = True       # Включить Noisy Networks
    
    # === внутренние счётчики ===
    global_step: int        = field(init=False, default=0)    
    
    # path
    model_path="dqn_model.pth"
    buffer_path="replay_buffer.pkl"
    
    # Определение устройства (GPU или CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    use_wandb: bool = False
        
    tick_every: int = 1000      # уменьшил для более частого логирования
    tick_slow_ms: float = 10.0  # уменьшил порог для логирования медленных операций
    
    
    
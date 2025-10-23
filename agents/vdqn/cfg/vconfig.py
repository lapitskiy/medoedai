from dataclasses import dataclass, field
from typing import Dict, Any
import os
import torch
from datetime import datetime
from .gpu_configs import get_optimal_config, apply_gpu_config_to_vconfig
   
@dataclass
class vDqnConfig:  
    # === ε‑greedy exploration ===
    eps_start: float       = 1.0     # начальное ε
    eps_final: float       = 0.005   # быстрее выйти в эксплуатацию
    eps_decay_steps: int   = 1_200_000 # ориентируемся на ~25% от плановых шагов

    # === replay‑buffer ===
    memory_size: int       = 200_000  # будет переопределено GPU-конфигом
    batch_size: int        = 4096     # будет переопределено GPU-конфигом
    prioritized: bool      = True     # Prioritized Experience Replay
    alpha: float           = 0.6      # приоритет для PER
    beta: float            = 0.4      # importance sampling для PER
    beta_increment: float  = 0.001    # увеличение beta

    # === сеть / обучение ===
    lr: float              = 3e-4     # базовый lr; может быть переопределен GPU-конфигом
    gamma: float           = 0.99     # discount factor
    soft_tau: float        = 1e-2     # мягкие обновления таргета
    soft_update_every: int = 1        # применяем каждый шаг
    hidden_sizes: tuple    = (512, 256, 128)  # компактная архитектура по умолчанию
    target_update_freq: int = 5_000   # увеличил для более стабильного обучения
    train_repeats: int     = 2        # меньше переобучения на свежем реплее
    
    # === улучшения сети ===
    dropout_rate: float    = 0.1      # умеренная регуляризация для DQN
    layer_norm: bool       = True     # Layer Normalization
    double_dqn: bool       = True     # Double DQN
    dueling_dqn: bool      = True     # Dueling DQN
    activation: str        = 'silu'    # Activation for MLP feature blocks
    use_residual_blocks: bool = True  # Residual skip connections inside MLP
    use_swiglu_gate: bool  = True     # SwiGLU gating inside MLP blocks
    
    # === градиентный клиппинг ===
    grad_clip: float       = 0.5      # более жесткий клиппинг градиентов
    
    # === GPU оптимизации ===
    device: str             = "cuda"      # GPU (быстрее, но DDR4 используется минимально)
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
        
        # Автоматическое определение GPU и применение оптимальных настроек
        self._apply_gpu_optimization()
    
    def _apply_gpu_optimization(self):
        """Автоматически определяет GPU и применяет оптимальные настройки"""
        try:
            # Получаем оптимальную конфигурацию для текущей GPU
            gpu_config = get_optimal_config()
            
            # Применяем настройки
            gpu_settings = apply_gpu_config_to_vconfig(gpu_config)
            
            # Обновляем параметры
            self.batch_size = gpu_settings['batch_size']
            self.memory_size = gpu_settings['memory_size']
            self.hidden_sizes = gpu_settings['hidden_sizes']
            self.train_repeats = gpu_settings['train_repeats']
            self.use_amp = gpu_settings['use_amp']
            self.use_gpu_storage = gpu_settings['use_gpu_storage']
            self.lr = gpu_settings['learning_rate']
            self.use_torch_compile = gpu_settings['use_torch_compile']
            # Новые параметры из GPU-конфига
            try:
                self.eps_decay_steps = gpu_settings.get('eps_decay_steps', self.eps_decay_steps)
            except Exception:
                pass
            try:
                self.dropout_rate = gpu_settings.get('dropout_rate', self.dropout_rate)
            except Exception:
                pass
            
            print(f"🚀 Применены оптимальные настройки для {gpu_config.name}")
            
        except Exception as e:
            print(f"⚠️ Ошибка применения GPU оптимизации: {e}")
            print("🔄 Используем настройки по умолчанию")
    
    # === оптимизации скорости ===
    use_mixed_precision: bool = True   # Mixed Precision Training
    use_amp: bool          = True      # Automatic Mixed Precision
    num_workers: int       = 8         # количество worker процессов (ускорение на V100)
    pin_memory: bool       = True      # pin memory для GPU
    prefetch_factor: int   = 4         # prefetch factor для DataLoader
    
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
    encoder_path="encoder_only.pth"
    
    # Определение устройства (GPU или CPU), учитываем GPU_COUNT
    _gpu_count_env = os.environ.get('GPU_COUNT', '').strip().lower()
    _want_gpu = False
    try:
        _want_gpu = (_gpu_count_env != '' and _gpu_count_env not in ('0', 'false', 'no', 'off'))
    except Exception:
        _want_gpu = False
    device = torch.device("cuda" if (_want_gpu and torch.cuda.is_available()) else "cpu")    
    
    use_wandb: bool = False
        
    tick_every: int = 1000      # уменьшил для более частого логирования
    tick_slow_ms: float = 10.0  # уменьшил порог для логирования медленных операций
    
    
    
"""Конфигурация SAC, основанная на параметрах DQN."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Tuple
import os

from agents.sac.cfg.sac_gpu_configs import apply_sac_gpu_config_to_sacconfig, get_optimal_sac_config, SACGPUConfig


def _symbol_code(sym: str) -> str:
    """Извлекает код символа для создания папки (аналогично DQN)"""
    if not isinstance(sym, str) or not sym:
        return "model"
    s = sym.upper().replace('/', '')
    for suffix in ["USDT", "USD", "USDC", "BUSD", "USDP"]:
        if s.endswith(suffix):
            s = s[:-len(suffix)]
            break
    s = s.lower() if s else "model"
    if s in ("мультивалюта", "multi", "multicrypto"):
        s = "multi"
    return s


@dataclass
class SacConfig:
    """Базовые гиперпараметры Soft Actor-Critic.

    Большая часть полей повторяет `vDqnConfig`, чтобы сохранить схожий pipeline.
    Специфичные для SAC параметры добавлены ниже.
    """

    # === Основные параметры обучения ===
    gamma: float = 0.99
    tau: float = 5e-3  # soft update
    lr_actor: float = 3e-4
    lr_critic: float = 1e-4 # Уменьшено по запросу
    lr_alpha: float = 3e-4
    target_entropy_scale: float = 1.0

    batch_size: int = 256  # Оптимально для GTX 1660 Super
    memory_size: int = 90_000  # Оптимально для GTX 1660 Super (~2.7GB VRAM)
    gradient_steps: int = 1
    updates_per_step: float = 1.0

    # === Сеть ===
    hidden_sizes: Tuple[int, ...] = (512, 256, 128)  # Оптимально для GTX 1660 Super
    activation: str = "silu"
    layer_norm: bool = True
    dropout_rate: float = 0.1
    use_residual_blocks: bool = True
    use_swiglu_gate: bool = False

    # === Обучение ===
    start_learning_after: int = 20_000 # Увеличено с 10_000 до 20_000
    target_update_interval: int = 1
    max_grad_norm: float | None = 1.0 # Включено по запросу
    min_lr_actor: float = 1e-6
    min_lr_critic: float = 1e-6
    min_lr_alpha: float = 1e-6
    grad_fail_patience: int = 3
    grad_fail_max_decays: int = 4
    grad_fail_lr_decay: float = 0.5
    grad_fail_clear_buffer: bool = True
    grad_fail_log_details: bool = True
    clear_buffer_on_nan: bool = False
    nan_critic_grad_count: int = 0  # Счетчик NaN в градиентах критика
    nan_alpha_count: int = 0      # Счетчик NaN в альфе
    use_amp: bool = True
    
    # === Обработка экстремальных входов ===
    obs_clip_value: float = 1e4
    obs_hard_limit: float = 1e6
    reward_clip_value: float = 1e3
    reward_hard_limit: float = 1e4
    drop_batch_on_extreme: bool = False
    warn_on_clipped_inputs: bool = True
    extreme_log_interval: int = 500
    
    # === Логирование и пути ===
    run_name: str = field(default_factory=lambda: f"sac-{datetime.utcnow():%Y%m%d-%H%M%S}")
    model_path: str = "sac/models/agent.pt"
    encoder_path: str = "sac/models/encoder_only.pth"
    replay_buffer_path: str = "sac/result/replay_buffer.pt"
    result_dir: str = "sac/result"

    # === Окружение ===
    reward_scale: float = 1.0
    normalize_rewards: bool = False
    use_adaptive_risk: bool = True
    train_episodes: int = 1000
    max_episode_steps: int | None = 1000 # Ограничено по запросу
    lookback_window: int = 60

    # === Валидация ===
    validation_interval: int = 50
    validation_episodes: int = 10
    
    # === Early Stopping параметры ===
    early_stopping_patience: int = 3000  # Базовый patience для early stopping
    min_episodes_before_stopping: int = 1000  # Минимум эпизодов до ранней остановки
    early_stopping_trend_threshold: float = 0.03  # Порог для анализа трендов
    long_term_patience_multiplier: float = 2.5  # Множитель для долгосрочного patience
    save_only_on_improvement: bool = True  # Сохранять только при улучшении метрик

    # === Технические параметры ===
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    use_torch_compile: bool = True
    torch_compile_mode: str = "default"
    torch_compile_force_disable: bool = False

    # === Для автоматического выбора GPU настроек ===
    _gpu_config_applied: bool = field(init=False, default=False)

    # === Логирование / мониторинг ===
    use_wandb: bool = False
    wandb_project: str | None = None
    tick_every: int = 1000
    tick_slow_ms: float = 10.0

    # === Сервисные поля ===
    extra: Dict[str, Any] = field(default_factory=dict)
    agent_type: str = "sac"
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        # Инициализируем extra если оно пустое
        if not hasattr(self, 'extra') or not self.extra:
            object.__setattr__(self, 'extra', {})

        self._apply_gpu_optimization()
        # Не вызываем _ensure_result_paths здесь, чтобы символ можно было установить позже

    def apply_gpu_config(self, gpu_config: SACGPUConfig) -> None:
        """Применяет настройки SAC GPU к конфигурации"""
        apply_sac_gpu_config_to_sacconfig(self, gpu_config)

    # GPU настройки используют SAC утилиты
    def _apply_gpu_optimization(self) -> None:
        try:
            gpu_config = get_optimal_sac_config()
            self.apply_gpu_config(gpu_config)
            self._gpu_config_applied = True
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Не удалось применить SAC GPU конфигурацию: {exc}")
            self._gpu_config_applied = False

    def _ensure_result_paths(self) -> None:
        """Создает структуру папок для результатов обучения"""
        # Получаем символ из extra или используем "sac" по умолчанию
        symbol = 'sac'  # По умолчанию
        if hasattr(self, 'extra') and self.extra and 'symbol' in self.extra:
            symbol = self.extra['symbol']
        symbol_code = _symbol_code(symbol)

        # Создаем структуру как в DQN: result/sac/{SYMBOL}/runs/{run_name}/
        base_dir = os.path.join("result", "sac", symbol_code)
        os.makedirs(base_dir, exist_ok=True)

        runs_dir = os.path.join(base_dir, "runs")
        os.makedirs(runs_dir, exist_ok=True)

        run_dir = os.path.join(runs_dir, self.run_name)
        os.makedirs(run_dir, exist_ok=True)

        self.result_dir = run_dir
        self.model_path = os.path.join(run_dir, "model.pth")
        self.encoder_path = os.path.join(run_dir, "encoder_only.pth")
        self.replay_buffer_path = os.path.join(run_dir, "replay.pkl")

    def update_result_paths(self) -> None:
        """Пересчитывает пути результатов после изменения символа"""
        self._ensure_result_paths()

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def build_sac_config(overrides: Dict[str, Any] | None = None) -> SacConfig:
    cfg = SacConfig()
    if overrides:
        for key, value in overrides.items():
            setattr(cfg, key, value)
    return cfg



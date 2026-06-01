"""
Конфигурации для разных GPU карт
Автоматический выбор batch_size и других параметров в зависимости от GPU
"""

from dataclasses import dataclass
from typing import Dict, Any
import os
import torch
from utils.time_log import msk_tag

@dataclass
class GPUConfig:
    """Конфигурация для конкретной GPU"""
    name: str
    vram_gb: float
    batch_size: int
    memory_size: int
    hidden_sizes: tuple
    train_repeats: int
    use_amp: bool
    use_gpu_storage: bool
    learning_rate: float
    description: str
    use_torch_compile: bool = False
    eps_decay_steps: int = 100000
    dropout_rate: float = 0.1

# Конфигурации для разных GPU
GPU_CONFIGS: Dict[str, GPUConfig] = {
    # Tesla P100 - максимальная скорость эпизодов
    "tesla_p100": GPUConfig(
        name="Tesla P100",
        vram_gb=16.0,
        batch_size=128,  # Уменьшено с 4096 для сокращения времени обучения
        memory_size=90_000,  # Увеличиваем для стабильности: ~7.5GB VRAM (47% от 16GB)
        hidden_sizes=(256, 128, 64),  # Сбалансированная архитектура
        train_repeats=2,  # Уменьшено с 4 для сокращения времени обучения
        use_amp=True,
        use_gpu_storage=False,  # Включаем для стабильности
        learning_rate=0.00015,  # Уменьшаем для стабильности с большим батчем
        description="Максимальная скорость эпизодов для Tesla P100 (используем все CPU ядра)",
        use_torch_compile=True,
        eps_decay_steps=2_500_000,
        dropout_rate=0.1
    ),
    
    # Tesla V100 - еще быстрее с Tensor Cores
    "tesla_v100": GPUConfig(
        name="Tesla V100",
        vram_gb=16.0, # 
        batch_size=192,
        memory_size=100_000,  # Уменьшено для снижения пиков памяти (реплей-буфер)
        hidden_sizes=(384, 256, 192), #hidden_sizes= (1024, 512, 256) (winrate 36%),  hidden_sizes=(384, 192, 96),
        train_repeats=2,
        use_amp=True,
        use_gpu_storage=False,  # Включаем GPU storage для синхронизации устройств
        learning_rate=0.00015,
        description="Оптимальная конфигурация для Tesla V100 с Tensor Cores",
        use_torch_compile=True,
        eps_decay_steps=2_500_000,
        dropout_rate=0.1
        #vram_gb=16.0,
        #batch_size=2048,
        #memory_size=300_000,
        #hidden_sizes=(1536, 768, 384),
        #train_repeats=2,
        #use_amp=True,
        #use_gpu_storage=True,
        #learning_rate=0.0001,
          
        #description="Максимальная производительность для Tesla V100 с Tensor Cores",
        #use_torch_compile=True
    ),
    
    # GTX 1660 Super - оптимальная для 6GB VRAM
    "gtx_1660_super": GPUConfig(
        name="GTX 1660 Super",
        vram_gb=6.0,
        batch_size=64,
        memory_size=50_000,  # Увеличиваем до ~2.7GB VRAM (45% от 6GB)
        hidden_sizes=(96, 64, 48),
        train_repeats=2,
        use_amp=False,
        use_gpu_storage=False,  # Включаем GPU storage для синхронизации устройств
        learning_rate=0.00015,        
        use_torch_compile=False,
        eps_decay_steps=5_000_000,
        dropout_rate=0.1,
        description="Оптимальная конфигурация для GTX 1660 Super"
    ),
    
    # RTX 3080 - высокопроизводительная карта
    "rtx_3080": GPUConfig(
        name="RTX 3080",
        vram_gb=10.0,
        batch_size=2048,
        memory_size=1_000_000,
        hidden_sizes=(1024, 512, 256),
        train_repeats=4,
        use_amp=True,
        use_gpu_storage=True,
        learning_rate=0.0001,
        description="Высокая производительность для RTX 3080",
        use_torch_compile=True,
        eps_decay_steps=3_000_000,
        dropout_rate=0.25
    ),
    
    # RTX 4090 - топовая карта
    "rtx_4090": GPUConfig(
        name="RTX 4090",
        vram_gb=24.0,
        batch_size=8192,
        memory_size=5_000_000,
        hidden_sizes=(4096, 2048, 1024),
        train_repeats=16,
        use_amp=True,
        use_gpu_storage=True,
        learning_rate=0.0001,
        description="Максимальная производительность для RTX 4090",
        use_torch_compile=True,
        eps_decay_steps=3_000_000,
        dropout_rate=0.25
    ),
    
    # CPU fallback
    "cpu": GPUConfig(
        name="CPU",
        vram_gb=0.0,
        batch_size=32,
        memory_size=50_000,
        hidden_sizes=(256, 128, 64),
        train_repeats=1,
        use_amp=False,
        use_gpu_storage=False,
        learning_rate=0.0001,
        description="Fallback конфигурация для CPU",
        use_torch_compile=False,
        eps_decay_steps=3_000_000,
        dropout_rate=0.25
    )
}

# Флаги для подавления дублирующегося вывода
_gpu_detect_printed = False
_gpu_info_printed = False
_gpu_forced_source_printed = False
_gpu_forced_invalid_printed = False
_gpu_compile_settings_printed = False

def _get_forced_gpu_from_settings() -> str:
    """
    Пробует получить FORCE_GPU_CONFIG из таблицы app_settings (UI /settings).
    Возвращает '' если не задано/ошибка/невалидно.
    """
    try:
        # Lazy import: чтобы не тянуть DB/SQLAlchemy на раннем импорте модулей
        from utils.settings_store import get_setting_value as _get_setting_value  # type: ignore

        # Сначала читаем из scope=rl, group=gpu, key=FORCE_GPU_CONFIG
        v = _get_setting_value('rl', 'gpu', 'FORCE_GPU_CONFIG')
        if v is None:
            # fallback: scope=rl, group=None
            v = _get_setting_value('rl', None, 'FORCE_GPU_CONFIG')
        s = str(v or '').strip().lower()
        if s in ('', 'auto', 'none', 'null'):
            return ''
        return s
    except Exception:
        return ''

def _get_bool_setting(scope: str, group: str | None, key: str) -> bool | None:
    """Читает bool из app_settings. Возвращает True/False или None если не задано/не распознано."""
    try:
        from utils.settings_store import get_setting_value as _get_setting_value  # type: ignore
        v = _get_setting_value(scope, group, key)
        if v is None:
            return None
        s = str(v).strip().lower()
        if s in ('1', 'true', 'yes', 'y', 'on'):
            return True
        if s in ('0', 'false', 'no', 'n', 'off'):
            return False
        return None
    except Exception:
        return None

def _apply_compile_override_from_settings(cfg: GPUConfig) -> GPUConfig:
    """
    Применяет override torch.compile из /settings (app_settings).
    Ключ: scope=rl, group=gpu, key=DISABLE_TORCH_COMPILE (bool).
    """
    try:
        disable = _get_bool_setting('rl', 'gpu', 'DISABLE_TORCH_COMPILE')
        if disable is None:
            # fallback: group=None
            disable = _get_bool_setting('rl', None, 'DISABLE_TORCH_COMPILE')
        if disable is None:
            return cfg
        if disable is True:
            if cfg.use_torch_compile:
                return GPUConfig(**{**cfg.__dict__, 'use_torch_compile': False})
            return cfg
        # disable == False -> явно разрешаем, оставляя значение из GPU профиля
        return cfg
    except Exception:
        return cfg

def detect_gpu() -> str:
    """
    Автоматически определяет тип GPU и возвращает ключ конфигурации
    """
    if not torch.cuda.is_available():
        return "cpu"
    
    try:
        # Получаем информацию о GPU
        gpu_name = torch.cuda.get_device_name(0).lower()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        global _gpu_detect_printed
        if not _gpu_detect_printed:
            print(msk_tag(f"🔍 Обнаружена GPU: {torch.cuda.get_device_name(0)}"))
            print(msk_tag(f"💾 VRAM: {gpu_memory:.1f} GB"))
            _gpu_detect_printed = True
        
        # Определяем тип GPU по названию
        if "tesla v100" in gpu_name:
            return "tesla_v100"
        elif "tesla p100" in gpu_name:
            return "tesla_p100"
        elif "gtx 1660" in gpu_name or "1660 super" in gpu_name:
            return "gtx_1660_super"
        elif "rtx 3080" in gpu_name:
            return "rtx_3080"
        elif "rtx 4090" in gpu_name:
            return "rtx_4090"
        else:
            # Для неизвестных GPU выбираем конфигурацию по VRAM
            if gpu_memory >= 20:
                return "rtx_4090"  # 24GB+
            elif gpu_memory >= 15:
                # Для 16GB выбираем V100 если доступны Tensor Cores, иначе P100
                if "volta" in gpu_name or "tensor" in gpu_name:
                    return "tesla_v100"  # 16GB с Tensor Cores
                else:
                    return "tesla_p100"  # 16GB без Tensor Cores
            elif gpu_memory >= 8:
                return "rtx_3080"  # 10GB
            elif gpu_memory >= 4:
                return "gtx_1660_super"  # 6GB
            else:
                return "cpu"  # <4GB
                
    except Exception as e:
        print(f"⚠️ Ошибка определения GPU: {e}")
        return "cpu"

def get_gpu_config(gpu_key: str = None) -> GPUConfig:
    global _gpu_info_printed
    """
    Получает конфигурацию для указанной GPU или автоматически определяет
    """
    if gpu_key is None:
        # 0) Пробуем принудительный выбор из /settings (app_settings)
        forced_gpu = _get_forced_gpu_from_settings()
        if forced_gpu:
            global _gpu_forced_source_printed
            if forced_gpu in GPU_CONFIGS:
                if not _gpu_forced_source_printed:
                    print(msk_tag(f"🔧 Принудительно выбрана GPU конфигурация (settings): {forced_gpu}"))
                    _gpu_forced_source_printed = True
                gpu_key = forced_gpu
            else:
                global _gpu_forced_invalid_printed
                if not _gpu_forced_invalid_printed:
                    print(msk_tag(f"⚠️ settings FORCE_GPU_CONFIG='{forced_gpu}' не распознана; доступно: {sorted(GPU_CONFIGS.keys())}"))
                    _gpu_forced_invalid_printed = True
                gpu_key = detect_gpu()
        else:
            # Никаких env-фолбэков: единственный источник forced — app_settings (/settings).
            gpu_key = detect_gpu()
    
    if gpu_key not in GPU_CONFIGS:
        print(msk_tag(f"⚠️ Неизвестная GPU: {gpu_key}, используем CPU конфигурацию"))
        gpu_key = "cpu"
    
    config = GPU_CONFIGS[gpu_key]
    # Применяем override torch.compile только из /settings (никаких env-фолбеков)
    config = _apply_compile_override_from_settings(config)
    if not _gpu_info_printed:
        print(msk_tag(f"✅ Выбрана конфигурация: {config.name}"))
        print(msk_tag(f"📊 Batch size: {config.batch_size}"))
        print(msk_tag(f"💾 Memory size: {config.memory_size}"))
        print(msk_tag(f"🧠 Hidden sizes: {config.hidden_sizes}"))
        print(msk_tag(f"🔄 Train repeats: {config.train_repeats}"))
        print(msk_tag(f"⚡ AMP: {config.use_amp}"))
        print(msk_tag(f"💾 GPU storage: {config.use_gpu_storage}"))
        print(msk_tag(f"🧩 torch.compile: {config.use_torch_compile}"))
        _gpu_info_printed = True

    # Отдельный лог про настройки compile из settings (один раз)
    global _gpu_compile_settings_printed
    if not _gpu_compile_settings_printed:
        try:
            disable = _get_bool_setting('rl', 'gpu', 'DISABLE_TORCH_COMPILE')
            if disable is None:
                disable = _get_bool_setting('rl', None, 'DISABLE_TORCH_COMPILE')
            if disable is True:
                print(msk_tag("⚠️ torch.compile отключен через настройки (/settings): DISABLE_TORCH_COMPILE=true"))
            elif disable is False:
                print(msk_tag("✅ torch.compile разрешен через настройки (/settings): DISABLE_TORCH_COMPILE=false"))
        except Exception:
            pass
        _gpu_compile_settings_printed = True
    
    return config

def apply_gpu_config_to_vconfig(gpu_config: GPUConfig) -> Dict[str, Any]:
    """
    Применяет GPU конфигурацию к vDqnConfig
    """
    return {
        'batch_size': gpu_config.batch_size,
        'memory_size': gpu_config.memory_size,
        'hidden_sizes': gpu_config.hidden_sizes,
        'train_repeats': gpu_config.train_repeats,
        'use_amp': gpu_config.use_amp,
        'use_gpu_storage': gpu_config.use_gpu_storage,
        'learning_rate': gpu_config.learning_rate,
        'use_torch_compile': gpu_config.use_torch_compile,
        'eps_decay_steps': gpu_config.eps_decay_steps,
        'dropout_rate': gpu_config.dropout_rate
    }

# Функция для быстрого получения настроек
def get_optimal_config() -> GPUConfig:
    """Получает оптимальную конфигурацию для текущей GPU"""
    return get_gpu_config()

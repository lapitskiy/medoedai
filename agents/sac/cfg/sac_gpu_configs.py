"""
Конфигурации для SAC агента под разные GPU карты
Автоматический выбор параметров в зависимости от GPU для оптимальной производительности SAC
"""

from dataclasses import dataclass
from typing import Dict, Any
import os
import torch

@dataclass
class SACGPUConfig:
    """Конфигурация для SAC агента под конкретную GPU"""
    name: str
    vram_gb: float
    batch_size: int
    memory_size: int
    hidden_sizes: tuple
    use_amp: bool
    learning_rate: float
    target_entropy_scale: float
    description: str

# Конфигурации для SAC под разные GPU
SAC_GPU_CONFIGS: Dict[str, SACGPUConfig] = {
    # Tesla P100 - максимальная скорость для SAC
    "tesla_p100": SACGPUConfig(
        name="Tesla P100",
        vram_gb=16.0,
        batch_size=512,  # SAC работает лучше с батчами 256-512
        memory_size=150_000,  # Больше памяти для стабильного обучения SAC
        hidden_sizes=(1024, 512, 256),
        use_amp=True,
        learning_rate=3e-4,
        target_entropy_scale=1.0,
        description="Высокая производительность для SAC на Tesla P100"
    ),

    # Tesla V100 - с Tensor Cores для SAC
    "tesla_v100": SACGPUConfig(
        name="Tesla V100",
        vram_gb=16.0,
        batch_size=384,
        memory_size=120_000,
        hidden_sizes=(1024, 512, 256),
        use_amp=False,
        learning_rate=1e-4,
        target_entropy_scale=0.9,
        description="Сбалансированная конфигурация SAC для Tesla V100 с упором на стабильность"
    ),

    # GTX 1660 Super - оптимальная для SAC
    "gtx_1660_super": SACGPUConfig(
        name="GTX 1660 Super",
        vram_gb=6.0,
        batch_size=256,  # Оптимально для SAC
        memory_size=20_000,  # Дополнительно уменьшено с 30_000
        hidden_sizes=(512, 256, 128),  # Компактные сети для SAC
        use_amp=True,  # Включаем для экономии памяти
        learning_rate=1e-5,
        target_entropy_scale=1.0,
        description="Оптимальная конфигурация SAC для GTX 1660 Super"
    ),

    # RTX 3080 - высокопроизводительная для SAC
    "rtx_3080": SACGPUConfig(
        name="RTX 3080",
        vram_gb=10.0,
        batch_size=512,
        memory_size=200_000,  # Можно позволить больше памяти
        hidden_sizes=(1024, 512, 256),
        use_amp=True,
        learning_rate=3e-4,
        target_entropy_scale=1.0,
        description="Высокая производительность SAC для RTX 3080"
    ),

    # RTX 4090 - топовая карта для SAC
    "rtx_4090": SACGPUConfig(
        name="RTX 4090",
        vram_gb=24.0,
        batch_size=1024,  # Максимальный батч для SAC
        memory_size=500_000,  # Много памяти для SAC
        hidden_sizes=(2048, 1024, 512),  # Большие сети для максимальной производительности
        use_amp=True,
        learning_rate=3e-4,
        target_entropy_scale=1.0,
        description="Максимальная производительность SAC для RTX 4090"
    ),

    # CPU fallback для SAC
    "cpu": SACGPUConfig(
        name="CPU",
        vram_gb=0.0,
        batch_size=64,  # Маленький батч для CPU
        memory_size=25_000,  # Минимум для SAC обучения
        hidden_sizes=(256, 128, 64),  # Компактные сети для CPU
        use_amp=False,  # Нет смысла на CPU
        learning_rate=1e-3,  # Ниже learning rate для стабильности на CPU
        target_entropy_scale=1.0,
        description="Минимальная конфигурация SAC для CPU"
    )
}

def detect_gpu() -> str:
    """
    Автоматически определяет тип GPU и возвращает ключ конфигурации для SAC
    """
    if not torch.cuda.is_available():
        return "cpu"

    try:
        # Получаем информацию о GPU
        gpu_name = torch.cuda.get_device_name(0).lower()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        print(f"🔍 Обнаружена GPU для SAC: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {gpu_memory:.1f} GB")

        # Определяем тип GPU по названию (адаптировано для SAC)
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
        print(f"⚠️ Ошибка определения GPU для SAC: {e}")
        return "cpu"

def get_sac_gpu_config(gpu_key: str = None) -> SACGPUConfig:
    """
    Получает конфигурацию для указанной GPU или автоматически определяет
    """
    if gpu_key is None:
        # Проверяем переменную окружения для принудительного выбора GPU
        forced_gpu = os.environ.get('FORCE_SAC_GPU_CONFIG', '').strip().lower()
        if forced_gpu and forced_gpu in SAC_GPU_CONFIGS:
            print(f"🔧 Принудительно выбрана SAC GPU конфигурация: {forced_gpu}")
            gpu_key = forced_gpu
        else:
            gpu_key = detect_gpu()

    if gpu_key not in SAC_GPU_CONFIGS:
        print(f"⚠️ Неизвестная SAC GPU: {gpu_key}, используем CPU конфигурацию")
        gpu_key = "cpu"

    config = SAC_GPU_CONFIGS[gpu_key]
    print(f"✅ Выбрана SAC конфигурация: {config.name}")
    print(f"📊 Batch size: {config.batch_size}")
    print(f"💾 Memory size: {config.memory_size}")
    print(f"🧠 Hidden sizes: {config.hidden_sizes}")
    print(f"⚡ AMP: {config.use_amp}")

    return config

def apply_sac_gpu_config_to_sacconfig(sac_config, gpu_config: SACGPUConfig) -> None:
    """
    Применяет SAC GPU конфигурацию к SacConfig
    """
    # Базовые параметры
    sac_config.batch_size = gpu_config.batch_size
    sac_config.memory_size = gpu_config.memory_size
    sac_config.hidden_sizes = gpu_config.hidden_sizes
    sac_config.use_amp = gpu_config.use_amp
    sac_config.lr_actor = gpu_config.learning_rate
    sac_config.lr_critic = gpu_config.learning_rate
    sac_config.lr_alpha = gpu_config.learning_rate
    sac_config.target_entropy_scale = gpu_config.target_entropy_scale

    print(f"🔧 SAC применены GPU настройки для {gpu_config.name}")

# Функция для быстрого получения настроек
def get_optimal_sac_config() -> SACGPUConfig:
    """Получает оптимальную конфигурацию для текущей GPU для SAC"""
    return get_sac_gpu_config()

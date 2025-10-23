"""
Конфигурации для разных GPU карт
Автоматический выбор batch_size и других параметров в зависимости от GPU
"""

from dataclasses import dataclass
from typing import Dict, Any
import os
import torch

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
        train_repeats=3,  # Уменьшено с 4 для сокращения времени обучения
        use_amp=False,
        use_gpu_storage=False,  # Включаем для стабильности
        learning_rate=0.00015,  # Уменьшаем для стабильности с большим батчем
        description="Максимальная скорость эпизодов для Tesla P100 (используем все CPU ядра)",
        use_torch_compile=True,
        eps_decay_steps=3_000_000,
        dropout_rate=0.25
    ),
    
    # Tesla V100 - еще быстрее с Tensor Cores
    "tesla_v100": GPUConfig(
        name="Tesla V100",
        vram_gb=16.0, # 
        batch_size=512,
        memory_size=200_000,  # Уменьшено для снижения пиков памяти (реплей-буфер)
        hidden_sizes=(512, 256, 128),
        train_repeats=3,
        use_amp=True,
        use_gpu_storage=True,  # Включаем GPU storage для синхронизации устройств
        learning_rate=0.00018,
        description="Оптимальная конфигурация для Tesla V100 с Tensor Cores",
        use_torch_compile=True,
        eps_decay_steps=3_000_000,
        dropout_rate=0.25
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
        batch_size=256,
        memory_size=90_000,  # Увеличиваем до ~2.7GB VRAM (45% от 6GB)
        hidden_sizes=(512, 256, 128),
        train_repeats=1,
        use_amp=True,
        use_gpu_storage=True,  # Включаем GPU storage для синхронизации устройств
        learning_rate=0.0001,
        description="Оптимальная конфигурация для GTX 1660 Super",
        use_torch_compile=True,
        eps_decay_steps=3_000_000,
        dropout_rate=0.25
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
            print(f"🔍 Обнаружена GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 VRAM: {gpu_memory:.1f} GB")
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
        # Проверяем переменную окружения для принудительного выбора GPU
        forced_gpu = os.environ.get('FORCE_GPU_CONFIG', '').strip().lower()
        if forced_gpu and forced_gpu in GPU_CONFIGS:
            if not _gpu_info_printed:
                print(f"🔧 Принудительно выбрана GPU конфигурация: {forced_gpu}")
            gpu_key = forced_gpu
        else:
            gpu_key = detect_gpu()
    
    if gpu_key not in GPU_CONFIGS:
        print(f"⚠️ Неизвестная GPU: {gpu_key}, используем CPU конфигурацию")
        gpu_key = "cpu"
    
    config = GPU_CONFIGS[gpu_key]
    if not _gpu_info_printed:
        print(f"✅ Выбрана конфигурация: {config.name}")
        print(f"📊 Batch size: {config.batch_size}")
        print(f"💾 Memory size: {config.memory_size}")
        print(f"🧠 Hidden sizes: {config.hidden_sizes}")
        print(f"🔄 Train repeats: {config.train_repeats}")
        print(f"⚡ AMP: {config.use_amp}")
        print(f"💾 GPU storage: {config.use_gpu_storage}")
        print(f"🧩 torch.compile: {config.use_torch_compile}")
        _gpu_info_printed = True
    
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

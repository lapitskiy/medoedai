#!/usr/bin/env python3
"""
🧪 Тест torch.compile для конкретного GPU
Проверяет совместимость и выбирает оптимальный режим
"""

import torch
import sys

def test_torch_compile_gpu():
    """Тестирует torch.compile на текущем GPU"""
    print("🧪 ТЕСТ TORCH.COMPILE ДЛЯ GPU")
    print("=" * 50)
    
    # Проверяем доступность PyTorch
    print(f"📦 PyTorch версия: {torch.__version__}")
    print(f"🔧 torch.compile доступен: {hasattr(torch, 'compile')}")
    
    if not hasattr(torch, 'compile'):
        print("❌ torch.compile недоступен в этой версии PyTorch")
        print("💡 Обновите до PyTorch 2.0+")
        return False
    
    # Проверяем CUDA
    print(f"🚀 CUDA доступен: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("ℹ️ CUDA недоступен, torch.compile будет работать в CPU режиме")
        return test_torch_compile_cpu()
    
    # Информация о GPU
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    
    print(f"🎯 GPU устройств: {device_count}")
    print(f"🎯 Текущий GPU: {device_name}")
    print(f"🎯 GPU индекс: {current_device}")
    
    # CUDA Capability
    device_capability = torch.cuda.get_device_capability()
    print(f"🔍 CUDA Capability: {device_capability[0]}.{device_capability[1]}")
    
    # Определяем поддерживаемые режимы
    if device_capability[0] >= 8:  # Ampere+ (A100, H100, RTX 4090, etc.)
        supported_modes = ['max-autotune', 'default', 'reduce-overhead']
        recommended_mode = 'max-autotune'
        print("✅ Современный GPU (Ampere+), поддерживает все режимы")
    elif device_capability[0] >= 7:  # Volta+ (V100, RTX 2080, etc.)
        supported_modes = ['max-autotune', 'default', 'reduce-overhead']
        recommended_mode = 'max-autotune'
        print("✅ Хороший GPU (Volta+), поддерживает большинство режимов")
    elif device_capability[0] >= 6:  # Pascal (P100, GTX 1080, etc.)
        supported_modes = ['default', 'reduce-overhead']
        recommended_mode = 'default'
        print("⚠️ Старый GPU (Pascal), ограниченная поддержка")
    else:  # Maxwell и старше
        supported_modes = ['default']
        recommended_mode = 'default'
        print("❌ Очень старый GPU, минимальная поддержка")
    
    print(f"🎯 Поддерживаемые режимы: {supported_modes}")
    print(f"🎯 Рекомендуемый режим: {recommended_mode}")
    
    # Тестируем torch.compile
    print(f"\n🚀 Тестирую torch.compile с режимом '{recommended_mode}'...")
    
    try:
        # Создаем простую модель для теста
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        ).to('cuda')
        
        # Компилируем модель
        compiled_model = torch.compile(model, mode=recommended_mode)
        print("✅ torch.compile успешен!")
        
        # Тестируем производительность
        print("\n📊 Тестирую производительность...")
        
        # Входные данные
        x = torch.randn(1000, 10, device='cuda')
        
        # Теплый запуск
        for _ in range(5):
            _ = compiled_model(x)
        
        torch.cuda.synchronize()
        
        # Тест скорости
        import time
        start_time = time.time()
        
        for _ in range(100):
            _ = compiled_model(x)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        compiled_time = end_time - start_time
        
        # Сравниваем с обычной моделью
        start_time = time.time()
        
        for _ in range(100):
            _ = model(x)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        normal_time = end_time - start_time
        
        speedup = normal_time / compiled_time
        print(f"⚡ Обычная модель: {normal_time:.4f}с")
        print(f"🚀 Скомпилированная модель: {compiled_time:.4f}с")
        print(f"🎯 Ускорение: {speedup:.2f}x")
        
        if speedup > 1.1:
            print("✅ torch.compile дает значительное ускорение!")
        elif speedup > 0.9:
            print("⚠️ torch.compile работает, но ускорение минимально")
        else:
            print("❌ torch.compile замедляет работу")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при torch.compile: {e}")
        
        # Пробуем fallback режим
        if recommended_mode != 'default':
            print(f"\n🔄 Пробую fallback режим 'default'...")
            try:
                compiled_model = torch.compile(model, mode='default')
                print("✅ torch.compile с режимом 'default' успешен!")
                return True
            except Exception as e2:
                print(f"❌ Fallback режим тоже не работает: {e2}")
        
        return False

def test_torch_compile_cpu():
    """Тестирует torch.compile на CPU"""
    print("\n🖥️ Тестирую torch.compile на CPU...")
    
    try:
        # Создаем простую модель для теста
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        
        # Компилируем модель
        compiled_model = torch.compile(model, mode='default')
        print("✅ torch.compile на CPU успешен!")
        
        # Тестируем производительность
        print("\n📊 Тестирую производительность на CPU...")
        
        # Входные данные
        x = torch.randn(1000, 10)
        
        # Теплый запуск
        for _ in range(5):
            _ = compiled_model(x)
        
        # Тест скорости
        import time
        start_time = time.time()
        
        for _ in range(100):
            _ = compiled_model(x)
        
        end_time = time.time()
        compiled_time = end_time - start_time
        
        # Сравниваем с обычной моделью
        start_time = time.time()
        
        for _ in range(100):
            _ = model(x)
        
        end_time = time.time()
        normal_time = end_time - start_time
        
        speedup = normal_time / compiled_time
        print(f"⚡ Обычная модель: {normal_time:.4f}с")
        print(f"🚀 Скомпилированная модель: {compiled_time:.4f}с")
        print(f"🎯 Ускорение: {speedup:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при torch.compile на CPU: {e}")
        return False

def main():
    """Главная функция"""
    print("🚀 ТЕСТИРОВАНИЕ TORCH.COMPILE")
    print("=" * 50)
    
    success = test_torch_compile_gpu()
    
    if success:
        print("\n🎉 torch.compile работает корректно!")
        print("\n💡 РЕКОМЕНДАЦИИ:")
        print("  • Для обучения: используйте torch.compile")
        print("  • Для инференса: torch.compile может дать ускорение")
        print("  • При проблемах: попробуйте режим 'default'")
    else:
        print("\n❌ torch.compile не работает")
        print("\n💡 РЕШЕНИЯ:")
        print("  • Отключите torch.compile в конфигурации")
        print("  • Обновите PyTorch до версии 2.0+")
        print("  • Проверьте совместимость GPU")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()

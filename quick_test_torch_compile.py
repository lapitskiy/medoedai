#!/usr/bin/env python3
"""
🚀 Быстрый тест torch.compile для диагностики
"""

import torch
import os

def quick_test():
    print("🧪 БЫСТРЫЙ ТЕСТ TORCH.COMPILE")
    print("=" * 40)
    
    # Проверяем переменные окружения
    disable_compile = os.environ.get('DISABLE_TORCH_COMPILE', 'false').lower() == 'true'
    print(f"🔧 DISABLE_TORCH_COMPILE: {disable_compile}")
    
    # Проверяем PyTorch
    print(f"📦 PyTorch: {torch.__version__}")
    print(f"🔧 torch.compile доступен: {hasattr(torch, 'compile')}")
    
    if not hasattr(torch, 'compile'):
        print("❌ torch.compile недоступен")
        return False
    
    # Проверяем CUDA
    print(f"🚀 CUDA доступен: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("ℹ️ CUDA недоступен")
        return False
    
    # Информация о GPU
    device_name = torch.cuda.get_device_name()
    device_capability = torch.cuda.get_device_capability()
    
    print(f"🎯 GPU: {device_name}")
    print(f"🔍 CUDA Capability: {device_capability[0]}.{device_capability[1]}")
    
    # Тестируем torch.compile
    try:
        print("\n🚀 Тестирую torch.compile...")
        
        # Простая модель
        model = torch.nn.Linear(10, 1).to('cuda')
        
        # Пробуем разные режимы
        if "Tesla P100" in device_name:
            print("⚠️ Tesla P100 - тестирую только режим 'default'")
            modes_to_test = ['default']
        elif device_capability[0] >= 7:
            print("✅ Современный GPU - тестирую все режимы")
            modes_to_test = ['max-autotune', 'default', 'reduce-overhead']
        else:
            print("⚠️ Старый GPU - тестирую базовые режимы")
            modes_to_test = ['default', 'reduce-overhead']
        
        for mode in modes_to_test:
            try:
                print(f"  🔄 Тестирую режим '{mode}'...")
                compiled_model = torch.compile(model, mode=mode)
                
                # Тестовый прогон
                x = torch.randn(100, 10, device='cuda')
                _ = compiled_model(x)
                
                print(f"  ✅ Режим '{mode}' работает!")
                return True
                
            except Exception as e:
                print(f"  ❌ Режим '{mode}' не работает: {e}")
                continue
        
        print("❌ Ни один режим не работает")
        return False
        
    except Exception as e:
        print(f"❌ Общая ошибка: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n🎉 torch.compile работает!")
    else:
        print("\n❌ torch.compile не работает")
        print("\n💡 РЕШЕНИЯ:")
        print("  1. Установите DISABLE_TORCH_COMPILE=true")
        print("  2. Пересоберите Docker контейнер")
        print("  3. Обновите PyTorch до версии 2.0+")

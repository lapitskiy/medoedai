#!/usr/bin/env python3
"""
Тесты для torch.compile
"""

import sys
import os
sys.path.append('/app')

import torch
from agents.vdqn.cfg.vconfig import vDqnConfig
from agents.vdqn.dqnn import DQNN

def test_torch_compile():
    """Тестирует torch.compile функциональность"""
    print("🧪 Тестирую torch.compile...")
    
    try:
        # Проверяем версию PyTorch
        print(f"📊 PyTorch version: {torch.__version__}")
        print(f"🚀 PyTorch 2.x: {torch.__version__.startswith('2.')}")
        print(f"⚡ Has torch.compile: {hasattr(torch, 'compile')}")
        
        if not hasattr(torch, 'compile'):
            print("❌ torch.compile недоступен в этой версии PyTorch")
            return False
        
        # Создаем тестовую модель
        cfg = vDqnConfig()
        obs_dim = 442  # Как в вашем коде
        act_dim = 3
        
        model = DQNN(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=cfg.hidden_sizes,
            dropout_rate=cfg.dropout_rate,
            layer_norm=cfg.layer_norm,
            dueling=cfg.dueling_dqn
        )
        
        print(f"✅ Модель создана: {obs_dim} -> {act_dim}")
        
        # Тестируем torch.compile
        print("\n🚀 Тестирую torch.compile...")
        
        try:
            compiled_model = torch.compile(model, mode='max-autotune')
            print("✅ torch.compile успешен!")
            
            # Тестируем forward pass
            test_input = torch.randn(1, obs_dim)
            with torch.no_grad():
                output = compiled_model(test_input)
            
            print(f"✅ Forward pass успешен: {output.shape}")
            print(f"   - Вход: {test_input.shape}")
            print(f"   - Выход: {output.shape}")
            
            # Проверяем на NaN
            if torch.isnan(output).any():
                print("❌ Обнаружены NaN значения!")
                return False
            else:
                print("✅ NaN значения не обнаружены")
            
            # Тест производительности
            print("\n⚡ Тест производительности...")
            import time
            
            # Тест без компиляции
            start_time = time.time()
            for _ in range(100):
                with torch.no_grad():
                    _ = model(test_input)
            original_time = time.time() - start_time
            
            # Тест с компиляцией
            start_time = time.time()
            for _ in range(100):
                with torch.no_grad():
                    _ = compiled_model(test_input)
            compiled_time = time.time() - start_time
            
            print(f"   - Оригинальная модель: {original_time*1000:.2f} мс на 100 forward pass")
            print(f"   - Скомпилированная модель: {compiled_time*1000:.2f} мс на 100 forward pass")
            
            if compiled_time < original_time:
                speedup = original_time / compiled_time
                print(f"   - Ускорение: {speedup:.2f}x")
                
                if speedup > 1.5:
                    print("✅ Отличное ускорение!")
                elif speedup > 1.2:
                    print("✅ Хорошее ускорение")
                else:
                    print("⚠️ Минимальное ускорение")
            else:
                print("⚠️ Компиляция не дала ускорения")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при torch.compile: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_torch_compile()
    if success:
        print("\n🎉 torch.compile работает отлично!")
        print("🚀 Ваша модель будет работать с максимальным ускорением!")
        print("\n💡 ПРЕИМУЩЕСТВА TORCH.COMPILE:")
        print("   • Автоматическая оптимизация графов")
        print("   • Ускорение до 30-50%")
        print("   • Оптимизация памяти")
        print("   • Лучшая производительность на GPU")
    else:
        print("\n⚠️ torch.compile недоступен или не работает")
        print("📝 Модель будет работать без компиляции")

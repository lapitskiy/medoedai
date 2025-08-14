#!/usr/bin/env python3
"""
Тесты для GPU Replay Buffer
"""

import sys
import os
sys.path.append('/app')

import time
import torch
import numpy as np
from agents.vdqn.dqnsolver import PrioritizedReplayBuffer
from agents.vdqn.cfg.vconfig import vDqnConfig

def test_replay_buffer_performance():
    """Тестирует производительность GPU-оптимизированного replay buffer"""
    print("🧪 Тестирование GPU Replay Buffer...")
    
    try:
        cfg = vDqnConfig()
        
        # Параметры теста
        capacity = 10000
        state_size = 144
        batch_size = 64
        
        print(f"📊 Параметры теста:")
        print(f"   - Емкость: {capacity}")
        print(f"   - Размер состояния: {state_size}")
        print(f"   - Размер батча: {batch_size}")
        print(f"   - GPU storage: {cfg.use_gpu_storage}")
        
        # Создаем replay buffer
        replay_buffer = PrioritizedReplayBuffer(
            capacity=capacity,
            state_size=state_size,
            alpha=cfg.alpha,
            beta=cfg.beta,
            beta_increment=cfg.beta_increment,
            use_gpu_storage=cfg.use_gpu_storage
        )
        
        print(f"✅ Replay buffer создан на {replay_buffer.device}")
        
        # Тест 1: Заполнение буфера
        print("\n🔵 Тест 1: Заполнение буфера...")
        start_time = time.time()
        
        for i in range(capacity):
            state = np.random.randn(state_size).astype(np.float32)
            next_state = np.random.randn(state_size).astype(np.float32)
            action = np.random.randint(0, 3)
            reward = np.random.randn()
            done = np.random.choice([True, False])
            
            replay_buffer.push(state, action, reward, next_state, done)
            
            if i % 1000 == 0:
                print(f"   - Добавлено {i}/{capacity} элементов")
        
        fill_time = time.time() - start_time
        fill_rate = capacity / fill_time
        print(f"✅ Заполнение завершено за {fill_time:.2f}с ({fill_rate:.1f} элементов/с)")
        
        # Тест 2: Сэмплирование
        print("\n🟡 Тест 2: Сэмплирование...")
        start_time = time.time()
        
        for i in range(100):
            batch = replay_buffer.sample(batch_size)
            if batch[0] is None:
                print("❌ Ошибка сэмплирования!")
                return False
        
        sample_time = time.time() - start_time
        sample_rate = 100 / sample_time
        print(f"✅ Сэмплирование завершено за {sample_time:.2f}с ({sample_rate:.1f} батчей/с)")
        
        # Тест 3: Обновление приоритетов
        print("\n🟢 Тест 3: Обновление приоритетов...")
        start_time = time.time()
        
        for i in range(100):
            indices = torch.randint(0, capacity, (batch_size,))
            priorities = torch.rand(batch_size)
            replay_buffer.update_priorities(indices, priorities)
        
        update_time = time.time() - start_time
        update_rate = 100 / update_time
        print(f"✅ Обновления завершены за {update_time:.2f}с ({update_rate:.1f} обновлений/с)")
        
        # Общая статистика
        total_time = fill_time + sample_time + update_time
        print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
        print(f"   • Заполнение: {fill_rate:.1f} элементов/с")
        print(f"   • Сэмплирование: {sample_rate:.1f} батчей/с")
        print(f"   • Обновления: {update_rate:.1f} обновлений/с")
        print(f"   • Общее время: {total_time:.2f}с")
        
        # GPU память
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"   • GPU память: {gpu_memory:.1f} MB")
            print(f"   • GPU память зарезервирована: {gpu_memory_reserved:.1f} MB")
        
        print(f"   • Тип хранения: {'GPU storage' if cfg.use_gpu_storage else 'Pinned memory'}")
        print(f"   • Устройство: {replay_buffer.device}")
        
        # Оценка производительности
        print(f"\n🎯 ОЦЕНКА ПРОИЗВОДИТЕЛЬНОСТИ:")
        if fill_rate > 500 and sample_rate > 20:
            print("✅ ОТЛИЧНО - Высокая производительность")
        elif fill_rate > 200 and sample_rate > 10:
            print("✅ ХОРОШО - GPU оптимизация работает хорошо")
        else:
            print("⚠️ УДОВЛЕТВОРИТЕЛЬНО - Есть возможности для оптимизации")
        
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        if cfg.use_gpu_storage:
            print("   • Используется GPU storage - отлично для производительности")
        else:
            print("   • Используется pinned memory - хорошо для CPU-GPU передачи")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_replay_buffer_performance()
    if success:
        print("\n🎉 Тест GPU Replay Buffer пройден успешно!")
    else:
        print("\n❌ Тест GPU Replay Buffer не пройден!")

#!/usr/bin/env python3
"""
Тесты для конфигурации DQN
"""

from agents.vdqn.cfg.vconfig import vDqnConfig

def test_configuration():
    """Тестирует конфигурацию"""
    print("\n⚙️ Тестирование конфигурации...")
    
    try:
        cfg = vDqnConfig()
        
        print("✅ Конфигурация загружена:")
        print(f"   - Epsilon: {cfg.eps_start} → {cfg.eps_final} за {cfg.eps_decay_steps} шагов")
        print(f"   - Архитектура: {cfg.hidden_sizes}")
        print(f"   - Обучение: lr={cfg.lr}, gamma={cfg.gamma}")
        print(f"   - Replay: size={cfg.memory_size}, batch={cfg.batch_size}")
        print(f"   - PER: {cfg.prioritized}, alpha={cfg.alpha}, beta={cfg.beta}")
        print(f"   - Улучшения: dropout={cfg.dropout_rate}, layer_norm={cfg.layer_norm}")
        print(f"   - DQN: double={cfg.double_dqn}, dueling={cfg.dueling_dqn}")
        print(f"   - GPU: use_gpu_storage={cfg.use_gpu_storage}, use_torch_compile={cfg.use_torch_compile}")
        
        # Проверяем совместимость параметров
        if cfg.batch_size > cfg.memory_size:
            print("❌ Batch size больше memory size!")
            return False, "Batch size больше memory size"
        else:
            print("✅ Параметры совместимы")
        
        if cfg.eps_final >= cfg.eps_start:
            print("❌ Epsilon final должен быть меньше eps start!")
            return False, "Epsilon final должен быть меньше eps start"
        else:
            print("✅ Epsilon параметры корректны")
        
        return True, "Конфигурация протестирована успешно"
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании конфигурации: {e}")
        return False, f"Ошибка при тестировании конфигурации: {str(e)}"

if __name__ == "__main__":
    success, message = test_configuration()
    if success:
        print("✅ Тест конфигурации пройден!")
    else:
        print(f"❌ Тест конфигурации не пройден: {message}")

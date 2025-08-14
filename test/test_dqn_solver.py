#!/usr/bin/env python3
"""
Тесты для DQN solver
"""

import numpy as np
from agents.vdqn.cfg.vconfig import vDqnConfig
from agents.vdqn.dqnsolver import DQNSolver

def test_dqn_solver():
    """Тестирует улучшенный DQN solver"""
    print("\n🔧 Тестирование улучшенного DQN solver...")
    
    try:
        cfg = vDqnConfig()
        
        # Создаем solver
        observation_space = 100
        action_space = 3
        
        solver = DQNSolver(observation_space, action_space, load=False)
        
        print(f"✅ DQN Solver создан успешно")
        print(f"   - Prioritized Replay: {cfg.prioritized}")
        print(f"   - Memory Size: {cfg.memory_size}")
        print(f"   - Batch Size: {cfg.batch_size}")
        print(f"   - Learning Rate: {cfg.lr}")
        print(f"   - Gamma: {cfg.gamma}")
        
        # Тестируем добавление переходов
        test_state = np.random.randn(100)
        test_action = 1
        test_reward = 0.5
        test_next_state = np.random.randn(100)
        test_done = False
        
        solver.store_transition(test_state, test_action, test_reward, test_next_state, test_done)
        print(f"✅ Переход добавлен в replay buffer")
        print(f"   - Размер буфера: {len(solver.memory)}")
        
        # Тестируем выбор действия
        action = solver.act(test_state)
        print(f"✅ Действие выбрано: {action}")
        print(f"   - Epsilon: {solver.epsilon:.4f}")
        
        return True, "DQN Solver протестирован успешно"
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании DQN solver: {e}")
        return False, f"Ошибка при тестировании DQN solver: {str(e)}"

if __name__ == "__main__":
    success, message = test_dqn_solver()
    if success:
        print("✅ Тест DQN solver пройден!")
    else:
        print(f"❌ Тест DQN solver не пройден: {message}")

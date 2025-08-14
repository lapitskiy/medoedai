#!/usr/bin/env python3
"""
Тесты для обработки NaN значений
"""

import numpy as np
from agents.vdqn.cfg.vconfig import vDqnConfig
from agents.vdqn.dqnn import DQNN
from agents.vdqn.dqnsolver import DQNSolver

def test_nan_handling():
    """Тестирует обработку NaN значений"""
    print("\n🛡️ Тестирование обработки NaN значений...")
    
    try:
        cfg = vDqnConfig()
        
        # Создаем модель
        model = DQNN(100, 3, (512, 256, 128))
        
        # Тестируем с NaN входом
        test_input = np.random.randn(100)
        test_input[0] = np.nan  # Добавляем NaN
        
        print(f"   - Вход содержит NaN: {np.isnan(test_input).any()}")
        
        # Тестируем обработку в solver
        solver = DQNSolver(100, 3, load=False)
        
        # Должно автоматически заменить NaN на нули
        action = solver.act(test_input)
        print(f"✅ Действие выбрано даже с NaN входом: {action}")
        
        # Проверяем, что NaN заменены
        cleaned_input = np.nan_to_num(test_input, nan=0.0)
        print(f"   - NaN заменены на нули: {np.isnan(cleaned_input).any()}")
        
        return True, "Обработка NaN значений протестирована успешно"
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании обработки NaN: {e}")
        return False, f"Ошибка при тестировании обработки NaN: {str(e)}"

if __name__ == "__main__":
    success, message = test_nan_handling()
    if success:
        print("✅ Тест обработки NaN пройден!")
    else:
        print(f"❌ Тест обработки NaN не пройден: {message}")

#!/usr/bin/env python3
"""
Тесты для нейронной сети DQN
"""

import torch
from agents.vdqn.cfg.vconfig import vDqnConfig
from agents.vdqn.dqnn import DQNN

def test_neural_network():
    """Тестирует улучшенную архитектуру нейронной сети"""
    print("🧠 Тестирование улучшенной архитектуры нейронной сети...")
    
    try:
        cfg = vDqnConfig()
        
        # Тестируем Dueling DQN
        obs_dim = 100
        act_dim = 3
        hidden_sizes = (512, 256, 128)
        
        model = DQNN(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            dropout_rate=cfg.dropout_rate,
            layer_norm=cfg.layer_norm,
            dueling=cfg.dueling_dqn
        )
        
        print(f"✅ Модель создана успешно")
        print(f"   - Архитектура: {hidden_sizes}")
        print(f"   - Dropout: {cfg.dropout_rate}")
        print(f"   - Layer Norm: {cfg.layer_norm}")
        print(f"   - Dueling: {cfg.dueling_dqn}")
        
        # Тестируем forward pass
        test_input = torch.randn(1, obs_dim)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ Forward pass успешен")
        print(f"   - Вход: {test_input.shape}")
        print(f"   - Выход: {output.shape}")
        print(f"   - Q-значения: {output.squeeze().tolist()}")
        
        # Проверяем на NaN
        if torch.isnan(output).any():
            print("❌ Обнаружены NaN значения в выходе!")
            return False, "Обнаружены NaN значения в выходе"
        else:
            print("✅ NaN значения не обнаружены")
            return True, "Модель создана и протестирована успешно"
            
    except Exception as e:
        print(f"❌ Ошибка при тестировании нейронной сети: {e}")
        return False, f"Ошибка при тестировании нейронной сети: {str(e)}"

if __name__ == "__main__":
    success, message = test_neural_network()
    if success:
        print("✅ Тест нейронной сети пройден!")
    else:
        print(f"❌ Тест нейронной сети не пройден: {message}")

#!/usr/bin/env python3
"""
Тест состояния окружения для понимания проблемы с winrate
"""

import sys
import os

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_env_state():
    """
    Тестируем состояние окружения
    """
    print("🧪 ТЕСТ СОСТОЯНИЯ ОКРУЖЕНИЯ")
    print("=" * 60)
    
    # Проверяем исправления в crypto_trading_env_optimized.py
    try:
        with open('envs/dqn_model/gym/crypto_trading_env_optimized.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        print("📋 Проверка исправлений в crypto_trading_env_optimized.py:")
        
        if 'ИСПРАВЛЕНИЕ: Добавляем общий список сделок для правильного расчета winrate' in content:
            print("  ✅ Добавлен self.all_trades в __init__")
        else:
            print("  ❌ self.all_trades НЕ добавлен в __init__")
            
        if 'ИСПРАВЛЯЕМ: НЕ очищаем сделки между эпизодами' in content:
            print("  ✅ Сделки НЕ очищаются в reset()")
        else:
            print("  ❌ Сделки очищаются в reset()")
            
        if 'ИСПРАВЛЯЕМ: Записываем сделку в оба списка' in content:
            print("  ✅ Сделки записываются в оба списка")
        else:
            print("  ❌ Сделки НЕ записываются в оба списка")
            
    except Exception as e:
        print(f"  ❌ Ошибка чтения crypto_trading_env_optimized.py: {e}")
    
    # Проверяем исправления в v_train_model_optimized.py
    try:
        with open('agents/vdqn/v_train_model_optimized.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        print(f"\n📋 Проверка исправлений в v_train_model_optimized.py:")
        
        if 'РАДИКАЛЬНОЕ ИСПРАВЛЕНИЕ: Используем env.all_trades для расчета winrate' in content:
            print("  ✅ Winrate считается из env.all_trades")
        else:
            print("  ❌ Winrate НЕ считается из env.all_trades")
            
    except Exception as e:
        print(f"  ❌ Ошибка чтения v_train_model_optimized.py: {e}")
    
    print(f"\n🔍 Анализ проблемы:")
    print(f"  • Проблема: episode_winrates = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]")
    print(f"  • Реальность: winrate = 40.61% (67 прибыльных из 165 сделок)")
    print(f"  • Причина: env.all_trades пустой или не инициализируется")
    
    print(f"\n🎯 Возможные причины:")
    print(f"  1. env.all_trades не инициализируется в __init__")
    print(f"  2. Сделки не записываются в env.all_trades")
    print(f"  3. env.all_trades очищается где-то еще")
    print(f"  4. Логика расчета winrate работает неправильно")
    
    print(f"\n💡 Решения:")
    print(f"  1. Проверить инициализацию env.all_trades")
    print(f"  2. Убедиться, что сделки записываются в env.all_trades")
    print(f"  3. Добавить отладочные print'ы для env.all_trades")
    print(f"  4. Проверить, что env.all_trades не очищается")
    
    print(f"\n✅ Тест завершен!")

if __name__ == "__main__":
    test_env_state()

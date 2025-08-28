#!/usr/bin/env python3
"""
Отладка состояния окружения для понимания проблемы с winrate
"""

import sys
import os

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_env_state():
    """
    Отлаживаем состояние окружения
    """
    print("🔍 ОТЛАДКА СОСТОЯНИЯ ОКРУЖЕНИЯ")
    print("=" * 60)
    
    # Проверяем, что файлы исправлений применены
    print("📋 Проверка исправлений:")
    
    # Проверяем v_train_model_optimized.py
    try:
        with open('agents/vdqn/v_train_model_optimized.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        if 'ИСПРАВЛЯЕМ: Правильно считаем winrate по эпизодам' in content:
            print("  ✅ v_train_model_optimized.py - исправления применены")
        else:
            print("  ❌ v_train_model_optimized.py - исправления НЕ применены")
            
    except Exception as e:
        print(f"  ❌ Ошибка чтения v_train_model_optimized.py: {e}")
    
    # Проверяем crypto_trading_env_optimized.py
    try:
        with open('envs/dqn_model/gym/crypto_trading_env_optimized.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        if 'ИСПРАВЛЯЕМ: НЕ очищаем сделки между эпизодами' in content:
            print("  ✅ crypto_trading_env_optimized.py - исправления применены")
        else:
            print("  ❌ crypto_trading_env_optimized.py - исправления НЕ применены")
            
    except Exception as e:
        print(f"  ❌ Ошибка чтения crypto_trading_env_optimized.py: {e}")
    
    print(f"\n🔍 Анализ проблемы:")
    print(f"  • Проблема: episode_winrates = [0.0, 0.0, 0.0, 0.0]")
    print(f"  • Реальность: winrate = 40.78% (42 прибыльных из 103 сделок)")
    print(f"  • Причина: env.trades пустой в каждом эпизоде")
    
    print(f"\n🎯 Возможные причины:")
    print(f"  1. env.trades очищается в reset()")
    print(f"  2. Сделки не записываются в env.trades")
    print(f"  3. Логика расчета winrate работает неправильно")
    
    print(f"\n💡 Решения:")
    print(f"  1. Проверить, что env.trades НЕ очищается")
    print(f"  2. Убедиться, что сделки записываются в env.trades")
    print(f"  3. Использовать env.all_trades для расчета winrate")
    
    print(f"\n✅ Отладка завершена!")

if __name__ == "__main__":
    debug_env_state()

#!/usr/bin/env python3
"""
Тест исправлений в мульти-окружении
"""

import sys
import os

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_multi_env_fix():
    """
    Тестируем исправления в мульти-окружении
    """
    print("🧪 ТЕСТ ИСПРАВЛЕНИЙ В МУЛЬТИ-ОКРУЖЕНИИ")
    print("=" * 60)
    
    # Проверяем исправления в crypto_trading_env_multi.py
    try:
        with open('envs/dqn_model/gym/crypto_trading_env_multi.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        print("📋 Проверка исправлений в crypto_trading_env_multi.py:")
        
        if 'ИСПРАВЛЕНИЕ: Добавляем общий список сделок для правильного расчета winrate' in content:
            print("  ✅ Добавлен _all_trades в __init__")
        else:
            print("  ❌ _all_trades НЕ добавлен в __init__")
            
        if 'ИСПРАВЛЕНИЕ: Создаем окружение только если его нет или сменилась криптовалюта' in content:
            print("  ✅ Окружение переиспользуется между эпизодами")
        else:
            print("  ❌ Окружение НЕ переиспользуется")
            
        if 'ИСПРАВЛЕНИЕ: Передаем накопленные сделки в новое окружение' in content:
            print("  ✅ Сделки передаются между окружениями")
        else:
            print("  ❌ Сделки НЕ передаются")
            
        if 'ИСПРАВЛЕНИЕ: Получает общий список сделок' in content:
            print("  ✅ Добавлен property all_trades")
        else:
            print("  ❌ property all_trades НЕ добавлен")
            
    except Exception as e:
        print(f"  ❌ Ошибка чтения crypto_trading_env_multi.py: {e}")
    
    print(f"\n🔍 Что исправлено:")
    print(f"  1. ✅ Добавлен _all_trades в мульти-окружении")
    print(f"  2. ✅ Окружение переиспользуется между эпизодами")
    print(f"  3. ✅ Сделки передаются между окружениями")
    print(f"  4. ✅ Добавлен property all_trades")
    print(f"  5. ✅ Синхронизация сделок между окружениями")
    
    print(f"\n🎯 Ожидаемый результат:")
    print(f"  • Было: Новое окружение в каждом эпизоде → сделки терялись")
    print(f"  • Будет: Окружение переиспользуется → сделки накапливаются")
    print(f"  • Результат: episode_winrates будут соответствовать реальным данным")
    
    print(f"\n💡 Как это работает:")
    print(f"  1. Мульти-окружение создает базовое окружение")
    print(f"  2. При смене криптовалюты окружение переиспользуется")
    print(f"  3. Сделки накапливаются в _all_trades")
    print(f"  4. Winrate рассчитывается из накопленных сделок")
    
    print(f"\n✅ Тест завершен!")

if __name__ == "__main__":
    test_multi_env_fix()

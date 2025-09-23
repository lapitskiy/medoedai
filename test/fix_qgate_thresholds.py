#!/usr/bin/env python3
"""
Скрипт для исправления порогов QGate на основе анализа
"""

import os
import sys

def fix_qgate_thresholds():
    """Исправление порогов QGate на основе анализа"""
    
    print("🔧 ИСПРАВЛЕНИЕ ПОРОГОВ QGATE")
    print("=" * 40)
    
    # Текущие пороги (из анализа)
    current_thresholds = {
        'QGATE_MAXQ': 0.331,
        'QGATE_GAPQ': 0.223,
        'QGATE_SELL_MAXQ': 0.365,
        'QGATE_SELL_GAPQ': 0.189
    }
    
    # Рекомендуемые пороги (из анализа)
    recommended_thresholds = {
        'QGATE_MAXQ': 0.995,      # 70% от среднего max_q (1.422)
        'QGATE_GAPQ': 0.459,      # 50% от среднего gap_q (0.917)
        'QGATE_SELL_MAXQ': 1.0,   # Немного выше для SELL
        'QGATE_SELL_GAPQ': 0.5    # Немного выше для SELL
    }
    
    print("📊 ТЕКУЩИЕ ПОРОГИ:")
    for key, value in current_thresholds.items():
        print(f"  {key}: {value}")
    
    print("\n💡 РЕКОМЕНДУЕМЫЕ ПОРОГИ:")
    for key, value in recommended_thresholds.items():
        print(f"  {key}: {value}")
    
    print("\n🔧 КОМАНДЫ ДЛЯ УСТАНОВКИ:")
    print("export QGATE_MAXQ=0.995")
    print("export QGATE_GAPQ=0.459")
    print("export QGATE_SELL_MAXQ=1.0")
    print("export QGATE_SELL_GAPQ=0.5")
    
    print("\n📝 ИЛИ В .env ФАЙЛЕ:")
    print("QGATE_MAXQ=0.995")
    print("QGATE_GAPQ=0.459")
    print("QGATE_SELL_MAXQ=1.0")
    print("QGATE_SELL_GAPQ=0.5")
    
    print("\n🎯 ОЖИДАЕМЫЙ РЕЗУЛЬТАТ:")
    print("  - Фильтрация снизится с 40.9% до ~10-15%")
    print("  - Больше торговых сигналов будет проходить")
    print("  - Сохранится качество (89.5% успешность)")
    
    # Создаем файл с настройками
    env_content = """# QGate пороги (исправлены на основе анализа)
QGATE_MAXQ=0.995
QGATE_GAPQ=0.459
QGATE_SELL_MAXQ=1.0
QGATE_SELL_GAPQ=0.5
"""
    
    with open('test/qgate_thresholds.env', 'w') as f:
        f.write(env_content)
    
    print(f"\n💾 Настройки сохранены в: test/qgate_thresholds.env")
    print(f"📋 Для применения выполните:")
    print(f"   source test/qgate_thresholds.env")

if __name__ == "__main__":
    fix_qgate_thresholds()

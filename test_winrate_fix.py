#!/usr/bin/env python3
"""
Тест исправлений winrate
"""

import sys
import os

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_winrate_calculation():
    """
    Тестируем логику расчета winrate
    """
    print("🧪 ТЕСТ ИСПРАВЛЕНИЙ WINRATE")
    print("=" * 50)
    
    # Симулируем данные сделок
    all_trades = [
        {'roi': 0.05, 'net': 50, 'reward': 0.5, 'duration': 120},
        {'roi': -0.02, 'net': -20, 'reward': -0.2, 'duration': 90},
        {'roi': 0.03, 'net': 30, 'reward': 0.3, 'duration': 150},
        {'roi': -0.01, 'net': -10, 'reward': -0.1, 'duration': 60},
        {'roi': 0.08, 'net': 80, 'reward': 0.8, 'duration': 180}
    ]
    
    # Рассчитываем winrate
    profitable_trades = [t for t in all_trades if t['roi'] > 0]
    total_trades = len(all_trades)
    winrate = len(profitable_trades) / total_trades if total_trades > 0 else 0
    
    print(f"📊 Тестовые данные:")
    print(f"  • Всего сделок: {total_trades}")
    print(f"  • Прибыльных: {len(profitable_trades)}")
    print(f"  • Убыточных: {total_trades - len(profitable_trades)}")
    print(f"  • Winrate: {winrate:.3f} ({winrate*100:.1f}%)")
    
    # Проверяем логику по эпизодам
    print(f"\n🔍 Логика по эпизодам:")
    
    # Эпизод 1: 2 сделки
    episode1_trades = all_trades[:2]
    episode1_profitable = [t for t in episode1_trades if t['roi'] > 0]
    episode1_winrate = len(episode1_profitable) / len(episode1_trades) if episode1_trades else 0
    print(f"  • Эпизод 1: {len(episode1_trades)} сделок, winrate={episode1_winrate:.3f}")
    
    # Эпизод 2: 3 сделки
    episode2_trades = all_trades[2:]
    episode2_profitable = [t for t in episode2_trades if t['roi'] > 0]
    episode2_winrate = len(episode2_profitable) / len(episode2_trades) if episode2_trades else 0
    print(f"  • Эпизод 2: {len(episode2_trades)} сделок, winrate={episode2_winrate:.3f}")
    
    # Общий winrate должен быть средним
    episode_winrates = [episode1_winrate, episode2_winrate]
    avg_episode_winrate = sum(episode_winrates) / len(episode_winrates)
    print(f"  • Средний winrate по эпизодам: {avg_episode_winrate:.3f}")
    print(f"  • Общий winrate: {winrate:.3f}")
    print(f"  • Совпадение: {'✅' if abs(avg_episode_winrate - winrate) < 0.001 else '❌'}")
    
    print(f"\n✅ Тест завершен!")

if __name__ == "__main__":
    test_winrate_calculation()

#!/usr/bin/env python3
"""
Простой скрипт для анализа результатов обучения
"""

import pickle
import os

def analyze_file():
    file_path = 'temp/train_results/training_results_1755911593.pkl'
    
    if not os.path.exists(file_path):
        print(f"❌ Файл {file_path} не найден!")
        return
    
    try:
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
        
        print("🔍 АНАЛИЗ ФАЙЛА С РЕЗУЛЬТАТАМИ")
        print("=" * 50)
        
        print(f"📋 Ключи в файле: {list(results.keys())}")
        
        # Анализируем winrates
        if 'episode_winrates' in results:
            winrates = results['episode_winrates']
            print(f"\n📊 Episode winrates:")
            print(f"  • Тип: {type(winrates)}")
            print(f"  • Длина: {len(winrates) if hasattr(winrates, '__len__') else 'N/A'}")
            print(f"  • Значения: {winrates}")
        else:
            print("\n❌ Ключ 'episode_winrates' не найден!")
        
        # Анализируем сделки
        if 'all_trades' in results:
            trades = results['all_trades']
            print(f"\n💰 All trades:")
            print(f"  • Тип: {type(trades)}")
            print(f"  • Количество: {len(trades) if hasattr(trades, '__len__') else 'N/A'}")
            if trades and len(trades) > 0:
                print(f"  • Первая сделка: {trades[0]}")
                print(f"  • Последняя сделка: {trades[-1] if len(trades) > 1 else 'N/A'}")
        else:
            print("\n❌ Ключ 'all_trades' не найден!")
        
        # Анализируем другие ключи
        for key in results:
            if key not in ['episode_winrates', 'all_trades']:
                value = results[key]
                print(f"\n🔍 {key}:")
                print(f"  • Тип: {type(value)}")
                if hasattr(value, '__len__'):
                    print(f"  • Длина: {len(value)}")
                print(f"  • Значение: {value}")
                
    except Exception as e:
        print(f"❌ Ошибка при чтении файла: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_file()

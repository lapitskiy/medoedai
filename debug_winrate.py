#!/usr/bin/env python3
"""
Анализ проблемы с нулевым winrate
"""

import pickle
import os

def analyze_winrate_problem():
    file_path = 'temp/train_results/training_results_1755911593.pkl'
    
    if not os.path.exists(file_path):
        print(f"❌ Файл {file_path} не найден!")
        return
    
    try:
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
        
        print("🔍 АНАЛИЗ ПРОБЛЕМЫ С WINRATE")
        print("=" * 60)
        
        print(f"📋 Ключи в файле: {list(results.keys())}")
        
        # Анализируем winrates
        if 'episode_winrates' in results:
            winrates = results['episode_winrates']
            print(f"\n📊 Episode winrates:")
            print(f"  • Тип: {type(winrates)}")
            print(f"  • Длина: {len(winrates) if hasattr(winrates, '__len__') else 'N/A'}")
            print(f"  • Значения: {winrates}")
            
            # Проверяем, все ли winrate равны 0
            if hasattr(winrates, '__len__') and len(winrates) > 0:
                all_zero = all(w == 0.0 for w in winrates)
                print(f"  • Все winrate равны 0: {all_zero}")
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
                if len(trades) > 1:
                    print(f"  • Последняя сделка: {trades[-1]}")
                
                # Анализируем ROI сделок
                profitable_count = 0
                loss_count = 0
                for trade in trades:
                    roi = trade.get('roi', 0)
                    if roi > 0:
                        profitable_count += 1
                    elif roi < 0:
                        loss_count += 1
                
                print(f"  • Прибыльных сделок: {profitable_count}")
                print(f"  • Убыточных сделок: {loss_count}")
                print(f"  • Нейтральных сделок: {len(trades) - profitable_count - loss_count}")
                
                if len(trades) > 0:
                    actual_winrate = profitable_count / len(trades)
                    print(f"  • Реальный winrate: {actual_winrate:.3f}")
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
    analyze_winrate_problem()

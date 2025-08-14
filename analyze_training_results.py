#!/usr/bin/env python3
"""
📊 Анализатор результатов обучения DQN модели

Использование:
    python analyze_training_results.py training_results_1234567890.pkl
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
from datetime import datetime

def analyze_training_results(results_file):
    """
    Анализирует результаты обучения и создает отчет
    """
    print(f"📊 Анализирую результаты из файла: {results_file}")
    
    # Загружаем результаты
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    print("\n" + "="*60)
    print("📈 ОТЧЕТ ОБ ОБУЧЕНИИ DQN МОДЕЛИ")
    print("="*60)
    
    # Основная информация
    print(f"📅 Дата обучения: {results['training_date']}")
    print(f"🎯 Количество эпизодов: {results['episodes']}")
    print(f"⏱️ Время обучения: {results['total_training_time']:.2f} секунд ({results['total_training_time']/60:.1f} минут)")
    print(f"🚀 Скорость: {results['episodes']/(results['total_training_time']/60):.1f} эпизодов/минуту")
    
    # Статистика winrate
    winrates = results['episode_winrates']
    if winrates:
        print(f"\n📊 СТАТИСТИКА WINRATE:")
        print(f"  • Средний winrate: {np.mean(winrates):.3f}")
        print(f"  • Лучший winrate: {results['best_winrate']:.3f}")
        print(f"  • Минимальный winrate: {np.min(winrates):.3f}")
        print(f"  • Максимальный winrate: {np.max(winrates):.3f}")
        print(f"  • Стандартное отклонение: {np.std(winrates):.3f}")
        
        # Тренд winrate
        if len(winrates) > 10:
            first_10 = np.mean(winrates[:10])
            last_10 = np.mean(winrates[-10:])
            print(f"  • Первые 10 эпизодов: {first_10:.3f}")
            print(f"  • Последние 10 эпизодов: {last_10:.3f}")
            print(f"  • Изменение: {last_10 - first_10:+.3f}")
    
    # Статистика сделок
    trades = results['all_trades']
    if trades:
        print(f"\n💰 СТАТИСТИКА СДЕЛОК:")
        print(f"  • Всего сделок: {len(trades)}")
        
        # Прибыльные и убыточные сделки
        profitable_trades = [t for t in trades if t.get('roi', 0) > 0]
        loss_trades = [t for t in trades if t.get('roi', 0) < 0]
        
        print(f"  • Прибыльных сделок: {len(profitable_trades)} ({len(profitable_trades)/len(trades)*100:.1f}%)")
        print(f"  • Убыточных сделок: {len(loss_trades)} ({len(loss_trades)/len(trades)*100:.1f}%)")
        
        if profitable_trades:
            avg_profit = np.mean([t.get('roi', 0) for t in profitable_trades])
            max_profit = np.max([t.get('roi', 0) for t in profitable_trades])
            print(f"  • Средняя прибыль: {avg_profit:.4f} ({avg_profit*100:.2f}%)")
            print(f"  • Максимальная прибыль: {max_profit:.4f} ({max_profit*100:.2f}%)")
        
        if loss_trades:
            avg_loss = np.mean([t.get('roi', 0) for t in loss_trades])
            max_loss = np.min([t.get('roi', 0) for t in loss_trades])
            print(f"  • Средний убыток: {avg_loss:.4f} ({avg_loss*100:.2f}%)")
            print(f"  • Максимальный убыток: {max_loss:.4f} ({max_loss*100:.2f}%)")
        
        # Длительность сделок
        durations = [t.get('duration', 0) for t in trades]
        if durations:
            print(f"  • Средняя длительность: {np.mean(durations):.1f} минут")
            print(f"  • Минимальная длительность: {np.min(durations):.1f} минут")
            print(f"  • Максимальная длительность: {np.max(durations):.1f} минут")
    
    # Финальная статистика
    if 'final_stats' in results:
        print(f"\n📈 ФИНАЛЬНАЯ СТАТИСТИКА:")
        for key, value in results['final_stats'].items():
            if isinstance(value, float):
                print(f"  • {key}: {value:.4f}")
            else:
                print(f"  • {key}: {value}")
    
    # Создаем графики
    create_plots(results, results_file)
    
    print(f"\n✅ Анализ завершен! Графики сохранены в папке plots/")

def create_plots(results, results_file):
    """
    Создает графики для визуализации результатов
    """
    # Создаем папку для графиков
    os.makedirs('plots', exist_ok=True)
    
    # График winrate по эпизодам
    if results['episode_winrates']:
        plt.figure(figsize=(12, 6))
        plt.plot(results['episode_winrates'], alpha=0.7, linewidth=1)
        plt.title('Winrate по эпизодам')
        plt.xlabel('Эпизод')
        plt.ylabel('Winrate')
        plt.grid(True, alpha=0.3)
        
        # Добавляем скользящее среднее
        if len(results['episode_winrates']) > 10:
            window = min(10, len(results['episode_winrates']) // 10)
            moving_avg = np.convolve(results['episode_winrates'], np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(results['episode_winrates'])), moving_avg, 
                    linewidth=2, color='red', label=f'Скользящее среднее (окно={window})')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('plots/winrate_progression.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # График распределения ROI
    if results['all_trades']:
        rois = [t.get('roi', 0) for t in results['all_trades']]
        
        plt.figure(figsize=(12, 6))
        plt.hist(rois, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Распределение ROI сделок')
        plt.xlabel('ROI')
        plt.ylabel('Количество сделок')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Безубыточность')
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/roi_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # График накопительной прибыли
        cumulative_roi = np.cumsum(rois)
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_roi, linewidth=1)
        plt.title('Накопительная прибыль')
        plt.xlabel('Номер сделки')
        plt.ylabel('Накопительный ROI')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('plots/cumulative_profit.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """
    Главная функция
    """
    if len(sys.argv) == 1:
        # Если файл не указан, ищем самый свежий в temp/train_results
        results_dir = "temp/train_results"
        if not os.path.exists(results_dir):
            print(f"❌ Папка {results_dir} не найдена!")
            print("Создайте папку temp/train_results и запустите обучение")
            return
        
        result_files = glob.glob(os.path.join(results_dir, 'training_results_*.pkl'))
        if not result_files:
            print(f"❌ Файлы результатов не найдены в {results_dir}")
            print("Сначала запустите обучение")
            return
        
        # Берем самый свежий файл
        results_file = max(result_files, key=os.path.getctime)
        print(f"📊 Автоматически выбран файл: {results_file}")
        
    elif len(sys.argv) == 2:
        results_file = sys.argv[1]
        # Если указан относительный путь, добавляем папку
        if not os.path.isabs(results_file) and not results_file.startswith('temp/'):
            results_file = os.path.join("temp/train_results", results_file)
    else:
        print("Использование: python analyze_training_results.py [results_file]")
        print("Пример: python analyze_training_results.py")
        print("Пример: python analyze_training_results.py training_results_1234567890.pkl")
        return
    
    if not os.path.exists(results_file):
        print(f"❌ Файл {results_file} не найден!")
        return
    
    try:
        analyze_training_results(results_file)
    except Exception as e:
        print(f"❌ Ошибка при анализе: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

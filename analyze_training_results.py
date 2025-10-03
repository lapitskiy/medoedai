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
    
    # ОТЛАДКА: Показываем все ключи в файле результатов
    print(f"🔍 ОТЛАДКА: Ключи в файле результатов:")
    for key in sorted(results.keys()):
        value = results[key]
        if isinstance(value, (list, tuple)):
            print(f"  • {key}: {type(value).__name__} с {len(value)} элементами")
        else:
            print(f"  • {key}: {value}")
    
    # Основная информация
    training_date = results.get('training_date') or results.get('created_at') or 'N/A'
    print(f"📅 Дата обучения: {training_date}")
    
    # Показываем планируемое и реальное количество эпизодов
    planned_episodes = results.get('episodes')
    if planned_episodes is None:
        planned_episodes = results.get('planned_episodes')
    if planned_episodes is None:
        planned_episodes = len(results.get('episode_winrates') or [])
    
    # Ищем реальное количество эпизодов в разных местах
    actual_episodes = None
    early_stopping_triggered = False
    
    print(f"\n🔍 ОТЛАДКА: Поиск реального количества эпизодов")
    print(f"  • Планируемые эпизоды: {planned_episodes}")
    
    # 1. Проверяем actual_episodes если есть (НО это может быть неправильно!)
    if 'actual_episodes' in results:
        actual_episodes = results['actual_episodes']
        print(f"🔍 1. Найден actual_episodes: {actual_episodes}")
        print(f"⚠️  ВНИМАНИЕ: actual_episodes может быть неправильным!")
    else:
        print(f"🔍 1. actual_episodes НЕ найден")
    
    # 2. Проверяем real_episodes если есть
    if actual_episodes is None and 'real_episodes' in results:
        actual_episodes = results['real_episodes']
        print(f"🔍 2. Найден real_episodes: {actual_episodes}")
    elif actual_episodes is None:
        print(f"🔍 2. real_episodes НЕ найден")
    
    # 3. Проверяем episode_winrates (но это может быть неточно)
    episode_winrates = results.get('episode_winrates') or []
    if actual_episodes is None and episode_winrates:
        # ВНИМАНИЕ: episode_winrates может содержать winrate для каждого эпизода, а не только завершенных
        # Поэтому используем это только как fallback
        episode_winrates_count = len(episode_winrates)
        print(f"🔍 3. Найден episode_winrates с {episode_winrates_count} элементами")
        if episode_winrates_count < planned_episodes * 0.8:  # Если меньше 80% от планируемых
            actual_episodes = episode_winrates_count
            early_stopping_triggered = True
            print(f"⚠️ 3. Обнаружен возможный early stopping: {episode_winrates_count} < {planned_episodes}")
        else:
            actual_episodes = episode_winrates_count
            print(f"🔍 3. Используем episode_winrates как actual_episodes: {actual_episodes}")
    elif actual_episodes is None:
        print(f"🔍 3. episode_winrates НЕ найден или пуст")
    
    # 4. Если ничего не нашли, используем планируемое
    if actual_episodes is None:
        actual_episodes = planned_episodes
        print(f"🔍 4. Не найдена информация о реальных эпизодах, используем планируемое: {actual_episodes}")
    
    print(f"🔍 После основного поиска: actual_episodes = {actual_episodes}")
    
    # Дополнительная проверка: если actual_episodes равен planned_episodes, 
    # но в episode_winrates меньше элементов, это может быть early stopping
    if (actual_episodes == planned_episodes and 
        episode_winrates and 
        len(episode_winrates) < planned_episodes):
        
        actual_episodes = len(episode_winrates)
        early_stopping_triggered = True
        print(f"🔍 5. Обнаружен early stopping по несоответствию: actual_episodes={actual_episodes}, episode_winrates={len(episode_winrates)}")
    
    # ИСПРАВЛЕНИЕ: Проверяем наличие early_stopping_triggered в результатах
    if 'early_stopping_triggered' in results and results['early_stopping_triggered']:
        early_stopping_triggered = True
        print(f"🔍 6. Обнаружен early stopping по флагу в результатах")
        
        # Если есть actual_episodes, используем его
        if 'actual_episodes' in results:
            actual_episodes = results['actual_episodes']
            print(f"🔍 6. Обновлен actual_episodes из результатов: {actual_episodes}")
    
    # ИСПРАВЛЕНИЕ: Дополнительная проверка по episode_winrates
    if episode_winrates:
        episode_winrates_count = len(episode_winrates)
        planned_episodes = results.get('episodes', planned_episodes)
        
        # ИСПРАВЛЕНИЕ: episode_winrates - это ЕДИНСТВЕННЫЙ надежный источник реального количества эпизодов
        # Если actual_episodes не равен episode_winrates_count, то actual_episodes неправильный
        if actual_episodes != episode_winrates_count:
            print(f"🔍 7. ОШИБКА: actual_episodes ({actual_episodes}) != episode_winrates_count ({episode_winrates_count})")
            print(f"🔍 7. Исправляем: actual_episodes = {episode_winrates_count}")
            actual_episodes = episode_winrates_count
            early_stopping_triggered = True
        
        # Если количество winrate'ов значительно меньше планируемых эпизодов, 
        # это явный признак early stopping
        elif episode_winrates_count < planned_episodes * 0.9:  # Если меньше 90% от планируемых
            early_stopping_triggered = True
            actual_episodes = episode_winrates_count
            print(f"🔍 7. Обнаружен early stopping по несоответствию episode_winrates: {episode_winrates_count} < {planned_episodes}")
            print(f"🔍 7. Обновлен actual_episodes: {actual_episodes}")
    
    print(f"🔍 ФИНАЛЬНЫЙ РЕЗУЛЬТАТ: actual_episodes = {actual_episodes}, early_stopping = {early_stopping_triggered}")
    
    # ФИНАЛЬНАЯ ПРОВЕРКА: Убеждаемся, что actual_episodes корректен
    if episode_winrates:
        episode_winrates_count = len(episode_winrates)
        if actual_episodes != episode_winrates_count:
            print(f"🔍 ФИНАЛЬНАЯ ПРОВЕРКА: Исправляем actual_episodes с {actual_episodes} на {episode_winrates_count}")
            actual_episodes = episode_winrates_count
            early_stopping_triggered = True
    
    if actual_episodes < planned_episodes:
        early_stopping_triggered = True
        print(f"🎯 Планируемое количество эпизодов: {planned_episodes}")
        print(f"⚠️ Реальное количество эпизодов: {actual_episodes}")
        print(f"🔄 Early Stopping сработал! Обучение остановлено на {actual_episodes} эпизоде из {planned_episodes}")
        print(f"📊 Причина: Достигнут стабильный winrate или сработали другие критерии остановки")
    else:
        print(f"🎯 Планируемое количество эпизодов: {planned_episodes}")
        print(f"✅ Реальное количество эпизодов: {actual_episodes}")
        print(f"✅ Обучение завершено полностью")
    
    if 'early_stopping_triggered' in results:
        early_stopping_triggered = results['early_stopping_triggered']
        print(f"🔄 Early Stopping: {'Сработал' if early_stopping_triggered else 'Не сработал'}")

    total_training_time = results.get('total_training_time') or results.get('training_time_seconds')
    if total_training_time is not None:
        try:
            total_training_time = float(total_training_time)
            print(f"⏱️ Время обучения: {total_training_time:.2f} секунд ({total_training_time/60:.1f} минут)")
            speed = (actual_episodes / (total_training_time / 60)) if total_training_time else 0.0
            print(f"🚀 Скорость: {speed:.1f} эпизодов/минуту")
        except Exception:
            print(f"⏱️ Время обучения: {total_training_time}")
    else:
        print("⏱️ Время обучения: N/A")

    if early_stopping_triggered:
        print(f"🔄 Early Stopping: Сработал на {actual_episodes} эпизоде")
        print(f"📊 Причина: Достигнут стабильный winrate или сработали другие критерии остановки")
        print(f"💡 Обучение остановлено автоматически для предотвращения переобучения")
    else:
        print(f"🔄 Early Stopping: Не сработал")
        print(f"✅ Обучение завершено по всем планируемым эпизодам")

    winrates = episode_winrates
    if winrates:
        print(f"\n📊 СТАТИСТИКА WINRATE:")
        print(f"  • Средний winrate: {np.mean(winrates):.3f}")
        best_winrate = results.get('best_winrate')
        if best_winrate is None and winrates:
            best_winrate = max(winrates)
        if best_winrate is not None:
            print(f"  • Лучший winrate: {float(best_winrate):.3f}")
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
            
            # Показываем реальное количество эпизодов в статистике
            print(f"  • Всего эпизодов в статистике: {len(winrates)}")
    
    # Статистика сделок
    trades = results.get('all_trades') or []
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
    final_stats = results.get('final_stats')
    if isinstance(final_stats, dict):
        print(f"\n📈 ФИНАЛЬНАЯ СТАТИСТИКА:")
        for key, value in final_stats.items():
            if isinstance(value, float):
                print(f"  • {key}: {value:.4f}")
            else:
                print(f"  • {key}: {value}")
    elif final_stats is not None:
        print(f"\n📈 ФИНАЛЬНАЯ СТАТИСТИКА: {final_stats}")
    
    # Дополнительная информация об обучении
    print(f"\n📊 ИНФОРМАЦИЯ ОБ ОБУЧЕНИИ:")
    print(f"  • Планируемые эпизоды: {planned_episodes}")
    print(f"  • Реальные эпизоды: {actual_episodes}")
    print(f"  • Early Stopping: {'Сработал' if early_stopping_triggered else 'Не сработал'}")
    if early_stopping_triggered:
        print(f"  • Остановка на: {actual_episodes} эпизоде")
        print(f"  • Причина: Достигнут стабильный winrate")
    
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

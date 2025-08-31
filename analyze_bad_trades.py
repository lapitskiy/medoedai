from typing import List, Dict, Any
import numpy as np


def analyze_bad_trades_detailed(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Простой анализ плохих сделок.
    Ожидается, что каждая сделка имеет ключи: 'roi' (прибыль в долях), 'duration' (в минутах, опц.).
    """
    if not trades:
        return {
            'bad_trades': [],
            'bad_trades_count': 0,
            'bad_trades_percentage': 0.0,
            'avg_bad_roi': 0.0,
            'avg_bad_duration': 0.0,
            'loss_distribution': {},
        }

    total_trades = len(trades)
    bad_trades = [t for t in trades if float(t.get('roi', 0.0)) < 0.0]
    bad_count = len(bad_trades)
    bad_pct = (bad_count / total_trades * 100.0) if total_trades else 0.0

    bad_rois = [float(t.get('roi', 0.0)) for t in bad_trades]
    bad_durations = [float(t.get('duration', 0.0)) for t in bad_trades if t.get('duration') is not None]

    avg_bad_roi = float(np.mean(bad_rois)) if bad_rois else 0.0
    avg_bad_duration = float(np.mean(bad_durations)) if bad_durations else 0.0

    # Грубая категоризация убытков
    loss_distribution = {
        'very_small_losses': sum(1 for r in bad_rois if -0.002 <= r < 0),     # до -0.2%
        'small_losses':      sum(1 for r in bad_rois if -0.01 <= r < -0.002), # до -1%
        'medium_losses':     sum(1 for r in bad_rois if -0.03 <= r < -0.01),  # до -3%
        'large_losses':      sum(1 for r in bad_rois if r < -0.03),           # больше -3%
    }

    return {
        'bad_trades': bad_trades,
        'bad_trades_count': bad_count,
        'bad_trades_percentage': bad_pct,
        'avg_bad_roi': avg_bad_roi,
        'avg_bad_duration': avg_bad_duration,
        'loss_distribution': loss_distribution,
    }


def print_bad_trades_analysis(analysis: Dict[str, Any]) -> None:
    print("============================================================")
    print("📉 АНАЛИЗ ПЛОХИХ СДЕЛОК")
    print("============================================================")
    print(f"Всего плохих сделок: {analysis.get('bad_trades_count', 0)}")
    print(f"Процент плохих сделок: {analysis.get('bad_trades_percentage', 0):.2f}%")
    print(f"Средний ROI плохих сделок: {analysis.get('avg_bad_roi', 0.0) * 100:.4f}%")
    print(f"Средняя длительность плохих сделок: {analysis.get('avg_bad_duration', 0.0):.1f} мин")
    dist = analysis.get('loss_distribution', {})
    if dist:
        print("\nРаспределение по убыткам:")
        for k, v in dist.items():
            print(f"  • {k}: {v}")


def print_detailed_recommendations(analysis: Dict[str, Any]) -> None:
    print("\n============================================================")
    print("🧠 РЕКОМЕНДАЦИИ ПО СНИЖЕНИЮ ПЛОХИХ СДЕЛОК")
    print("============================================================")
    bad_pct = analysis.get('bad_trades_percentage', 0)
    avg_bad_roi = analysis.get('avg_bad_roi', 0.0)
    if bad_pct > 5:
        print("• Ужесточить фильтр объема/волатильности в фазе exploitation (eps<=0.2)")
    if avg_bad_roi < -0.01:
        print("• Поднять стоп-лосс (меньше просадка), увеличить минимальное время удержания")
    print("• Проверьте распределение: very_small_losses → можно игнорировать как шум")
    print("• Подкрутите take-profit/stop-loss через адаптивные параметры")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ плохих сделок из результатов обучения DQN модели
Помогает понять, где модель ошибается и как её улучшить
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import sys
import os

# Добавляем путь к модулям
sys.path.append('agents/vdqn')
from analysis import analyze_bad_trades, print_bad_trades_analysis

def load_training_results(file_path: str) -> Dict[str, Any]:
    """Загружает результаты обучения из файла"""
    try:
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
        print(f"✅ Результаты загружены из: {file_path}")
        return results
    except Exception as e:
        print(f"❌ Ошибка загрузки файла: {e}")
        return {}

def analyze_bad_trades_detailed(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Расширенный анализ плохих сделок с дополнительными метриками"""
    
    if not trades:
        return {}
    
    df = pd.DataFrame(trades)
    
    # Определяем плохие сделки (ROI < 0.1%)
    bad_trades = df[abs(df['roi']) < 0.001]
    
    if len(bad_trades) == 0:
        return {'bad_trades_count': 0, 'message': 'Плохих сделок не найдено'}
    
    analysis = {
        'bad_trades_count': len(bad_trades),
        'bad_trades_percentage': len(bad_trades) / len(trades) * 100,
        'bad_trades_details': []
    }
    
    # Анализируем каждую плохую сделку
    for idx, trade in bad_trades.iterrows():
        trade_analysis = {
            'trade_id': idx,
            'roi': trade.get('roi', 0),
            'duration': trade.get('duration', 0),
            'entry_time': trade.get('entry_time', 'N/A'),
            'exit_time': trade.get('exit_time', 'N/A'),
            'entry_price': trade.get('entry_price', 0),
            'exit_price': trade.get('exit_price', 0),
            'volume': trade.get('volume', 0),
            'action': trade.get('action', 'N/A'),
            'market_conditions': trade.get('market_conditions', 'N/A')
        }
        analysis['bad_trades_details'].append(trade_analysis)
    
    # Статистика плохих сделок
    bad_rois = bad_trades['roi']
    analysis.update({
        'avg_bad_roi': bad_rois.mean(),
        'min_bad_roi': bad_rois.min(),
        'max_bad_roi': bad_rois.max(),
        'bad_roi_std': bad_rois.std(),
        
        # Анализ по времени
        'avg_bad_duration': bad_trades['duration'].mean() if 'duration' in bad_trades.columns else 0,
        'min_bad_duration': bad_trades['duration'].min() if 'duration' in bad_trades.columns else 0,
        'max_bad_duration': bad_trades['duration'].max() if 'duration' in bad_trades.columns else 0,
        
        # Анализ по объему
        'avg_bad_volume': bad_trades['volume'].mean() if 'volume' in bad_trades.columns else 0,
        'min_bad_volume': bad_trades['volume'].min() if 'volume' in bad_trades.columns else 0,
        'max_bad_volume': bad_trades['volume'].max() if 'volume' in bad_trades.columns else 0
    })
    
    # Группировка по причинам
    very_small_losses = bad_trades[bad_trades['roi'] < 0]
    very_small_profits = bad_trades[bad_trades['roi'] > 0]
    
    analysis['loss_distribution'] = {
        'very_small_losses': len(very_small_losses),
        'very_small_profits': len(very_small_profits),
        'neutral_trades': len(bad_trades) - len(very_small_losses) - len(very_small_profits)
    }
    
    return analysis

def create_bad_trades_plots(analysis: Dict[str, Any], save_path: str = 'plots'):
    """Создает графики для анализа плохих сделок"""
    
    if not analysis or analysis.get('bad_trades_count', 0) == 0:
        print("📊 Нет данных для создания графиков")
        return
    
    # Создаем папку для графиков
    Path(save_path).mkdir(exist_ok=True)
    
    # График 1: Распределение ROI плохих сделок
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    bad_trades_df = pd.DataFrame(analysis['bad_trades_details'])
    plt.hist(bad_trades_df['roi'], bins=20, alpha=0.7, color='red', edgecolor='black')
    plt.title('Распределение ROI плохих сделок')
    plt.xlabel('ROI')
    plt.ylabel('Количество сделок')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    # График 2: Длительность плохих сделок
    plt.subplot(2, 2, 2)
    plt.hist(bad_trades_df['duration'], bins=15, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Распределение длительности плохих сделок')
    plt.xlabel('Длительность (минуты)')
    plt.ylabel('Количество сделок')
    plt.grid(True, alpha=0.3)
    
    # График 3: Круговая диаграмма типов плохих сделок
    plt.subplot(2, 2, 3)
    dist = analysis['loss_distribution']
    labels = ['Убытки', 'Прибыли', 'Нейтральные']
    sizes = [dist['very_small_losses'], dist['very_small_profits'], dist['neutral_trades']]
    colors = ['red', 'green', 'gray']
    
    if sum(sizes) > 0:
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Типы плохих сделок')
    
    # График 4: Сравнение с хорошими сделками
    plt.subplot(2, 2, 4)
    all_trades_df = pd.DataFrame(analysis.get('all_trades', []))
    if not all_trades_df.empty and 'roi' in all_trades_df.columns:
        good_trades = all_trades_df[abs(all_trades_df['roi']) >= 0.001]
        bad_trades = all_trades_df[abs(all_trades_df['roi']) < 0.001]
        
        plt.hist([good_trades['roi'], bad_trades['roi']], 
                bins=20, alpha=0.7, label=['Хорошие сделки', 'Плохие сделки'],
                color=['green', 'red'])
        plt.title('Сравнение ROI: хорошие vs плохие сделки')
        plt.xlabel('ROI')
        plt.ylabel('Количество сделок')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/bad_trades_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Графики сохранены в: {save_path}/bad_trades_analysis.png")

def print_detailed_recommendations(analysis: Dict[str, Any]):
    """Выводит детальные рекомендации по улучшению модели"""
    
    if not analysis or analysis.get('bad_trades_count', 0) == 0:
        return
    
    print(f"\n🎯 ДЕТАЛЬНЫЕ РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ")
    print("=" * 60)
    
    dist = analysis['loss_distribution']
    total_bad = analysis['bad_trades_count']
    
    # Анализ по типам сделок
    if dist['very_small_losses'] > dist['very_small_profits']:
        print(f"🔴 ПРОБЛЕМА: Большинство плохих сделок - убыточные ({dist['very_small_losses']}/{total_bad})")
        print(f"   💡 РЕШЕНИЯ:")
        print(f"      • Улучшить точность входа в позицию")
        print(f"      • Добавить дополнительные фильтры для входа")
        print(f"      • Увеличить минимальный порог для открытия позиции")
        print(f"      • Улучшить анализ рыночных условий")
    else:
        print(f"🟡 ПРОБЛЕМА: Большинство плохих сделок - малоприбыльные ({dist['very_small_profits']}/{total_bad})")
        print(f"   💡 РЕШЕНИЯ:")
        print(f"      • Улучшить управление выходом из позиции")
        print(f"      • Добавить trailing stop-loss")
        print(f"      • Увеличить целевую прибыль")
        print(f"      • Улучшить анализ тренда")
    
    # Анализ по времени
    avg_duration = analysis['avg_bad_duration']
    if avg_duration < 300:  # Меньше 5 часов
        print(f"⏰ ПРОБЛЕМА: Плохие сделки слишком короткие ({avg_duration:.1f} мин)")
        print(f"   💡 РЕШЕНИЯ:")
        print(f"      • Увеличить минимальное время удержания позиции")
        print(f"      • Добавить фильтр по волатильности")
        print(f"      • Улучшить анализ краткосрочных движений")
    elif avg_duration > 600:  # Больше 10 часов
        print(f"⏰ ПРОБЛЕМА: Плохие сделки слишком длительные ({avg_duration:.1f} мин)")
        print(f"   💡 РЕШЕНИЯ:")
        print(f"      • Добавить максимальное время удержания")
        print(f"      • Улучшить stop-loss стратегию")
        print(f"      • Добавить анализ изменения тренда")
    
    # Общие рекомендации
    print(f"\n🔧 ОБЩИЕ РЕКОМЕНДАЦИИ:")
    print(f"   • Процент плохих сделок: {analysis['bad_trades_percentage']:.2f}%")
    if analysis['bad_trades_percentage'] > 5:
        print(f"   • ⚠️  Высокий процент плохих сделок - требуется оптимизация")
    elif analysis['bad_trades_percentage'] > 3:
        print(f"   • ⚠️  Средний процент плохих сделок - можно улучшить")
    else:
        print(f"   • ✅  Низкий процент плохих сделок - хороший результат")
    
    print(f"   • Стандартное отклонение ROI: {analysis['bad_roi_std']:.6f}")
    if analysis['bad_roi_std'] > 0.002:
        print(f"   • ⚠️  Высокая волатильность плохих сделок")
    else:
        print(f"   • ✅  Стабильные результаты плохих сделок")

def main():
    """Основная функция анализа"""
    
    print("🔍 АНАЛИЗ ПЛОХИХ СДЕЛОК ИЗ РЕЗУЛЬТАТОВ ОБУЧЕНИЯ")
    print("=" * 60)
    
    # Ищем файлы с результатами
    results_files = list(Path('temp/train_results').glob('*.pkl'))
    
    if not results_files:
        print("❌ Файлы с результатами не найдены в temp/train_results/")
        return
    
    # Показываем доступные файлы
    print(f"📁 Найдено файлов с результатами: {len(results_files)}")
    for i, file_path in enumerate(results_files):
        print(f"  {i+1}. {file_path.name}")
    
    # Загружаем последний файл
    latest_file = max(results_files, key=lambda x: x.stat().st_mtime)
    print(f"\n📥 Загружаем последний файл: {latest_file.name}")
    
    results = load_training_results(str(latest_file))
    if not results:
        return
    
    # Проверяем наличие сделок
    if 'all_trades' not in results:
        print("❌ В файле нет данных о сделках")
        return
    
    trades = results['all_trades']
    print(f"📊 Загружено сделок: {len(trades)}")
    
    # Анализируем плохие сделки
    bad_trades_analysis = analyze_bad_trades_detailed(trades)
    
    # Добавляем все сделки для сравнения
    bad_trades_analysis['all_trades'] = trades
    
    # Выводим анализ
    print_bad_trades_analysis(bad_trades_analysis)
    
    # Детальные рекомендации
    print_detailed_recommendations(bad_trades_analysis)
    
    # Создаем графики
    print(f"\n📈 Создание графиков...")
    create_bad_trades_plots(bad_trades_analysis)
    
    print(f"\n✅ Анализ завершен!")

if __name__ == "__main__":
    main()

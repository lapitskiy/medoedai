#!/usr/bin/env python3
"""
Полный анализ ошибок CNN модели на всех символах
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cnn_training.test.error_analyzer import CNNErrorAnalyzer
import json
from datetime import datetime

def full_analysis():
    """Полный анализ ошибок на всех символах"""
    print("🔍 ПОЛНЫЙ АНАЛИЗ ОШИБОК CNN МОДЕЛИ")
    print("=" * 50)
    
    # Настройки
    model_path = "cnn_training/result/multi/runs/93b1/cnn_model_multi_best.pth"
    test_symbols = ['SOLUSDT', 'XRPUSDT', 'TONUSDT']
    
    print(f"📊 Модель: {model_path}")
    print(f"🎯 Тестовые символы: {test_symbols}")
    print(f"⏰ Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Создаем анализатор
    analyzer = CNNErrorAnalyzer()
    
    # Анализируем все символы
    all_results = {}
    for i, symbol in enumerate(test_symbols, 1):
        print(f"🔍 [{i}/{len(test_symbols)}] Анализируем {symbol}...")
        result = analyzer.analyze_single_symbol(model_path, symbol)
        all_results[symbol] = result
        
        if 'error' not in result:
            print(f"✅ {symbol}: точность {result['accuracy']:.2%}, образцов {result['samples']}")
        else:
            print(f"❌ {symbol}: {result['error']}")
        print()
    
    # Общий анализ
    print("📊 ОБЩИЙ АНАЛИЗ:")
    print("-" * 30)
    
    successful_results = {k: v for k, v in all_results.items() if 'error' not in v}
    
    if successful_results:
        accuracies = [r['accuracy'] for r in successful_results.values()]
        avg_accuracy = sum(accuracies) / len(accuracies)
        
        best_symbol = max(successful_results.items(), key=lambda x: x[1]['accuracy'])
        worst_symbol = min(successful_results.items(), key=lambda x: x[1]['accuracy'])
        
        print(f"Средняя точность: {avg_accuracy:.2%}")
        print(f"Лучший символ: {best_symbol[0]} ({best_symbol[1]['accuracy']:.2%})")
        print(f"Худший символ: {worst_symbol[0]} ({worst_symbol[1]['accuracy']:.2%})")
        print(f"Успешных тестов: {len(successful_results)}/{len(test_symbols)}")
        
        # Детальный анализ лучшего и худшего символов
        print(f"\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ЛУЧШЕГО СИМВОЛА ({best_symbol[0]}):")
        analyzer.print_detailed_analysis(best_symbol[1])
        
        print(f"\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ХУДШЕГО СИМВОЛА ({worst_symbol[0]}):")
        analyzer.print_detailed_analysis(worst_symbol[1])
        
        # Общие рекомендации
        print(f"\n💡 ОБЩИЕ РЕКОМЕНДАЦИИ:")
        print("-" * 30)
        
        if avg_accuracy < 0.55:
            print("🔴 КРИТИЧЕСКИЕ ПРОБЛЕМЫ:")
            print("  • Средняя точность близка к случайной (50%)")
            print("  • Модель не улавливает универсальные паттерны")
            print("  • Требуется кардинальная переработка")
            print("\n🚀 ПЛАН ДЕЙСТВИЙ:")
            print("  1. Обучить новую модель с:")
            print("     - 10+ криптовалютами в обучении")
            print("     - Улучшенной архитектурой")
            print("     - Большим количеством признаков")
            print("  2. Проанализировать качество данных обучения")
            print("  3. Попробовать другие подходы (LSTM, Transformer)")
        elif avg_accuracy < 0.65:
            print("🟡 ПРОБЛЕМЫ СРЕДНЕЙ СЕРЬЕЗНОСТИ:")
            print("  • Точность выше случайной, но недостаточная")
            print("  • Модель частично работает на новых символах")
            print("  • Требуются улучшения")
            print("\n🚀 ПЛАН ДЕЙСТВИЙ:")
            print("  1. Улучшить текущую модель:")
            print("     - Добавить технические индикаторы")
            print("     - Увеличить данные обучения")
            print("     - Настроить гиперпараметры")
            print("  2. Использовать с осторожностью")
            print("  3. Постоянно мониторить")
        else:
            print("🟢 ПРИЕМЛЕМЫЕ РЕЗУЛЬТАТЫ:")
            print("  • Средняя точность достаточная")
            print("  • Модель показывает обобщение")
            print("  • Можно использовать в торговле")
            print("\n🚀 ПЛАН ДЕЙСТВИЙ:")
            print("  1. Продолжить улучшение:")
            print("     - Добавить больше символов в валидацию")
            print("     - Оптимизировать пороги")
            print("     - Реализовать риск-менеджмент")
            print("  2. Начать осторожное использование")
            print("  3. Постоянно мониторить производительность")
        
        # Сохраняем результаты в файл
        results_file = f"cnn_training/test/error_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'model_path': model_path,
                'test_symbols': test_symbols,
                'results': all_results,
                'summary': {
                    'avg_accuracy': avg_accuracy,
                    'best_symbol': best_symbol[0],
                    'worst_symbol': worst_symbol[0],
                    'successful_tests': len(successful_results),
                    'total_tests': len(test_symbols)
                }
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Результаты сохранены в: {results_file}")
    
    else:
        print("❌ Не удалось проанализировать ни одного символа")

if __name__ == "__main__":
    full_analysis()

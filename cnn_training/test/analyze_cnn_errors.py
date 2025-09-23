#!/usr/bin/env python3
"""
Скрипт для детального анализа ошибок CNN модели
Показывает, на каких типах паттернов модель ошибается
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cnn_training.model_validator import validate_cnn_model
import json
import numpy as np
from datetime import datetime

def analyze_cnn_errors():
    """Детальный анализ ошибок CNN модели"""
    
    print("🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ОШИБОК CNN МОДЕЛИ")
    print("=" * 50)
    
    # Путь к модели
    model_path = "cnn_training/result/multi/runs/93b1/cnn_model_multi_best.pth"
    
    # Тестовые символы
    test_symbols = ['SOLUSDT', 'XRPUSDT', 'TONUSDT']
    
    print(f"📊 Модель: {model_path}")
    print(f"🎯 Тестовые символы: {test_symbols}")
    print()
    
    # Запускаем валидацию
    print("🧪 Запускаем валидацию...")
    results = validate_cnn_model(
        model_path=model_path,
        test_symbols=test_symbols,
        test_period='last_year'
    )
    
    if not results['success']:
        print(f"❌ Ошибка валидации: {results.get('error', 'Неизвестная ошибка')}")
        return
    
    print("✅ Валидация завершена")
    print()
    
    # Анализируем результаты
    symbol_results = results.get('symbol_results', [])
    overall_accuracy = results.get('overall_accuracy', 0)
    
    print(f"📈 ОБЩАЯ ТОЧНОСТЬ: {overall_accuracy:.2%}")
    print()
    
    # Детальный анализ по символам
    for symbol_result in symbol_results:
        if not symbol_result.get('success', False):
            continue
            
        symbol = symbol_result['symbol']
        accuracy = symbol_result['accuracy']
        samples = symbol_result['samples_tested']
        error_analysis = symbol_result.get('error_analysis', {})
        
        print(f"🔍 АНАЛИЗ СИМВОЛА: {symbol}")
        print("-" * 30)
        print(f"Точность: {accuracy:.2%}")
        print(f"Образцов: {samples}")
        print()
        
        # Анализ ошибок
        if error_analysis:
            print("📊 АНАЛИЗ ОШИБОК:")
            
            # Матрица ошибок
            cm = error_analysis.get('confusion_matrix', [])
            if cm:
                print(f"Матрица ошибок: {cm}")
            
            # Типы ошибок
            error_patterns = error_analysis.get('error_patterns', {})
            if error_patterns:
                print(f"Всего ошибок: {error_patterns.get('total_errors', 0)}")
                print(f"Правильных: {error_patterns.get('total_correct', 0)}")
                
                error_types = error_patterns.get('error_types', {})
                if error_types:
                    print("Типы ошибок:")
                    for error_type, count in error_types.items():
                        print(f"  {error_type}: {count} раз")
                
                most_common = error_patterns.get('most_common_error')
                if most_common:
                    print(f"Самая частая ошибка: {most_common[0]} ({most_common[1]} раз)")
            
            # Баланс классов
            class_balance = error_analysis.get('class_balance', {})
            if class_balance:
                print("\n⚖️ БАЛАНС КЛАССОВ:")
                distribution = class_balance.get('class_distribution', {})
                for class_id, info in distribution.items():
                    print(f"  Класс {class_id}: {info['count']} ({info['percentage']:.1f}%)")
                
                is_balanced = class_balance.get('is_balanced', False)
                imbalance_ratio = class_balance.get('imbalance_ratio', 1.0)
                print(f"  Сбалансированность: {'✅ Да' if is_balanced else '❌ Нет'} (соотношение: {imbalance_ratio:.2f})")
            
            # Уверенность предсказаний
            confidence_analysis = error_analysis.get('prediction_confidence', {})
            if confidence_analysis:
                print("\n🎯 УВЕРЕННОСТЬ ПРЕДСКАЗАНИЙ:")
                dist = confidence_analysis.get('confidence_distribution', {})
                print(f"  Высокая уверенность: {dist.get('high', 0):.1f}%")
                print(f"  Средняя уверенность: {dist.get('medium', 0):.1f}%")
                print(f"  Низкая уверенность: {dist.get('low', 0):.1f}%")
                print(f"  Средняя длина паттерна: {confidence_analysis.get('avg_pattern_length', 0):.1f}")
        
        print()
    
    # Общие выводы и рекомендации
    print("💡 ВЫВОДЫ И РЕКОМЕНДАЦИИ:")
    print("-" * 30)
    
    if overall_accuracy < 0.55:
        print("🔴 КРИТИЧЕСКИЕ ПРОБЛЕМЫ:")
        print("  • Точность близка к случайной (50%)")
        print("  • Модель не улавливает паттерны на новых символах")
        print("  • Требуется пересмотр архитектуры или данных обучения")
    elif overall_accuracy < 0.65:
        print("🟡 ПРОБЛЕМЫ СРЕДНЕЙ СЕРЬЕЗНОСТИ:")
        print("  • Точность выше случайной, но недостаточная")
        print("  • Модель частично улавливает паттерны")
        print("  • Рекомендуется дообучение или улучшение признаков")
    else:
        print("🟢 ХОРОШИЕ РЕЗУЛЬТАТЫ:")
        print("  • Точность приемлемая для торговли")
        print("  • Модель показывает обобщающую способность")
        print("  • Можно использовать с осторожностью")
    
    # Анализ по символам
    symbol_accuracies = [r['accuracy'] for r in symbol_results if r.get('success', False)]
    if symbol_accuracies:
        best_symbol = max(symbol_results, key=lambda x: x.get('accuracy', 0))
        worst_symbol = min(symbol_results, key=lambda x: x.get('accuracy', 1))
        
        print(f"\n📊 ЛУЧШИЙ СИМВОЛ: {best_symbol['symbol']} ({best_symbol['accuracy']:.2%})")
        print(f"📊 ХУДШИЙ СИМВОЛ: {worst_symbol['symbol']} ({worst_symbol['accuracy']:.2%})")
        
        # Рекомендации по символам
        for symbol_result in symbol_results:
            if not symbol_result.get('success', False):
                continue
                
            symbol = symbol_result['symbol']
            accuracy = symbol_result['accuracy']
            
            if accuracy < 0.5:
                print(f"  • {symbol}: Требует специализированного обучения")
            elif accuracy < 0.6:
                print(f"  • {symbol}: Нужно больше данных или улучшение признаков")
            else:
                print(f"  • {symbol}: Показывает хорошие результаты")
    
    print("\n🚀 СЛЕДУЮЩИЕ ШАГИ:")
    print("  1. Проанализировать ошибки на конкретных паттернах")
    print("  2. Увеличить объем данных для обучения")
    print("  3. Попробовать другие архитектуры CNN")
    print("  4. Добавить больше технических индикаторов")
    print("  5. Использовать ансамбли моделей")

if __name__ == "__main__":
    analyze_cnn_errors()

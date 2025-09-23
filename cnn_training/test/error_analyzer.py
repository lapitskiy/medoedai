#!/usr/bin/env python3
"""
Анализатор ошибок CNN модели
Показывает детальную информацию о том, на каких паттернах модель ошибается
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cnn_training.model_validator import CNNModelValidator
import numpy as np
import json
from datetime import datetime

class CNNErrorAnalyzer:
    """Класс для анализа ошибок CNN модели"""
    
    def __init__(self):
        self.validator = CNNModelValidator()
    
    def analyze_single_symbol(self, model_path: str, symbol: str) -> dict:
        """Анализ ошибок на одном символе"""
        print(f"🔍 Анализ ошибок для {symbol}...")
        
        # Загружаем модель
        model = self.validator._load_model(model_path)
        if model is None:
            return {"error": "Не удалось загрузить модель"}
        
        # Валидируем символ (используем публичный метод validate_cnn_model)
        from cnn_training.model_validator import validate_cnn_model
        results = validate_cnn_model(
            model_path=model_path,
            test_symbols=[symbol],
            test_period="last_year"
        )
        
        if not results['success']:
            return {"error": f"Ошибка валидации: {results.get('error', 'Неизвестная ошибка')}"}
        
        symbol_results = results.get('symbol_results', [])
        if not symbol_results:
            return {"error": "Нет результатов для символа"}
        
        result = symbol_results[0]
        
        if not result.get('success', False):
            return {"error": f"Ошибка валидации: {result.get('error', 'Неизвестная ошибка')}"}
        
        # Детальный анализ
        analysis = {
            "symbol": symbol,
            "accuracy": result['accuracy'],
            "samples": result['samples_tested'],
            "error_analysis": result.get('error_analysis', {}),
            "patterns": result.get('patterns_detected', []),
            "confidence": result.get('avg_confidence', 0)
        }
        
        return analysis
    
    def analyze_multiple_symbols(self, model_path: str, symbols: list) -> dict:
        """Анализ ошибок на нескольких символах"""
        print(f"🔍 Анализ ошибок для символов: {symbols}")
        
        results = {}
        for symbol in symbols:
            results[symbol] = self.analyze_single_symbol(model_path, symbol)
        
        return results
    
    def print_detailed_analysis(self, analysis: dict):
        """Вывод детального анализа"""
        symbol = analysis['symbol']
        accuracy = analysis['accuracy']
        samples = analysis['samples']
        error_analysis = analysis.get('error_analysis', {})
        
        print(f"\n📊 РЕЗУЛЬТАТЫ ДЛЯ {symbol}:")
        print(f"  Точность: {accuracy:.2%}")
        print(f"  Образцов: {samples}")
        print(f"  Уверенность: {analysis.get('confidence', 0):.2%}")
        print()
        
        if not error_analysis:
            print("❌ Нет данных для анализа ошибок")
            return
        
        # Матрица ошибок
        cm = error_analysis.get('confusion_matrix', [])
        if cm and len(cm) == 2:
            print("🔢 МАТРИЦА ОШИБОК:")
            print(f"  Правильно предсказали падение (0): {cm[0][0]}")
            print(f"  Ошибочно предсказали рост вместо падения: {cm[0][1]}")
            print(f"  Ошибочно предсказали падение вместо роста: {cm[1][0]}")
            print(f"  Правильно предсказали рост (1): {cm[1][1]}")
            print()
        
        # Типы ошибок
        error_patterns = error_analysis.get('error_patterns', {})
        if error_patterns:
            total_errors = error_patterns.get('total_errors', 0)
            total_correct = error_patterns.get('total_correct', 0)
            
            print("❌ АНАЛИЗ ОШИБОК:")
            print(f"  Всего ошибок: {total_errors}")
            print(f"  Правильных предсказаний: {total_correct}")
            print(f"  Процент ошибок: {total_errors/(total_errors+total_correct)*100:.1f}%")
            
            error_types = error_patterns.get('error_types', {})
            if error_types:
                print(f"\n  Типы ошибок:")
                for error_type, count in error_types.items():
                    error_desc = self._describe_error_type(error_type)
                    print(f"    {error_desc}: {count} раз ({count/total_errors*100:.1f}%)")
            print()
        
        # Баланс классов
        class_balance = error_analysis.get('class_balance', {})
        if class_balance:
            print("⚖️ БАЛАНС КЛАССОВ:")
            distribution = class_balance.get('class_distribution', {})
            for class_id, info in distribution.items():
                class_name = "Падение" if class_id == 0 else "Рост"
                print(f"  {class_name} (класс {class_id}): {info['count']} ({info['percentage']:.1f}%)")
            
            is_balanced = class_balance.get('is_balanced', False)
            imbalance_ratio = class_balance.get('imbalance_ratio', 1.0)
            print(f"  Сбалансированность: {'✅ Да' if is_balanced else '❌ Нет'}")
            print(f"  Соотношение классов: {imbalance_ratio:.2f}")
            print()
        
        # Уверенность предсказаний
        confidence_analysis = error_analysis.get('prediction_confidence', {})
        if confidence_analysis:
            print("🎯 УВЕРЕННОСТЬ ПРЕДСКАЗАНИЙ:")
            dist = confidence_analysis.get('confidence_distribution', {})
            print(f"  Высокая уверенность: {dist.get('high', 0):.1f}%")
            print(f"  Средняя уверенность: {dist.get('medium', 0):.1f}%")
            print(f"  Низкая уверенность: {dist.get('low', 0):.1f}%")
            print(f"  Средняя длина паттерна: {confidence_analysis.get('avg_pattern_length', 0):.1f}")
            print()
    
    def _describe_error_type(self, error_type: str) -> str:
        """Описание типа ошибки"""
        if "True_0_Pred_1" in error_type:
            return "Ложный рост (предсказали рост, а было падение)"
        elif "True_1_Pred_0" in error_type:
            return "Ложное падение (предсказали падение, а был рост)"
        else:
            return error_type
    
    def generate_recommendations(self, analysis: dict) -> list:
        """Генерация рекомендаций на основе анализа"""
        recommendations = []
        accuracy = analysis['accuracy']
        error_analysis = analysis.get('error_analysis', {})
        
        if accuracy < 0.55:
            recommendations.extend([
                "🔴 КРИТИЧЕСКИЕ ПРОБЛЕМЫ:",
                "  • Точность близка к случайной (50%)",
                "  • Модель не улавливает паттерны на новых символах",
                "  • Требуется кардинальная переработка архитектуры",
                "",
                "🚀 РЕКОМЕНДАЦИИ:",
                "  1. Обучить новую модель с:",
                "     - Больше символов в обучении (10+ криптовалют)",
                "     - Улучшенной архитектурой (более глубокие слои)",
                "     - Лучшими признаками (технические индикаторы)",
                "  2. Проанализировать данные обучения на качество",
                "  3. Попробовать другие подходы (LSTM, Transformer)",
                "  4. Использовать ансамбли моделей"
            ])
        elif accuracy < 0.65:
            recommendations.extend([
                "🟡 ПРОБЛЕМЫ СРЕДНЕЙ СЕРЬЕЗНОСТИ:",
                "  • Точность выше случайной, но недостаточная",
                "  • Модель частично улавливает паттерны",
                "  • Требуются улучшения",
                "",
                "🚀 РЕКОМЕНДАЦИИ:",
                "  1. Улучшить текущую модель:",
                "     - Добавить больше признаков (RSI, MACD, Bollinger Bands)",
                "     - Увеличить объем данных обучения",
                "     - Настроить гиперпараметры",
                "  2. Использовать с осторожностью в торговле",
                "  3. Мониторить производительность в реальном времени"
            ])
        else:
            recommendations.extend([
                "🟢 ПРИЕМЛЕМЫЕ РЕЗУЛЬТАТЫ:",
                "  • Точность достаточная для торговли",
                "  • Модель показывает обобщающую способность",
                "  • Можно использовать с осторожностью",
                "",
                "🚀 РЕКОМЕНДАЦИИ:",
                "  1. Продолжить улучшение модели:",
                "     - Добавить больше символов в валидацию",
                "     - Оптимизировать пороги принятия решений",
                "     - Реализовать риск-менеджмент",
                "  2. Начать осторожное использование в торговле",
                "  3. Постоянно мониторить производительность"
            ])
        
        # Дополнительные рекомендации на основе анализа ошибок
        if error_analysis:
            error_patterns = error_analysis.get('error_patterns', {})
            if error_patterns:
                most_common_error = error_patterns.get('most_common_error')
                if most_common_error:
                    error_desc = self._describe_error_type(most_common_error[0])
                    recommendations.append(f"  4. Обратить внимание на: {error_desc}")
            
            class_balance = error_analysis.get('class_balance', {})
            if class_balance and not class_balance.get('is_balanced', True):
                recommendations.append("  5. Исправить дисбаланс классов в данных обучения")
        
        return recommendations

def main():
    """Основная функция для запуска анализа ошибок"""
    print("🔍 АНАЛИЗ ОШИБОК CNN МОДЕЛИ")
    print("=" * 50)
    
    # Настройки
    model_path = "cnn_training/result/multi/runs/93b1/cnn_model_multi_best.pth"
    test_symbols = ['SOLUSDT', 'XRPUSDT', 'TONUSDT']
    
    print(f"📊 Модель: {model_path}")
    print(f"🎯 Тестовые символы: {test_symbols}")
    print()
    
    # Создаем анализатор
    analyzer = CNNErrorAnalyzer()
    
    # Анализируем каждый символ
    all_results = {}
    for symbol in test_symbols:
        print(f"🔍 Анализируем {symbol}...")
        result = analyzer.analyze_single_symbol(model_path, symbol)
        all_results[symbol] = result
        
        if 'error' not in result:
            analyzer.print_detailed_analysis(result)
            recommendations = analyzer.generate_recommendations(result)
            print("\n".join(recommendations))
            print("\n" + "="*50 + "\n")
    
    # Общий анализ
    print("📊 ОБЩИЙ АНАЛИЗ:")
    print("-" * 30)
    
    accuracies = [r['accuracy'] for r in all_results.values() if 'error' not in r]
    if accuracies:
        avg_accuracy = np.mean(accuracies)
        best_symbol = max(all_results.items(), key=lambda x: x[1].get('accuracy', 0))
        worst_symbol = min(all_results.items(), key=lambda x: x[1].get('accuracy', 1))
        
        print(f"Средняя точность: {avg_accuracy:.2%}")
        print(f"Лучший символ: {best_symbol[0]} ({best_symbol[1]['accuracy']:.2%})")
        print(f"Худший символ: {worst_symbol[0]} ({worst_symbol[1]['accuracy']:.2%})")
        
        if avg_accuracy < 0.55:
            print("\n🔴 ВЫВОД: Модель требует кардинальной переработки")
        elif avg_accuracy < 0.65:
            print("\n🟡 ВЫВОД: Модель нуждается в улучшениях")
        else:
            print("\n🟢 ВЫВОД: Модель показывает приемлемые результаты")

if __name__ == "__main__":
    main()

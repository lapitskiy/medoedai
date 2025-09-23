#!/usr/bin/env python3
"""
Быстрый запуск анализа ошибок CNN модели
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cnn_training.test.error_analyzer import CNNErrorAnalyzer

def quick_analysis():
    """Быстрый анализ ошибок"""
    print("🚀 БЫСТРЫЙ АНАЛИЗ ОШИБОК CNN")
    print("=" * 40)
    
    # Настройки
    model_path = "cnn_training/result/multi/runs/93b1/cnn_model_multi_best.pth"
    test_symbol = "SOLUSDT"  # Только один символ для быстроты
    
    print(f"📊 Модель: {model_path}")
    print(f"🎯 Тестовый символ: {test_symbol}")
    print()
    
    # Создаем анализатор
    analyzer = CNNErrorAnalyzer()
    
    # Анализируем
    result = analyzer.analyze_single_symbol(model_path, test_symbol)
    
    if 'error' in result:
        print(f"❌ Ошибка: {result['error']}")
        return
    
    # Выводим результаты
    analyzer.print_detailed_analysis(result)
    
    # Генерируем рекомендации
    recommendations = analyzer.generate_recommendations(result)
    print("\n".join(recommendations))

if __name__ == "__main__":
    quick_analysis()

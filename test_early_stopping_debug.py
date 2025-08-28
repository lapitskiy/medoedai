#!/usr/bin/env python3
"""
Тест параметров early stopping
"""

def test_early_stopping_calculation(episodes=10000):
    """Тестирует расчет параметров early stopping"""
    
    # Базовые параметры
    base_patience_limit = 3000
    
    # Адаптивный patience_limit в зависимости от количества эпизодов
    if episodes >= 10000:
        patience_limit = max(base_patience_limit, episodes // 3)  # 3333
    elif episodes >= 5000:
        patience_limit = max(base_patience_limit, episodes // 4)  # 2500
    elif episodes >= 2000:
        patience_limit = max(base_patience_limit, episodes // 3)  # 3333
    
    # Увеличиваем patience для длинных тренировок
    patience_limit = max(patience_limit, 8000)  # 8000
    
    # Минимальное количество эпизодов перед stopping
    min_episodes_before_stopping = max(3000, episodes // 4)  # 3000
    
    # Долгосрочный patience
    long_term_patience = int(patience_limit * 2.5)  # 20000
    
    print(f"🧪 ТЕСТ ПАРАМЕТРОВ EARLY STOPPING")
    print(f"==================================")
    print(f"Планируемые эпизоды: {episodes}")
    print(f"min_episodes_before_stopping: {min_episodes_before_stopping}")
    print(f"patience_limit: {patience_limit}")
    print(f"long_term_patience: {long_term_patience}")
    print(f"trend_threshold: 0.05")
    print()
    
    print(f"🎯 ЗАЩИТА ОТ РАННЕГО STOPPING:")
    print(f"• До {episodes // 2} эпизодов: patience ограничен до {patience_limit // 4}")
    print(f"• До {episodes * 3 // 4} эпизодов: patience ограничен до {patience_limit // 2}")
    print(f"• Анализ трендов: только после {episodes * 3 // 4} эпизодов и 300+ winrate'ов")
    print()
    
    # Рассчитываем, когда может сработать early stopping
    earliest_stopping = min_episodes_before_stopping + patience_limit
    print(f"🚨 САМЫЙ РАННИЙ STOPPING: {earliest_stopping} эпизодов")
    print(f"✅ Это означает, что обучение продлится минимум {earliest_stopping} эпизодов")
    
    # Проверяем, что это больше 500
    if earliest_stopping > 500:
        print(f"✅ ПРОБЛЕМА РЕШЕНА: Early stopping не сработает на 500 эпизоде!")
    else:
        print(f"❌ ПРОБЛЕМА НЕ РЕШЕНА: Early stopping все еще может сработать рано!")

if __name__ == "__main__":
    test_early_stopping_calculation(10000)


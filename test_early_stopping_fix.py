#!/usr/bin/env python3
"""
🧪 Тест исправлений early stopping
"""

def test_early_stopping_settings():
    """Тестирует новые настройки early stopping"""
    
    # Симулируем настройки для 10000 эпизодов
    episodes = 10000
    base_patience_limit = 3000
    
    print("🧪 ТЕСТ НОВЫХ НАСТРОЕК EARLY STOPPING")
    print("=" * 50)
    
    # Адаптивный patience_limit
    if episodes >= 10000:
        patience_limit = max(base_patience_limit, episodes // 2)  # 5000
    elif episodes >= 5000:
        patience_limit = max(base_patience_limit, episodes // 3)
    elif episodes >= 2000:
        patience_limit = max(base_patience_limit, episodes // 2)
    
    # Увеличиваем patience для длинных тренировок
    patience_limit = max(patience_limit, 5000)  # Минимум 5000 эпизодов
    
    long_term_patience = int(patience_limit * 2.5)
    trend_threshold = 0.03
    
    print(f"📊 Настройки для {episodes} эпизодов:")
    print(f"  • Базовый patience_limit: {base_patience_limit}")
    print(f"  • Адаптивный patience_limit: {patience_limit}")
    print(f"  • Долгосрочный patience: {long_term_patience}")
    print(f"  • Порог тренда: {trend_threshold}")
    
    # Защита от раннего stopping
    protection_1 = episodes // 3  # 3333
    protection_2 = episodes // 2  # 5000
    min_episodes = max(1000, episodes // 8)  # 1250
    
    print(f"\n🛡️ Защита от раннего stopping:")
    print(f"  • Минимум эпизодов: {min_episodes}")
    print(f"  • Защита 1: первые {protection_1} эпизодов")
    print(f"  • Защита 2: до {protection_2} эпизода")
    
    # Анализ трендов
    trend_min_episodes = max(200, episodes // 2)  # 5000
    trend_window = 90  # Увеличили с 60 до 90
    
    print(f"\n📈 Анализ трендов:")
    print(f"  • Минимум эпизодов: {trend_min_episodes}")
    print(f"  • Окно анализа: {trend_window} эпизодов")
    print(f"  • Порог тренда: {trend_threshold * 1.5:.3f}")
    
    # Проверяем, когда может сработать early stopping
    print(f"\n⚠️ Early stopping может сработать:")
    print(f"  • По patience: после {patience_limit} эпизодов без улучшений")
    print(f"  • По тренду: после {trend_min_episodes + trend_window} эпизодов")
    print(f"  • По долгосрочному patience: после {long_term_patience} эпизодов")
    
    # Расчеты для понимания
    print(f"\n🧮 Расчеты:")
    print(f"  • 1/3 от {episodes}: {episodes // 3}")
    print(f"  • 1/2 от {episodes}: {episodes // 2}")
    print(f"  • 2/3 от {episodes}: {episodes * 2 // 3}")
    
    print(f"\n✅ Результат: Early stopping теперь сработает не раньше {min(patience_limit, trend_min_episodes + trend_window)} эпизода")

if __name__ == "__main__":
    test_early_stopping_settings()

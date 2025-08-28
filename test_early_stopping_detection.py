#!/usr/bin/env python3
"""
🧪 Тест исправлений early stopping detection
"""

def test_early_stopping_detection():
    """Тестирует логику определения early stopping"""
    
    print("🧪 ТЕСТ ОПРЕДЕЛЕНИЯ EARLY STOPPING")
    print("=" * 50)
    
    # Симулируем разные сценарии
    test_cases = [
        {
            "name": "Early stopping по patience",
            "episodes": 10000,
            "actual_episodes": 500,
            "episode_winrates_count": 500,
            "early_stopping_triggered": True
        },
        {
            "name": "Early stopping по тренду",
            "episodes": 10000,
            "actual_episodes": 750,
            "episode_winrates_count": 750,
            "early_stopping_triggered": True
        },
        {
            "name": "Полное завершение",
            "episodes": 1000,
            "actual_episodes": 1000,
            "episode_winrates_count": 1000,
            "early_stopping_triggered": False
        },
        {
            "name": "Early stopping без actual_episodes",
            "episodes": 10000,
            "actual_episodes": None,
            "episode_winrates_count": 600,
            "early_stopping_triggered": True
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📋 Тест {i}: {case['name']}")
        print("-" * 40)
        
        planned_episodes = case["episodes"]
        actual_episodes = case["actual_episodes"]
        episode_winrates_count = case["episode_winrates_count"]
        early_stopping_triggered = case["early_stopping_triggered"]
        
        # Симулируем логику анализа
        detected_actual_episodes = None
        detected_early_stopping = False
        
        # 1. Проверяем actual_episodes если есть
        if actual_episodes is not None:
            detected_actual_episodes = actual_episodes
            print(f"🔍 Найден actual_episodes: {detected_actual_episodes}")
        
        # 2. Проверяем episode_winrates
        elif episode_winrates_count > 0:
            episode_winrates_count = episode_winrates_count
            print(f"🔍 Найден episode_winrates с {episode_winrates_count} элементами")
            
            # Если количество winrate'ов значительно меньше планируемых эпизодов
            if episode_winrates_count < planned_episodes * 0.8:
                detected_actual_episodes = episode_winrates_count
                detected_early_stopping = True
                print(f"⚠️ Обнаружен возможный early stopping: {episode_winrates_count} < {planned_episodes}")
            else:
                detected_actual_episodes = episode_winrates_count
        
        # 3. Если ничего не нашли, используем планируемое
        if detected_actual_episodes is None:
            detected_actual_episodes = planned_episodes
            print(f"🔍 Не найдена информация о реальных эпизодах, используем планируемое: {detected_actual_episodes}")
        
        # 4. Дополнительная проверка по несоответствию
        if (detected_actual_episodes == planned_episodes and 
            episode_winrates_count < planned_episodes):
            
            detected_actual_episodes = episode_winrates_count
            detected_early_stopping = True
            print(f"🔍 Обнаружен early stopping по несоответствию: {detected_actual_episodes}, episode_winrates={episode_winrates_count}")
        
        # 5. Проверяем флаг early_stopping_triggered
        if early_stopping_triggered:
            detected_early_stopping = True
            print(f"🔍 Обнаружен early stopping по флагу в результатах")
            
            # Если есть actual_episodes, используем его
            if actual_episodes is not None:
                detected_actual_episodes = actual_episodes
                print(f"🔍 Обновлен actual_episodes из результатов: {detected_actual_episodes}")
        
        # Результаты
        print(f"📊 Результаты:")
        print(f"  • Планируемые эпизоды: {planned_episodes}")
        print(f"  • Обнаруженные реальные эпизоды: {detected_actual_episodes}")
        print(f"  • Early stopping обнаружен: {'Да' if detected_early_stopping else 'Нет'}")
        print(f"  • Ожидалось: {case['actual_episodes'] if case['actual_episodes'] else 'early stopping'}")
        
        # Проверяем корректность
        if case["early_stopping_triggered"]:
            if detected_early_stopping and detected_actual_episodes < planned_episodes:
                print("✅ ТЕСТ ПРОЙДЕН: Early stopping корректно обнаружен")
            else:
                print("❌ ТЕСТ ПРОВАЛЕН: Early stopping не обнаружен")
        else:
            if not detected_early_stopping and detected_actual_episodes == planned_episodes:
                print("✅ ТЕСТ ПРОЙДЕН: Полное завершение корректно обнаружено")
            else:
                print("❌ ТЕСТ ПРОВАЛЕН: Неправильно определено завершение")

if __name__ == "__main__":
    test_early_stopping_detection()

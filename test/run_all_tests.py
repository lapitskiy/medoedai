#!/usr/bin/env python3
"""
Главный файл для запуска всех тестов DQN
"""

import sys
import os
import time
sys.path.append('/app')

def run_all_tests():
    """Запускает все тесты DQN"""
    print("🚀 ЗАПУСК ВСЕХ ТЕСТОВ DQN")
    print("=" * 60)
    
    start_time = time.time()
    test_results = {}
    overall_success = True
    
    # Список всех тестов
    tests = [
        ("Конфигурация", "test_configuration", "test_configuration"),
        ("Нейронная сеть", "test_neural_network", "test_neural_network"),
        ("DQN Solver", "test_dqn_solver", "test_dqn_solver"),
        ("Market STATE + Mask", "test_market_state_mask", "test_market_state_masking"),
        ("Обработка NaN", "test_nan_handling", "test_nan_handling"),
        ("GPU Replay Buffer", "test_gpu_replay", "test_replay_buffer_performance"),
        ("Предвычисление состояний", "test_precomputed_states", "test_precomputed_states"),
        ("torch.compile", "test_torch_compile", "test_torch_compile")
    ]
    
    for i, (test_name, module_name, function_name) in enumerate(tests, 1):
        print(f"\n{i}️⃣ {test_name}...")
        
        try:
            # Импортируем модуль
            module = __import__(module_name)
            test_function = getattr(module, function_name)
            
            # Запускаем тест
            if function_name in ["test_replay_buffer_performance", "test_precomputed_states", "test_torch_compile"]:
                # Эти тесты возвращают только success
                success = test_function()
                message = "Тест пройден" if success else "Тест не пройден"
            else:
                # Остальные тесты возвращают (success, message)
                success, message = test_function()
            
            test_results[test_name] = {
                'success': success,
                'message': message,
                'timestamp': time.time()
            }
            
            if success:
                print(f"✅ {test_name}: {message}")
            else:
                print(f"❌ {test_name}: {message}")
                overall_success = False
                
        except Exception as e:
            print(f"❌ {test_name}: Ошибка импорта/выполнения - {e}")
            test_results[test_name] = {
                'success': False,
                'message': f"Ошибка: {str(e)}",
                'timestamp': time.time()
            }
            overall_success = False
    
    # Финальные результаты
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("📊 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    # Подсчитываем статистику
    total_tests = len(tests)
    passed_tests = sum(1 for result in test_results.values() if result['success'])
    failed_tests = total_tests - passed_tests
    
    print(f"⏱️ Время выполнения: {duration:.2f} секунд")
    print(f"📊 Всего тестов: {total_tests}")
    print(f"✅ Пройдено: {passed_tests}")
    print(f"❌ Не пройдено: {failed_tests}")
    print(f"📈 Успешность: {(passed_tests/total_tests)*100:.1f}%")
    
    if overall_success:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("🚀 DQN агент готов к использованию")
        print("⚡ Все оптимизации работают корректно")
    else:
        print("\n⚠️ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ")
        print("🔧 Требуется дополнительная настройка")
        
        # Показываем детали неудачных тестов
        print("\n📋 ДЕТАЛИ НЕУДАЧНЫХ ТЕСТОВ:")
        for test_name, result in test_results.items():
            if not result['success']:
                print(f"   • {test_name}: {result['message']}")
    
    print("\n💡 РЕКОМЕНДАЦИИ:")
    if overall_success:
        print("   • Система готова к обучению")
        print("   • Все оптимизации активны")
        print("   • Ожидается высокая производительность")
    else:
        print("   • Проверьте зависимости и конфигурацию")
        print("   • Убедитесь, что PyTorch установлен корректно")
        print("   • Проверьте доступность GPU")
    
    return overall_success, test_results

if __name__ == "__main__":
    success, results = run_all_tests()
    
    if success:
        print("\n🎯 Система готова к работе!")
        exit(0)
    else:
        print("\n⚠️ Требуется дополнительная настройка")
        exit(1)

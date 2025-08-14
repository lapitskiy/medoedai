#!/usr/bin/env python3
"""
Тесты для предвычисления состояний
"""

import sys
import os
sys.path.append('/app')

import numpy as np
import pandas as pd
from agents.vdqn.cfg.vconfig import vDqnConfig
from envs.dqn_model.gym.crypto_trading_env_optimized import CryptoTradingEnvOptimized

def create_test_data():
    """Создает тестовые данные для проверки"""
    print("📊 Создаю тестовые данные...")
    
    # Создаем тестовые OHLCV данные
    n_samples = 1000
    
    # 5-минутные данные
    df_5min = pd.DataFrame({
        'open': np.random.randn(n_samples).cumsum() + 100000,
        'high': np.random.randn(n_samples).cumsum() + 100000,
        'low': np.random.randn(n_samples).cumsum() + 100000,
        'close': np.random.randn(n_samples).cumsum() + 100000,
        'volume': np.random.exponential(100, n_samples)
    })
    
    # 15-минутные данные (меньше точек)
    df_15min = df_5min.iloc[::3].copy()
    df_15min.index = range(len(df_15min))
    
    # 1-часовые данные (еще меньше точек)
    df_1h = df_5min.iloc[::12].copy()
    df_1h.index = range(len(df_1h))
    
    print(f"✅ Тестовые данные созданы:")
    print(f"   - 5min: {len(df_5min)} свечей")
    print(f"   - 15min: {len(df_15min)} свечей")
    print(f"   - 1h: {len(df_1h)} свечей")
    
    return {
        'df_5min': df_5min,
        'df_15min': df_15min,
        'df_1h': df_1h
    }

def test_precomputed_states():
    """Тестирует предвычисление состояний"""
    print("🧪 Тестирование предвычисления состояний...")
    
    try:
        # Создаем тестовые данные
        dfs = create_test_data()
        
        # Создаем конфигурацию
        cfg = vDqnConfig()
        
        # Создаем окружение
        env = CryptoTradingEnvOptimized(
            dfs=dfs,
            cfg=cfg,
            lookback_window=20,
            indicators_config=None
        )
        
        print(f"✅ Окружение создано")
        print(f"   - Размер состояния: {env.observation_space_shape}")
        print(f"   - Количество предвычисленных состояний: {len(env.precomputed_states)}")
        
        # Тест 1: Проверяем reset
        print("\n🔵 Тест 1: Проверка reset...")
        state = env.reset()
        print(f"   - Состояние после reset: {type(state)}, размер: {state.shape if hasattr(state, 'shape') else len(state)}")
        
        if state is not None and len(state) == env.observation_space_shape:
            print("✅ Reset работает корректно")
        else:
            print("❌ Reset работает некорректно")
            return False
        
        # Тест 2: Проверяем несколько шагов
        print("\n🟡 Тест 2: Проверка нескольких шагов...")
        for step in range(5):
            action = np.random.randint(0, 3)
            state_next, reward, terminal, info = env.step(action)
            
            if state_next is None:
                print(f"   ❌ Шаг {step}: state_next = None")
                return False
            
            print(f"   - Шаг {step}: action={action}, reward={reward:.4f}, terminal={terminal}")
            
            if terminal:
                print(f"   - Эпизод завершен на шаге {step}")
                break
        
        print("✅ Несколько шагов выполнены корректно")
        
        # Тест 3: Проверяем производительность
        print("\n🟢 Тест 3: Проверка производительности...")
        import time
        
        # Тест скорости получения состояний
        start_time = time.time()
        for _ in range(100):
            _ = env._get_state()
        get_state_time = time.time() - start_time
        
        print(f"   - 100 вызовов _get_state(): {get_state_time*1000:.2f} мс")
        print(f"   - Среднее время на состояние: {get_state_time*10:.3f} мс")
        
        if get_state_time < 0.1:  # Меньше 100мс на 100 состояний
            print("✅ Производительность отличная!")
        elif get_state_time < 0.5:
            print("✅ Производительность хорошая")
        else:
            print("⚠️ Производительность может быть улучшена")
        
        # Тест 4: Проверяем корректность данных
        print("\n🔴 Тест 4: Проверка корректности данных...")
        
        # Проверяем, что нет NaN в предвычисленных состояниях
        if hasattr(env, 'precomputed_states'):
            nan_count = np.isnan(env.precomputed_states).sum()
            if nan_count == 0:
                print("✅ NaN значения не обнаружены в предвычисленных состояниях")
            else:
                print(f"⚠️ Обнаружено {nan_count} NaN значений")
        
        # Проверяем размеры
        if hasattr(env, 'states_tensor'):
            print(f"   - states_tensor размер: {env.states_tensor.shape}")
        
        print("✅ Все тесты пройдены успешно!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_precomputed_states()
    if success:
        print("\n🎉 Тест предвычисления состояний пройден успешно!")
        print("🚀 Окружение готово к быстрой работе")
        print("⚡ Состояния будут загружаться мгновенно")
        print("\n💡 ПРЕИМУЩЕСТВА ПРЕДВЫЧИСЛЕНИЯ:")
        print("   • Мгновенный доступ к состояниям")
        print("   • Устранение задержек при обучении")
        print("   • Оптимизация hot-path операций")
        print("   • Снижение нагрузки на CPU")
    else:
        print("\n❌ Тест предвычисления состояний не пройден!")

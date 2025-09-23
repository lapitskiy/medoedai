#!/usr/bin/env python3
"""
Пример использования CNN Training Module

Этот скрипт демонстрирует:
1. Обучение CNN модели на данных криптовалют
2. Извлечение признаков из предобученной модели
3. Интеграцию с DQN средой
"""

import numpy as np
import os
import sys

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnn_training.config import CNNTrainingConfig
from cnn_training.trainer import CNNTrainer
from cnn_training.feature_extractor import create_cnn_wrapper
from cnn_training.data_loader import CryptoDataLoader


def example_training():
    """Пример обучения CNN модели"""
    print("🚀 Пример обучения CNN модели")
    
    # Создаем конфигурацию
    config = CNNTrainingConfig(
        symbols=["BTCUSDT"],
        timeframes=["5m"],
        sequence_length=20,  # Уменьшаем для примера
        batch_size=32,
        learning_rate=0.001,
        num_epochs=5,  # Уменьшаем для примера
        prediction_horizon=3,
        prediction_threshold=0.01,
        output_features=32,
        validation_split=0.2
    )
    
    print(f"📊 Конфигурация: {config.symbols}, {config.timeframes}")
    print(f"🎯 Параметры: seq_len={config.sequence_length}, epochs={config.num_epochs}")
    
    # Создаем тренер
    trainer = CNNTrainer(config)
    
    # Проверяем наличие данных
    data_loader = CryptoDataLoader(config)
    df = data_loader.load_symbol_data("BTCUSDT", "5m")
    
    if df is None:
        print("❌ Данные не найдены. Создаем синтетические данные для демонстрации...")
        
        # Создаем синтетические данные для демонстрации
        np.random.seed(42)
        n_samples = 1000
        
        # Генерируем OHLCV данные
        base_price = 50000
        prices = []
        current_price = base_price
        
        for i in range(n_samples):
            # Простое случайное блуждание с трендом
            change = np.random.normal(0, 0.02)  # 2% волатильность
            current_price *= (1 + change)
            
            # Генерируем OHLCV
            high = current_price * (1 + abs(np.random.normal(0, 0.01)))
            low = current_price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = current_price * (1 + np.random.normal(0, 0.005))
            close_price = current_price
            volume = np.random.uniform(100, 1000)
            
            prices.append([open_price, high, low, close_price, volume])
        
        # Создаем DataFrame
        import pandas as pd
        df = pd.DataFrame(prices, columns=['open', 'high', 'low', 'close', 'volume'])
        
        # Сохраняем для демонстрации
        os.makedirs(config.data_dir, exist_ok=True)
        df.to_csv(os.path.join(config.data_dir, "btcusdt_5m.csv"), index=False)
        print(f"✅ Созданы синтетические данные: {len(df)} записей")
    
    # Обучение модели
    try:
        result = trainer.train_single_model("BTCUSDT", "5m", "prediction")
        
        if result:
            print(f"✅ Обучение завершено!")
            print(f"📈 Лучшая точность: {result['best_val_accuracy']:.2f}%")
            print(f"📉 Финальная train loss: {result['train_losses'][-1]:.4f}")
            print(f"📉 Финальная val loss: {result['val_losses'][-1]:.4f}")
        else:
            print("❌ Ошибка обучения")
    
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")


def example_feature_extraction():
    """Пример извлечения признаков"""
    print("\n🔧 Пример извлечения признаков")
    
    # Создаем конфигурацию
    config = CNNTrainingConfig(
        symbols=["BTCUSDT"],
        timeframes=["5m"],
        sequence_length=20,
        output_features=32
    )
    
    # Создаем обертку для извлечения признаков
    try:
        cnn_wrapper = create_cnn_wrapper(config)
        
        # Создаем тестовые данные
        np.random.seed(42)
        test_data = {
            "5m": np.random.randn(20, 5)  # 20 свечей, 5 признаков OHLCV
        }
        
        # Извлекаем признаки
        features = cnn_wrapper.get_cnn_features("BTCUSDT", test_data)
        
        print(f"✅ CNN признаки извлечены: {features.shape}")
        print(f"📊 Статистики признаков:")
        print(f"   Среднее: {np.mean(features):.4f}")
        print(f"   Стандартное отклонение: {np.std(features):.4f}")
        print(f"   Мин: {np.min(features):.4f}")
        print(f"   Макс: {np.max(features):.4f}")
        
        # Проверяем доступность экстрактора
        is_available = cnn_wrapper.feature_extractor.is_available("BTCUSDT", "5m")
        print(f"🔍 Экстрактор доступен: {is_available}")
        
    except Exception as e:
        print(f"❌ Ошибка извлечения признаков: {e}")


def example_integration():
    """Пример интеграции с DQN"""
    print("\n🤖 Пример интеграции с DQN")
    
    # Создаем конфигурацию
    config = CNNTrainingConfig(
        symbols=["BTCUSDT"],
        timeframes=["5m"],
        sequence_length=20,
        output_features=32
    )
    
    try:
        # Создаем обертку
        cnn_wrapper = create_cnn_wrapper(config)
        
        # Симулируем состояние DQN
        base_state_size = 100  # Размер обычного состояния DQN
        base_state = np.random.randn(base_state_size)
        
        # OHLCV данные для CNN
        ohlcv_data = {
            "5m": np.random.randn(20, 5)
        }
        
        # Получаем объединенное состояние
        combined_state = cnn_wrapper.get_state_with_cnn("BTCUSDT", base_state, ohlcv_data)
        
        print(f"✅ Интеграция с DQN:")
        print(f"   Размер базового состояния: {base_state.shape}")
        print(f"   Размер CNN признаков: {config.output_features}")
        print(f"   Размер объединенного состояния: {combined_state.shape}")
        
        # Проверяем корректность размеров
        expected_size = base_state_size + config.output_features
        assert len(combined_state) == expected_size, f"Неверный размер состояния: {len(combined_state)} != {expected_size}"
        
        print("✅ Размеры состояния корректны!")
        
    except Exception as e:
        print(f"❌ Ошибка интеграции: {e}")


def main():
    """Основная функция"""
    print("🎯 CNN Training Module - Примеры использования")
    print("=" * 50)
    
    # Пример обучения
    example_training()
    
    # Пример извлечения признаков
    example_feature_extraction()
    
    # Пример интеграции
    example_integration()
    
    print("\n🏁 Все примеры завершены!")
    print("\n💡 Для реального использования:")
    print("1. Подготовьте данные криптовалют в формате CSV")
    print("2. Настройте конфигурацию под ваши данные")
    print("3. Обучите CNN модель на исторических данных")
    print("4. Интегрируйте извлеченные признаки в DQN среду")


if __name__ == "__main__":
    main()

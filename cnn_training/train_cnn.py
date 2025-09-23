#!/usr/bin/env python3
"""
Скрипт для обучения CNN моделей на данных криптовалют

Использование:
    python cnn_training/train_cnn.py --symbol BTCUSDT --timeframe 5m --model_type prediction
    python cnn_training/train_cnn.py --symbols BTCUSDT,ETHUSDT --timeframes 5m,15m,1h --model_type multiframe
"""

import argparse
import os
import sys
from typing import List

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnn_training.config import CNNTrainingConfig
from cnn_training.trainer import CNNTrainer


def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description="Обучение CNN моделей для анализа криптовалют")
    
    # Основные параметры
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,TONUSDT",
                       help="Символы криптовалют (через запятую)")
    parser.add_argument("--timeframes", type=str, default="5m,15m,1h",
                       help="Временные фреймы (через запятую)")
    parser.add_argument("--model_type", type=str, default="prediction",
                       choices=["single", "multiframe", "prediction"],
                       help="Тип модели для обучения")
    
    # Параметры обучения
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Размер батча")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Скорость обучения")
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Количество эпох")
    parser.add_argument("--sequence_length", type=int, default=50,
                       help="Длина последовательности для CNN")
    parser.add_argument("--label_scheme", type=str, default="binary", choices=["binary", "ternary"],
                       help="Схема меток: binary (0/1) или ternary (0/1/2)")
    parser.add_argument("--class_balance", type=str, default="auto", choices=["auto", "none"],
                       help="Балансировка классов: auto включает веса/самплер")
    parser.add_argument("--label_smoothing", type=float, default=0.05,
                       help="Label smoothing для CrossEntropyLoss")
    
    # Параметры предсказания
    parser.add_argument("--prediction_horizon", type=int, default=1,
                       help="Горизонт предсказания (количество свечей)")
    parser.add_argument("--prediction_threshold", type=float, default=0.01,
                       help="Порог для определения значимого движения цены")
    parser.add_argument("--label_noise", type=float, default=0.0,
                       help="Доля случайного шума в метках (0 — отключено)")
    
    # Пути
    parser.add_argument("--data_dir", type=str, default="temp/binance_data",
                       help="Директория с данными")
    parser.add_argument("--model_save_dir", type=str, default="cnn_training/models",
                       help="Директория для сохранения моделей")
    
    # Устройство
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Устройство для обучения")
    
    # Логирование
    parser.add_argument("--use_wandb", action="store_true",
                       help="Использовать Weights & Biases для логирования")
    parser.add_argument("--wandb_project", type=str, default="crypto-cnn-training",
                       help="Название проекта в Wandb")
    
    return parser.parse_args()


def main():
    """Основная функция"""
    args = parse_arguments()
    
    # Парсим списки символов и временных фреймов
    symbols = [s.strip() for s in args.symbols.split(",")]
    timeframes = [t.strip() for t in args.timeframes.split(",")]
    
    print(f"🚀 Начинаем обучение CNN модели")
    print(f"📊 Символы: {symbols}")
    print(f"⏰ Временные фреймы: {timeframes}")
    print(f"🤖 Тип модели: {args.model_type}")
    
    # Создаем конфигурацию
    config = CNNTrainingConfig(
        symbols=symbols,
        timeframes=timeframes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        sequence_length=args.sequence_length,
        label_scheme=args.label_scheme,
        class_balance=args.class_balance,
        label_smoothing=args.label_smoothing,
        prediction_horizon=args.prediction_horizon,
        prediction_threshold=args.prediction_threshold,
        label_noise=args.label_noise,
        data_dir=args.data_dir,
        model_save_dir=args.model_save_dir,
        device=args.device,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project
    )
    
    # Создаем тренер
    trainer = CNNTrainer(config)
    
    try:
        if args.model_type == "single":
            # Обучение отдельных моделей для каждого символа и фрейма
            for symbol in symbols:
                for timeframe in timeframes:
                    print(f"\n🎯 Обучение модели для {symbol} {timeframe}")
                    result = trainer.train_single_model(symbol, timeframe, "prediction")
                    
                    if result:
                        print(f"✅ Обучение {symbol} {timeframe} завершено")
                        print(f"📈 Лучшая точность: {result['best_val_accuracy']:.2f}%")
                    else:
                        print(f"❌ Ошибка обучения {symbol} {timeframe}")
        
        elif args.model_type == "multiframe":
            # Обучение мультифреймовой модели
            print(f"\n🎯 Обучение мультифреймовой модели")
            result = trainer.train_multiframe_model(symbols)
            
            if result:
                print(f"✅ Обучение мультифреймовой модели завершено")
                print(f"📈 Лучшая точность: {result['best_val_accuracy']:.2f}%")
            else:
                print(f"❌ Ошибка обучения мультифреймовой модели")
        
        elif args.model_type == "prediction":
            # Обучение моделей предсказания для каждого символа
            for symbol in symbols:
                print(f"\n🎯 Обучение модели предсказания для {symbol}")
                result = trainer.train_single_model(symbol, timeframes[0], "prediction")
                
                if result:
                    print(f"✅ Обучение {symbol} завершено")
                    print(f"📈 Лучшая точность: {result['best_val_accuracy']:.2f}%")
                else:
                    print(f"❌ Ошибка обучения {symbol}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Обучение прервано пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка обучения: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Закрываем Wandb если используется
        if trainer.wandb:
            trainer.wandb.finish()
    
    print("\n🏁 Обучение завершено!")


if __name__ == "__main__":
    main()

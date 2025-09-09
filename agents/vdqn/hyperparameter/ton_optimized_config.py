# Оптимизированная конфигурация для TON
# Анализ показал низкий winrate (45.7%) и много убыточных сделок (53.9%)

TON_OPTIMIZED_CONFIG = {
    # Более консервативные параметры риска
    'risk_management': {
        'STOP_LOSS_PCT': -0.025,  # Ужесточаем стоп-лосс с -4% до -2.5%
        'TAKE_PROFIT_PCT': 0.04,  # Снижаем тейк-профит с 6% до 4%
        'min_hold_steps': 20,     # Уменьшаем минимальное время удержания с 30 до 20 (1.7 часа)
        'volume_threshold': 0.005, # Повышаем порог объема с 0.003 до 0.005
    },
    
    # Более строгие параметры позиционирования
    'position_sizing': {
        'base_position_fraction': 0.2,  # Снижаем размер позиции с 0.3 до 0.2
        'position_fraction': 0.2,
        'position_confidence_threshold': 0.8,  # Повышаем порог уверенности с 0.7 до 0.8
    },
    
    # Оптимизированные индикаторы
    'indicators_config': {
        'rsi': {'length': 21},  # Увеличиваем период RSI с 14 до 21 для менее шумных сигналов
        'ema': {'lengths': [50, 100, 200]},  # Добавляем EMA 50 для лучшего тренда
        'ema_cross': {
            'pairs': [(50, 100), (100, 200)],  # Добавляем пересечение 50/100
            'include_cross_signal': True
        },
        'sma': {'length': 21},  # Увеличиваем период SMA с 14 до 21
        # Добавляем новые индикаторы
        'bb': {'length': 20, 'std': 2},  # Bollinger Bands для волатильности
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},  # MACD для тренда
    },
    
    # Более консервативные параметры обучения
    'training_params': {
        'eps_start': 0.8,  # Снижаем начальную эксплорацию с 1.0 до 0.8
        'eps_final': 0.02,  # Снижаем финальную эксплорацию с 0.05 до 0.02
        'eps_decay_steps': 1500000,  # Увеличиваем время затухания эксплорации
        'lr': 0.0005,  # Снижаем learning rate с 0.001 до 0.0005
        'gamma': 0.995,  # Увеличиваем gamma с 0.99 до 0.995 для долгосрочного планирования
        'batch_size': 512,  # Увеличиваем batch size с 256 до 512
        'memory_size': 500000,  # Увеличиваем размер памяти с 200000 до 500000
        'hidden_sizes': (1024, 512, 256),  # Увеличиваем размер сети
        'dropout_rate': 0.3,  # Увеличиваем dropout с 0.2 до 0.3 для регуляризации
        'train_repeats': 4,
        'soft_update_every': 40,
        'target_update_freq': 800,
        'early_stopping_patience': 2000,  # Уменьшаем терпение с 3000 до 2000
        'min_episodes_before_stopping': 1500,  # Увеличиваем минимум эпизодов с 1000 до 1500
    },
    
    # Параметры среды
    'gym_config': {
        'lookback_window': 30,  # Увеличиваем окно истории с 20 до 30
        'step_minutes': 5,
        'funding_features': {
            'included': True,
            'weight': 0.1  # Добавляем вес для funding rate
        }
    }
}

# Дополнительные рекомендации для TON
TON_RECOMMENDATIONS = {
    'market_analysis': {
        'ton_volatility': 'TON имеет высокую волатильность, поэтому нужны более консервативные параметры',
        'funding_rate_impact': 'TON часто имеет высокие funding rates, учитываем это в модели',
        'volume_patterns': 'TON имеет специфические паттерны объема, увеличиваем порог'
    },
    
    'optimization_strategy': {
        'step_1': 'Сначала обучите с новыми параметрами риска',
        'step_2': 'Проанализируйте результаты и при необходимости скорректируйте',
        'step_3': 'Добавьте дополнительные индикаторы если winrate < 55%',
        'step_4': 'Рассмотрите использование ensemble из нескольких моделей'
    },
    
    'expected_improvements': {
        'winrate_target': '55-65% (вместо текущих 45.7%)',
        'pl_ratio_target': '1.3-1.5 (вместо текущих 1.095)',
        'bad_trades_target': '<40% (вместо текущих 53.9%)',
        'avg_roi_target': '>0.002 (вместо текущих -0.0007)'
    }
}

if __name__ == "__main__":
    print("🔧 Оптимизированная конфигурация для TON")
    print("=" * 50)
    print(f"🎯 Целевой winrate: {TON_RECOMMENDATIONS['expected_improvements']['winrate_target']}")
    print(f"💰 Целевой P&L ratio: {TON_RECOMMENDATIONS['expected_improvements']['pl_ratio_target']}")
    print(f"📉 Целевые плохие сделки: {TON_RECOMMENDATIONS['expected_improvements']['bad_trades_target']}")
    print(f"📈 Целевой ROI: {TON_RECOMMENDATIONS['expected_improvements']['avg_roi_target']}")
    print("\n🔧 Основные изменения:")
    print("• STOP_LOSS: -4% → -2.5%")
    print("• TAKE_PROFIT: +6% → +4%")
    print("• min_hold_steps: 30 → 20")
    print("• position_fraction: 0.3 → 0.2")
    print("• confidence_threshold: 0.7 → 0.8")
    print("• learning_rate: 0.001 → 0.0005")
    print("• Добавлены новые индикаторы: BB, MACD")
    print("• Увеличен lookback_window: 20 → 30")

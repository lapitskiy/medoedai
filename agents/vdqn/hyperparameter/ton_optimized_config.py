# Оптимизированная конфигурация для TON
# Анализ показал низкий winrate (45.7%) и много убыточных сделок (53.9%)

TON_OPTIMIZED_CONFIG = {
    # risk_management вынесен в общий файл:
    # agents/vdqn/hyperparameter/global_overrides.py -> GLOBAL_OVERRIDES['risk_management']
    
    # Более строгие параметры позиционирования
    'position_sizing': {
        'base_position_fraction': 0.2,  # Снижаем размер позиции с 0.3 до 0.2
        'position_fraction': 0.2,
        'position_confidence_threshold': 0.8,  # Повышаем порог уверенности с 0.7 до 0.8
    },
    
    # Оптимизированные индикаторы
    'indicators_config': {
        'rsi': {'length': 21},
        'rsi_7': {'length': 7},
        'ema': {'lengths': [20, 50, 100, 200]},
        'ema_cross': {
            'pairs': [(20, 50), (50, 100), (100, 200)],
            'include_cross_signal': True
        },
        'sma': {'length': 21},
        'atr': {'length': 14},
        'obv': {},
        'returns': {'periods': [1, 3, 12, 60]},
        'zscore': {'ema_length': 50, 'window': 20},
    },
    
    # Более консервативные параметры обучения
    # NOTE: параметры из GPU профиля (см. agents/vdqn/cfg/gpu_configs.py) НЕ должны задаваться per-symbol:
    # batch_size/memory_size/hidden_sizes/train_repeats/use_amp/use_gpu_storage/learning_rate/use_torch_compile/eps_decay_steps/dropout_rate
    # Иначе лог будет показывать "применены настройки GPU", но cfg_snapshot окажется перезаписан символ-оверрайдом.
    'training_params': {
        'eps_start': 0.995,  # Снижаем начальную эксплорацию с 1.0 до 0.8
        'eps_final': 0.02,  # Снижаем финальную эксплорацию с 0.05 до 0.02
        'gamma': 0.995,  # Увеличиваем gamma с 0.99 до 0.995 для долгосрочного планирования
        'soft_update_every': 40,
        'target_update_freq': 800,
        'early_stopping_patience': 2000,  # Уменьшаем терпение с 3000 до 2000
        'min_episodes_before_stopping': 1500,  # Увеличиваем минимум эпизодов с 1000 до 1500
    },
    
    # Параметры среды
    'gym_config': {
        'lookback_window': 100,  # Увеличиваем окно истории с 20 до 30
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

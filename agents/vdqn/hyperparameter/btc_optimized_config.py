"""
Оптимизированная конфигурация для BTC (спот/фьючерсы, 5m таймфрейм)

Цели:
- Стабильный winrate 55–65%
- Контролируемый риск и адекватная эксплорация
"""

BTC_OPTIMIZED_CONFIG = {
    # Риск-менеджмент и фильтры
    'risk_management': {
        'STOP_LOSS_PCT': -0.03,     # более широкий стоп из-за волатильности
        'TAKE_PROFIT_PCT': 0.05,    # умеренный тейк для частой фиксации
        'min_hold_steps': 12,       # ~60 минут удержания
        'volume_threshold': 0.001,  # фильтр по объёму
    },

    # Размер позиции и уверенность
    'position_sizing': {
        'base_position_fraction': 0.25,
        'position_fraction': 0.25,
        'position_confidence_threshold': 0.65,
    },

    # Индикаторы (базовый, расширяемый набор)
    'indicators_config': {
        'rsi': {'length': 21},
        'ema': {'lengths': [50, 100, 200]},
        'ema_cross': {'pairs': [(50, 100), (100, 200)], 'include_cross_signal': True},
        'sma': {'length': 21},
        'bb': {'length': 20, 'std': 2},
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
    },

    # Параметры обучения
    # NOTE: GPU-owned параметры (batch_size/memory_size/train_repeats/use_amp/use_gpu_storage/use_torch_compile)
    # НЕ должны задаваться per-symbol. Это делает прогоны непрозрачными и ломает hardware-профиль.
    'training_params': {
        'eps_start': 0.6,
        'eps_final': 0.05,
        'eps_decay_steps': 1_200_000,

        'lr': 1e-4,
        'gamma': 0.99,
        'hidden_sizes': (1024, 512, 256),
        'dropout_rate': 0.2,
        'soft_update_every': 20,
        'target_update_freq': 1000,

        'early_stopping_patience': 2500,
        'min_episodes_before_stopping': 1200,
    },

    # Конфигурация среды
    'gym_config': {
        'lookback_window': 100,
        'step_minutes': 5,
        'funding_features': {'included': True, 'weight': 0.05},
    },
}

if __name__ == "__main__":
    import pprint
    print("🔧 Оптимизированная конфигурация для BTC")
    pprint.pprint(BTC_OPTIMIZED_CONFIG)



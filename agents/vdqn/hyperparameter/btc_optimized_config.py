"""
Оптимизированная конфигурация для BTC (спот/фьючерсы, 5m таймфрейм)

Цели:
- Стабильный winrate 55–65%
- Контролируемый риск и адекватная эксплорация
"""

BTC_OPTIMIZED_CONFIG = {
    # risk_management вынесен в общий файл:
    # agents/vdqn/hyperparameter/global_overrides.py -> GLOBAL_OVERRIDES['risk_management']

    # Размер позиции и уверенность
    'position_sizing': {
        'base_position_fraction': 0.25,
        'position_fraction': 0.25,
        'position_confidence_threshold': 0.65,
    },

    # Индикаторы (базовый, расширяемый набор)
    'indicators_config': {
        'rsi': {'length': 21},
        'rsi_7': {'length': 7},
        'ema': {'lengths': [20, 50, 100, 200]},
        'ema_cross': {'pairs': [(20, 50), (50, 100), (100, 200)], 'include_cross_signal': True},
        'sma': {'length': 21},
        'atr': {'length': 14},
        'obv': {},
        'returns': {'periods': [1, 3, 12, 60]},
        'zscore': {'ema_length': 50, 'window': 20},
    },

    # Параметры обучения
    # NOTE: параметры из GPU профиля (см. agents/vdqn/cfg/gpu_configs.py) НЕ должны задаваться per-symbol:
    # batch_size/memory_size/hidden_sizes/train_repeats/use_amp/use_gpu_storage/learning_rate/use_torch_compile/eps_decay_steps/dropout_rate
    'training_params': {
        'eps_start': 0.6,
        'eps_final': 0.05,
        'gamma': 0.99,
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



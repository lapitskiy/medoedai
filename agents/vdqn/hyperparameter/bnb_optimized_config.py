"""
Оптимизированная конфигурация для BNB (по результатам прогона cf09)

Цели:
- Увеличить долю эпизодов со сделками (episodes_with_trade_ratio ~ 7.25% → 15–25%)
- Повысить winrate и итоговый PnL через смягчение фильтров и выходов
"""

BNB_OPTIMIZED_CONFIG = {
    # risk_management вынесен в общий файл:
    # agents/vdqn/hyperparameter/global_overrides.py -> GLOBAL_OVERRIDES['risk_management']

    # Размер позиции и уверенность
    'position_sizing': {
        'base_position_fraction': 0.30,
        'position_fraction': 0.30,
        'position_confidence_threshold': 0.58  # было 0.7 → чаще брать сигналы
    },

    # Индикаторы (минимальные правки, можно расширять позже)
    'indicators_config': {
        'rsi': {'length': 14},
        'ema': {'lengths': [100, 200]},
        'ema_cross': {'pairs': [(100, 200)], 'include_cross_signal': True},
        'sma': {'length': 14},
    },

    # Параметры обучения
    # NOTE: параметры из GPU профиля (см. agents/vdqn/cfg/gpu_configs.py) НЕ должны задаваться per-symbol:
    # batch_size/memory_size/hidden_sizes/train_repeats/use_amp/use_gpu_storage/learning_rate/use_torch_compile/eps_decay_steps/dropout_rate
    'training_params': {
        'use_noisy_networks': True,
        'eps_start': 0.10,            # меньше случайности при NoisyNet
        'eps_final': 0.02,
        'gamma': 0.99,

        'soft_tau': 0.005,
        'soft_update_every': 1,
        'target_update_freq': 5000,   # оставим, но в будущем можно перейти на чистый soft

        'layer_norm': True,

        'use_distributional_rl': True,
        'n_atoms': 51,
        'v_min': -0.2,
        'v_max': 0.2,
    },

    # Конфигурация среды
    'gym_config': {
        'lookback_window': 100,
        'step_minutes': 5,
        'funding_features': {'included': True},
    },
}

if __name__ == "__main__":
    import pprint
    print("🔧 Оптимизированная конфигурация для BNB")
    pprint.pprint(BNB_OPTIMIZED_CONFIG)



"""
Оптимизированная конфигурация для BNB (по результатам прогона cf09)

Цели:
- Увеличить долю эпизодов со сделками (episodes_with_trade_ratio ~ 7.25% → 15–25%)
- Повысить winrate и итоговый PnL через смягчение фильтров и выходов
"""

BNB_OPTIMIZED_CONFIG = {
    # Риск-менеджмент и фильтры
    'risk_management': {
        'STOP_LOSS_PCT': -0.035,   # было -0.045 → гибче удержание, но контролируем риск
        'TAKE_PROFIT_PCT': 0.06,   # было 0.07 → быстрее фиксация профита
        'min_hold_steps': 8,       # было 18 (≈90 мин) → 8 (≈40 мин)
        'volume_threshold': 0.0011 # было ~0.00162 → пропускаем больше входов
    },

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
    'training_params': {
        'use_noisy_networks': True,
        'eps_start': 0.10,            # меньше случайности при NoisyNet
        'eps_final': 0.02,
        'eps_decay_steps': 2_000_000,

        'memory_size': 1_000_000,     # больше разнообразие опыта
        'batch_size': 256,
        'train_repeats': 2,           # больше шагов обучения на шаг среды

        'lr': 1e-4,                   # AdamW предпочтительнее, но поле lr здесь
        'gamma': 0.99,

        'soft_tau': 0.005,
        'soft_update_every': 1,
        'target_update_freq': 5000,   # оставим, но в будущем можно перейти на чистый soft

        'dropout_rate': 0.0,          # отключаем dropout для DQN
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



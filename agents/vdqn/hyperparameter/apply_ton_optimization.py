#!/usr/bin/env python3
"""
Скрипт для применения оптимизированных настроек для TON
"""

import os
import sys
import json

# Добавляем пути для импортов
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..', '..')
sys.path.append(project_root)

from agents.vdqn.hyperparameter.ton_optimized_config import TON_OPTIMIZED_CONFIG, TON_RECOMMENDATIONS

def update_gym_config():
    """Обновляет конфигурацию gym для TON"""
    gym_config_path = os.path.join(project_root, "envs", "dqn_model", "gym", "gconfig.py")
    
    if not os.path.exists(gym_config_path):
        print(f"❌ Файл {gym_config_path} не найден")
        return False
    
    print("🔧 Обновление конфигурации gym...")
    
    # Читаем текущий файл
    with open(gym_config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Создаем резервную копию
    backup_path = gym_config_path + ".backup"
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"📋 Создана резервная копия: {backup_path}")
    
    # Добавляем оптимизированную конфигурацию TON
    ton_config = f"""
# Оптимизированная конфигурация для TON (добавлено автоматически)
TON_OPTIMIZED_CONFIG = {json.dumps(TON_OPTIMIZED_CONFIG, indent=2, ensure_ascii=False)}

def get_ton_optimized_config():
    '''Возвращает оптимизированную конфигурацию для TON'''
    return TON_OPTIMIZED_CONFIG
"""
    
    # Добавляем в конец файла
    with open(gym_config_path, 'a', encoding='utf-8') as f:
        f.write(ton_config)
    
    print("✅ Конфигурация gym обновлена")
    return True

def create_ton_training_script():
    """Создает скрипт для обучения TON с оптимизированными параметрами"""
    script_content = '''#!/usr/bin/env python3
"""
Скрипт для обучения TON с оптимизированными параметрами
"""

import sys
import os
sys.path.append('/workspace')

from agents.vdqn.v_train_model_optimized import main as train_main
from envs.dqn_model.gym.gconfig import get_ton_optimized_config

def train_ton_optimized():
    """Обучает TON модель с оптимизированными параметрами"""
    
    # Получаем оптимизированную конфигурацию
    config = get_ton_optimized_config()
    
    print("🚀 Запуск обучения TON с оптимизированными параметрами")
    print("=" * 60)
    print(f"📊 Символ: TONUSDT")
    print(f"🎯 Целевой winrate: 55-65%")
    print(f"💰 Целевой P&L ratio: 1.3-1.5")
    print(f"📉 Целевые плохие сделки: <40%")
    print("=" * 60)
    
    # Параметры обучения
    args = {
        'symbol': 'TONUSDT',
        'episodes': 15000,  # Увеличиваем количество эпизодов
        'timeframe': '5m',
        'config': config,
        'save_frequency': 100,
        'early_stopping': True,
        'verbose': True
    }
    
    try:
        # Запускаем обучение
        result = train_main(**args)
        
        if result:
            print("✅ Обучение TON завершено успешно!")
            print("📊 Проанализируйте результаты и при необходимости скорректируйте параметры")
        else:
            print("❌ Обучение TON завершилось с ошибкой")
            
    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        return False
    
    return True

if __name__ == "__main__":
    train_ton_optimized()
'''
    
    script_path = os.path.join(project_root, "train_ton_optimized.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # Делаем скрипт исполняемым
    os.chmod(script_path, 0o755)
    
    print(f"✅ Создан скрипт обучения: {script_path}")
    return script_path

def print_optimization_summary():
    """Выводит сводку оптимизации"""
    print("\n" + "="*60)
    print("🔧 ОПТИМИЗАЦИЯ TON - СВОДКА ИЗМЕНЕНИЙ")
    print("="*60)
    
    print("\n📊 ТЕКУЩИЕ ПРОБЛЕМЫ:")
    print("• Winrate: 45.7% (низкий)")
    print("• Убыточные сделки: 53.9% (много)")
    print("• Средний ROI: -0.0007 (отрицательный)")
    print("• P&L ratio: 1.095 (близко к 1)")
    
    print("\n🎯 ЦЕЛЕВЫЕ УЛУЧШЕНИЯ:")
    for key, value in TON_RECOMMENDATIONS['expected_improvements'].items():
        print(f"• {key}: {value}")
    
    print("\n🔧 ОСНОВНЫЕ ИЗМЕНЕНИЯ:")
    risk = TON_OPTIMIZED_CONFIG['risk_management']
    pos = TON_OPTIMIZED_CONFIG['position_sizing']
    train = TON_OPTIMIZED_CONFIG['training_params']
    
    print(f"• STOP_LOSS: -4% → {risk['STOP_LOSS_PCT']*100:.1f}%")
    print(f"• TAKE_PROFIT: +6% → {risk['TAKE_PROFIT_PCT']*100:.1f}%")
    print(f"• min_hold_steps: 30 → {risk['min_hold_steps']}")
    print(f"• position_fraction: 0.3 → {pos['base_position_fraction']}")
    print(f"• confidence_threshold: 0.7 → {pos['position_confidence_threshold']}")
    print(f"• learning_rate: 0.001 → {train['lr']}")
    print(f"• lookback_window: 20 → {TON_OPTIMIZED_CONFIG['gym_config']['lookback_window']}")
    
    print("\n📈 НОВЫЕ ИНДИКАТОРЫ:")
    indicators = TON_OPTIMIZED_CONFIG['indicators_config']
    print(f"• RSI: {indicators['rsi']['length']} (было 14)")
    print(f"• EMA: {indicators['ema']['lengths']} (добавлена EMA 50)")
    print(f"• Bollinger Bands: {indicators['bb']['length']}, std={indicators['bb']['std']}")
    print(f"• MACD: fast={indicators['macd']['fast']}, slow={indicators['macd']['slow']}")
    
    print("\n🚀 СЛЕДУЮЩИЕ ШАГИ:")
    for step, desc in TON_RECOMMENDATIONS['optimization_strategy'].items():
        print(f"• {step}: {desc}")

def main():
    """Основная функция"""
    print("🔧 Применение оптимизированных настроек для TON")
    print("=" * 50)
    
    # Обновляем конфигурацию gym
    if not update_gym_config():
        return False
    
    # Создаем скрипт обучения
    script_path = create_ton_training_script()
    
    # Выводим сводку
    print_optimization_summary()
    
    print(f"\n✅ Оптимизация завершена!")
    print(f"🚀 Для запуска обучения выполните:")
    print(f"   python {script_path}")
    print(f"\n📊 После обучения проанализируйте результаты и при необходимости скорректируйте параметры")
    
    return True

if __name__ == "__main__":
    main()

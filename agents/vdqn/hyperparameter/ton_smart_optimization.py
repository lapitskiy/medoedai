#!/usr/bin/env python3
"""
Умная оптимизация для TON - комбинирует быстрые улучшения с автоматической оптимизацией
"""

import sys
import os
import json

# Добавляем пути для импортов
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..', '..')
sys.path.append(project_root)

from agents.vdqn.hyperparameter.ton_optimized_config import TON_OPTIMIZED_CONFIG
from agents.vdqn.hyperparameter_optimizer import HyperparameterOptimizer
from agents.vdqn.cfg.vconfig import vDqnConfig

class TONSmartOptimizer:
    """Умный оптимизатор для TON, сочетающий быстрые улучшения с автоматической оптимизацией"""
    
    def __init__(self):
        self.ton_config = TON_OPTIMIZED_CONFIG
        self.base_config = vDqnConfig()
        
    def apply_quick_fixes(self):
        """Применяет быстрые улучшения на основе анализа TON"""
        print("🚀 ПРИМЕНЕНИЕ БЫСТРЫХ УЛУЧШЕНИЙ ДЛЯ TON")
        print("=" * 50)
        
        # Применяем оптимизированные настройки
        quick_fixes = {
            'risk_management': {
                'STOP_LOSS_PCT': -0.025,  # -2.5% вместо -4%
                'TAKE_PROFIT_PCT': 0.04,  # +4% вместо +6%
                'min_hold_steps': 20,     # 1.7 часа вместо 2.5
                'volume_threshold': 0.005, # Повышаем порог объема
            },
            'position_sizing': {
                'base_position_fraction': 0.2,  # Меньший размер позиции
                'position_confidence_threshold': 0.8,  # Выше порог уверенности
            },
            'training_params': {
                'lr': 0.0005,  # Меньший learning rate
                'gamma': 0.995,  # Больше долгосрочного планирования
                'eps_start': 0.8,  # Меньше эксплорации
                'eps_final': 0.02,
            }
        }
        
        print("✅ Быстрые улучшения применены:")
        for category, params in quick_fixes.items():
            print(f"  📊 {category}:")
            for param, value in params.items():
                print(f"    • {param}: {value}")
        
        return quick_fixes
    
    def create_ton_specific_grid(self):
        """Создает специфичную сетку параметров для TON"""
        
        # Базовые улучшения уже применены, теперь тонкая настройка
        ton_specific_grid = [
            # Learning rate (около оптимального 0.0005)
            {'lr': 0.0003},
            {'lr': 0.0004},
            {'lr': 0.0005},
            {'lr': 0.0006},
            {'lr': 0.0007},
            
            # Gamma (около оптимального 0.995)
            {'gamma': 0.992},
            {'gamma': 0.994},
            {'gamma': 0.995},
            {'gamma': 0.996},
            {'gamma': 0.998},
            
            # Batch size (для стабильности)
            {'batch_size': 256},
            {'batch_size': 512},
            {'batch_size': 1024},
            
            # Memory size (больше опыта)
            {'memory_size': 300000},
            {'memory_size': 500000},
            {'memory_size': 750000},
            
            # Epsilon decay (более консервативная эксплорация)
            {'eps_decay_steps': 1200000},
            {'eps_decay_steps': 1500000},
            {'eps_decay_steps': 1800000},
            
            # Target update frequency
            {'target_update_freq': 3000},
            {'target_update_freq': 4000},
            {'target_update_freq': 5000},
        ]
        
        return ton_specific_grid
    
    def run_smart_optimization(self, max_iterations=15, test_episodes=200):
        """Запускает умную оптимизацию для TON"""
        
        print("🧠 УМНАЯ ОПТИМИЗАЦИЯ TON")
        print("=" * 50)
        
        # Шаг 1: Применяем быстрые улучшения
        quick_fixes = self.apply_quick_fixes()
        
        # Шаг 2: Создаем базовую конфигурацию с улучшениями
        base_config = vDqnConfig()
        
        # Применяем быстрые улучшения к базовой конфигурации
        for category, params in quick_fixes.items():
            for param, value in params.items():
                if hasattr(base_config, param):
                    setattr(base_config, param, value)
        
        print(f"\n🔧 Базовая конфигурация с улучшениями создана")
        
        # Шаг 3: Создаем специфичную сетку для TON
        ton_grid = self.create_ton_specific_grid()
        print(f"📊 Создана специфичная сетка: {len(ton_grid)} комбинаций")
        
        # Шаг 4: Запускаем автоматическую оптимизацию
        print(f"\n🚀 Запуск автоматической оптимизации...")
        print(f"   • Максимум итераций: {max_iterations}")
        print(f"   • Эпизодов для тестирования: {test_episodes}")
        
        # Создаем оптимизатор с TON-специфичной сеткой
        optimizer = TONHyperparameterOptimizer(
            dfs={},  # Нужно передать данные
            base_config=base_config,
            custom_grid=ton_grid
        )
        
        # Запускаем оптимизацию
        results = optimizer.optimize(
            max_iterations=max_iterations,
            test_episodes=test_episodes
        )
        
        return results

class TONHyperparameterOptimizer(HyperparameterOptimizer):
    """Специализированный оптимизатор для TON"""
    
    def __init__(self, dfs, base_config=None, custom_grid=None):
        super().__init__(dfs, base_config)
        self.custom_grid = custom_grid or []
    
    def define_parameter_grid(self):
        """Использует TON-специфичную сетку"""
        if self.custom_grid:
            return self.custom_grid
        return super().define_parameter_grid()
    
    def calculate_score(self, results):
        """Специализированная функция оценки для TON"""
        
        if not results or 'final_stats' not in results:
            return -np.inf
        
        stats = results['final_stats']
        
        # Метрики для TON
        winrate = stats.get('winrate', 0)
        pl_ratio = stats.get('pl_ratio', 0)
        trades_count = stats.get('trades_count', 0)
        bad_trades_count = stats.get('bad_trades_count', 0)
        avg_roi = stats.get('avg_roi', 0)
        
        # Специальная оценка для TON (учитываем проблемы)
        score = 0
        
        # Winrate (вес: 35%) - критично для TON
        if winrate > 0.5:  # Хороший winrate
            score += (winrate - 0.5) * 2 * 0.35  # Бонус за winrate > 50%
        else:
            score += winrate * 0.35
        
        # P/L ratio (вес: 25%) - важно для прибыльности
        if pl_ratio > 1.2:  # Хороший P/L ratio
            score += (pl_ratio - 1.0) * 0.5 * 0.25
        else:
            score += max(0, pl_ratio - 1.0) * 0.25
        
        # Количество сделок (вес: 15%) - активность
        if trades_count > 1000:
            score += min(trades_count / 2000.0, 1.0) * 0.15
        
        # Штраф за плохие сделки (вес: 15%)
        bad_trades_ratio = bad_trades_count / max(trades_count, 1)
        if bad_trades_ratio < 0.4:  # Меньше 40% плохих сделок
            score += (0.4 - bad_trades_ratio) * 2 * 0.15
        else:
            score -= (bad_trades_ratio - 0.4) * 0.15
        
        # ROI (вес: 10%) - общая прибыльность
        if avg_roi > 0:
            score += min(avg_roi * 100, 1.0) * 0.10
        else:
            score += max(avg_roi * 50, -0.1) * 0.10
        
        return score

def main():
    """Основная функция"""
    print("🧠 УМНАЯ ОПТИМИЗАЦИЯ TON")
    print("=" * 60)
    print("🎯 Цель: Улучшить winrate с 45.7% до 55-65%")
    print("💰 Цель: Улучшить P&L ratio с 1.095 до 1.3-1.5")
    print("📉 Цель: Снизить плохие сделки с 53.9% до <40%")
    print("=" * 60)
    
    optimizer = TONSmartOptimizer()
    
    # Запускаем умную оптимизацию
    results = optimizer.run_smart_optimization(
        max_iterations=12,  # Ограничиваем для TON
        test_episodes=150   # Больше эпизодов для стабильности
    )
    
    if results:
        print(f"\n✅ ОПТИМИЗАЦИЯ TON ЗАВЕРШЕНА!")
        print(f"🏆 Лучший score: {results['best_score']:.4f}")
        print(f"🔧 Лучшая конфигурация: {results['best_config']}")
        
        # Сохраняем результаты
        with open('ton_optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Результаты сохранены в ton_optimization_results.json")
    else:
        print(f"\n❌ ОПТИМИЗАЦИЯ НЕ ДАЛА РЕЗУЛЬТАТОВ")

if __name__ == "__main__":
    main()

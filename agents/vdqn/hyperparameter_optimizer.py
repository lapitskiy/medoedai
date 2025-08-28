#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оптимизатор гиперпараметров для DQN модели
Помогает найти оптимальные настройки для улучшения соотношения P/L
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import sys
import os
import itertools
from datetime import datetime
import json

# Добавляем путь к модулям
sys.path.append('.')
from agents.vdqn.cfg.vconfig import vDqnConfig
from agents.vdqn.v_train_model_optimized import train_model_optimized

class HyperparameterOptimizer:
    """Класс для оптимизации гиперпараметров DQN модели"""
    
    def __init__(self, dfs: Dict, base_config: vDqnConfig = None):
        self.dfs = dfs
        self.base_config = base_config or vDqnConfig()
        self.results = []
        self.best_config = None
        self.best_score = -np.inf
        
    def define_parameter_grid(self) -> List[Dict]:
        """Определяет сетку параметров для перебора"""
        
        # Основные параметры для оптимизации
        parameter_grid = [
            # Learning rate
            {'learning_rate': 0.0001},
            {'learning_rate': 0.0005},
            {'learning_rate': 0.001},
            {'learning_rate': 0.002},
            
            # Epsilon decay
            {'eps_decay_rate': 0.995},
            {'eps_decay_rate': 0.997},
            {'eps_decay_rate': 0.999},
            {'eps_decay_rate': 0.9995},
            
            # Batch size
            {'batch_size': 32},
            {'batch_size': 64},
            {'batch_size': 128},
            {'batch_size': 256},
            
            # Memory size
            {'memory_size': 10000},
            {'memory_size': 20000},
            {'memory_size': 50000},
            {'memory_size': 100000},
            
            # Target update frequency
            {'target_update_freq': 100},
            {'target_update_freq': 200},
            {'target_update_freq': 500},
            {'target_update_freq': 1000},
        ]
        
        # Комбинируем параметры
        combinations = []
        for i in range(1, min(5, len(parameter_grid) + 1)):  # Максимум 4 параметра одновременно
            for combo in itertools.combinations(parameter_grid, i):
                combined_config = {}
                for param_dict in combo:
                    combined_config.update(param_dict)
                combinations.append(combined_config)
        
        return combinations[:20]  # Ограничиваем количество комбинаций
    
    def evaluate_config(self, config_params: Dict, episodes: int = 100) -> Dict[str, Any]:
        """Оценивает конфигурацию на небольшом количестве эпизодов"""
        
        try:
            print(f"🔍 Тестирование конфигурации: {config_params}")
            
            # Создаем новую конфигурацию
            test_config = vDqnConfig()
            for param, value in config_params.items():
                if hasattr(test_config, param):
                    setattr(test_config, param, value)
            
            # Запускаем обучение на небольшом количестве эпизодов
            results = train_model_optimized(
                dfs=self.dfs,
                cfg=test_config,
                episodes=episodes,
                patience_limit=episodes // 2,  # Быстрая остановка для тестирования
                use_wandb=False
            )
            
            # Извлекаем метрики
            if isinstance(results, str) and 'dqn_model.pth' in results:
                # Модель сохранена, загружаем результаты
                try:
                    with open('temp/train_results/latest_results.pkl', 'rb') as f:
                        training_results = pickle.load(f)
                    
                    # Рассчитываем score
                    score = self.calculate_score(training_results)
                    
                    return {
                        'config': config_params,
                        'score': score,
                        'results': training_results,
                        'status': 'success'
                    }
                except Exception as e:
                    print(f"⚠️ Ошибка загрузки результатов: {e}")
                    return {
                        'config': config_params,
                        'score': -np.inf,
                        'results': None,
                        'status': 'load_error'
                    }
            else:
                return {
                    'config': config_params,
                    'score': -np.inf,
                    'results': None,
                    'status': 'training_error'
                }
                
        except Exception as e:
            print(f"❌ Ошибка тестирования конфигурации: {e}")
            return {
                'config': config_params,
                'score': -np.inf,
                'results': None,
                'status': 'error'
            }
    
    def calculate_score(self, results: Dict[str, Any]) -> float:
        """Рассчитывает score для конфигурации"""
        
        if not results or 'final_stats' not in results:
            return -np.inf
        
        stats = results['final_stats']
        
        # Основные метрики
        winrate = stats.get('winrate', 0)
        pl_ratio = stats.get('pl_ratio', 0)
        trades_count = stats.get('trades_count', 0)
        
        # Дополнительные метрики
        avg_profit = stats.get('avg_profit', 0)
        avg_loss = abs(stats.get('avg_loss', 0))
        
        # Рассчитываем score (чем выше, тем лучше)
        score = 0
        
        # Winrate (вес: 40%)
        score += winrate * 0.4
        
        # P/L ratio (вес: 30%)
        if pl_ratio > 0:
            score += min(pl_ratio / 3.0, 1.0) * 0.3  # Нормализуем до 1.0
        
        # Количество сделок (вес: 15%)
        if trades_count > 0:
            score += min(trades_count / 1000.0, 1.0) * 0.15
        
        # Соотношение прибыли/убытка (вес: 15%)
        if avg_loss > 0:
            profit_loss_ratio = avg_profit / avg_loss
            score += min(profit_loss_ratio / 5.0, 1.0) * 0.15
        
        return score
    
    def optimize(self, max_iterations: int = 20, test_episodes: int = 100) -> Dict[str, Any]:
        """Основной метод оптимизации"""
        
        print(f"🚀 ЗАПУСК ОПТИМИЗАЦИИ ГИПЕРПАРАМЕТРОВ")
        print(f"=" * 60)
        print(f"📊 Максимум итераций: {max_iterations}")
        print(f"🎯 Эпизодов для тестирования: {test_episodes}")
        
        # Получаем сетку параметров
        parameter_combinations = self.define_parameter_grid()
        print(f"🔍 Найдено комбинаций параметров: {len(parameter_combinations)}")
        
        # Ограничиваем количество итераций
        parameter_combinations = parameter_combinations[:max_iterations]
        
        # Тестируем каждую конфигурацию
        for i, config_params in enumerate(parameter_combinations):
            print(f"\n📋 Итерация {i+1}/{len(parameter_combinations)}")
            
            # Оцениваем конфигурацию
            result = self.evaluate_config(config_params, test_episodes)
            self.results.append(result)
            
            # Обновляем лучший результат
            if result['status'] == 'success' and result['score'] > self.best_score:
                self.best_score = result['score']
                self.best_config = result['config']
                print(f"🏆 Новый лучший score: {self.best_score:.4f}")
                print(f"   Конфигурация: {self.best_config}")
            
            # Сохраняем промежуточные результаты
            self.save_intermediate_results()
        
        # Финальный анализ
        return self.analyze_results()
    
    def analyze_results(self) -> Dict[str, Any]:
        """Анализирует результаты оптимизации"""
        
        print(f"\n📊 АНАЛИЗ РЕЗУЛЬТАТОВ ОПТИМИЗАЦИИ")
        print(f"=" * 60)
        
        # Фильтруем успешные результаты
        successful_results = [r for r in self.results if r['status'] == 'success']
        
        if not successful_results:
            print("❌ Нет успешных результатов для анализа")
            return {}
        
        # Сортируем по score
        successful_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Топ-5 результатов
        print(f"🏆 ТОП-5 ЛУЧШИХ КОНФИГУРАЦИЙ:")
        for i, result in enumerate(successful_results[:5]):
            print(f"  {i+1}. Score: {result['score']:.4f}")
            print(f"     Параметры: {result['config']}")
        
        # Анализ по отдельным параметрам
        self.analyze_parameter_importance(successful_results)
        
        # Сохраняем результаты
        self.save_final_results()
        
        return {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'top_results': successful_results[:5],
            'total_tested': len(self.results),
            'successful_tests': len(successful_results)
        }
    
    def analyze_parameter_importance(self, results: List[Dict[str, Any]]):
        """Анализирует важность отдельных параметров"""
        
        print(f"\n🔍 АНАЛИЗ ВАЖНОСТИ ПАРАМЕТРОВ:")
        
        # Собираем все уникальные параметры
        all_params = set()
        for result in results:
            all_params.update(result['config'].keys())
        
        # Анализируем каждый параметр
        for param in all_params:
            param_values = []
            param_scores = []
            
            for result in results:
                if param in result['config']:
                    param_values.append(result['config'][param])
                    param_scores.append(result['score'])
            
            if param_values:
                # Группируем по значениям параметра
                unique_values = list(set(param_values))
                avg_scores = []
                
                for value in unique_values:
                    value_scores = [score for val, score in zip(param_values, param_scores) if val == value]
                    avg_scores.append(np.mean(value_scores))
                
                # Находим лучшее значение
                best_value_idx = np.argmax(avg_scores)
                best_value = unique_values[best_value_idx]
                best_avg_score = avg_scores[best_value_idx]
                
                print(f"  • {param}:")
                print(f"    Лучшее значение: {best_value} (score: {best_avg_score:.4f})")
                print(f"    Все значения: {dict(zip(unique_values, [f'{s:.4f}' for s in avg_scores]))}")
    
    def save_intermediate_results(self):
        """Сохраняет промежуточные результаты"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"temp/hyperparameter_optimization_{timestamp}.json"
        
        # Создаем папку если не существует
        Path('temp').mkdir(exist_ok=True)
        
        # Подготавливаем данные для сохранения
        save_data = {
            'timestamp': timestamp,
            'results': [
                {
                    'config': r['config'],
                    'score': r['score'],
                    'status': r['status']
                }
                for r in self.results
            ],
            'best_config': self.best_config,
            'best_score': self.best_score
        }
        
        # Сохраняем в JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Промежуточные результаты сохранены: {filename}")
    
    def save_final_results(self):
        """Сохраняет финальные результаты оптимизации"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"temp/final_hyperparameter_optimization_{timestamp}.json"
        
        # Создаем папку если не существует
        Path('temp').mkdir(exist_ok=True)
        
        # Подготавливаем данные для сохранения
        save_data = {
            'timestamp': timestamp,
            'optimization_summary': {
                'total_tested': len(self.results),
                'successful_tests': len([r for r in self.results if r['status'] == 'success']),
                'best_score': self.best_score,
                'best_config': self.best_config
            },
            'all_results': [
                {
                    'config': r['config'],
                    'score': r['score'],
                    'status': r['status']
                }
                for r in self.results
            ],
            'recommendations': self.generate_recommendations()
        }
        
        # Сохраняем в JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Финальные результаты сохранены: {filename}")
    
    def generate_recommendations(self) -> List[str]:
        """Генерирует рекомендации на основе результатов"""
        
        recommendations = []
        
        if self.best_score > 0.8:
            recommendations.append("🎯 Отличный результат! Модель показывает высокую производительность")
        elif self.best_score > 0.6:
            recommendations.append("✅ Хороший результат! Есть возможности для улучшения")
        else:
            recommendations.append("⚠️ Результат ниже ожидаемого. Требуется дополнительная оптимизация")
        
        if self.best_config:
            recommendations.append(f"🔧 Рекомендуемая конфигурация: {self.best_config}")
        
        # Анализ по параметрам
        successful_results = [r for r in self.results if r['status'] == 'success']
        if successful_results:
            avg_score = np.mean([r['score'] for r in successful_results])
            recommendations.append(f"📊 Средний score успешных тестов: {avg_score:.4f}")
        
        return recommendations

def main():
    """Основная функция для запуска оптимизации"""
    
    print("🚀 ОПТИМИЗАТОР ГИПЕРПАРАМЕТРОВ DQN МОДЕЛИ")
    print("=" * 60)
    
    # Проверяем наличие данных
    if not Path('temp/binance_data').exists():
        print("❌ Данные для обучения не найдены в temp/binance_data/")
        print("   Сначала запустите download.py для загрузки данных")
        return
    
    # Загружаем данные (здесь нужно будет адаптировать под вашу структуру)
    print("📥 Загрузка данных...")
    # TODO: Добавить загрузку данных
    
    # Создаем оптимизатор
    optimizer = HyperparameterOptimizer(dfs={})  # Передайте ваши данные
    
    # Запускаем оптимизацию
    results = optimizer.optimize(max_iterations=10, test_episodes=50)
    
    if results:
        print(f"\n✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
        print(f"🏆 Лучший score: {results['best_score']:.4f}")
        print(f"🔧 Лучшая конфигурация: {results['best_config']}")
    else:
        print(f"\n❌ ОПТИМИЗАЦИЯ НЕ ДАЛА РЕЗУЛЬТАТОВ")

if __name__ == "__main__":
    main()

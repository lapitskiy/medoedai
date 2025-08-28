#!/usr/bin/env python3
"""
Мультивалютный тренер DQN
Обучает модель одновременно на нескольких криптовалютах
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import torch

# Добавляем пути
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from agents.vdqn.v_train_model_optimized import train_model_optimized
from agents.vdqn.cfg.vconfig import vDqnConfig
from utils.adaptive_normalization import adaptive_normalizer

logger = logging.getLogger(__name__)

class MultiCryptoTrainer:
    """
    Тренер для обучения DQN на нескольких криптовалютах одновременно
    """
    
    def __init__(self, config: Optional[vDqnConfig] = None):
        self.config = config or vDqnConfig()
        self.training_history = defaultdict(list)
        self.model_performance = {}
        
    def prepare_multi_crypto_data(self, crypto_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Подготавливает данные для мультивалютного обучения
        
        Args:
            crypto_data: {symbol: {df_5min, df_15min, df_1h}}
            
        Returns:
            Подготовленные данные с адаптивной нормализацией
        """
        prepared_data = {}
        
        for symbol, data in crypto_data.items():
            print(f"🔧 Подготовка данных для {symbol}...")
            
            try:
                # Адаптивная нормализация для каждой криптовалюты
                if hasattr(adaptive_normalizer, 'normalize_features'):
                    # Нормализуем 5-минутные данные
                    df_5min_normalized = adaptive_normalizer.normalize_features(
                        data['df_5min'], symbol
                    )
                    
                    # Получаем адаптивные параметры
                    trading_params = adaptive_normalizer.get_trading_params(symbol, data['df_5min'])
                    
                    # Добавляем параметры в данные
                    data['trading_params'] = trading_params
                    data['df_5min'] = df_5min_normalized
                    
                    print(f"  ✅ {symbol}: volatility_mult={trading_params['volatility_multiplier']:.2f}, "
                          f"min_hold={trading_params['min_hold_steps']}")
                
                prepared_data[symbol] = data
                
            except Exception as e:
                logger.error(f"Ошибка подготовки данных для {symbol}: {e}")
                print(f"  ❌ {symbol}: ошибка подготовки - {e}")
                continue
        
        return prepared_data
    
    def train_on_single_crypto(self, symbol: str, data: Dict, 
                              episodes: int = 1000, 
                              model_path: Optional[str] = None) -> Dict:
        """
        Обучает модель на одной криптовалюте
        
        Args:
            symbol: Символ криптовалюты
            data: Данные для обучения
            episodes: Количество эпизодов
            model_path: Путь для сохранения модели
            
        Returns:
            Результаты обучения
        """
        print(f"\n🚀 Обучение на {symbol} ({episodes} эпизодов)")
        print("=" * 60)
        
        try:
            # Настраиваем конфигурацию под криптовалюту
            crypto_config = self.config.copy()
            
            # Адаптируем параметры под криптовалюту
            if 'trading_params' in data:
                params = data['trading_params']
                
                # Обновляем конфигурацию
                crypto_config.min_episodes_before_stopping = max(500, episodes // 10)
                crypto_config.early_stopping_patience = max(2000, episodes // 3)
                
                print(f"🔧 Адаптивные параметры для {symbol}:")
                print(f"  • Stop Loss: {params['stop_loss_pct']:.3f}")
                print(f"  • Take Profit: {params['take_profit_pct']:.3f}")
                print(f"  • Min Hold: {params['min_hold_steps']} шагов")
                print(f"  • Volume Threshold: {params['volume_threshold']:.4f}")
            
            # Запускаем обучение
            start_time = time.time()
            model_path = train_model_optimized(
                dfs=data,
                cfg=crypto_config,
                episodes=episodes,
                patience_limit=crypto_config.early_stopping_patience,
                use_wandb=False
            )
            training_time = time.time() - start_time
            
            # Сохраняем результаты
            result = {
                'symbol': symbol,
                'episodes': episodes,
                'model_path': model_path,
                'training_time': training_time,
                'trading_params': data.get('trading_params', {}),
                'status': 'success'
            }
            
            print(f"✅ {symbol}: обучение завершено за {training_time/60:.1f} минут")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка обучения на {symbol}: {e}")
            print(f"❌ {symbol}: ошибка обучения - {e}")
            
            return {
                'symbol': symbol,
                'episodes': episodes,
                'model_path': None,
                'training_time': 0,
                'trading_params': {},
                'status': 'error',
                'error': str(e)
            }
    
    def train_sequentially(self, crypto_data: Dict[str, Dict], 
                          episodes_per_crypto: int = 1000) -> Dict:
        """
        Последовательное обучение на каждой криптовалюте
        
        Args:
            crypto_data: Данные для всех криптовалют
            episodes_per_crypto: Эпизодов на каждую криптовалюту
            
        Returns:
            Результаты обучения
        """
        print(f"🎯 ПОСЛЕДОВАТЕЛЬНОЕ ОБУЧЕНИЕ")
        print(f"📊 Криптовалют: {len(crypto_data)}")
        print(f"🎬 Эпизодов на криптовалюту: {episodes_per_crypto}")
        print("=" * 80)
        
        # Подготавливаем данные
        prepared_data = self.prepare_multi_crypto_data(crypto_data)
        
        results = {}
        total_episodes = 0
        
        for symbol, data in prepared_data.items():
            print(f"\n{'='*20} {symbol} {'='*20}")
            
            # Обучаем на текущей криптовалюте
            result = self.train_on_single_crypto(
                symbol=symbol,
                data=data,
                episodes=episodes_per_crypto
            )
            
            results[symbol] = result
            total_episodes += episodes_per_crypto
            
            # Сохраняем историю
            self.training_history[symbol].append(result)
            
            # Небольшая пауза между криптовалютами
            time.sleep(2)
        
        # Сводка по всем криптовалютам
        self._print_training_summary(results, total_episodes)
        
        return results
    
    def train_parallel_episodes(self, crypto_data: Dict[str, Dict], 
                               total_episodes: int = 10000) -> Dict:
        """
        Параллельное обучение: чередуем эпизоды между криптовалютами
        
        Args:
            crypto_data: Данные для всех криптовалют
            total_episodes: Общее количество эпизодов
            
        Returns:
            Результаты обучения
        """
        print(f"🎯 ПАРАЛЛЕЛЬНОЕ ОБУЧЕНИЕ")
        print(f"📊 Криптовалют: {len(crypto_data)}")
        print(f"🎬 Общее количество эпизодов: {total_episodes}")
        print("=" * 80)
        
        # Подготавливаем данные
        prepared_data = self.prepare_multi_crypto_data(crypto_data)
        
        # Распределяем эпизоды между криптовалютами
        episodes_per_crypto = total_episodes // len(prepared_data)
        remaining_episodes = total_episodes % len(prepared_data)
        
        print(f"📈 Эпизодов на криптовалюту: {episodes_per_crypto}")
        print(f"📈 Дополнительных эпизодов: {remaining_episodes}")
        
        results = {}
        
        # Обучаем на каждой криптовалюте
        for i, (symbol, data) in enumerate(prepared_data.items()):
            # Первые криптовалюты получают дополнительный эпизод
            crypto_episodes = episodes_per_crypto + (1 if i < remaining_episodes else 0)
            
            print(f"\n{'='*20} {symbol} ({crypto_episodes} эпизодов) {'='*20}")
            
            result = self.train_on_single_crypto(
                symbol=symbol,
                data=data,
                episodes=crypto_episodes
            )
            
            results[symbol] = result
            self.training_history[symbol].append(result)
            
            time.sleep(1)
        
        # Сводка по всем криптовалютам
        self._print_training_summary(results, total_episodes)
        
        return results
    
    def _print_training_summary(self, results: Dict, total_episodes: int):
        """Выводит сводку по обучению"""
        
        print(f"\n{'='*80}")
        print(f"📊 СВОДКА ПО ОБУЧЕНИЮ")
        print(f"{'='*80}")
        
        successful = 0
        total_time = 0
        
        for symbol, result in results.items():
            if result['status'] == 'success':
                successful += 1
                total_time += result['training_time']
                print(f"✅ {symbol}: {result['episodes']} эпизодов, "
                      f"{result['training_time']/60:.1f} мин")
            else:
                print(f"❌ {symbol}: ошибка - {result.get('error', 'неизвестно')}")
        
        print(f"\n📈 ИТОГО:")
        print(f"  • Успешно обучено: {successful}/{len(results)}")
        print(f"  • Общее время: {total_time/60:.1f} минут")
        print(f"  • Среднее время на криптовалюту: {total_time/successful/60:.1f} минут")
        
        # Сохраняем результаты
        self._save_training_results(results)
    
    def _save_training_results(self, results: Dict):
        """Сохраняет результаты обучения"""
        
        timestamp = int(time.time())
        results_file = f"temp/multi_crypto_results_{timestamp}.pkl"
        
        try:
            import pickle
            os.makedirs("temp", exist_ok=True)
            
            with open(results_file, 'wb') as f:
                pickle.dump({
                    'results': results,
                    'training_history': dict(self.training_history),
                    'timestamp': timestamp
                }, f)
            
            print(f"💾 Результаты сохранены в {results_file}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения результатов: {e}")
            print(f"❌ Ошибка сохранения: {e}")

def main():
    """Основная функция для тестирования"""
    
    # Создаем тестовые данные для разных криптовалют
    test_data = {
        'BTCUSDT': {
            'df_5min': pd.DataFrame({
                'open': np.random.uniform(45000, 55000, 1000),
                'high': np.random.uniform(45000, 55000, 1000),
                'low': np.random.uniform(45000, 55000, 1000),
                'close': np.random.uniform(45000, 55000, 1000),
                'volume': np.random.uniform(1000, 10000, 1000)
            }),
            'df_15min': pd.DataFrame({
                'open': np.random.uniform(45000, 55000, 1000),
                'high': np.random.uniform(45000, 55000, 1000),
                'low': np.random.uniform(45000, 55000, 1000),
                'close': np.random.uniform(45000, 55000, 1000),
                'volume': np.random.uniform(1000, 10000, 1000)
            }),
            'df_1h': pd.DataFrame({
                'open': np.random.uniform(45000, 55000, 1000),
                'high': np.random.uniform(45000, 55000, 1000),
                'low': np.random.uniform(45000, 55000, 1000),
                'close': np.random.uniform(45000, 55000, 1000),
                'volume': np.random.uniform(1000, 10000, 1000)
            })
        },
        'TONUSDT': {
            'df_5min': pd.DataFrame({
                'open': np.random.uniform(1.5, 2.5, 1000),
                'high': np.random.uniform(1.5, 2.5, 1000),
                'low': np.random.uniform(1.5, 2.5, 1000),
                'close': np.random.uniform(1.5, 2.5, 1000),
                'volume': np.random.uniform(50000, 500000, 1000)
            }),
            'df_15min': pd.DataFrame({
                'open': np.random.uniform(1.5, 2.5, 1000),
                'high': np.random.uniform(1.5, 2.5, 1000),
                'low': np.random.uniform(1.5, 2.5, 1000),
                'close': np.random.uniform(1.5, 2.5, 1000),
                'volume': np.random.uniform(50000, 500000, 1000)
            }),
            'df_1h': pd.DataFrame({
                'open': np.random.uniform(1.5, 2.5, 1000),
                'high': np.random.uniform(1.5, 2.5, 1000),
                'low': np.random.uniform(1.5, 2.5, 1000),
                'close': np.random.uniform(1.5, 2.5, 1000),
                'volume': np.random.uniform(50000, 500000, 1000)
            })
        }
    }
    
    # Создаем тренер
    trainer = MultiCryptoTrainer()
    
    # Тестируем последовательное обучение
    print("🧪 ТЕСТИРОВАНИЕ МУЛЬТИВАЛЮТНОГО ОБУЧЕНИЯ")
    print("=" * 80)
    
    # Последовательное обучение (как у вас сейчас)
    print("\n1️⃣ ПОСЛЕДОВАТЕЛЬНОЕ ОБУЧЕНИЕ:")
    sequential_results = trainer.train_sequentially(test_data, episodes_per_crypto=100)
    
    # Параллельное обучение (новая стратегия)
    print("\n2️⃣ ПАРАЛЛЕЛЬНОЕ ОБУЧЕНИЕ:")
    parallel_results = trainer.train_parallel_episodes(test_data, total_episodes=200)

if __name__ == "__main__":
    main()

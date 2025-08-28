#!/usr/bin/env python3
"""
Мультивалютное окружение для обучения DQN.
Автоматически переключается между криптовалютами для каждого эпизода.
"""

import random
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from .crypto_trading_env_optimized import CryptoTradingEnvOptimized as CryptoTradingEnv

class MultiCryptoTradingEnv:
    """
    Окружение для мультивалютного обучения DQN.
    Автоматически переключается между криптовалютами для каждого эпизода.
    """
    
    def __init__(self, dfs: dict, cfg=None):
        """
        Args:
            dfs (dict): Словарь с данными криптовалют в формате:
                {symbol: {'df_5min': DataFrame, 'symbol': str, 'candle_count': int}}
            cfg: Конфигурация для n-step learning
        """
        self.dfs = dfs
        self.symbols = list(dfs.keys())
        self.current_symbol = None
        self.current_env = None
        self.cfg = cfg
        
        # ИСПРАВЛЕНИЕ: Добавляем общий список сделок для правильного расчета winrate
        self._all_trades = []
        
        # Добавляем атрибут all_trades для совместимости с train_model_optimized
        self.all_trades = []
        
        # Добавляем внутренний атрибут _trades для совместимости
        self._trades = []
        
        # Создаем базовое окружение для определения размеров
        if not self.symbols:
            raise ValueError("Не переданы данные криптовалют")
            
        first_symbol = self.symbols[0]
        first_data = self.dfs[first_symbol]
        temp_dfs = {
            'df_5min': first_data['df_5min'],
            'df_15min': first_data['df_15min'],
            'df_1h': first_data['df_1h']
        }
        
        temp_env = CryptoTradingEnv(dfs=temp_dfs, cfg=cfg)
        self.observation_space = temp_env.observation_space
        self.action_space = temp_env.action_space
        
        # Добавляем недостающий атрибут для совместимости с train_model_optimized
        if hasattr(temp_env, 'observation_space_shape'):
            self.observation_space_shape = temp_env.observation_space_shape
        else:
            # Fallback: вычисляем размер из observation_space
            if hasattr(self.observation_space, 'shape'):
                self.observation_space_shape = self.observation_space.shape[0]
            else:
                # Если не можем определить, используем размер по умолчанию
                self.observation_space_shape = 100  # Примерный размер для DQN
        
        # N-step learning параметры
        self.n_step = getattr(cfg, 'n_step', 3) if cfg else 3
        self.n_step_buffer = []
        self.gamma = getattr(cfg, 'gamma', 0.99) if cfg else 0.99
        
        print(f"🌍 Мультивалютное окружение инициализировано для {len(self.symbols)} криптовалют")
        print(f"📊 Доступные символы: {', '.join(self.symbols)}")
        print(f"🔄 N-step learning: {self.n_step} шагов, gamma: {self.gamma}")
        
        # Статистика использования криптовалют
        self.episode_stats = {symbol: 0 for symbol in self.symbols}
        
        # Добавляем атрибут symbol для совместимости с логированием
        self.symbol = "МУЛЬТИВАЛЮТА"
    
    def reset(self):
        """Сбрасывает окружение и случайно выбирает криптовалюту для эпизода"""
        # Случайно выбираем криптовалюту
        self.current_symbol = random.choice(self.symbols)
        current_data = self.dfs[self.current_symbol]
        
        # Обновляем статистику
        self.episode_stats[self.current_symbol] += 1
        
        # ИСПРАВЛЕНИЕ: Создаем окружение только если его нет или сменилась криптовалюта
        if (self.current_env is None or 
            getattr(self.current_env, 'symbol', None) != self.current_symbol):
            
            temp_dfs = {
                'df_5min': current_data['df_5min'],
                'df_15min': current_data['df_15min'],
                'df_1h': current_data['df_1h']
            }
                        
            
            self.current_env = CryptoTradingEnv(dfs=temp_dfs, cfg=self.cfg)
            
            # ИСПРАВЛЕНИЕ: Передаем накопленные сделки в новое окружение
            if hasattr(self, '_all_trades') and self._all_trades:
                self.current_env.all_trades = self._all_trades.copy()
            
            # Обновляем наш атрибут all_trades
            if hasattr(self.current_env, 'all_trades'):
                self.all_trades = self.current_env.all_trades
            
            # Обновляем наш внутренний атрибут _trades
            if hasattr(self.current_env, 'trades'):
                self._trades = getattr(self.current_env, 'trades', []).copy()

        
        # Очищаем n-step buffer при смене криптовалюты
        self.n_step_buffer.clear()
        
        print(f"🔄 Эпизод: выбрана {self.current_symbol} ({current_data['candle_count']} свечей)")
        
        # Сбрасываем окружение
        return self.current_env.reset()
    
    def get_current_symbol(self):
        """Возвращает текущую выбранную криптовалюту"""
        return self.current_symbol
    
    def get_n_step_return(self, n_steps: int = None) -> list:
        """
        Возвращает n-step transitions для обучения
        
        Args:
            n_steps: количество шагов (по умолчанию использует self.n_step)
            
        Returns:
            list: список n-step transitions
        """
        if n_steps is None:
            n_steps = self.n_step
            
        if len(self.n_step_buffer) < n_steps:
            return []
            
        transitions = []
        for i in range(len(self.n_step_buffer) - n_steps + 1):
            # Берем n последовательных переходов
            n_step_transitions = list(self.n_step_buffer)[i:i+n_steps]
            
            # Проверяем, что все next_state заполнены
            if any(t['next_state'] is None for t in n_step_transitions):
                continue  # Пропускаем неполные transitions
            
            # Рассчитываем n-step return
            total_reward = 0
            for j, transition in enumerate(n_step_transitions):
                total_reward += transition['reward'] * (self.gamma ** j)
            
            # Создаем n-step transition
            first_transition = n_step_transitions[0]
            last_transition = n_step_transitions[-1]
            
            n_step_transition = {
                'state': first_transition['state'],
                'action': first_transition['action'],
                'reward': total_reward,
                'next_state': last_transition['next_state'],
                'done': last_transition['done']
            }
            
            transitions.append(n_step_transition)
            
        return transitions
    
    def step(self, action):
        """Выполняет шаг в текущем окружении"""
        if self.current_env is None:
            raise ValueError("Окружение не инициализировано. Вызовите reset()")
        
        # Запоминаем количество сделок до шага
        trades_before = len(getattr(self.current_env, 'all_trades', []))
        
        result = self.current_env.step(action)
        
        # Проверяем, появились ли новые сделки
        trades_after = len(getattr(self.current_env, 'all_trades', []))
        if trades_after > trades_before:            
            # Синхронизируем сделки
            self._all_trades = getattr(self.current_env, 'all_trades', []).copy()
        
        return result
    
    @property
    def epsilon(self):
        """Получает epsilon из текущего окружения"""
        if self.current_env:
            return getattr(self.current_env, 'epsilon', 0.1)
        return 0.1
    
    @epsilon.setter
    def epsilon(self, value):
        """Устанавливает epsilon в текущее окружение"""
        if self.current_env:
            self.current_env.epsilon = value
    
    @property
    def trades(self):
        """Получает сделки из текущего окружения"""
        if self.current_env:
            # Синхронизируем сделки между окружением и мульти-окружением
            env_trades = getattr(self.current_env, 'trades', [])
            if len(env_trades) > len(self._trades):
                self._trades = env_trades.copy()
            return self._trades
        return self._trades
    
    @trades.setter
    def trades(self, value):
        """Устанавливает сделки"""
        self._trades = value.copy() if value else []
        if self.current_env:
            self.current_env.trades = value.copy() if value else []
    
    @property
    def all_trades(self):
        """ИСПРАВЛЕНИЕ: Получает общий список сделок"""
        if self.current_env:
            # Синхронизируем сделки между окружением и мульти-окружением
            env_all_trades = getattr(self.current_env, 'all_trades', [])
            if len(env_all_trades) > len(self._all_trades):
                self._all_trades = env_all_trades.copy()
            return self._all_trades
        return self._all_trades
    
    @all_trades.setter
    def all_trades(self, value):
        """ИСПРАВЛЕНИЕ: Устанавливает общий список сделок"""
        self._all_trades = value
        if self.current_env:
            self.current_env.all_trades = value.copy()
    
    def get_current_symbol(self):
        """Возвращает текущую выбранную криптовалюту"""
        return self.current_symbol
    
    def get_episode_stats(self):
        """Возвращает статистику использования криптовалют"""
        total_episodes = sum(self.episode_stats.values())
        if total_episodes == 0:
            return {}
        
        stats = {}
        for symbol, count in self.episode_stats.items():
            percentage = (count / total_episodes) * 100
            stats[symbol] = {
                'episodes': count,
                'percentage': percentage
            }
        
        return stats
    
    def print_episode_stats(self):
        """Выводит статистику использования криптовалют"""
        stats = self.get_episode_stats()
        if not stats:
            print("📊 Статистика эпизодов недоступна")
            return
        
        print("\n📊 СТАТИСТИКА ИСПОЛЬЗОВАНИЯ КРИПТОВАЛЮТ:")
        print("=" * 50)
        
        total_episodes = sum([s['episodes'] for s in stats.values()])
        print(f"Всего эпизодов: {total_episodes}")
        print()
        
        # Сортируем по количеству эпизодов
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['episodes'], reverse=True)
        
        for symbol, data in sorted_stats:
            episodes = data['episodes']
            percentage = data['percentage']
            bar_length = int(percentage / 2)  # Максимум 50 символов
            bar = "█" * bar_length
            
            print(f"{symbol:>10}: {episodes:>4} эпизодов ({percentage:>5.1f}%) {bar}")
        
        print("=" * 50)
    
    def get_env_info(self):
        """Возвращает информацию об окружении"""
        return {
            'total_cryptos': len(self.symbols),
            'available_symbols': self.symbols,
            'current_symbol': self.current_symbol,
            'observation_space': str(self.observation_space),
            'action_space': str(self.action_space),
            'episode_stats': self.get_episode_stats()
        }
    
    def set_symbol_weights(self, weights: Dict[str, float]):
        """
        Устанавливает веса для выбора криптовалют.
        
        Args:
            weights (dict): Словарь {symbol: weight} где weight > 0
        """
        if not weights:
            return
        
        # Проверяем, что все символы существуют
        for symbol in weights:
            if symbol not in self.symbols:
                print(f"⚠️ Символ {symbol} не найден в доступных криптовалютах")
                return
        
        # Создаем список для взвешенного выбора
        self.weighted_symbols = []
        for symbol, weight in weights.items():
            if weight > 0:
                self.weighted_symbols.extend([symbol] * int(weight * 100))
        
        if self.weighted_symbols:
            print(f"⚖️ Установлены веса для криптовалют: {weights}")
            # Переопределяем метод выбора символа
            self._select_symbol = self._select_weighted_symbol
        else:
            print("⚠️ Некорректные веса, используется случайный выбор")
    
    def _select_weighted_symbol(self):
        """Выбирает криптовалюту с учетом весов"""
        if hasattr(self, 'weighted_symbols') and self.weighted_symbols:
            return random.choice(self.weighted_symbols)
        return random.choice(self.symbols)
    
    def _select_symbol(self):
        """Выбирает криптовалюту случайным образом (по умолчанию)"""
        return random.choice(self.symbols)
    
    def reset_with_symbol(self, symbol: str):
        """
        Сбрасывает окружение с указанной криптовалютой.
        
        Args:
            symbol (str): Символ криптовалюты для использования
        """
        if symbol not in self.symbols:
            print(f"⚠️ Символ {symbol} не найден, используется случайный выбор")
            return self.reset()
        
        self.current_symbol = symbol
        current_data = self.dfs[symbol]
        
        # Обновляем статистику
        self.episode_stats[symbol] += 1
        
        # Создаем окружение для указанной криптовалюты
        temp_dfs = {
            'df_5min': current_data['df_5min'],
            'df_15min': current_data['df_15min'],
            'df_1h': current_data['df_1h']
        }
        self.current_env = CryptoTradingEnv(dfs=temp_dfs, cfg=self.cfg)
        
        # ИСПРАВЛЕНИЕ: Передаем накопленные сделки в новое окружение
        if hasattr(self, '_all_trades') and self._all_trades:
            self.current_env.all_trades = self._all_trades.copy()
            print(f"    📊 Передано {len(self._all_trades)} сделок в новое окружение")
        
        print(f"🔄 Эпизод: принудительно выбрана {symbol} ({current_data['candle_count']} свечей)")
        
        # Сбрасываем окружение
        return self.current_env.reset()

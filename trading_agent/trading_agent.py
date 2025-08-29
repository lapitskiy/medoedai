import os
import time
import logging
from typing import Dict, Optional, Tuple
import ccxt
import numpy as np
import torch
from datetime import datetime, timedelta

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingAgent:
    def __init__(self, model_path: str = "/workspace/good_model/dqn_model.pth"):
        """
        Инициализация торгового агента
        
        Args:
            model_path: путь к обученной модели
        """
        self.model_path = model_path
        self.exchange = None
        self.model = None
        self.is_trading = False
        self.current_position = None
        self.trading_history = []
        
        # Загружаем модель
        self._load_model()
        
        # Инициализируем биржу
        self._init_exchange()
    
    def _load_model(self):
        """Загрузка обученной модели (поддержка torch.save(obj) и torch.save(state_dict))"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Модель не найдена: {self.model_path}")
                return

            checkpoint = torch.load(self.model_path, map_location='cpu')

            # Если сохранён state_dict (dict), нужно создать модель и загрузить веса
            if isinstance(checkpoint, dict) and not hasattr(checkpoint, 'eval'):
                # Попытка восстановить размерности из окружения (по умолчанию)
                try:
                    from envs.dqn_model.gym.crypto_trading_env_optimized import CryptoTradingEnv
                    temp_env = CryptoTradingEnv(symbol='BTC/USDT', timeframe='5m')
                    obs_dim = getattr(temp_env, 'observation_space_shape', None)
                    if obs_dim is None and hasattr(temp_env, 'observation_space'):
                        obs_dim = temp_env.observation_space.shape[0]
                    act_dim = 3
                except Exception:
                    # Фолбэк значения, если окружение недоступно
                    obs_dim = 100
                    act_dim = 3

                # Импортируем архитектуру сети
                try:
                    from agents.vdqn.dqnn import DQNN
                    model = DQNN(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=(512, 256, 128))
                except Exception as arch_err:
                    logger.error(f"Не удалось создать архитектуру сети: {arch_err}")
                    return

                # Если в checkpoint есть вложенный ключ state_dict
                state_dict = checkpoint.get('state_dict', checkpoint)
                model.load_state_dict(state_dict, strict=False)
                self.model = model
                self.model.eval()
                logger.info(f"Загружен state_dict модели из {self.model_path}")
            else:
                # Сохранён целый объект модели
                self.model = checkpoint
                self.model.eval()
                logger.info(f"Модель загружена из {self.model_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
    
    def _init_exchange(self):
        """Инициализация подключения к бирже"""
        try:
            # API ключи из переменных окружения
            api_key = os.getenv('BYBIT_API_KEY')
            secret_key = os.getenv('BYBIT_SECRET_KEY')
            
            if not api_key or not secret_key:
                logger.error("API ключи не настроены")
                return
            
            self.exchange = ccxt.bybit({
                'apiKey': api_key,
                'secret': secret_key,
                'sandbox': False,  # True для тестового режима
                'enableRateLimit': True
            })
            
            logger.info("Подключение к Bybit установлено")
        except Exception as e:
            logger.error(f"Ошибка подключения к бирже: {e}")
    
    def start_trading(self, symbols: list) -> Dict:
        """
        Запуск торговли
        
        Args:
            symbols: список торговых пар
            
        Returns:
            Dict с результатом запуска
        """
        if not self.exchange:
            return {"success": False, "error": "Биржа не инициализирована"}
        
        if not self.model:
            return {"success": False, "error": "Модель не загружена"}
        
        try:
            self.is_trading = True
            self.symbols = symbols
            
            logger.info(f"Торговля запущена для {symbols}")
            
            # Запускаем торговый цикл в отдельном потоке
            import threading
            self.trading_thread = threading.Thread(target=self._trading_loop)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
            return {
                "success": True, 
                "message": f"Торговля запущена для {symbols}"
            }
            
        except Exception as e:
            self.is_trading = False
            logger.error(f"Ошибка запуска торговли: {e}")
            return {"success": False, "error": str(e)}
    
    def stop_trading(self) -> Dict:
        """Остановка торговли"""
        try:
            self.is_trading = False
            if hasattr(self, 'trading_thread'):
                self.trading_thread.join(timeout=5)
            
            logger.info("Торговля остановлена")
            return {"success": True, "message": "Торговля остановлена"}
            
        except Exception as e:
            logger.error(f"Ошибка остановки торговли: {e}")
            return {"success": False, "error": str(e)}
    
    def get_trading_status(self) -> Dict:
        """Получение статуса торговли"""
        return {
            "is_trading": self.is_trading,
            "symbol": getattr(self, 'symbol', None),
            "amount": getattr(self, 'trade_amount', None),
            "position": self.current_position,
            "trades_count": len(self.trading_history)
        }
    
    def get_balance(self) -> Dict:
        """Получение баланса"""
        try:
            if not self.exchange:
                return {"success": False, "error": "Биржа не инициализирована"}
            
            balance = self.exchange.fetch_balance()
            return {
                "success": True,
                "balance": {
                    "USDT": balance.get('USDT', {}).get('free', 0),
                    "BTC": balance.get('BTC', {}).get('free', 0)
                }
            }
        except Exception as e:
            logger.error(f"Ошибка получения баланса: {e}")
            return {"success": False, "error": str(e)}
    
    def _trading_loop(self):
        """Основной торговый цикл"""
        logger.info("Торговый цикл запущен")
        
        while self.is_trading:
            try:
                # Получаем текущие данные
                ticker = self.exchange.fetch_ticker(self.symbol)
                
                # Получаем предсказание от модели
                action = self._get_model_prediction(ticker)
                
                # Выполняем торговую операцию
                if action == 'buy' and not self.current_position:
                    self._execute_buy()
                elif action == 'sell' and self.current_position:
                    self._execute_sell()
                
                # Пауза между итерациями
                time.sleep(60)  # 1 минута
                
            except Exception as e:
                logger.error(f"Ошибка в торговом цикле: {e}")
                time.sleep(60)
        
        logger.info("Торговый цикл остановлен")
    
    def _get_model_prediction(self, ticker: Dict) -> str:
        """Получение предсказания от модели"""
        try:
            # Здесь должна быть логика подготовки данных для модели
            # и получения предсказания
            # Пока возвращаем случайное действие для примера
            
            # TODO: Реализовать реальное предсказание
            import random
            return random.choice(['buy', 'sell', 'hold'])
            
        except Exception as e:
            logger.error(f"Ошибка получения предсказания: {e}")
            return 'hold'
    
    def _execute_buy(self):
        """Выполнение покупки"""
        try:
            order = self.exchange.create_market_buy_order(
                self.symbol, 
                self.trade_amount
            )
            
            self.current_position = {
                'type': 'long',
                'amount': self.trade_amount,
                'entry_price': order['price'],
                'entry_time': datetime.now()
            }
            
            self.trading_history.append({
                'action': 'buy',
                'price': order['price'],
                'amount': self.trade_amount,
                'time': datetime.now()
            })
            
            logger.info(f"Покупка выполнена: {order}")
            
        except Exception as e:
            logger.error(f"Ошибка покупки: {e}")
    
    def _execute_sell(self):
        """Выполнение продажи"""
        try:
            order = self.exchange.create_market_sell_order(
                self.symbol, 
                self.current_position['amount']
            )
            
            # Расчет P&L
            exit_price = order['price']
            entry_price = self.current_position['entry_price']
            pnl = (exit_price - entry_price) * self.current_position['amount']
            
            self.trading_history.append({
                'action': 'sell',
                'price': exit_price,
                'amount': self.current_position['amount'],
                'time': datetime.now(),
                'pnl': pnl
            })
            
            logger.info(f"Продажа выполнена: {order}, P&L: {pnl}")
            
            self.current_position = None
            
        except Exception as e:
            logger.error(f"Ошибка продажи: {e}")
    
    def get_trading_history(self) -> Dict:
        """Получение истории торговли"""
        return {
            "success": True,
            "trades": self.trading_history,
            "total_trades": len(self.trading_history)
        }

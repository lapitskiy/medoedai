import os
import time
import logging
from typing import Dict, Optional, Tuple
import ccxt
import numpy as np
import torch
from datetime import datetime, timedelta
from utils.trade_utils import create_trade_record, update_trade_status

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
                    temp_env = CryptoTradingEnv(symbol='BTCUSDT', timeframe='5m')
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
        Запуск торговли (одноразовое выполнение)
        
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
            self.symbols = symbols
            self.symbol = symbols[0] if symbols else 'BTCUSDT'  # Устанавливаем первый символ как основной
            
            # Рассчитываем количество для торговли на основе баланса
            self.trade_amount = self._calculate_trade_amount()
            
            logger.info(f"Торговля запущена для {symbols}, основной символ: {self.symbol}, количество: {self.trade_amount}")
            
            # Выполняем один торговый шаг
            result = self._execute_trading_step()
            
            return {
                "success": True, 
                "message": f"Торговый шаг выполнен для {symbols}",
                "trading_result": result
            }
            
        except Exception as e:
            logger.error(f"Ошибка выполнения торгового шага: {e}")
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
        balance_info = self.get_balance()
        current_price = self._get_current_price()
        
        return {
            "is_trading": self.is_trading,
            "symbol": getattr(self, 'symbol', None),
            "amount": getattr(self, 'trade_amount', None),
            "amount_usdt": getattr(self, 'trade_amount', 0) * current_price if current_price > 0 else 0,
            "position": self.current_position,
            "trades_count": len(self.trading_history),
            "balance": balance_info.get('balance', {}) if balance_info.get('success') else {},
            "current_price": current_price,
            "risk_management": {
                "risk_percentage": 0.15,  # 15% от баланса
                "min_trade_usdt": 10.0,
                "max_trade_usdt": 100.0
            }
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
    
    def _calculate_trade_amount(self) -> float:
        """
        Рассчитывает количество для торговли на основе баланса и риск-менеджмента
        
        Returns:
            float: количество в BTC для торговли
        """
        try:
            # Получаем баланс
            balance_result = self.get_balance()
            if not balance_result.get('success'):
                logger.warning("Не удалось получить баланс, используем минимальное количество")
                return 0.001  # Минимальное количество
            
            usdt_balance = balance_result['balance']['USDT']
            btc_balance = balance_result['balance']['BTC']
            
            # Настройки риск-менеджмента
            risk_percentage = 0.15  # 15% от баланса на одну сделку
            min_trade_usdt = 10.0   # Минимальная сделка $10
            max_trade_usdt = 100.0  # Максимальная сделка $100
            
            # Рассчитываем количество USDT для торговли
            trade_usdt = usdt_balance * risk_percentage
            
            # Ограничиваем минимальным и максимальным значением
            trade_usdt = max(min_trade_usdt, min(trade_usdt, max_trade_usdt))
            
            # Если USDT недостаточно, используем BTC баланс
            if trade_usdt > usdt_balance:
                if btc_balance > 0.001:  # Минимум 0.001 BTC
                    trade_btc = btc_balance * risk_percentage
                    trade_btc = max(0.001, min(trade_btc, 0.01))  # Ограничиваем 0.001-0.01 BTC
                    logger.info(f"Используем BTC баланс: {trade_btc} BTC (${trade_btc * self._get_current_price():.2f})")
                    return trade_btc
                else:
                    logger.warning("Недостаточно средств для торговли")
                    return 0.001  # Минимальное количество
            
            # Конвертируем USDT в BTC
            current_price = self._get_current_price()
            if current_price > 0:
                trade_btc = trade_usdt / current_price
                logger.info(f"Рассчитано количество: {trade_btc:.6f} BTC (${trade_usdt:.2f})")
                return trade_btc
            else:
                logger.warning("Не удалось получить текущую цену, используем минимальное количество")
                return 0.001
                
        except Exception as e:
            logger.error(f"Ошибка расчета количества торговли: {e}")
            return 0.001  # Фолбэк на минимальное количество
    
    def _get_current_price(self) -> float:
        """Получает текущую цену из базы данных или с биржи"""
        try:
            # Сначала пробуем получить из БД
            from utils.db_utils import db_get_or_fetch_ohlcv
            
            df_5min = db_get_or_fetch_ohlcv(
                symbol_name=self.symbol,
                timeframe='5m',
                limit_candles=1  # Только последняя свеча
            )
            
            if df_5min is not None and not df_5min.empty:
                # Берем цену закрытия последней свечи
                current_price = df_5min['close'].iloc[-1]
                logger.debug(f"Цена из БД: ${current_price:.2f}")
                return current_price
            else:
                # Фолбэк на биржу
                ticker = self.exchange.fetch_ticker(self.symbol)
                current_price = ticker['last']
                logger.debug(f"Цена с биржи: ${current_price:.2f}")
                return current_price
                
        except Exception as e:
            logger.error(f"Ошибка получения цены: {e}")
            return 0.0
    
    def _get_current_balance(self) -> float:
        """Получает текущий баланс USDT"""
        try:
            balance_result = self.get_balance()
            if balance_result.get('success'):
                return balance_result['balance'].get('USDT', 0.0)
            return 0.0
        except Exception as e:
            logger.error(f"Ошибка получения баланса: {e}")
            return 0.0
    
    def _execute_trading_step(self) -> Dict:
        """
        Выполняет один торговый шаг (вызывается каждые 5 минут через Celery)
        
        Returns:
            Dict с результатом торгового шага
        """
        try:
            # Получаем текущие данные из БД (эффективнее чем с биржи)
            current_price = self._get_current_price()
            
            if current_price <= 0:
                logger.error("Не удалось получить текущую цену")
                return {
                    "error": "Не удалось получить текущую цену",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Обновляем количество для торговли (каждые 10 шагов)
            if not hasattr(self, '_trade_counter'):
                self._trade_counter = 0
            self._trade_counter += 1
            
            if self._trade_counter % 10 == 0:  # Каждые 50 минут (10 * 5 минут)
                new_amount = self._calculate_trade_amount()
                if abs(new_amount - self.trade_amount) > 0.0001:  # Если изменение больше 0.0001 BTC
                    logger.info(f"Обновляем количество торговли: {self.trade_amount:.6f} -> {new_amount:.6f} BTC")
                    self.trade_amount = new_amount
            
            # Получаем предсказание от модели на основе свежих данных
            action = self._get_model_prediction()
            
            # Логируем текущую цену и действие
            logger.info(f"Цена {self.symbol}: ${current_price:.2f}, Действие: {action}")
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "symbol": self.symbol,
                "price": current_price,
                "action": action,
                "trade_amount": self.trade_amount,
                "position": self.current_position
            }
            
            # Выполняем торговую операцию
            if action == 'buy' and not self.current_position:
                logger.info(f"🟢 Выполняем покупку {self.trade_amount} BTC по цене ${current_price:.2f}")
                buy_result = self._execute_buy()
                result["trade_executed"] = "buy"
                result["trade_details"] = buy_result
            elif action == 'sell' and self.current_position:
                logger.info(f"🔴 Выполняем продажу {self.current_position['amount']} BTC по цене ${current_price:.2f}")
                sell_result = self._execute_sell()
                result["trade_executed"] = "sell"
                result["trade_details"] = sell_result
            elif action == 'hold':
                if self.current_position:
                    # Показываем текущий P&L для открытой позиции
                    entry_price = self.current_position['entry_price']
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    logger.info(f"📊 Удерживаем позицию: P&L {pnl_pct:.2f}% (${current_price:.2f} vs ${entry_price:.2f})")
                    result["position_pnl"] = pnl_pct
                else:
                    logger.info(f"⏸️ Ожидаем сигнал для входа в позицию")
                result["trade_executed"] = "hold"
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка в торговом шаге: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_model_prediction(self) -> str:
        """Получение предсказания от модели"""
        try:
            if not self.model:
                logger.warning("Модель не загружена, возвращаем 'hold'")
                self.last_model_prediction = 'hold'
                return 'hold'
            
            # Получаем исторические данные для формирования состояния
            state = self._prepare_state_for_model()
            if state is None:
                logger.warning("Не удалось подготовить состояние, возвращаем 'hold'")
                self.last_model_prediction = 'hold'
                return 'hold'
            
            # Получаем предсказание от модели
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Добавляем batch dimension
                q_values = self.model(state_tensor)
                action = torch.argmax(q_values, dim=1).item()
            
            # Преобразуем действие в строку
            action_map = {0: 'hold', 1: 'buy', 2: 'sell'}
            action_str = action_map.get(action, 'hold')
            
            # Сохраняем последнее предсказание для записи в БД
            self.last_model_prediction = action_str
            
            logger.info(f"Предсказание модели: {action_str} (action={action}, q_values={q_values[0].tolist()})")
            return action_str
            
        except Exception as e:
            logger.error(f"Ошибка получения предсказания: {e}")
            self.last_model_prediction = 'hold'
            return 'hold'
    
    def _prepare_state_for_model(self) -> Optional[np.ndarray]:
        """
        Подготавливает состояние для модели так же, как во время обучения
        Использует базу данных для получения данных, докачивает недостающие
        
        Returns:
            np.ndarray: состояние для модели или None при ошибке
        """
        try:
            # Используем ту же логику, что и в train_dqn_symbol
            from utils.db_utils import db_get_or_fetch_ohlcv
            
            # Получаем данные из БД, докачиваем недостающие
            df_5min = db_get_or_fetch_ohlcv(
                symbol_name=self.symbol,
                timeframe='5m',
                limit_candles=100  # Нам нужно 100 свечей для индикаторов
            )
            
            if df_5min is None or df_5min.empty:
                logger.warning(f"Не удалось получить данные для {self.symbol}")
                return None
            
            if len(df_5min) < 50:  # Минимум 50 свечей для расчета индикаторов
                logger.warning(f"Недостаточно исторических данных: {len(df_5min)} свечей")
                return None
            
            logger.info(f"Получено {len(df_5min)} свечей из БД для {self.symbol}")
            
            # Преобразуем в numpy array (только OHLCV колонки)
            ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
            df_5min_array = df_5min[ohlcv_columns].values.astype(np.float32)
            
            # Подготавливаем данные как в окружении обучения
            state = self._create_state_from_ohlcv(df_5min_array)
            return state
            
        except Exception as e:
            logger.error(f"Ошибка подготовки состояния: {e}")
            return None
    
    def _create_state_from_ohlcv(self, df_5min: np.ndarray) -> np.ndarray:
        """
        Создает состояние из OHLCV данных, имитируя логику окружения обучения
        """
        try:
            # Конфигурация индикаторов (как в обучении)
            indicators_config = {
                'rsi': {'length': 14},
                'ema': {'lengths': [12, 26]},
                'sma': {'length': 20},
                'ema_cross': {'pairs': [(12, 26)]}
            }
            
            # Рассчитываем индикаторы
            from envs.dqn_model.gym.indicators_optimized import preprocess_dataframes
            
            # Создаем заглушки для 15m и 1h (используем 5m данные)
            df_15min = df_5min[::3]  # Каждая 3-я свеча из 5m
            df_1h = df_5min[::12]    # Каждая 12-я свеча из 5m
            
            # Подготавливаем данные
            df_5min_clean, df_15min_clean, df_1h_clean, indicators_array, individual_indicators = \
                preprocess_dataframes(df_5min, df_15min, df_1h, indicators_config)
            
            # Создаем состояние как в окружении
            lookback_window = 20  # Как в обучении
            current_step = len(df_5min_clean)
            
            if current_step < lookback_window:
                # Недостаточно данных, возвращаем нулевое состояние
                return np.zeros(100, dtype=np.float32)  # Примерный размер состояния
            
            # Берем последние lookback_window свечей
            start_idx = current_step - lookback_window
            end_idx = current_step
            
            # Получаем OHLCV данные для окна
            window_ohlcv = df_5min_clean[start_idx:end_idx]
            
            # Получаем индикаторы для окна
            window_indicators = indicators_array[start_idx:end_idx]
            
            # Нормализуем OHLCV данные
            normalized_ohlcv = self._normalize_ohlcv(window_ohlcv)
            
            # Объединяем OHLCV и индикаторы
            state_features = np.concatenate([
                normalized_ohlcv.flatten(),
                window_indicators.flatten()
            ], axis=0)
            
            # Добавляем информацию о позиции (как в окружении)
            position_info = np.array([
                0.0,  # Нормализованный баланс (пока 0)
                0.0   # Нормализованная криптовалюта (пока 0)
            ], dtype=np.float32)
            
            # Объединяем все в финальное состояние
            final_state = np.concatenate([state_features, position_info])
            
            # Убеждаемся, что размер правильный
            if len(final_state) > 100:
                final_state = final_state[:100]
            elif len(final_state) < 100:
                # Дополняем нулями
                padding = np.zeros(100 - len(final_state), dtype=np.float32)
                final_state = np.concatenate([final_state, padding])
            
            return final_state.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Ошибка создания состояния: {e}")
            return np.zeros(100, dtype=np.float32)
    
    def _normalize_ohlcv(self, ohlcv_data: np.ndarray) -> np.ndarray:
        """
        Нормализует OHLCV данные
        """
        try:
            # Простая нормализация: делим на максимальное значение
            max_values = np.max(ohlcv_data, axis=0)
            max_values = np.where(max_values == 0, 1, max_values)  # Избегаем деления на 0
            
            normalized = ohlcv_data / max_values
            return normalized
            
        except Exception as e:
            logger.error(f"Ошибка нормализации OHLCV: {e}")
            return ohlcv_data
    
    def _execute_buy(self) -> Dict:
        """Выполнение покупки"""
        try:
            # Получаем текущий баланс перед покупкой
            balance = self._get_current_balance()
            
            # Создаем запись о сделке в БД
            trade_record = create_trade_record(
                symbol_name=self.symbol,
                action='buy',
                status='pending',
                quantity=self.trade_amount,
                price=0,  # Будет обновлено после исполнения
                model_prediction=self.last_model_prediction,
                current_balance=balance,
                is_successful=False
            )
            
            # Выполняем покупку
            order = self.exchange.create_market_buy_order(
                self.symbol, 
                self.trade_amount
            )
            
            # Обновляем запись о сделке
            update_trade_status(
                trade_record.trade_number,
                status='executed',
                price=order['price'],
                exchange_order_id=order.get('id'),
                is_successful=True
            )
            
            self.current_position = {
                'type': 'long',
                'amount': self.trade_amount,
                'entry_price': order['price'],
                'entry_time': datetime.now(),
                'trade_number': trade_record.trade_number
            }
            
            self.trading_history.append({
                'action': 'buy',
                'price': order['price'],
                'amount': self.trade_amount,
                'time': datetime.now(),
                'trade_number': trade_record.trade_number
            })
            
            logger.info(f"Покупка выполнена: {order}, Trade #: {trade_record.trade_number}")
            
            return {
                "success": True,
                "order": order,
                "position": self.current_position,
                "trade_number": trade_record.trade_number
            }
            
        except Exception as e:
            logger.error(f"Ошибка покупки: {e}")
    
            # Обновляем запись о сделке с ошибкой
            if 'trade_record' in locals():
                update_trade_status(
                    trade_record.trade_number,
                    status='failed',
                    error_message=str(e),
                    is_successful=False
                )
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def _execute_sell(self) -> Dict:
        """Выполнение продажи"""
        try:
            # Получаем текущий баланс перед продажей
            balance = self._get_current_balance()
            
            # Создаем запись о сделке в БД
            trade_record = create_trade_record(
                symbol_name=self.symbol,
                action='sell',
                status='pending',
                quantity=self.current_position['amount'],
                price=0,  # Будет обновлено после исполнения
                model_prediction=self.last_model_prediction,
                current_balance=balance,
                is_successful=False
            )
            
            # Выполняем продажу
            order = self.exchange.create_market_sell_order(
                self.symbol, 
                self.current_position['amount']
            )
            
            # Расчет P&L
            exit_price = order['price']
            entry_price = self.current_position['entry_price']
            pnl = (exit_price - entry_price) * self.current_position['amount']
            
            # Обновляем запись о сделке
            update_trade_status(
                trade_record.trade_number,
                status='executed',
                price=order['price'],
                exchange_order_id=order.get('id'),
                position_pnl=pnl,
                is_successful=True
            )
            
            self.trading_history.append({
                'action': 'sell',
                'price': exit_price,
                'amount': self.current_position['amount'],
                'time': datetime.now(),
                'pnl': pnl,
                'trade_number': trade_record.trade_number
            })
            
            logger.info(f"Продажа выполнена: {order}, P&L: {pnl}, Trade #: {trade_record.trade_number}")
            
            old_position = self.current_position
            self.current_position = None
            
            return {
                "success": True,
                "order": order,
                "pnl": pnl,
                "closed_position": old_position,
                "trade_number": trade_record.trade_number
            }
            
        except Exception as e:
            logger.error(f"Ошибка продажи: {e}")
            
            # Обновляем запись о сделке с ошибкой
            if 'trade_record' in locals():
                update_trade_status(
                    trade_record.trade_number,
                    status='failed',
                    error_message=str(e),
                    is_successful=False
                )
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_trading_history(self) -> Dict:
        """Получение истории торговли"""
        return {
            "success": True,
            "trades": self.trading_history,
            "total_trades": len(self.trading_history)
        }

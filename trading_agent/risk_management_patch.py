# Патч для trading_agent.py
# Добавить эти функции перед _execute_buy (около строки 1620)

from typing import Dict, Optional, Tuple
import redis
import json
from datetime import datetime

def _get_risk_management_params(self) -> Tuple[float, float, str]:
    """
    Получает параметры риск-менеджмента из Redis
    
    Returns:
        Tuple[float, float, str]: (take_profit_pct, stop_loss_pct, risk_type)
    """
    try:
        rc = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
        take_profit_pct = float(rc.get('trading:take_profit_pct') or '5.0')
        stop_loss_pct = float(rc.get('trading:stop_loss_pct') or '3.0')
        risk_type = rc.get('trading:risk_management_type') or 'exchange_orders'
        return take_profit_pct, abs(stop_loss_pct), risk_type
    except Exception as e:
        logger.warning(f"Ошибка получения параметров риск-менеджмента: {e}")
        return 5.0, 3.0, 'exchange_orders'

def _cancel_existing_risk_orders(self):
    """Отменяет существующие ордера тейк-профита и стоп-лосса"""
    try:
        if not hasattr(self, '_risk_orders') or not self._risk_orders:
            return
        
        for order_type, order_id in self._risk_orders.items():
            try:
                self.exchange.cancel_order(order_id, self.symbol)
                logger.info(f"Отменен {order_type} ордер {order_id}")
            except Exception as e:
                logger.warning(f"Не удалось отменить {order_type} ордер {order_id}: {e}")
        
        self._risk_orders = {}
        
    except Exception as e:
        logger.warning(f"Ошибка отмены ордеров: {e}")

def _set_take_profit_stop_loss_orders(self, entry_price: float, amount: float) -> Dict:
    """
    Устанавливает ордера тейк-профита и стоп-лосса на бирже
    
    Args:
        entry_price: Цена входа в позицию
        amount: Количество для продажи
        
    Returns:
        Dict: Результат установки ордеров
    """
    try:
        take_profit_pct, stop_loss_pct, risk_type = self._get_risk_management_params()
        
        if risk_type not in ['exchange_orders', 'both']:
            logger.info(f"Риск-менеджмент через ордера отключен (type={risk_type})")
            return {"success": True, "message": "Exchange orders disabled"}
        
        # Отменяем существующие ордера
        self._cancel_existing_risk_orders()
        
        # Рассчитываем цены
        take_profit_price = entry_price * (1 + take_profit_pct / 100)
        stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
        
        # Нормализуем цены под tickSize биржи
        take_profit_price = self._normalize_price(take_profit_price)
        stop_loss_price = self._normalize_price(stop_loss_price)
        
        orders_result = {}
        self._risk_orders = {}
        
        # Устанавливаем ордер тейк-профита (лимитный ордер)
        try:
            tp_order = self.exchange.create_limit_sell_order(
                self.symbol,
                amount,
                take_profit_price,
                {
                    'reduceOnly': True,
                    'timeInForce': 'GTC',  # Good Till Cancelled
                    'postOnly': False
                }
            )
            self._risk_orders['take_profit'] = tp_order['id']
            orders_result['take_profit'] = {
                'order_id': tp_order['id'],
                'price': take_profit_price,
                'amount': amount
            }
            logger.info(f"✅ Тейк-профит установлен: {take_profit_price:.4f} (ордер {tp_order['id']})")
        except Exception as e:
            logger.error(f"❌ Ошибка установки тейк-профита: {e}")
            orders_result['take_profit_error'] = str(e)
        
        # Устанавливаем ордер стоп-лосса (стоп-маркет ордер)
        try:
            sl_order = self.exchange.create_stop_market_sell_order(
                self.symbol,
                amount,
                stop_loss_price,
                {
                    'reduceOnly': True,
                    'stopPrice': stop_loss_price
                }
            )
            self._risk_orders['stop_loss'] = sl_order['id']
            orders_result['stop_loss'] = {
                'order_id': sl_order['id'],
                'stop_price': stop_loss_price,
                'amount': amount
            }
            logger.info(f"🛡️ Стоп-лосс установлен: {stop_loss_price:.4f} (ордер {sl_order['id']})")
        except Exception as e:
            logger.error(f"❌ Ошибка установки стоп-лосса: {e}")
            orders_result['stop_loss_error'] = str(e)
        
        # Сохраняем информацию в Redis для мониторинга
        try:
            rc = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
            risk_info = {
                'entry_price': entry_price,
                'take_profit_price': take_profit_price,
                'stop_loss_price': stop_loss_price,
                'take_profit_pct': take_profit_pct,
                'stop_loss_pct': stop_loss_pct,
                'orders': self._risk_orders,
                'timestamp': datetime.now().isoformat()
            }
            rc.set(f'trading:risk_orders:{self.symbol}', json.dumps(risk_info))
        except Exception as e:
            logger.warning(f"Не удалось сохранить информацию о рисках: {e}")
        
        return {
            "success": True,
            "orders": orders_result,
            "take_profit_price": take_profit_price,
            "stop_loss_price": stop_loss_price
        }
        
    except Exception as e:
        logger.error(f"Критическая ошибка установки ордеров: {e}")
        return {"success": False, "error": str(e)}

def _normalize_price(self, price: float) -> float:
    """Нормализует цену под требования биржи (tickSize)"""
    try:
        # Получаем информацию о рынке
        market = self.exchange.market(self.symbol)
        tick_size = market.get('precision', {}).get('price', 0.01)
        
        # Округляем цену до нужного шага
        normalized = round(price / tick_size) * tick_size
        return normalized
    except Exception as e:
        logger.warning(f"Ошибка нормализации цены: {e}")
        # Используем стандартное округление до 4 знаков
        return round(price, 4)

def _check_take_profit_stop_loss(self) -> Optional[Dict]:
    """
    Резервная проверка условий тейк-профита и стоп-лосса (для режима bot_monitoring или both)
    
    Returns:
        Dict: Результат автоматической продажи или None
    """
    try:
        if not self.current_position:
            return None
        
        take_profit_pct, stop_loss_pct, risk_type = self._get_risk_management_params()
        
        if risk_type not in ['bot_monitoring', 'both']:
            return None  # Мониторинг ботом отключен
        
        current_price = self._get_current_price()
        if current_price <= 0:
            return None
        
        entry_price = self.current_position['entry_price']
        pnl_percentage = ((current_price - entry_price) / entry_price) * 100
        
        should_sell = False
        reason = ""
        
        # Проверяем стоп-лосс
        if pnl_percentage <= -stop_loss_pct:
            should_sell = True
            reason = f"🛡️ Резервный стоп-лосс: убыток {pnl_percentage:.2f}% превысил порог -{stop_loss_pct:.2f}%"
        
        # Проверяем тейк-профит
        elif pnl_percentage >= take_profit_pct:
            should_sell = True
            reason = f"🎯 Резервный тейк-профит: прибыль {pnl_percentage:.2f}% достигла цели {take_profit_pct:.2f}%"
        
        if should_sell:
            logger.info(f"Резервная автоматическая продажа: {reason}")
            # Отменяем ордера перед продажей
            self._cancel_existing_risk_orders()
            return self._execute_sell()
        
        return None
        
    except Exception as e:
        logger.warning(f"Ошибка резервной проверки тейк-профита/стоп-лосса: {e}")
        return None

# ИНСТРУКЦИИ ПО ПРИМЕНЕНИЮ:

# 1. В __init__ добавить:
# self._risk_orders = {}  # Словарь для отслеживания ордеров риск-менеджмента

# 2. В _execute_buy после строки "logger.info(f"Покупка выполнена: {order}, Trade #: {trade_record.trade_number}")":
# Добавить:
# 
# # Устанавливаем ордера тейк-профита и стоп-лосса
# try:
#     risk_orders_result = self._set_take_profit_stop_loss_orders(executed_price, amount)
#     if risk_orders_result.get('success'):
#         logger.info(f"Ордера риск-менеджмента установлены: TP={risk_orders_result.get('take_profit_price')}, SL={risk_orders_result.get('stop_loss_price')}")
#     else:
#         logger.warning(f"Проблема с установкой ордеров: {risk_orders_result.get('error', 'Unknown')}")
# except Exception as e:
#     logger.error(f"Критическая ошибка установки ордеров риск-менеджмента: {e}")

# 3. В make_trading_decision добавить в начале:
# 
# # Проверяем резервные условия тейк-профита/стоп-лосса
# if self.current_position:
#     backup_sell_result = self._check_take_profit_stop_loss()
#     if backup_sell_result:
#         return 'sell', backup_sell_result

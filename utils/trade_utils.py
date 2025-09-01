import uuid
from datetime import datetime
from sqlalchemy.orm import Session, joinedload
from orm.models import Trade, Symbol
from orm.database import get_db_session

def generate_trade_number():
    """Генерирует уникальный номер сделки"""
    return f"TRADE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

def get_or_create_symbol(session: Session, symbol_name: str) -> Symbol:
    """Получает или создает символ в базе данных"""
    symbol = session.query(Symbol).filter(Symbol.name == symbol_name).first()
    if not symbol:
        symbol = Symbol(name=symbol_name)
        session.add(symbol)
        session.commit()
        session.refresh(symbol)
    return symbol

def create_trade_record(
    symbol_name: str,
    action: str,
    status: str,
    quantity: float,
    price: float,
    model_prediction: str = None,
    confidence: float = None,
    current_balance: float = None,
    position_pnl: float = None,
    exchange_order_id: str = None,
    error_message: str = None,
    is_successful: bool = False
) -> Trade:
    """
    Создает запись о сделке в базе данных
    
    Args:
        symbol_name: Название торговой пары (например, 'BTC/USDT')
        action: Действие ('buy', 'sell', 'hold')
        status: Статус сделки ('executed', 'pending', 'cancelled', 'failed')
        quantity: Количество торгуемого актива
        price: Цена исполнения
        model_prediction: Предсказание модели
        confidence: Уверенность модели
        current_balance: Текущий баланс
        position_pnl: P&L позиции
        exchange_order_id: ID ордера на бирже
        error_message: Сообщение об ошибке
        is_successful: Успешность сделки
    
    Returns:
        Trade: Созданная запись о сделке
    """
    session = get_db_session()
    try:
        # Получаем или создаем символ
        symbol = get_or_create_symbol(session, symbol_name)
        
        # Создаем запись о сделке
        trade = Trade(
            trade_number=generate_trade_number(),
            symbol_id=symbol.id,
            action=action,
            status=status,
            quantity=quantity,
            price=price,
            total_value=quantity * price,
            model_prediction=model_prediction,
            confidence=confidence,
            current_balance=current_balance,
            position_pnl=position_pnl,
            exchange_order_id=exchange_order_id,
            error_message=error_message,
            is_successful=is_successful,
            executed_at=datetime.utcnow() if status == 'executed' else None
        )
        
        session.add(trade)
        session.commit()
        session.refresh(trade)
        
        return trade
        
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def update_trade_status(trade_number: str, status: str, **kwargs):
    """
    Обновляет статус сделки и дополнительные поля
    
    Args:
        trade_number: Номер сделки
        status: Новый статус
        **kwargs: Дополнительные поля для обновления
    """
    session = get_db_session()
    try:
        trade = session.query(Trade).filter(Trade.trade_number == trade_number).first()
        if trade:
            trade.status = status
            if status == 'executed':
                trade.executed_at = datetime.utcnow()
                trade.is_successful = True
            
            # Обновляем дополнительные поля
            for key, value in kwargs.items():
                if hasattr(trade, key):
                    setattr(trade, key, value)
            
            session.commit()
            return trade
        else:
            raise ValueError(f"Сделка с номером {trade_number} не найдена")
            
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_trades_by_symbol(symbol_name: str, limit: int = 100):
    """
    Получает сделки по символу
    
    Args:
        symbol_name: Название торговой пары
        limit: Максимальное количество записей
    
    Returns:
        List[Trade]: Список сделок
    """
    session = get_db_session()
    try:
        symbol = session.query(Symbol).filter(Symbol.name == symbol_name).first()
        if symbol:
            trades = session.query(Trade).options(
                joinedload(Trade.symbol)
            ).filter(
                Trade.symbol_id == symbol.id
            ).order_by(Trade.created_at.desc()).limit(limit).all()
            return trades
        return []
        
    finally:
        session.close()

def get_recent_trades(limit: int = 50):
    """
    Получает последние сделки
    
    Args:
        limit: Максимальное количество записей
    
    Returns:
        List[Trade]: Список сделок
    """
    session = get_db_session()
    try:
        trades = session.query(Trade).options(
            joinedload(Trade.symbol)
        ).order_by(Trade.created_at.desc()).limit(limit).all()
        return trades
    finally:
        session.close()

def get_trade_statistics(symbol_name: str = None):
    """
    Получает статистику по сделкам
    
    Args:
        symbol_name: Название торговой пары (опционально)
    
    Returns:
        dict: Статистика сделок
    """
    session = get_db_session()
    try:
        query = session.query(Trade)
        
        if symbol_name:
            symbol = session.query(Symbol).filter(Symbol.name == symbol_name).first()
            if symbol:
                query = query.filter(Trade.symbol_id == symbol.id)
            else:
                return {
                    'total_trades': 0,
                    'successful_trades': 0,
                    'failed_trades': 0,
                    'total_volume': 0,
                    'total_value': 0,
                    'success_rate': 0
                }
        
        total_trades = query.count()
        successful_trades = query.filter(Trade.is_successful == True).count()
        failed_trades = query.filter(Trade.is_successful == False).count()
        
        # Суммы по объему и стоимости
        volume_sum = session.query(Trade.quantity).filter(
            Trade.is_successful == True
        ).scalar() or 0
        value_sum = session.query(Trade.total_value).filter(
            Trade.is_successful == True
        ).scalar() or 0
        
        success_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'successful_trades': successful_trades,
            'failed_trades': failed_trades,
            'total_volume': volume_sum,
            'total_value': value_sum,
            'success_rate': round(success_rate, 2)
        }
        
    finally:
        session.close()

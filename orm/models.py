from sqlalchemy import (
    create_engine, Column, Integer, Float, String, BigInteger,
    ForeignKey, UniqueConstraint, DateTime, Text, Boolean
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()

class Symbol(Base):
    __tablename__ = 'symbols'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)  # Например, 'BTC/USDT'

    # Связь "один ко многим" с OHLCV
    ohlcvs = relationship("OHLCV", back_populates="symbol")
    # Связь "один ко многим" с Trade
    trades = relationship("Trade", back_populates="symbol")

class OHLCV(Base):
    __tablename__ = 'ohlcv'
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id'), nullable=False)
    timeframe = Column(String, nullable=False)  # '5m', '15m', '1h' и т.п.
    timestamp = Column(BigInteger, nullable=False)  # UNIX время в мс
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

    symbol = relationship("Symbol", back_populates="ohlcvs")

    __table_args__ = (
        UniqueConstraint('symbol_id', 'timeframe', 'timestamp', name='uix_symbol_timeframe_timestamp'),
    )

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_number = Column(String, unique=True, nullable=False)  # Уникальный номер сделки
    symbol_id = Column(Integer, ForeignKey('symbols.id'), nullable=False)
    
    # Основные данные сделки
    action = Column(String, nullable=False)  # 'buy', 'sell', 'hold'
    status = Column(String, nullable=False)  # 'executed', 'pending', 'cancelled', 'failed'
    quantity = Column(Float, nullable=False)  # Количество торгуемого актива
    price = Column(Float, nullable=False)  # Цена исполнения
    total_value = Column(Float, nullable=False)  # Общая стоимость сделки (quantity * price)
    
    # Дополнительные данные
    model_prediction = Column(String, nullable=True)  # Предсказание модели
    confidence = Column(Float, nullable=True)  # Уверенность модели (если доступна)
    current_balance = Column(Float, nullable=True)  # Баланс на момент сделки
    position_pnl = Column(Float, nullable=True)  # P&L позиции
    
    # Метаданные
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    executed_at = Column(DateTime, nullable=True)  # Время исполнения
    exchange_order_id = Column(String, nullable=True)  # ID ордера на бирже
    error_message = Column(Text, nullable=True)  # Сообщение об ошибке, если есть
    
    # Флаги
    is_successful = Column(Boolean, default=False)  # Успешность сделки
    
    # Связи
    symbol = relationship("Symbol", back_populates="trades")
    
    __table_args__ = (
        UniqueConstraint('trade_number', name='uix_trade_number'),
    )


class ModelPrediction(Base):
    __tablename__ = 'model_predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String, nullable=False)
    action = Column(String, nullable=False)  # buy, sell, hold
    q_values = Column(Text)  # JSON string с Q-values
    current_price = Column(Float)
    position_status = Column(String)  # open, closed, none
    confidence = Column(Float)  # max(q_values) - уверенность модели
    model_path = Column(String)  # путь к модели
    market_conditions = Column(Text)  # JSON с условиями рынка (RSI, EMA и т.д.)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ModelPrediction(symbol='{self.symbol}', action='{self.action}', confidence={self.confidence:.3f}, timestamp='{self.timestamp}')>"


class FundingRate(Base):
    __tablename__ = 'funding_rates'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)  # Например, 'BTCUSDT' или 'BTC/USDT'
    timestamp = Column(BigInteger, nullable=False)  # UNIX время в мс времени применения ставки
    rate = Column(Float, nullable=False)  # В долях (0.0001 == 1 bp)
    source = Column(String, nullable=True)  # bybit_v5/ccxt/etc

    __table_args__ = (
        UniqueConstraint('symbol', 'timestamp', name='uix_funding_symbol_timestamp'),
    )
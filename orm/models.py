from sqlalchemy import (
    create_engine, Column, Integer, Float, String, BigInteger,
    ForeignKey, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()

class Symbol(Base):
    __tablename__ = 'symbols'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)  # Например, 'BTC/USDT'

    # Связь "один ко многим" с OHLCV
    ohlcvs = relationship("OHLCV", back_populates="symbol")

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

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


class MarketIndicator(Base):
    """OI (Open Interest) и Long/Short ratio от Bybit, привязанные к 5m-баркам."""
    __tablename__ = 'market_indicators'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id'), nullable=False)
    timeframe = Column(String, nullable=False)
    timestamp = Column(BigInteger, nullable=False)  # UNIX ms
    open_interest = Column(Float, nullable=True)
    buy_ratio = Column(Float, nullable=True)   # long ratio (0..1)
    sell_ratio = Column(Float, nullable=True)  # short ratio (0..1)

    symbol = relationship("Symbol")

    __table_args__ = (
        UniqueConstraint('symbol_id', 'timeframe', 'timestamp', name='uix_mkt_ind_sym_tf_ts'),
    )


class AppSetting(Base):
    """
    Универсальные настройки приложения (в т.ч. secret).

    Ключ уникален в рамках (scope, group, key).
    """
    __tablename__ = 'app_settings'

    id = Column(Integer, primary_key=True, autoincrement=True)

    scope = Column(String, nullable=False)                 # например: analyzer / api / trading
    group = Column(String, nullable=True)                  # опционально: bybit / stage1 / etc
    key = Column(String, nullable=False)                   # например: ANALYZER_MAX_POOLS_PER_SYMBOL

    value_type = Column(String, nullable=False, default='string')  # string/number/bool/json
    label = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    is_secret = Column(Boolean, nullable=False, default=False)

    value = Column(Text, nullable=True)

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint('scope', 'group', 'key', name='uix_app_settings_scope_group_key'),
    )


class BotUser(Base):
    """
    Человек, который пользуется ботами MedoedAI.

    Не привязан к Telegram напрямую: один user может иметь identity в telegram,
    max или другой платформе.
    """
    __tablename__ = 'bot_users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    status = Column(String, nullable=False, default='active')  # active/blocked/deleted
    role = Column(String, nullable=False, default='user')      # user/admin/support

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_seen_at = Column(DateTime, nullable=True)

    identities = relationship("BotUserIdentity", back_populates="user")
    subscriptions = relationship("BotSubscription", back_populates="user")


class BotUserIdentity(Base):
    """
    Аккаунт пользователя на конкретной платформе: telegram/max/etc.
    platform_user_id должен быть строкой, потому что у разных платформ разные форматы ID.
    """
    __tablename__ = 'bot_user_identities'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('bot_users.id'), nullable=False)

    platform = Column(String, nullable=False)          # telegram / max / etc
    platform_user_id = Column(String, nullable=False)  # telegram_id, max user id, etc
    username = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    display_name = Column(String, nullable=True)
    language_code = Column(String, nullable=True)
    raw_profile = Column(Text, nullable=True)          # JSON snapshot от платформы

    bybit_api_key = Column(String, nullable=True)
    bybit_api_secret = Column(String, nullable=True)
    bybit_leverage = Column(Integer, nullable=True, default=1)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_seen_at = Column(DateTime, nullable=True)

    user = relationship("BotUser", back_populates="identities")

    __table_args__ = (
        UniqueConstraint('platform', 'platform_user_id', name='uix_bot_identity_platform_user'),
    )


class BotPromoCode(Base):
    """Выпущенные промокоды для доступа к боту."""
    __tablename__ = 'bot_promo_codes'

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String, unique=True, nullable=False)    # Например: MEDOED-A7F3-K2
    duration_days = Column(Integer, nullable=False)       # На сколько дней выдается доступ
    max_uses = Column(Integer, nullable=False, default=1) # Лимит активаций
    used_count = Column(Integer, nullable=False, default=0)
    valid_from = Column(DateTime, nullable=True)          # С какой даты действует
    valid_until = Column(DateTime, nullable=True)         # До какой даты можно активировать
    is_active = Column(Boolean, nullable=False, default=True) # Вкл/выкл вручную
    note = Column(String, nullable=True)                  # Комментарий для себя

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    redemptions = relationship("BotPromoRedemption", back_populates="promo_code")


class BotPromoRedemption(Base):
    """Аудит активаций промокодов пользователями."""
    __tablename__ = 'bot_promo_redemptions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    promo_code_id = Column(Integer, ForeignKey('bot_promo_codes.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('bot_users.id'), nullable=False)
    
    paid_until_after = Column(DateTime, nullable=False)   # До какой даты продлили
    redeemed_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    promo_code = relationship("BotPromoCode", back_populates="redemptions")
    user = relationship("BotUser")

    __table_args__ = (
        UniqueConstraint('promo_code_id', 'user_id', name='uix_promo_redemption_code_user'),
    )


class BotSubscription(Base):
    """Платный доступ пользователя к продукту/функции."""
    __tablename__ = 'bot_subscriptions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('bot_users.id'), nullable=False)

    product_code = Column(String, nullable=False, default='signals')  # signals/pro/etc
    plan_code = Column(String, nullable=False, default='monthly')
    status = Column(String, nullable=False, default='inactive')       # active/inactive/canceled/expired
    paid_until = Column(DateTime, nullable=True)

    provider = Column(String, nullable=True)              # telegram_stars/yookassa/manual/etc
    provider_payment_id = Column(String, nullable=True)
    provider_subscription_id = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("BotUser", back_populates="subscriptions")

    __table_args__ = (
        UniqueConstraint('user_id', 'product_code', name='uix_bot_subscription_user_product'),
    )
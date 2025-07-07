from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, Symbol, OHLCV  # импорт из твоего файла с моделями
from datetime import datetime

engine = create_engine("postgresql://medoed_user:medoed@postgres:5432/medoed_db")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def get_session():
    return Session()

def add_symbol(session, symbol_name):
    sym = session.query(Symbol).filter_by(name=symbol_name).first()
    if not sym:
        sym = Symbol(name=symbol_name)
        session.add(sym)
        session.commit()
    return sym

def add_ohlcv(session, symbol_obj, timeframe, timestamp, open_, high, low, close, volume):
    existing = session.query(OHLCV).filter_by(
        symbol_id=symbol_obj.id,
        timeframe=timeframe,
        timestamp=timestamp
    ).first()

    if existing:
        existing.open = open_
        existing.high = high
        existing.low = low
        existing.close = close
        existing.volume = volume
    else:
        new_candle = OHLCV(
            symbol_id=symbol_obj.id,
            timeframe=timeframe,
            timestamp=timestamp,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume
        )
        session.add(new_candle)
    session.commit()
    
    
def db_get_or_fetch_ohlcv(symbol, timeframe, episode_length=10000):
    session = Session()
    # Проверяем, есть ли символ в базе
    symbol_obj = session.query(Symbol).filter(Symbol.name == symbol).first()
    if not symbol_obj:
        symbol_obj = Symbol(name=symbol)
        session.add(symbol_obj)
        session.commit()
        print(f"Добавлен новый символ в базу: {symbol}")
    else:
        print(f"Символ {symbol} уже есть в базе")

    # Получаем последнюю свечу из базы для данного таймфрейма
    last_candle = (
        session.query(OHLCV)
        .filter(OHLCV.symbol_id == symbol_obj.id, OHLCV.timeframe == timeframe)
        .order_by(OHLCV.timestamp.desc())
        .first()
    )

    now = int(datetime.utcnow().timestamp() * 1000)  # текущий момент в ms

    # Определяем длительность таймфрейма в миллисекундах
    if timeframe.endswith('m'):
        tf_ms = int(timeframe.replace('m', '')) * 60 * 1000
    elif timeframe.endswith('h'):
        tf_ms = int(timeframe.replace('h', '')) * 60 * 60 * 1000
    elif timeframe.endswith('d'):
        tf_ms = int(timeframe.replace('d', '')) * 24 * 60 * 60 * 1000
    else:
        tf_ms = 60 * 1000  # по умолчанию 1 минута

    lookback_ms = episode_length * tf_ms
    start_time = now - lookback_ms

    if last_candle:
        last_timestamp = last_candle.timestamp
        # fetch_start должен быть выровнен по таймфрейму
        if last_timestamp > start_time:
            fetch_start = start_time - (start_time % tf_ms)
        else:
            fetch_start = last_timestamp + tf_ms
    else:
        fetch_start = start_time - (start_time % tf_ms)

    # Догружаем недостающие свечи, если нужно
    if fetch_start < now:
        new_data = fetch_ohlcv('bybit', symbol, timeframe, since=fetch_start, limit=episode_length)
        for _, row in new_data.iterrows():
            candle = OHLCV(
                symbol_id=symbol_obj.id,
                timeframe=timeframe,
                timestamp=int(row['timestamp']),
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
            )
            session.merge(candle)
        session.commit()

    # Получаем последние episode_length свечей из базы
    candles = (
        session.query(OHLCV)
        .filter(OHLCV.symbol_id == symbol_obj.id, OHLCV.timeframe == timeframe)
        .order_by(OHLCV.timestamp.desc())
        .limit(episode_length)
        .all()
    )
    # Преобразуем в DataFrame
    df = pd.DataFrame([{
        'timestamp': c.timestamp,
        'open': c.open,
        'high': c.high,
        'low': c.low,
        'close': c.close,
        'volume': c.volume
    } for c in reversed(candles)])
    return df    
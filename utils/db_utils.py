import os
import pandas as pd
import ccxt
from datetime import datetime, timezone, timedelta

from sqlalchemy import create_engine, func, cast
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.expression import or_
from sqlalchemy import Numeric # Import Numeric for casting

from sqlalchemy.dialects.postgresql import insert

# Import your models from model.py
from orm.models import Base, Symbol, OHLCV

import logging

from utils.cctx_utils import normalize_symbol

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import math # Для isclose
import time

# --- Database Engine and Session setup ---
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://medoed_user:medoed@postgres:5432/medoed_db")
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

def get_db_session():
    """Returns a new SQLAlchemy session."""
    db = Session()
    try:
        yield db
    finally:
        db.close()

def add_symbol(session, symbol_name):
    """Adds a new symbol to the database if it doesn't exist."""
    sym = session.query(Symbol).filter_by(name=symbol_name).first()
    if not sym:
        sym = Symbol(name=symbol_name)
        session.add(sym)
        session.commit()
        session.refresh(sym) # Refresh to get the ID
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
    

def db_get_or_fetch_ohlcv(
    symbol_name: str,
    timeframe: str,
    limit_candles: int = 1000,
    exchange_id: str = 'binance',
    max_retries: int = 3,
    csv_file_path: str ='./temp/binance_data/bigdata_5m_klines.csv',
    dry_run: bool = False # Если True, только загружает, но не сохраняет в БД
                        ) -> pd.DataFrame:
    """
    Загружает OHLCV данные из базы данных. Если данных недостаточно или есть разрывы,
    догружает их с биржи и сохраняет. Включает проверку на разрывы во временных метках.

    Args:
        symbol_name (str): Имя торговой пары (например, 'BTC/USDT').
        timeframe (str): Таймфрейм (например, '1m', '5m', '1h').
        limit_candles (int): Желаемое количество свечей для возврата.
        exchange_id (str): ID биржи для загрузки данных.
        max_retries (int): Максимальное количество попыток загрузки с биржи.
        dry_run (bool): Если True, данные загружаются, но не сохраняются в БД.
                        Полезно для проверки без изменения БД.

    Returns:
        pd.DataFrame: DataFrame с OHLCV данными, отсортированными по timestamp.
                      Может быть пустым, если данные не удалось загрузить.
    """
    session = next(get_db_session())
    df = pd.DataFrame()
    exchange = None # Инициализируем exchange вне try-блока

    try:
        # Получаем или создаем запись символа
        
        # 1. Получить или создать символ
        symbol_obj = _get_or_create_symbol(session, symbol_name)

        total_count = session.query(OHLCV).filter_by(
            symbol_id=symbol_obj.id,
            timeframe=timeframe
        ).count()

        print(f"Всего свечей в БД: {total_count}")

        last_db_candle = session.query(OHLCV).filter_by(
            symbol_id=symbol_obj.id,
            timeframe=timeframe
        ).order_by(OHLCV.timestamp.desc()).first()

        if last_db_candle:
            last_db_timestamp = last_db_candle.timestamp
            logging.info(f"Последняя свеча в БД по {symbol_name}, {timeframe} имеет timestamp {last_db_timestamp} ({datetime.fromtimestamp(last_db_timestamp/1000)})")
        else:
            last_db_timestamp = None
            logging.info(f"Свечей в базе для {symbol_name}, {timeframe} нет. Начинаем загрузку с нуля (30 дней назад).")

        # Инициализация биржи
        try:            
            
            symbol_cctx = normalize_symbol(symbol_name)
            
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                'enableRateLimit': True,
                'timeout': 30000, # Увеличение таймаута
            })
            exchange.load_markets()
            if symbol_cctx not in exchange.symbols:
                raise ValueError(f"Символ {symbol_cctx} не найден на бирже {exchange_id}.")
            logging.info(f"Биржа {exchange_id} успешно инициализирована.")
        except Exception as e:
            logging.error(f"Не удалось инициализировать биржу {exchange_id}: {e}")
            return pd.DataFrame()


        tf_ms = exchange.parse_timeframe(timeframe) * 1000

        # Определяем from какого момента качать данные
        if last_db_timestamp:
            since_ms = last_db_timestamp + tf_ms
        else:
            since_ms = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)
            
        # Если since_ms в будущем — корректируем
        now_ms = exchange.milliseconds()
        if since_ms > now_ms:
            logging.warning("Начальная дата больше текущей. Сбрасываем на текущий момент.")
            since_ms = now_ms - tf_ms

        all_new_data = []
        retries = 0
        # Докачиваем данные, пока не дойдём до сейчас или не наберём достаточно
        while True:
            try:
                limit_fetch = min(exchange.options.get('fetchOHLCVLimit', 1000), 1000)
                ohlcv = exchange.fetch_ohlcv(symbol_cctx, timeframe, since=since_ms, limit=limit_fetch)

                if not ohlcv:
                    logging.info("Новых данных с биржи нет. Докачка завершена.")
                    break

                new_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

                # Отфильтровываем свечи, которые уже есть в БД
                if last_db_timestamp:
                    new_df = new_df[new_df['timestamp'] > last_db_timestamp]

                if new_df.empty:
                    logging.info("Все загруженные свечи уже есть в БД.")
                    break

                all_new_data.extend(new_df.to_dict('records'))

                last_ts_in_batch = new_df['timestamp'].max()
                since_ms = last_ts_in_batch + tf_ms

                # Если докачали достаточно (по времени или по объёму), прекращаем
                if since_ms > now_ms:
                    logging.info("Достигнуто текущее время. Докачка завершена.")
                    break

                # Ждём, чтобы не превысить лимиты API
                time.sleep(getattr(exchange, 'rateLimit', 1000) / 1000)

            except ccxt.RateLimitExceeded:
                logging.warning("Превышен лимит запросов, ждем 30 секунд.")
                time.sleep(30)
                retries += 1
                if retries > max_retries:
                    logging.error("Максимум попыток превышен, прекращаем докачку.")
                    break
            except Exception as e:
                logging.error(f"Ошибка при загрузке OHLCV: {e}", exc_info=True)
                retries += 1
                time.sleep(5)
                if retries > max_retries:
                    logging.error("Максимум попыток превышен, прекращаем докачку.")
                    break

        # Если есть новые данные — добавляем в БД
        if all_new_data:
            new_data_df = pd.DataFrame(all_new_data)
            # Загружаем из базы все последние limit_candles, чтобы обновить итоговый DataFrame
            if not dry_run:
                existing_timestamps = set(
                    r[0] for r in session.query(OHLCV.timestamp).filter_by(
                        symbol_id=symbol_obj.id,
                        timeframe=timeframe
                    ).filter(OHLCV.timestamp.in_(new_data_df['timestamp'].tolist())).all()
                )
                new_records = [
                    {
                        'symbol_id': symbol_obj.id,
                        'timeframe': timeframe,
                        'timestamp': int(row['timestamp']),
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    }
                    for _, row in new_data_df.iterrows() if int(row['timestamp']) not in existing_timestamps
                ]
                if new_records:
                    session.bulk_insert_mappings(OHLCV, new_records)
                    session.commit()
                    logging.info(f"Добавлено {len(new_records)} новых свечей {timeframe} для {symbol_name}.")
                else:
                    logging.info("Новых свечей для добавления не найдено.")
            else:
                logging.info(f"Dry run: {len(new_data_df)} свечей не сохранены в БД.")

        # В итоге загружаем из БД последние limit_candles свечей
        db_candles = session.query(OHLCV).filter_by(
            symbol_id=symbol_obj.id,
            timeframe=timeframe
        ).order_by(OHLCV.timestamp.desc()).limit(limit_candles).all()

        df = pd.DataFrame([{
            'timestamp': c.timestamp,
            'open': c.open,
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume
        } for c in reversed(db_candles)])

        logging.info(f"Итоговое количество свечей для {symbol_name} {timeframe}: {len(df)}")

        # --- Финальная проверка на разрывы во всем загруженном DataFrame ---
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
            tf_ms = exchange.parse_timeframe(timeframe) * 1000
            
            # Проверка на разрывы во времени (ожидаем tf_ms между последовательными свечами)
            # Добавим небольшой допуск, например, 10% от tf_ms
            allowed_gap_tolerance = 0.1 * tf_ms 

            # Проверка разницы между текущей свечой и предыдущей
            time_diffs = df['timestamp'].diff().iloc[1:]
            
            # Проверяем, что разница больше ожидаемого таймфрейма + допуск ИЛИ разница отрицательна (неправильный порядок)
            anomalous_gaps_mask = (time_diffs > (tf_ms + allowed_gap_tolerance)) | (time_diffs < (tf_ms - allowed_gap_tolerance * 5)) 
            
            if any(anomalous_gaps_mask):
                gap_indices = time_diffs[anomalous_gaps_mask].index.tolist()
                
                logging.warning(f"Обнаружены аномальные разрывы/наложения во времени в загруженных данных {symbol_name}, {timeframe}:")
                for idx in gap_indices:
                    prev_row = df.iloc[idx - 1]
                    curr_row = df.iloc[idx]
                    
                    gap_duration_ms = curr_row['timestamp'] - prev_row['timestamp']
                    
                    logging.warning(f"Разрыв/Наложение между свечами с индексами {idx-1} и {idx}:")
                    logging.warning(f"  Длительность: {gap_duration_ms/1000:.2f} сек (ожидается {tf_ms/1000:.2f} сек)")
                    logging.warning(f"  Предыдущая: {datetime.fromtimestamp(prev_row['timestamp']/1000)} (Close: {prev_row['close']:.2f})")
                    logging.warning(f"  Текущая: {datetime.fromtimestamp(curr_row['timestamp']/1000)} (Open: {curr_row['open']:.2f})")
                    
                logging.info(f"Проверка целостности данных {symbol_name}, {timeframe} завершена: обнаружены разрывы.")
            else:
                logging.info(f"Проверка целостности данных {symbol_name}, {timeframe} завершена: разрывы не обнаружены.")

        return df
    except Exception as e:
        logging.error(f"Критическая ошибка в db_get_or_fetch_ohlcv: {e}", exc_info=True)
        return pd.DataFrame()
    finally:
        session.close()


def _get_or_create_symbol(session, symbol_name: str) -> Symbol:
    """Получает или создает объект символа в базе данных."""
    symbol_obj = session.query(Symbol).filter_by(name=symbol_name).first()
    if not symbol_obj:
        symbol_obj = Symbol(name=symbol_name)
        session.add(symbol_obj)
        session.commit()
        session.refresh(symbol_obj)
        logging.info(f"Добавлен новый символ в БД: {symbol_name}")
    return symbol_obj

def clean_ohlcv_data(timeframe: str,
                     symbol_name: str = None,
                     max_close_change_percent: float = 15.0, # Максимальное % изменение цены (Open-Close)
                     max_hl_range_percent: float = 20.0,     # Максимальный % диапазон (High-Low)
                     volume_multiplier: float = 30.0,        # Максимальный объем (от среднего)
                     round_precision: int = 6,               # Точность округления для финансовых данных
                     epsilon: float = 1e-9):                  # Малый допуск для сравнений с плавающей точкой
    """
    Очищает OHLCV данные от аномалий в PostgreSQL с использованием SQLAlchemy.
    Обрезает аномальные значения внутри свечей, вместо их удаления.
    Это НЕ Celery-задача.
    """
    print(f"Начало очистки данных для {symbol_name if symbol_name else 'всех символов'} ({timeframe})...")
    
    db = next(get_db_session()) # Get a session
    
    try:
        symbol_id_filter = None
        if symbol_name:
            symbol_obj = db.query(Symbol).filter(Symbol.name == symbol_name).first()
            if not symbol_obj:
                raise ValueError(f"Символ '{symbol_name}' не найден в базе данных.")
            symbol_id_filter = symbol_obj.id
        
        # 1. Рассчитываем средний объем для данного таймфрейма и символа (если указан)
        avg_volume_query = db.query(func.avg(OHLCV.volume)).filter(OHLCV.timeframe == timeframe, OHLCV.volume > 0)
        if symbol_id_filter is not None:
            avg_volume_query = avg_volume_query.filter(OHLCV.symbol_id == symbol_id_filter)
        
        avg_volume_result = avg_volume_query.scalar()
        avg_volume = float(avg_volume_result) if avg_volume_result is not None else 0.000001 # Избегаем деления на ноль, приводим к float

        print(f"Рассчитан средний объем ({avg_volume:.{round_precision}f}) для {symbol_name if symbol_name else 'всех символов'} ({timeframe}).")

        # 2. Идентифицируем аномальные свечи
        query_anomalies = db.query(OHLCV).filter(OHLCV.timeframe == timeframe)
        if symbol_id_filter is not None:
            query_anomalies = query_anomalies.filter(OHLCV.symbol_id == symbol_id_filter)

        # Условия для идентификации аномалий (используем cast для точности)
        # Важно: open_price должен быть не равен 0 для этих расчетов
        # Используем float(OHLCV.open) для сравнения, так как в модели Float
        price_change_expr = func.abs((OHLCV.close - OHLCV.open) / cast(OHLCV.open, Numeric) * 100)
        hl_range_expr = (OHLCV.high - OHLCV.low) / cast(OHLCV.open, Numeric) * 100
        volume_ratio_expr = OHLCV.volume / cast(avg_volume, Numeric)

        anomalous_candles = query_anomalies.filter(
            or_(
                OHLCV.open == 0, # Если open_price равен 0, это уже аномалия, обрабатываем отдельно
                price_change_expr > (max_close_change_percent + epsilon), # Увеличиваем порог на epsilon
                hl_range_expr > (max_hl_range_percent + epsilon),         # Увеличиваем порог на epsilon
                volume_ratio_expr > (volume_multiplier + epsilon)         # Увеличиваем порог на epsilon
            )
        ).all()

        modified_count = 0
        total_anomalies = len(anomalous_candles)
        
        print(f"Найдено {total_anomalies} аномальных свечей. Начало корректировки...")

        # 3. Обрезаем значения в каждой аномальной свече
        for i, candle in enumerate(anomalous_candles):
            # Сохраняем оригинальные значения для финального сравнения
            _original_open = float(candle.open)
            _original_high = float(candle.high)
            _original_low = float(candle.low)
            _original_close = float(candle.close)
            _original_volume = float(candle.volume)

            # Флаг, который будет true, если хоть одно поле свечи было изменено
            current_candle_changed = False

            # Обработка open_price == 0 (или очень маленького)
            if _original_open == 0:
                new_open = round(0.000001, round_precision) # Устанавливаем минимально допустимое значение
                if new_open != _original_open: # Прямое сравнение после округления
                    candle.open = new_open
                    current_candle_changed = True
                
                # Если open изменился, остальные цены также должны быть скорректированы относительно этого нового open
                # Приравниваем close, high, low к новому open, если они были 0 или невалидны
                if round(float(candle.close), round_precision) != round(_original_close, round_precision) or _original_close == 0:
                    new_close = round(new_open, round_precision)
                    if new_close != round(float(candle.close), round_precision):
                        candle.close = new_close
                        current_candle_changed = True
                
                if round(float(candle.high), round_precision) != round(_original_high, round_precision) or _original_high == 0:
                    new_high = round(new_open, round_precision)
                    if new_high != round(float(candle.high), round_precision):
                        candle.high = new_high
                        current_candle_changed = True
                
                if round(float(candle.low), round_precision) != round(_original_low, round_precision) or _original_low == 0:
                    new_low = round(new_open, round_precision)
                    if new_low != round(float(candle.low), round_precision):
                        candle.low = new_low
                        current_candle_changed = True
            
            # Используем текущие (потенциально уже скорректированные) значения для дальнейших расчетов
            current_open = float(candle.open)
            current_high = float(candle.high)
            current_low = float(candle.low)
            current_close = float(candle.close)
            current_volume = float(candle.volume)

            # 1. Обрезаем close_price
            if current_open > 0: # Проверяем, чтобы избежать деления на ноль после корректировки open=0
                current_price_change_percent = (current_close - current_open) / current_open * 100
                if abs(current_price_change_percent) > max_close_change_percent:
                    sign = 1 if current_price_change_percent > 0 else -1
                    new_close = current_open * (1 + sign * (max_close_change_percent - epsilon) / 100)
                    new_close_rounded = round(new_close, round_precision) 
                    if new_close_rounded != current_close: # Прямое сравнение после округления
                        print(f"  Свеча {datetime.fromtimestamp(candle.timestamp / 1000)}: Close изм. с {current_close:.{round_precision}f} на {new_close_rounded:.{round_precision}f} (было {current_price_change_percent:.2f}%).")
                        candle.close = new_close_rounded
                        current_candle_changed = True
            
            # Обновляем current_close после возможной обрезки
            current_close = float(candle.close)

            # 2. Обрезаем high_price
            base_price_for_high = max(current_open, current_close)
            if base_price_for_high > 0:
                current_spike_percent = (current_high - base_price_for_high) / base_price_for_high * 100
                if current_spike_percent > max_hl_range_percent:
                    new_high = base_price_for_high * (1 + (max_hl_range_percent - epsilon) / 100)
                    new_high_rounded = round(new_high, round_precision)
                    if new_high_rounded != current_high: # Прямое сравнение после округления
                        print(f"  Свеча {datetime.fromtimestamp(candle.timestamp / 1000)}: High изм. с {current_high:.{round_precision}f} на {new_high_rounded:.{round_precision}f} (было {current_spike_percent:.2f}% шип).")
                        candle.high = new_high_rounded
                        current_candle_changed = True
            else: # Если base_price_for_high == 0, то high_price тоже должен быть 0 или очень маленьким
                if current_high > 0.000001:
                    new_high = round(0.000001, round_precision)
                    if new_high != current_high: # Прямое сравнение после округления
                        print(f"  Свеча {datetime.fromtimestamp(candle.timestamp / 1000)}: High изм. с {current_high:.{round_precision}f} на {new_high:.{round_precision}f} (base_price_for_high был 0).")
                        candle.high = new_high
                        current_candle_changed = True

            # 3. Обрезаем low_price
            base_price_for_low = min(current_open, current_close)
            if base_price_for_low > 0:
                current_dip_percent = (base_price_for_low - current_low) / base_price_for_low * 100
                if current_dip_percent > max_hl_range_percent:
                    new_low = base_price_for_low * (1 - (max_hl_range_percent - epsilon) / 100)
                    new_low_rounded = round(new_low, round_precision)
                    if new_low_rounded != current_low: # Прямое сравнение после округления
                        print(f"  Свеча {datetime.fromtimestamp(candle.timestamp / 1000)}: Low изм. с {current_low:.{round_precision}f} на {new_low_rounded:.{round_precision}f} (было {current_dip_percent:.2f}% провал).")
                        candle.low = new_low_rounded
                        current_candle_changed = True
            else:
                if current_low > 0.000001:
                    new_low = round(0.000001, round_precision)
                    if new_low != current_low: # Прямое сравнение после округления
                        print(f"  Свеча {datetime.fromtimestamp(candle.timestamp / 1000)}: Low изм. с {current_low:.{round_precision}f} на {new_low:.{round_precision}f} (base_price_for_low был 0).")
                        candle.low = new_low
                        current_candle_changed = True

            # 4. Гарантируем, что high >= low (критично для валидности свечи)
            # Применяем после всех обрезок, чтобы high и low были согласованы
            if float(candle.high) < float(candle.low):
                # Если high стал меньше low после обрезки, устанавливаем high = low
                # print(f"Warning: high < low for candle {candle.timestamp}. Adjusting.")
                new_high_from_low = round(float(candle.low), round_precision)
                if new_high_from_low != float(candle.high): # Прямое сравнение после округления
                    print(f"  Свеча {datetime.fromtimestamp(candle.timestamp / 1000)}: High < Low. High изм. с {float(candle.high):.{round_precision}f} на {new_high_from_low:.{round_precision}f} (приравнено к Low).")
                    candle.high = new_high_from_low
                    current_candle_changed = True
            
            # 5. Обрезаем volume
            if avg_volume > 0.000001 and current_volume / avg_volume > volume_multiplier:
                new_volume = avg_volume * (volume_multiplier - epsilon)
                new_volume_rounded = round(new_volume, round_precision) # Округляем объем
                if new_volume_rounded != current_volume: # Прямое сравнение после округления
                    print(f"  Свеча {datetime.fromtimestamp(candle.timestamp / 1000)}: Volume изм. с {current_volume:.{round_precision}f} на {new_volume_rounded:.{round_precision}f} (было {current_volume / avg_volume:.2f}x от среднего).")
                    candle.volume = new_volume_rounded
                    current_candle_changed = True

            # Если хотя бы одно поле было изменено, увеличиваем счетчик
            if current_candle_changed:
                db.add(candle) # Отмечаем свечу как измененную для обновления
                modified_count += 1
            
            # Логирование прогресса
            if (total_anomalies > 0) and ((i + 1) % (total_anomalies // 10 + 1) == 0):
                print(f"Прогресс очистки: {i+1}/{total_anomalies} аномальных свечей скорректировано.")


        db.commit() # Коммитим все изменения
        
        print(f"Очистка данных для {symbol_name if symbol_name else 'всех символов'} ({timeframe}) завершена. Скорректировано {modified_count} аномальных свечей.")
        return {"status": "success", "message": f"Очистка данных для {symbol_name if symbol_name else 'всех символов'} ({timeframe}) завершена. Скорректировано {modified_count} аномальных свечей."}

    except Exception as e:
        db.rollback()
        print(f"Ошибка при очистке данных для {symbol_name if symbol_name else 'всех символов'} ({timeframe}): {e}")
        return {"status": "error", "message": str(e)}
    finally:
        db.close() # Закрываем сессию




# --- ВАША ФУНКЦИЯ load_ohlcv_from_csv (без изменений) ---
def load_ohlcv_from_csv(file_path: str) -> pd.DataFrame:
    """
    Загружает данные OHLCV из CSV-файла, переименовывает колонки и фильтрует аномальные timestamp.
    """
    try:
        df = pd.read_csv(file_path)
        
        print(f"Всего строк: {len(df)}")
        print(f"Уникальных timestamp: {df['timestamp'].nunique()}")
        
        required_columns_csv = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns_csv):
            logging.error(f"CSV файл по пути {file_path} не содержит все необходимые колонки: {required_columns_csv}")
            return pd.DataFrame()
        
        df_ohlcv = df[required_columns_csv].copy()
        df_ohlcv.rename(columns={
            'Open time': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        
        df_ohlcv['timestamp'] = pd.to_numeric(df_ohlcv['timestamp'], errors='coerce')
        df_ohlcv.dropna(subset=['timestamp'], inplace=True) 
        df_ohlcv['timestamp'] = df_ohlcv['timestamp'].astype(int) 
        
        logging.info(f"Успешно загружено {len(df_ohlcv)} записей из CSV: {file_path}")
        return df_ohlcv.sort_values('timestamp').reset_index(drop=True)
    except FileNotFoundError:
        logging.error(f"CSV файл не найден: {file_path}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        logging.error(f"CSV файл пуст: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Ошибка при чтении CSV файла {file_path}: {e}", exc_info=True)
        return pd.DataFrame()

# --- НОВАЯ ФУНКЦИЯ ДЛЯ ЗАГРУЗКИ CSV В БД С ПЕРЕЗАПИСЬЮ ---
def load_latest_candles_from_csv_to_db(
    file_path: str,
    symbol_name: str,
    timeframe: str
) -> int:
    session = next(get_db_session())
    try:
        df_csv = load_ohlcv_from_csv(file_path)
        if df_csv.empty:
            logging.warning(f"CSV файл {file_path} пуст или нечитаем. Нечего загружать в БД.")
            return 0

        symbol_obj = _get_or_create_symbol(session, symbol_name)

        records_to_insert = []
        for _, row in df_csv.iterrows():
            records_to_insert.append({
                'symbol_id': symbol_obj.id,
                'timeframe': timeframe,
                'timestamp': int(row['timestamp']),
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            })       
        
        stmt = insert(OHLCV).values(records_to_insert)
        stmt = stmt.on_conflict_do_update(
            index_elements=['symbol_id', 'timeframe', 'timestamp'],
            set_={
                'open': stmt.excluded.open,
                'high': stmt.excluded.high,
                'low': stmt.excluded.low,
                'close': stmt.excluded.close,
                'volume': stmt.excluded.volume
            }
        )
        session.execute(stmt)
        session.commit()
        logging.info(f"Успешно загружено {len(records_to_insert)} записей OHLCV из CSV в БД для {symbol_name} ({timeframe}).")

        # --- Новый код: считаем количество записей в базе для данного symbol_id и timeframe ---
        total_count = session.query(func.count(OHLCV.id)).filter_by(
            symbol_id=symbol_obj.id,
            timeframe=timeframe
        ).scalar()
        logging.info(f"В базе всего {total_count} записей OHLCV для {symbol_name} ({timeframe}).")

        return len(records_to_insert)
    
    except Exception as e:
        session.rollback()
        logging.error(f"Ошибка при загрузке данных из CSV в БД: {e}", exc_info=True)
        return 0
    finally:
        session.close()
        
        
def delete_ohlcv_for_symbol_timeframe(symbol: str, timeframe: str):
    session = Session()
    symbol_obj = session.query(Symbol).filter(Symbol.name == symbol).first()
    if not symbol_obj:
        print(f"Символ {symbol} не найден в базе, удалять нечего.")
        return

    deleted_count = session.query(OHLCV).filter(
        OHLCV.symbol_id == symbol_obj.id,
        OHLCV.timeframe == timeframe
    ).delete(synchronize_session=False)
    session.commit()
    print(f"Удалено {deleted_count} записей OHLCV для символа '{symbol}' и таймфрейма '{timeframe}'.")
    

def _get_or_create_symbol(session, symbol_name: str) -> Symbol:
    """Получает или создает объект символа в базе данных."""
    symbol_obj = session.query(Symbol).filter_by(name=symbol_name).first()
    if not symbol_obj:
        symbol_obj = Symbol(name=symbol_name)
        session.add(symbol_obj)
        session.commit()
        session.refresh(symbol_obj)
        logging.info(f"Добавлен новый символ в БД: {symbol_name}")
    return symbol_obj
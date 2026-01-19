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
from orm.models import Base, Symbol, OHLCV, FundingRate

import logging

from utils.cctx_utils import normalize_symbol, normalize_to_db

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import math # Для isclose
import time
import json # Для логирования сырых данных
# --- Helpers: Bybit API keys discovery from Postgres (app_settings via utils.settings_store) ---
def _key_prefix_4(key: str | None) -> str:
    """Safe prefix for logs: first 4 chars only (never full key)."""
    try:
        s = (key or "").strip()
        return s[:4] if len(s) >= 4 else "***"
    except Exception:
        return "***"


def _discover_bybit_api_keys(symbol: str | None = None) -> tuple[str | None, str | None, str | None]:
    try:
        from utils.settings_store import ensure_settings_table, get_setting_value, list_settings

        ensure_settings_table()

        def _norm(v: str | None) -> str | None:
            try:
                s = (v or "").strip()
                return s if s else None
            except Exception:
                return None

        def _get_pair(idx: str) -> tuple[str | None, str | None]:
            ak = _norm(get_setting_value('api', 'bybit', f'BYBIT_{idx}_API_KEY'))
            sk = _norm(get_setting_value('api', 'bybit', f'BYBIT_{idx}_SECRET_KEY'))
            return ak, sk

        # 0) Если выбран account_id в Redis — используем его (без фолбэков на другие аккаунты).
        # Приоритет: per-symbol ключ -> глобальный ключ (backward-compatible).
        selected: str | None = None
        try:
            import redis  # type: ignore
            r = redis.Redis(host='redis', port=6379, db=0, decode_responses=True, socket_connect_timeout=2)
            sel = None
            try:
                if symbol:
                    sym = str(symbol).strip()
                    if sym:
                        sel = r.get(f'trading:account_id:{sym}')
            except Exception:
                sel = None
            if sel is None:
                sel = r.get('trading:account_id')
            selected = str(sel).strip() if sel else None
        except Exception:
            selected = None

        if selected:
            api_key, secret_key = _get_pair(selected)
            src = f'BYBIT_{selected}_API_KEY'
            if api_key and secret_key:
                return api_key, secret_key, src
            return None, None, src

        # 1) По умолчанию берём аккаунт 1 из БД
        api_key, secret_key = _get_pair('1')
        if api_key and secret_key:
            return api_key, secret_key, 'BYBIT_1_API_KEY'

        # 2) Автоскан в БД: BYBIT_<ID>_API_KEY / BYBIT_<ID>_SECRET_KEY
        ids: list[int] = []
        try:
            rows = list_settings(scope='api', group='bybit')
            for r in rows:
                k = str(r.get('key') or '').strip()
                if k.startswith('BYBIT_') and k.endswith('_API_KEY'):
                    mid = k[len('BYBIT_'):-len('_API_KEY')]
                    if str(mid).isdigit():
                        ids.append(int(mid))
        except Exception:
            ids = []

        for idx in sorted(set(ids)):
            api_key, secret_key = _get_pair(str(idx))
            if api_key and secret_key:
                return api_key, secret_key, f'BYBIT_{idx}_API_KEY'

        return None, None, None
    except Exception:
        return None, None, None


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
    dry_run: bool = False, # Если True, только загружает, но не сохраняет в БД
    include_error: bool = False
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
    error_reason: str | None = None
 
    # Управление подробностью логов внутри этой функции
    detailed_logs = True
    #audit_enabled = str(os.getenv('OHLCV_AUDIT', '0')).lower() in ('1','true','yes','on')
    audit_enabled = True
    

    def _wrap_result(result_df: pd.DataFrame):
        return (result_df, error_reason) if include_error else result_df

    try:
        # Получаем или создаем запись символа в БД в формате БЕЗ слеша (TONUSDT)
        # CCXT-формат (TON/USDT) используем только для биржи
        # 1. Получить или создать символ
        try:
            symbol_db = normalize_to_db(symbol_name)
        except Exception:
            symbol_db = str(symbol_name).replace('/', '').upper()
        symbol_obj = _get_or_create_symbol(session, symbol_db)

        total_count = session.query(OHLCV).filter_by(
            symbol_id=symbol_obj.id,
            timeframe=timeframe
        ).count()

        if detailed_logs:
            print(f"Всего свечей в БД: {total_count}")

        last_db_candle = session.query(OHLCV).filter_by(
            symbol_id=symbol_obj.id,
            timeframe=timeframe
        ).order_by(OHLCV.timestamp.desc()).first()

        if last_db_candle:
            last_db_timestamp = last_db_candle.timestamp
            if detailed_logs:
                logging.info(f"Последняя свеча в БД по {symbol_name}, {timeframe} имеет timestamp {last_db_timestamp} ({datetime.fromtimestamp(last_db_timestamp/1000)})")
        else:
            last_db_timestamp = None
            if detailed_logs:
                logging.info(f"Свечей в базе для {symbol_name}, {timeframe} нет. Начинаем загрузку с нуля.")

        # Инициализация биржи
        try:            
            init_t0 = time.monotonic()
            if detailed_logs:
                logging.info(f"[OHLCV] EXCHANGE_INIT start: exchange={exchange_id}, symbol={symbol_name}, tf={timeframe}")
            
            symbol_cctx = normalize_symbol(symbol_name)
            
            exchange_class = getattr(ccxt, exchange_id)
            
            bybit_auth_used = None
            bybit_api_key_var = None
            bybit_api_key_prefix4 = None
            # Для Bybit: используем API ключи, если заданы, иначе публичные эндпоинты (они достаточны для OHLCV)
            if exchange_id == 'bybit':
                # Поддержка BYBIT_API_KEY/BYBIT_SECRET_KEY и множественных BYBIT_<N>_*
                api_key, secret_key, bybit_api_key_var = _discover_bybit_api_keys(symbol_name)
                bybit_api_key_prefix4 = _key_prefix_4(api_key)
                bybit_auth_used = bool(api_key and secret_key)
                if api_key and secret_key:
                    exchange = exchange_class({
                        'apiKey': api_key,
                        'secret': secret_key,
                        'enableRateLimit': True,
                        'timeout': 30000, # Увеличение таймаута
                        'options': {
                            'recv_window': 20000,
                            'recvWindow': 20000,
                            'adjustForTimeDifference': True,
                            'timeDifference': True,
                            # Для Bybit: linear = perpetual futures (swap), не spot
                            'defaultType': 'linear',
                        }
                    })
                else:
                    logging.warning("API ключи Bybit не настроены — использую публичные эндпоинты для OHLCV")
                    exchange = exchange_class({
                        'enableRateLimit': True,
                        'timeout': 30000,
                        'options': {
                            'recv_window': 20000,
                            'recvWindow': 20000,
                            'adjustForTimeDifference': True,
                            'timeDifference': True,
                            # Для Bybit: linear = perpetual futures (swap), не spot
                            'defaultType': 'linear',
                        }
                    })
            else:
                exchange = exchange_class({
                    'enableRateLimit': True,
                    'timeout': 30000, # Увеличение таймаута
                })
            if detailed_logs:
                logging.info(f"[OHLCV] EXCHANGE_OBJ created: exchange={exchange_id}, bybit_auth={bybit_auth_used}")
            
            lm_t0 = time.monotonic()
            if detailed_logs:
                logging.info("[OHLCV] load_markets() start")
            exchange.load_markets()
            lm_dt = (time.monotonic() - lm_t0) * 1000
            if detailed_logs:
                logging.info(f"[OHLCV] load_markets() done in {lm_dt:.0f} ms; symbols={len(getattr(exchange,'symbols',[]) or [])}")
            
            # Синхронизация времени для избежания retCode 10002 (recv_window)
            try:
                if hasattr(exchange, 'load_time_difference'):
                    exchange.load_time_difference()
            except Exception as _te:
                if detailed_logs:
                    logging.warning(f"Не удалось синхронизировать время с биржей {exchange_id}: {_te}")
            # Если на Bybit не найден символ в текущем типе рынка (по умолчанию linear) — пробуем spot
            if symbol_cctx not in getattr(exchange, 'symbols', []):
                # Без fallback на spot: работаем строго в linear
                markets = getattr(exchange, 'markets', {}) or {}
                symbol_fetch = None
                try:
                    # Прямое совпадение unified
                    if symbol_cctx in markets:
                        symbol_fetch = symbol_cctx
                    else:
                        # Совпадение по market id без слеша (BNBUSDT и т.п.)
                        sym_no_sep = normalize_to_db(symbol_name)
                        for uni, m in markets.items():
                            mid = m.get('id') if isinstance(m, dict) else None
                            if mid and str(mid).upper() == str(sym_no_sep).upper():
                                symbol_fetch = uni
                                break
                except Exception:
                    symbol_fetch = None
                if not symbol_fetch:
                    raise ValueError(f"Символ {symbol_cctx} не найден в Bybit linear (fallback на spot отключён).")
            else:
                # Символ найден как unified
                markets = getattr(exchange, 'markets', {}) or {}
                symbol_fetch = symbol_cctx if symbol_cctx in markets else None
            if detailed_logs:
                logging.info(f"[OHLCV] EXCHANGE_INIT done in {(time.monotonic()-init_t0)*1000:.0f} ms; symbol_fetch={symbol_fetch}")
        except Exception as e:
            if exchange_id == 'bybit':
                logging.error(
                    f"Не удалось инициализировать биржу {exchange_id}: {e} "
                    f"(api_key_var={bybit_api_key_var}, api_key_prefix4={bybit_api_key_prefix4}, auth={bybit_auth_used})"
                )
            else:
                logging.error(f"Не удалось инициализировать биржу {exchange_id}: {e}")
            error_reason = f"exchange_init_failed: {exchange_id}: {e}"
            if include_error:
                return _wrap_result(pd.DataFrame())
            return pd.DataFrame()


        tf_ms = exchange.parse_timeframe(timeframe) * 1000
        # Текущее время биржи (нужно раньше, чтобы рассчитать старт при пустой БД)
        now_ms = exchange.milliseconds()

        # Определяем from какого момента качать данные
        if last_db_timestamp:
            since_ms = last_db_timestamp + tf_ms
        else:
            # При пустой БД: для 5m загружаем минимум 100000 свечей (или больше, если limit_candles больше)
            try:
                desired_candles = int(limit_candles) if limit_candles is not None else 0
            except Exception:
                desired_candles = 0
            if timeframe == '5m':
                desired_candles = max(desired_candles, 100000)
            else:
                desired_candles = max(desired_candles, 1000)
            since_ms = max(0, now_ms - desired_candles * tf_ms)
            if detailed_logs:
                logging.info(
                    f"Свечей в базе для {symbol_name}, {timeframe} нет. Начинаем загрузку с нуля (цель ~{desired_candles} свечей, с {datetime.fromtimestamp(since_ms/1000)})."
                )
        
        # Если since_ms в будущем — корректируем
        if since_ms > now_ms:
            if detailed_logs:
                logging.warning("Начальная дата больше текущей. Сбрасываем на текущий момент.")
            since_ms = now_ms - tf_ms

        all_new_data = []
        retries = 0
        # Докачиваем данные, пока не дойдём до сейчас или не наберём достаточно
        while True:
            try:
                limit_fetch = min(exchange.options.get('fetchOHLCVLimit', 1000), 1000)
                # Используем symbol_fetch (unified) если он определён, иначе fallback к symbol_cctx
                _sym_to_use = symbol_fetch or symbol_cctx
                print(f"[DB_OHLCV] Fetching {limit_fetch} candles for {_sym_to_use} {timeframe} since {datetime.fromtimestamp(since_ms/1000)}")
                if exchange_id.lower() == 'bybit':
                    raw_ohlcv = exchange.fetch_ohlcv(_sym_to_use, timeframe, since=since_ms, limit=limit_fetch, params={'category': 'linear'})
                else:
                    raw_ohlcv = exchange.fetch_ohlcv(_sym_to_use, timeframe, since=since_ms, limit=limit_fetch)

                if (detailed_logs or audit_enabled) and raw_ohlcv:
                    try:
                        first_ts = raw_ohlcv[0][0]
                        last_ts_b = raw_ohlcv[-1][0]
                        print(f"[AUDIT][FETCH] count={len(raw_ohlcv)} ts_range={datetime.fromtimestamp(first_ts/1000)}..{datetime.fromtimestamp(last_ts_b/1000)}")
                        # Логируем только хвост, чтобы не засорять
                        print(f"[DB_OHLCV] Raw OHLCV (last 10) for {_sym_to_use} {timeframe}: {json.dumps(raw_ohlcv[-10:], ensure_ascii=False)}")
                    except Exception as _e:
                        print(f"[AUDIT][FETCH] log error: {_e}")

                ohlcv = raw_ohlcv

                if not ohlcv:
                    if detailed_logs:
                        logging.info("Новых данных с биржи нет. Докачка завершена.")
                    break

                new_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

                # Отфильтровываем свечи, которые уже есть в БД
                if last_db_timestamp:
                    new_df = new_df[new_df['timestamp'] > last_db_timestamp]

                if new_df.empty:
                    if detailed_logs:
                        logging.info("Все загруженные свечи уже есть в БД.")
                    if audit_enabled and isinstance(ohlcv, list) and len(ohlcv) > 0:
                        try:
                            # Сравним хвост полученных баров с БД, даже если вставлять нечего
                            tail = ohlcv[-20:]
                            ts_list = [int(r[0]) for r in tail]
                            db_rows = session.query(OHLCV).filter(
                                OHLCV.symbol_id == symbol_obj.id,
                                OHLCV.timeframe == timeframe,
                                OHLCV.timestamp.in_(ts_list)
                            ).all()
                            db_map = {int(r.timestamp): r for r in db_rows}
                            def _diff(a,b, rel=1e-8, abs_tol=1e-6):
                                try:
                                    return not math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_tol)
                                except Exception:
                                    return True
                            for r in tail:
                                ts = int(r[0]); o=float(r[1]); h=float(r[2]); l=float(r[3]); c=float(r[4]); v=float(r[5])
                                db_r = db_map.get(ts)
                                if not db_r:
                                    print(f"[AUDIT][TAIL_COMPARE] ts={datetime.fromtimestamp(ts/1000)} missing_in_db")
                                    continue
                                diffs=[]
                                if _diff(db_r.open,o): diffs.append(("open", db_r.open, o))
                                if _diff(db_r.high,h): diffs.append(("high", db_r.high, h))
                                if _diff(db_r.low,l): diffs.append(("low", db_r.low, l))
                                if _diff(db_r.close,c): diffs.append(("close", db_r.close, c))
                                if _diff(db_r.volume,v): diffs.append(("volume", db_r.volume, v))
                                if diffs:
                                    print(f"[AUDIT][TAIL_MISMATCH] ts={datetime.fromtimestamp(ts/1000)} diffs={diffs}")
                                else:
                                    print(f"[AUDIT][TAIL_OK] ts={datetime.fromtimestamp(ts/1000)}")
                        except Exception as _e:
                            print(f"[AUDIT][TAIL_COMPARE] error: {_e}")
                    break

                if audit_enabled:
                    try:
                        tmin = int(new_df['timestamp'].min()); tmax = int(new_df['timestamp'].max())
                        print(f"[AUDIT][PREP] new_df size={len(new_df)} ts_range={datetime.fromtimestamp(tmin/1000)}..{datetime.fromtimestamp(tmax/1000)} last5={json.dumps(new_df.tail(5).to_dict('records'), ensure_ascii=False)}")
                    except Exception as _e:
                        print(f"[AUDIT][PREP] log error: {_e}")

                all_new_data.extend(new_df.to_dict('records'))

                last_ts_in_batch = new_df['timestamp'].max()
                since_ms = last_ts_in_batch + tf_ms

                # Если докачали достаточно (по времени или по объёму), прекращаем
                if since_ms > now_ms:
                    if detailed_logs:
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
                    error_reason = f"exchange_fetch_failed: {exchange_id}: {e}"
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
                audit_ts_list = [rec['timestamp'] for rec in new_records[-20:]] if audit_enabled else []
                if audit_enabled and new_records:
                    try:
                        print(f"[AUDIT][INSERT] count={len(new_records)} tail10={json.dumps(new_records[-10:], ensure_ascii=False)}")
                    except Exception as _e:
                        print(f"[AUDIT][INSERT] log error: {_e}")
                # --- ВСТРОЕННАЯ СВЕРКА: перепроверяем весь диапазон new_data_df, даже если вставлять нечего ---
                try:
                    if detailed_logs:
                        logging.info("Запуск сверки диапазона после обработки новых баров...")
                    from scripts.reconcile_ohlcv_bybit import reconcile_range_ohlcv
                    range_since = int(new_data_df['timestamp'].min())
                    range_until = int(new_data_df['timestamp'].max())
                    reconcile_range_ohlcv(symbol_name, timeframe, exchange_id, range_since, range_until, verbose=False)
                except Exception as _re:
                    logging.warning(f"Сверка диапазона не выполнена: {_re}")

                if new_records:
                    # Используем ON CONFLICT DO UPDATE для обработки дубликатов
                    stmt = insert(OHLCV).values(new_records)
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
                    if audit_enabled and audit_ts_list:
                        # читаем обратно и сравниваем
                        stored_rows = session.query(OHLCV).filter(
                            OHLCV.symbol_id == symbol_obj.id,
                            OHLCV.timeframe == timeframe,
                            OHLCV.timestamp.in_(audit_ts_list)
                        ).all()
                        stored_map = {int(r.timestamp): r for r in stored_rows}
                        def _diff(a,b, rel=1e-8, abs_tol=1e-6):
                            try:
                                return not math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_tol)
                            except Exception:
                                return True
                        for rec in new_records[-20:]:
                            ts = int(rec['timestamp'])
                            st = stored_map.get(ts)
                            if not st:
                                print(f"[AUDIT][MISSING_AFTER_INSERT] ts={datetime.fromtimestamp(ts/1000)}")
                                continue
                            diffs = []
                            for f in ('open','high','low','close','volume'):
                                if _diff(rec[f], getattr(st, f)):
                                    diffs.append((f, rec[f], getattr(st, f)))
                            if diffs:
                                print(f"[AUDIT][MISMATCH_AFTER_INSERT] ts={datetime.fromtimestamp(ts/1000)} diffs={diffs}")
                            else:
                                print(f"[AUDIT][OK_AFTER_INSERT] ts={datetime.fromtimestamp(ts/1000)}")
                    if detailed_logs:
                        logging.info(f"Добавлено {len(new_records)} новых свечей {timeframe} для {symbol_name}.")
                else:
                    if detailed_logs:
                        logging.info("Новых свечей для добавления не найдено.")
            else:
                if detailed_logs:
                    logging.info(f"Dry run: {len(new_data_df)} свечей не сохранены в БД.")

        # --- ДОКАЧКА ИЗ ПРОШЛОГО: гарантируем минимальный объём истории в БД ---
        try:
            try:
                min_needed = int(limit_candles) if limit_candles is not None else 0
            except Exception:
                min_needed = 0
            if timeframe == '5m':
                min_needed = max(min_needed, 100000)
            else:
                min_needed = max(min_needed, 1000)

            current_count = session.query(OHLCV).filter_by(
                symbol_id=symbol_obj.id,
                timeframe=timeframe
            ).count()

            if current_count < min_needed:
                if detailed_logs:
                    logging.info(f"Недостаточно свечей в БД: {current_count} < {min_needed}. Докачиваем из прошлого.")

                earliest_row = session.query(OHLCV).filter_by(
                    symbol_id=symbol_obj.id,
                    timeframe=timeframe
                ).order_by(OHLCV.timestamp.asc()).first()

                if earliest_row is not None:
                    earliest_ts = int(earliest_row.timestamp)
                    safety_iter = 0
                    while current_count < min_needed:
                        safety_iter += 1
                        limit_fetch = min(getattr(exchange, 'options', {}).get('fetchOHLCVLimit', 1000), 1000)
                        back_since = max(0, earliest_ts - limit_fetch * tf_ms)
                        _sym_to_use = symbol_fetch or symbol_cctx
                        print(f"[DB_OHLCV] Backfilling older {limit_fetch} candles for {_sym_to_use} {timeframe} since {datetime.fromtimestamp(back_since/1000)}")
                        raw_older = []
                        try:
                            raw_older = exchange.fetch_ohlcv(_sym_to_use, timeframe, since=back_since, limit=limit_fetch)
                        except Exception as _be:
                            logging.warning(f"Ошибка обратной докачки: {_be}")
                            break

                        if not raw_older:
                            if detailed_logs:
                                logging.info("Биржа не вернула более старые данные. Останавливаем обратную догрузку.")
                            break

                        older_df = pd.DataFrame(raw_older, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        # Оставляем только свечи строго старше самой ранней имеющейся
                        older_df = older_df[older_df['timestamp'] < earliest_ts]

                        if older_df.empty:
                            if detailed_logs:
                                logging.info("Нет более старых свечей (после фильтрации). Останавливаем обратную догрузку.")
                            break

                        # Апсертим только те, которых ещё нет
                        existing_ts_older = set(
                            r[0] for r in session.query(OHLCV.timestamp).filter_by(
                                symbol_id=symbol_obj.id,
                                timeframe=timeframe
                            ).filter(OHLCV.timestamp.in_(older_df['timestamp'].tolist())).all()
                        )
                        older_records = [
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
                            for _, row in older_df.iterrows() if int(row['timestamp']) not in existing_ts_older
                        ]

                        if older_records:
                            stmt = insert(OHLCV).values(older_records)
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
                            if detailed_logs:
                                logging.info(f"Обратно добавлено {len(older_records)} старых свечей {timeframe} для {symbol_name}.")

                            # Обновляем опорные значения
                            earliest_ts_new = int(older_df['timestamp'].min())
                            if earliest_ts_new >= earliest_ts:
                                logging.warning("Не удалось продвинуться к более старым данным (earliest не изменился). Прерываем.")
                                break
                            earliest_ts = earliest_ts_new
                            current_count = session.query(OHLCV).filter_by(
                                symbol_id=symbol_obj.id,
                                timeframe=timeframe
                            ).count()
                        else:
                            if detailed_logs:
                                logging.info("Все загруженные старые свечи уже есть в БД. Продвижения нет.")
                            break

                        # Пауза согласно rateLimit
                        time.sleep(getattr(exchange, 'rateLimit', 1000) / 1000)

                        if safety_iter > 200:
                            logging.warning("Достигнут лимит итераций обратной докачки. Останавливаемся, чтобы избежать бесконечного цикла.")
                            break
        except Exception as _bf:
            logging.warning(f"Ошибка процедуры обратной докачки: {_bf}")

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

        if detailed_logs:
            logging.info(f"Итоговое количество свечей для {symbol_name} {timeframe}: {len(df)}")

        # --- Funding Join (forward-fill) ---
        try:
            if not df.empty:
                sym_norm = normalize_symbol(symbol_name)
                # Приведём к формату без слешей для хранения funding
                sym_key = sym_norm.replace('/', '')
                # Получаем funding за диапазон по времени выборки OHLCV с запасом 2 суток
                ts_min = int(df['timestamp'].min()) - 2 * 24 * 3600 * 1000
                ts_max = int(df['timestamp'].max()) + 2 * 24 * 3600 * 1000
                q = session.query(FundingRate).filter(
                    FundingRate.symbol.in_([sym_key, sym_norm]),
                    FundingRate.timestamp >= ts_min,
                    FundingRate.timestamp <= ts_max
                ).order_by(FundingRate.timestamp.asc())
                fr_rows = q.all()
                fr_df = pd.DataFrame([
                    {'timestamp': r.timestamp, 'funding_rate': float(r.rate)} for r in fr_rows
                ])
                if fr_df.empty:
                    # Пытаемся догрузить funding и повторить
                    try:
                        _download_and_store_funding_rates(session, sym_key, ts_min, ts_max)
                        fr_rows = session.query(FundingRate).filter(
                            FundingRate.symbol == sym_key,
                            FundingRate.timestamp >= ts_min,
                            FundingRate.timestamp <= ts_max
                        ).order_by(FundingRate.timestamp.asc()).all()
                        fr_df = pd.DataFrame([
                            {'timestamp': r.timestamp, 'funding_rate': float(r.rate)} for r in fr_rows
                        ])
                    except Exception as _e:
                        logging.warning(f"Не удалось догрузить funding rates: {_e}")
                if not fr_df.empty:
                    fr_df = fr_df.sort_values('timestamp')
                    # Forward-fill funding по 5m барам
                    df_ff = df[['timestamp']].merge(fr_df, on='timestamp', how='left').sort_values('timestamp')
                    df_ff['funding_rate'] = df_ff['funding_rate'].ffill()
                    df = df.merge(df_ff, on='timestamp', how='left')
                    # Добавляем бипсы и индикаторы
                    df['funding_rate_bp'] = (df['funding_rate'] * 10000).clip(-50, 50)
                    # EMA по funding (окно ~24 часа для 5m: 288 баров)
                    window = 288 if timeframe == '5m' else max(1, int((60*24) / int(timeframe.replace('m','')))) if timeframe.endswith('m') else 24
                    df['funding_rate_ema'] = df['funding_rate'].ewm(span=window, adjust=False).mean()
                    # Дельта за сутки: разница между текущим и значением 24 часа назад
                    shift_bars = 288 if timeframe == '5m' else window
                    df['funding_rate_change'] = df['funding_rate'].diff(shift_bars)
                    # Знак funding
                    df['funding_sign'] = df['funding_rate'].apply(lambda x: 1 if x and x > 0 else (-1 if x and x < 0 else 0))
        except Exception as fe:
            logging.warning(f"Funding join warning: {fe}")

        if detailed_logs:
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
                    
                if detailed_logs:
                    logging.info(f"Проверка целостности данных {symbol_name}, {timeframe} завершена: обнаружены разрывы.")
            else:
                if detailed_logs:
                    logging.info(f"Проверка целостности данных {symbol_name}, {timeframe} завершена: разрывы не обнаружены.")

        # Единый финальный лог
        #logging.info(f"✅ Все необходимые свечи для {symbol_name} {timeframe} загружены (доступно {len(df)}).")

        return _wrap_result(df)

    except Exception as e:
        logging.error(f"Критическая ошибка в db_get_or_fetch_ohlcv: {e}", exc_info=True)
        return pd.DataFrame()
    finally:
        session.close()


def db_get_ohlcv_only(symbol_name: str, timeframe: str, limit_candles: int = 1000) -> pd.DataFrame:
    """
    Возвращает OHLCV строго из Postgres (без докачки с биржи).

    symbol_name может быть 'BTCUSDT' или 'BTC/USDT' — внутри нормализуем к формату БД.
    """
    session = next(get_db_session())
    try:
        try:
            symbol_db = normalize_to_db(symbol_name)
        except Exception:
            symbol_db = str(symbol_name).replace("/", "").upper()
        symbol_obj = session.query(Symbol).filter_by(name=symbol_db).first()
        if not symbol_obj:
            return pd.DataFrame()

        rows = (
            session.query(OHLCV)
            .filter_by(symbol_id=symbol_obj.id, timeframe=timeframe)
            .order_by(OHLCV.timestamp.desc())
            .limit(int(limit_candles))
            .all()
        )
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            [
                {
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
                for c in reversed(rows)
            ]
        )
        return df
    finally:
        try:
            session.close()
        except Exception:
            pass


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


def _download_and_store_funding_rates(session, symbol_no_slash: str, ts_from_ms: int, ts_to_ms: int) -> int:
    """Скачивает funding rates Bybit v5 через ccxt и сохраняет в БД.
    symbol_no_slash: 'BTCUSDT'
    Возвращает количество вставленных записей.
    """
    try:
        ex = ccxt.bybit({'enableRateLimit': True, 'timeout': 30000})
        ex.load_markets()
        inserted = 0
        # Bybit funding раз в 8 часов: шаг 8h
        cursor = ts_from_ms
        while cursor <= ts_to_ms:
            params = {'category': 'linear', 'symbol': symbol_no_slash}
            # ccxt не всегда имеет унифицированный метод; используем raw v5, если доступен
            resp = None
            if hasattr(ex, 'v5PublicGetMarketFundingHistory'):
                resp = ex.v5PublicGetMarketFundingHistory(params)
            elif hasattr(ex, 'publicGetV5MarketFundingHistory'):
                resp = ex.publicGetV5MarketFundingHistory(params)
            else:
                # fallback: попытаться fetchFundingRate/ fetchFundingRates
                try:
                    fr = ex.fetchFundingRate(symbol_no_slash)
                    resp = {'result': {'list': [{'symbol': symbol_no_slash, 'fundingRate': fr.get('fundingRate'), 'fundingRateTimestamp': fr.get('timestamp')} ]}}
                except Exception:
                    resp = None
            if not resp:
                break
            data = resp.get('result') or resp
            items = data.get('list') if isinstance(data, dict) else None
            if not (items and isinstance(items, list)):
                break
            recs = []
            for it in items:
                try:
                    rate = it.get('fundingRate') or it.get('rate')
                    t = it.get('fundingRateTimestamp') or it.get('timestamp') or it.get('time')
                    if rate is None or t is None:
                        continue
                    t_ms = int(t)
                    if t_ms < ts_from_ms or t_ms > ts_to_ms:
                        continue
                    recs.append({'symbol': symbol_no_slash, 'timestamp': t_ms, 'rate': float(rate), 'source': 'bybit_v5'})
                except Exception:
                    continue
            if recs:
                _upsert_funding_records(session, recs)
                inserted += len(recs)
            # Выходим — Bybit отдаёт последние события; для простоты одной порции достаточно
            break
        return inserted
    except Exception as e:
        logging.warning(f"download funding error: {e}")
        return 0


def _upsert_funding_records(session, records: list) -> None:
    if not records:
        return
    stmt = insert(FundingRate).values(records)
    stmt = stmt.on_conflict_do_nothing(index_elements=['symbol', 'timestamp'])
    session.execute(stmt)
    session.commit()
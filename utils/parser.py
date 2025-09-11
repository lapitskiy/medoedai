from binance_historical_data import BinanceDataDumper
from datetime import date, timedelta, datetime
import os
import time
import pandas as pd
try:
    import ccxt
except Exception:
    ccxt = None

def parser_download_and_combine_with_library(
    symbol: str = 'BTCUSDT',
    interval: str = '5m',
    data_type: str = 'klines',
    months_to_fetch: int = 12,
    desired_candles: int = 100000,    
    temp_data_dir: str = './temp/binance_data/'  # Временная директория для скачанных данных
):
    
    output_file = f"{temp_data_dir}/bigdata_5m_klines.csv"
    print(f"Начало загрузки данных {symbol} {interval} с использованием binance-historical-data.")

    end_date = date.today()
    start_date = end_date - timedelta(days=30 * months_to_fetch)

    dumper = BinanceDataDumper(
        path_dir_where_to_dump=temp_data_dir,
        asset_class="spot",
        data_type=data_type,
        data_frequency=interval,
    )

    try:
        print(f"Загрузка данных с {start_date} по {end_date}...")
        dumper.dump_data(
            date_start=start_date,
            date_end=end_date,
            tickers=[symbol],
            is_to_update_existing=False,
        )
        print("Загрузка завершена. Чтение и объединение данных...")

        # Путь в исходной библиотеке использует формат без разделителя (BTCUSDT)
        _sym_flat = str(symbol).replace('/', '').replace('-', '').replace('_', '').upper()
        data_path = f"{temp_data_dir}/spot/monthly/klines/{_sym_flat}/{interval}"
        all_files = []

        if os.path.exists(data_path):
            for root, _, files in os.walk(data_path):
                for file in files:
                    if file.endswith('.csv'):
                        all_files.append(os.path.join(root, file))

        if not all_files:
            print("Не найдено CSV-файлов от Binance, пробуем fallback на Bybit через ccxt...")
            # Попробуем fallback на Bybit
            bybit_ok = _fallback_download_bybit(symbol=symbol, interval=interval, months_to_fetch=months_to_fetch, desired_candles=desired_candles, output_file=output_file)
            if bybit_ok:
                return output_file
            print("Fallback на Bybit не дал данных.")
            return

        column_names = [
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ]

        all_data = []
        for f_path in all_files:
            try:
                df = pd.read_csv(f_path, header=None, names=column_names)
                all_data.append(df)
            except Exception as e:
                print(f"Ошибка чтения {f_path}: {e}")

        if not all_data:
            print("Нет данных для объединения.")
            return

        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.drop_duplicates(subset=['Open time'], inplace=True)
        combined_df.sort_values(by='Open time', inplace=True)

        # --- Преобразуем и фильтруем timestamp ---
        combined_df['timestamp'] = pd.to_numeric(combined_df['Open time'], errors='coerce')
        combined_df.dropna(subset=['timestamp'], inplace=True)
        combined_df['timestamp'] = combined_df['timestamp'].astype('int64')

        # Если максимальный слишком большой, попробуем "подрезать" лишние нули, например делением на 1000, если слишком большой
        max_reasonable = int(datetime(2030, 1, 1).timestamp() * 1000)  # например 2030 год

        def fix_timestamp(ts):
            while ts > max_reasonable:
                ts = ts // 1000  # делим пока не станет в разумных пределах
            return ts

        combined_df['timestamp'] = combined_df['timestamp'].apply(fix_timestamp)

# --- Диагностика до фильтрации ---
        print(f"Всего строк до фильтрации: {len(combined_df)}")
        print(f"Типы данных:\n{combined_df.dtypes}")
        print("Минимальный timestamp:", combined_df['timestamp'].min())
        print("Максимальный timestamp:", combined_df['timestamp'].max())
        min_ts = combined_df['timestamp'].min()
        max_ts = combined_df['timestamp'].max()
        print(f"Минимальный timestamp: {min_ts} -> {pd.to_datetime(min_ts, unit='ms')}")
        print(f"Максимальный timestamp: {max_ts} -> {pd.to_datetime(max_ts, unit='ms')}")
        print("Примеры дат:")
        print(pd.to_datetime(combined_df['timestamp'].sample(5), unit='ms'))

        # --- Подготовка итогового DataFrame ---
        prepared_df = combined_df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        prepared_df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)

        # Оставляем только нужное количество свечей
        if len(prepared_df) > desired_candles:
            prepared_df = prepared_df.tail(desired_candles)
            print(f"Оставлено {desired_candles} последних свечей.")
        else:
            print(f"Свечей всего: {len(prepared_df)}")

        prepared_df.to_csv(output_file, index=False)
        print(f"Файл сохранён: {output_file}")
        return output_file

    except Exception as e:
        print(f"Ошибка в процессе: {e}")


def _fallback_download_bybit(symbol: str, interval: str, months_to_fetch: int, desired_candles: int, output_file: str) -> bool:
    """
    Скачивает OHLCV через ccxt.bybit итерациями (limit ~200) за указанный период, сохраняет в CSV.
    Возвращает True, если данные получены и сохранены, иначе False.
    """
    try:
        if ccxt is None:
            print("ccxt недоступен, пропускаю Bybit fallback")
            return False

        # Подготовим кандидатов символов и типов рынков
        s = symbol.upper().replace('-', '').replace('_', '')
        base = s[:-4] if s.endswith('USDT') else s
        candidates = [
            s,  # XMRUSDT (как просишь)
            f"{base}/USDT",
            f"{base}/USDT:USDT",
        ]
        market_types = ['spot', 'swap']  # пробуем спот и своп (перпеты)
        found = False
        chosen_symbol = None

        timeframe = interval  # '5m'
        now_ms = int(time.time() * 1000)
        since_ms_start = now_ms - int(months_to_fetch * 30 * 24 * 60 * 60 * 1000)
        max_limit = 200
        all_rows = []

        for mtype in market_types:
            try:
                exchange = ccxt.bybit({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': mtype,
                        'defaultSettle': 'USDT'
                    }
                })
                markets = exchange.load_markets()
            except Exception as e:
                print(f"Bybit load_markets ошибка для {mtype}: {e}")
                continue

            # Собираем доступные ключи и id
            market_ids = set([m.get('id') for m in markets.values() if isinstance(m, dict) and m.get('id')])
            market_syms = set(markets.keys())

            # Пытаемся найти совпадение по кандидатам
            c_sym = None
            for cand in candidates:
                if cand in market_syms:
                    c_sym = cand
                    break
                if cand in market_ids:
                    # найдём unified symbol по id
                    for k, v in markets.items():
                        if v.get('id') == cand:
                            c_sym = v.get('symbol', k)
                            break
                if c_sym:
                    break

            if not c_sym:
                print(f"Bybit {mtype}: не найден символ среди кандидатов {candidates}")
                continue

            print(f"Bybit fallback найден: type={mtype}, symbol={c_sym}")
            print(f"Начинаю итеративную загрузку: timeframe={timeframe}, целево {desired_candles} свечей, шаг {max_limit}")

            # Итерируемся, собирая свечи
            since_ms = since_ms_start
            tmp_rows = []
            safety_iters = 0
            max_iters = 20000
            while since_ms < now_ms and len(tmp_rows) < desired_candles and safety_iters < max_iters:
                safety_iters += 1
                try:
                    ohlcv = exchange.fetch_ohlcv(c_sym, timeframe=timeframe, since=since_ms, limit=max_limit)
                except ccxt.RateLimitExceeded:
                    time.sleep(1.0)
                    continue
                except Exception as err:
                    print(f"Ошибка fetch_ohlcv({mtype},{c_sym}): {err}")
                    break

                if not ohlcv:
                    break

                tmp_rows.extend(ohlcv)
                last_ts = ohlcv[-1][0]
                tf_ms = 5 * 60 * 1000 if timeframe == '5m' else 60 * 1000
                since_ms = last_ts + tf_ms
                # Прогресс
                try:
                    from datetime import datetime as _dt
                    total = len(tmp_rows)
                    pct = min(100, int(total * 100 / max(1, desired_candles)))
                    print(f"[Bybit {mtype} {c_sym}] загружено {total}/{desired_candles} ({pct}%), последняя свеча: {_dt.fromtimestamp(last_ts/1000)}", flush=True)
                except Exception:
                    pass
                time.sleep(0.1)

            if tmp_rows:
                all_rows = tmp_rows
                chosen_symbol = c_sym
                found = True
                break

        if not all_rows:
            print("Bybit fallback: данных не получено")
            return False

        # В DataFrame
        df = pd.DataFrame(all_rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = df.drop_duplicates(subset=['timestamp']).sort_values(by='timestamp')
        if len(df) > desired_candles:
            df = df.tail(desired_candles)

        # Гарантируем каталог
        out_dir = os.path.dirname(output_file)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        df.to_csv(output_file, index=False)
        print(f"Bybit fallback: сохранено {len(df)} свечей в {output_file}")
        return True
    except Exception as e:
        print(f"Bybit fallback ошибка: {e}")
        return False
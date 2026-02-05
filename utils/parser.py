from binance_historical_data import BinanceDataDumper
from datetime import date, timedelta, datetime
import os
import time
import pandas as pd
def parser_download_and_combine_with_library(
    symbol: str = 'BTCUSDT',
    interval: str = '5m',
    data_type: str = 'klines',
    months_to_fetch: int = 12,
    desired_candles: int = 100000,    
    temp_data_dir: str = './temp/binance_data/'  # Временная директория для скачанных данных
):
    
    output_file = f"{temp_data_dir}/bigdata_5m_klines.csv"
    print(f"Начало загрузки данных {symbol} {interval} (основной источник: Binance).")

    end_date = date.today()
    start_date = end_date - timedelta(days=30 * months_to_fetch)

    dumper = BinanceDataDumper(
        path_dir_where_to_dump=temp_data_dir,
        asset_class="spot",
        data_type=data_type,
        data_frequency=interval,
    )

    try:
        # Путь в исходной библиотеке использует формат без разделителя (BTCUSDT)
        _sym_flat = str(symbol).replace('/', '').replace('-', '').replace('_', '').upper()
        data_path = f"{temp_data_dir}/spot/monthly/klines/{_sym_flat}/{interval}"
        all_files = []

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            print(f"Загрузка данных с {start_date} по {end_date} (попытка {attempt}/{max_attempts})...")
            try:
                dumper.dump_data(
                    date_start=start_date,
                    date_end=end_date,
                    tickers=[symbol],
                    is_to_update_existing=False,
                )
            except Exception as e:
                print(f"Ошибка загрузки Binance: {e}")

            all_files = []
            if os.path.exists(data_path):
                for root, _, files in os.walk(data_path):
                    for file in files:
                        if file.endswith('.csv'):
                            all_files.append(os.path.join(root, file))

            if all_files:
                print("Загрузка завершена. Чтение и объединение данных...")
                break

            if attempt < max_attempts:
                print("CSV-файлы Binance не найдены, повторяем попытку...")
                time.sleep(5)

        if not all_files:
            print("Не найдено CSV-файлов от Binance после повторных попыток.")
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
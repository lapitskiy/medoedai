from utils.path import create_dataframe, is_hashfile_in_folder
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
from utils.path import generate_uuid
import os

import hashlib
import json
import pandas as pd

def run_multiprocessing_rolling_window(config):
    with ProcessPoolExecutor(max_workers=config.CPU_COUNT) as executor:
        futures = []
        for current_period in config.period:
            df = create_dataframe(coin=config.coin, period=current_period, data=config.date_df)
            for current_window in config.window_size:
                for current_threshold in config.threshold:
                    # Запускаем процесс
                    futures.append(executor.submit(process_data, df.copy(), current_window, current_threshold, current_period, config))
        # Обработка завершенных задач
        for future in as_completed(futures):
            x_path, y_path = future.result()

def process_data(df, current_window, current_threshold, current_period, config):
    df.ffill(inplace=True)
    df['pct_change'] = df['close'].pct_change(periods=current_window)
    df['bullish_volume'] = df['volume'] * (df['close'] > df['open'])
    df['bearish_volume'] = df['volume'] * (df['close'] < df['open'])
    df[config.numeric] = df[config.numeric].astype(np.float32)
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[config.numeric] = scaler.fit_transform(df_scaled[config.numeric])
    joblib.dump(scaler, f'temp/{config.ii_path}/scaler/scaler_ct{current_threshold}_cw{current_window}_cp{current_period}.gz')
    x_path, y_path, num_samples, hash_value = create_rolling_windows(df, df_scaled, current_threshold, current_window, config)
    try:
        with open(f'temp/{config.ii_path}/roll_win/roll_path_ct{current_threshold}_cw{current_window}_cp{current_period}.txt', 'w') as f:
            f.write(x_path + '\n')
            f.write(y_path + '\n')
            f.write(str(num_samples) + '\n')
            f.write(str(hash_value) + '\n')
    except Exception as e:
        print(f"Failed to write paths to file: {e}")
    return x_path, y_path

def create_rolling_windows(df, df_scaled, current_threshold, input_window, config): # work BTC and TON and VOLUME
    feature_columns = config.numeric
    output_window = input_window  # Предсказываем на столько же периодов вперед, сколько и входных данных
    num_samples = len(df) - input_window - output_window
    num_features = len(feature_columns)

    hash_value = hash_data_blake2b(df, df_scaled, current_threshold, input_window, feature_columns)
    #print("Уникальный хеш данных:", hash_value)
    if is_hashfile_in_folder(f'temp/{config.ii_path}/mmap/', hash_value):
        print(f"Файл rolling window c hash {hash_value} существует в папке temp/{config.ii_path}/mmap/")
        return f'temp/{config.ii_path}/mmap/{hash_value}_x.dat', f'temp/{config.ii_path}/mmap/{hash_value}_y.dat', num_samples, hash_value

    # Создание memmap файлов
    #uuid_mmap = generate_uuid()
    if not os.path.exists('temp'):
        os.makedirs('temp')
    x_mmap = np.memmap(f'temp/{config.ii_path}/mmap/{hash_value}_x.dat', dtype=np.float32, mode='w+', shape=(num_samples, input_window, num_features))
    y_mmap = np.memmap(f'temp/{config.ii_path}/mmap/{hash_value}_y.dat', dtype=np.int8, mode='w+', shape=(num_samples, output_window))

    for i in range(num_samples):
        if i % 20000 == 0:
            print(f'create window {i} from {len(df)}')
        x_mmap[i] = df_scaled[feature_columns].iloc[i:(i + input_window)].values
        future_prices = df['close'].iloc[(i + input_window):(i + input_window + output_window)]
        closing_price = df['close'].iloc[i + input_window - 1]
        changes = (future_prices - closing_price) / closing_price
        y_mmap[i] = (np.any(changes >= current_threshold)).astype(int)

    # Синхронизация данных с диском и закрытие файлов
    x_mmap.flush()
    y_mmap.flush()
    x_mmap._mmap.close()
    y_mmap._mmap.close()
    del x_mmap, y_mmap
    return f'temp/{config.ii_path}/mmap/{hash_value}_x.dat', f'temp/{config.ii_path}/mmap/{hash_value}_y.dat', num_samples, hash_value


def hash_data_blake2b(df, df_scaled, current_threshold, current_window, feature_columns):
    df_str = df.to_json()
    df_scaled_str = df_scaled.to_json()
    threshold_str = str(current_threshold)
    window_str = str(current_window)
    feature_columns_str = json.dumps(feature_columns, sort_keys=True)

    combined_str = df_str + df_scaled_str + threshold_str + window_str + feature_columns_str
    hash_object = hashlib.blake2b(combined_str.encode(), digest_size=8)  # Хеш длиной 8 байт (16 символов)

    return hash_object.hexdigest()
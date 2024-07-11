from utils.path import create_dataframe
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
from utils.path import generate_uuid
import os

def run_multiprocessing_rolling_window(coin, period, date_df, window_size, threshold, numeric, ii_path):
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        for current_period in period:
            df = create_dataframe(coin=coin, period=current_period, data=date_df)
            for current_window in window_size:
                for current_threshold in threshold:
                    # Запускаем процесс
                    futures.append(executor.submit(process_data, df.copy(), current_window, current_threshold, current_period, numeric, ii_path))
        # Обработка завершенных задач
        for future in as_completed(futures):
            x_path, y_path = future.result()

def process_data(df, current_window, current_threshold, current_period, numeric, ii_path):
    df.ffill(inplace=True)
    df['pct_change'] = df['close'].pct_change(periods=current_window)
    df['bullish_volume'] = df['volume'] * (df['close'] > df['open'])
    df['bearish_volume'] = df['volume'] * (df['close'] < df['open'])
    df[numeric] = df[numeric].astype(np.float32)
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[numeric] = scaler.fit_transform(df_scaled[numeric])
    joblib.dump(scaler, f'temp/{ii_path}/scaler/scaler_ct{current_threshold}_cw{current_window}_cp{current_period}.gz')
    x_path, y_path, num_samples = create_rolling_windows(df, df_scaled, current_threshold, current_window, numeric, ii_path)
    try:
        with open(f'temp/{ii_path}/roll_win/roll_path_ct{current_threshold}_cw{current_window}_cp{current_period}.txt', 'w') as f:
            f.write(x_path + '\n')
            f.write(y_path + '\n')
            f.write(str(num_samples) + '\n')
    except Exception as e:
        print(f"Failed to write paths to file: {e}")
    return x_path, y_path

def create_rolling_windows(df, df_scaled, current_threshold, input_window, numeric, ii_path): # work BTC and TON and VOLUME
    output_window = input_window  # Предсказываем на столько же периодов вперед, сколько и входных данных
    num_samples = len(df) - input_window - output_window
    feature_columns = numeric
    num_features = len(feature_columns)

    # Создание memmap файлов
    uuid_mmap = generate_uuid()
    if not os.path.exists('temp'):
        os.makedirs('temp')
    x_mmap = np.memmap(f'temp/{ii_path}/mmap/{uuid_mmap}_x.dat', dtype=np.float32, mode='w+', shape=(num_samples, input_window, num_features))
    y_mmap = np.memmap(f'temp/{ii_path}/mmap/{uuid_mmap}_y.dat', dtype=np.int8, mode='w+', shape=(num_samples, output_window))

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
    return f'temp/{ii_path}/mmap/{uuid_mmap}_x.dat', f'temp/{ii_path}/mmap/{uuid_mmap}_y.dat', num_samples

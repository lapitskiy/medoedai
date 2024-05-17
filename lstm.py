import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping


#from test import goTest
from utils import create_dataframe

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import joblib
window_size = [3, 5, 7, 9, 11, 13, 15]
threshold = [0.01, 0.02]
date_df = ['2024-04','2024-05'] # 1m
period = ["1m", "5m", "15m", "30m"]
coin = 'TONUSDT'
layer = '45day-1layer-50'
numeric = ['open', 'high', 'low', 'close', 'volume', 'bullish_volume', 'bearish_volume']


def goLSTM(current_period: str, current_window: int, current_threshold: float):

    directory_save = f'keras_model/lstm/{coin}/{layer}/{current_period}/{current_window}/{current_threshold}'
    if not os.path.exists(directory_save):
        os.makedirs(directory_save)
    df = create_dataframe(coin=coin, period=current_period, data=date_df)
    df['pct_change'] = df['high'].pct_change(periods=current_window)
    # Рассчитываем Bullish и Bearish объемы
    df['bullish_volume'] = df.apply(lambda row: row['volume'] if row['close'] > row['open'] else 0, axis=1)
    df['bearish_volume'] = df.apply(lambda row: row['volume'] if row['close'] < row['open'] else 0, axis=1)

    # Удаление строк с NaN, образовавшихся после pct_change
    df.dropna(subset=['pct_change'], inplace=True)
    scaler = MinMaxScaler()
    # Нормализация данных
    numeric_features = numeric
    df_scaled = df.copy()
    df_scaled[numeric_features] = scaler.fit_transform(df_scaled[numeric_features])

    joblib.dump(scaler, f'{directory_save}/{current_period}.gz')

    df_scaled = pd.DataFrame(df[numeric_features], columns=numeric_features)
    print(df.tail(2))

    # Установка размера окна
    x_path, y_path = create_rolling_windows(df, df_scaled, current_threshold, current_window)

    # Загрузка memmap массивов
    num_samples = len(df_scaled) - current_window
    feature_columns = numeric
    num_features = len(feature_columns)
    x = np.memmap(x_path, dtype=np.float32, mode='r', shape=(num_samples, current_window, num_features))
    y = np.memmap(y_path, dtype=np.int8, mode='r', shape=(num_samples,))

    # X теперь содержит входные данные для сети, y - целевые значения
    print("Shape of X:", x.shape)
    print("Shape of y:", y.shape)

    # Создание модели LSTM
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1, activation='sigmoid'))  # Используем сигмоидальную функцию активации для бинарной классификации
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])  # Используем бинарную кроссэнтропию
    # Обучение модели
    model.fit(x, y, batch_size=1, epochs=1)

    # Сохранение модели
    model.save(f'{directory_save}/{current_period}.h5')

    # Вывод структуры модели
    model.summary()

    # Вывод подготовленных данных для проверки
    print(f'X shape: {x.shape}')
    print(f'y shape: {y.shape}')
    print(df_scaled.tail(2))

def create_rolling_windows(df_not_scaled, df, current_threshold ,current_window): # work BTC and TON and VOLUME
    num_samples = len(df) - current_window
    feature_columns = numeric
    num_features = len(feature_columns)

    # Создание memmap файлов
    x_mmap = np.memmap('x_mmap.dat', dtype=np.float32, mode='w+', shape=(num_samples, current_window, num_features))
    y_mmap = np.memmap('y_mmap.dat', dtype=np.int8, mode='w+', shape=(num_samples,))

    for i in range(num_samples):
        if i % 10000 == 0:
            print(f'create window {i} from {len(df)}')
        change = df_not_scaled['pct_change'].iloc[i + current_window]

        # Запись в memmap
        x_mmap[i] = df[feature_columns].iloc[i:i + current_window].values
        y_mmap[i] = 1 if abs(change) >= current_threshold else 0

    # Синхронизация данных с диском и закрытие файлов
    x_mmap.flush()
    y_mmap.flush()
    del x_mmap, y_mmap
    return 'x_mmap.dat', 'y_mmap.dat'



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for current_period in period:
        for current_window in window_size:
            for current_threshold in threshold:
                goLSTM(current_period=current_period, current_window=current_window, current_threshold=current_threshold)



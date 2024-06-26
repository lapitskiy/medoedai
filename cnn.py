import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall, AUC

#from test import goTest
from utils import create_dataframe

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import joblib
window_size = [3, 4, 5, 6, 7, 8, 9, 10]
threshold = [0.01, 0.02, 0.03]
date_df = ['2024-03','2024-04','2024-05']
period = ["1m", "3m", "5m", "15m", "30m"]
coin = 'TONUSDT'

def goCNN(current_period: str, current_window: int, current_threshold: float):

    directory_save = f'keras_model/{coin}/{current_period}/{current_window}/{current_threshold}'
    if not os.path.exists(directory_save):
        os.makedirs(directory_save)
    df = create_dataframe(coin=coin, period=current_period, data=date_df)
    df['pct_change'] = df['close'].pct_change(periods=current_window)
    scaler = MinMaxScaler()

    # Нормализация данных
    numeric_features = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    joblib.dump(scaler, f'{directory_save}/{current_period}.gz')

    df_scaled = pd.DataFrame(df[numeric_features], columns=numeric_features)
    print(df.tail(2))

    # Вычисление количества 5-минутных интервалов в апреле (29 дней)
    intervals_per_day = 24 * 60 // 5  # Количество интервалов в одних сутках
    total_intervals = intervals_per_day * 29  # Общее количество интервалов в месяце

    # Установка размера окна
    x, y = create_rolling_windows(df, df_scaled, current_threshold, current_window)

    # X теперь содержит входные данные для сети, y - целевые значения
    print("Shape of X:", x.shape)
    print("Shape of y:", y.shape)

    # Для начала разделите ваши данные на обучающий и тестовый наборы.
    # Обучающий набор будет использоваться для тренировки модели,
    # а тестовый набор — для оценки её производительности.
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Создайте архитектуру CNN. Пример простой CNN для временных рядов:
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1, activation='sigmoid')  # Используем сигмоид для бинарной классификации
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Устанавливаем EarlyStopping
    early_stopper = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopper])

    # Теперь оцениваем модель на основе точности
    accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f'Test Accuracy: {accuracy[1] * 100:.2f}%')


    unique, counts = np.unique(y_train, return_counts=True)
    print("Training class distribution:", dict(zip(unique, counts)))



    model.save(f'{directory_save}/{current_period}.keras')

    '''
    predictions = model.predict(X_test)
    # Переводим предсказания в классы
    predicted_classes = (predictions > 0.5).astype(int)

    # Отображение истории обучения
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('teach.png')

    # Сравнение прогнозов с реальностью
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('Actual vs Predicted')
    plt.ylabel('Value')
    plt.xlabel('Sample Index')
    plt.legend()
    plt.savefig('plot.png')
    plt.close()
    '''
def create_rolling_windows(df_not_scaled, df, current_threshold ,current_window): # work BTC and TON and VOLUME
    x = []
    y = []      # 1% изменение

    for i in range(len(df) - current_window):
        if i % 4000 == 0:
            print(f'create window {i} from {len(df)}')

        #start_price = df['close'].iloc[i]
        #end_price = df['close'].iloc[i + window_size]
        # Вычисление процентного изменения
        change = df_not_scaled['pct_change'].iloc[i + current_window]  # Использование ранее рассчитанного изменения
        #print(f'change {i}: {change}; start_price {start_price}; end_price {end_price}')
        x.append(df[['open', 'high', 'low', 'close', 'volume']].iloc[i:i + current_window].values) # 'volume', 'count', 'taker_buy_volume'
        # Создание бинарной целевой переменной
        y.append(1 if abs(change) >= current_threshold else 0)
    return np.array(x), np.array(y)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for current_period in period:
        for current_window in window_size:
            for current_threshold in threshold:
                goCNN(current_period=current_period, current_window=current_window, current_threshold=current_threshold)



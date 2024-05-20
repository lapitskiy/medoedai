

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall, AUC



import tensorflow as tf
from tensorflow.keras import backend as K

import concurrent.futures
import logging
import os
import psutil

#from test import goTest
from utils import create_dataframe, delete_folder, generate_uuid

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import joblib
date_df = ['2024-03','2024-04','2024-05'] # 1m
coin = 'TONUSDT'
#layer = '75day-2layer250-Dropout02'
numeric = ['open', 'high', 'low', 'close', 'bullish_volume', 'bearish_volume']


def goLSTM(current_period: str, current_window: int, current_threshold: float, current_neiron: int, current_dropout: float):
    try:
        df = create_dataframe(coin=coin, period=current_period, data=date_df)
        df['pct_change'] = df['close'].pct_change(periods=current_window)
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

        df_scaled = pd.DataFrame(df[numeric_features], columns=numeric_features)
        print(df.tail(15))

        # Установка размера окна
        x_path, y_path = create_rolling_windows(df, df_scaled, current_threshold, current_window)

        # Загрузка memmap массивов
        num_samples = len(df_scaled) - current_window
        feature_columns = numeric
        num_features = len(feature_columns)
        x = np.memmap(f'{x_path}', dtype=np.float32, mode='r', shape=(num_samples, current_window, num_features))
        y = np.memmap(f'{y_path}', dtype=np.int8, mode='r', shape=(num_samples,))

        # X теперь содержит входные данные для сети, y - целевые значения
        #print("Shape of X:", x.shape)
        #print("Shape of y:", y.shape)


        # Сначала разделите данные на обучающий+валидационный и тестовый наборы
        x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Затем разделите обучающий+валидационный набор на обучающий и валидационный наборы
        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25,
                                                          random_state=42)  # 0.25 x 0.8 = 0.2

        # Создание модели LSTM
        model = Sequential()
        model.add(LSTM(current_neiron, return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
        model.add(Dropout(current_dropout))
        model.add(LSTM(current_neiron, return_sequences=False))
        model.add(Dropout(current_dropout))  # Добавление слоя Dropout
        model.add(Dense(25))
        model.add(Dense(1, activation='sigmoid'))  # Используем сигмоидальную функцию активации для бинарной классификации
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy', Precision(), Recall(), AUC(), f1_score])  # Используем бинарную кроссэнтропию
        # Обучение модели
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = dict(enumerate(class_weights))

        history = model.fit(x_train, y_train, batch_size=1, epochs=1, validation_split=0.2, class_weight=class_weights_dict)  # добавление некоторого количества данных для валидации

        # Подсчёт количества примеров каждого класса и вывод информации
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))


        # Оценка модели
        scores = model.evaluate(x_test, y_test, verbose=1)

        #
        # подбора порога классификации
        #
        y_val_probs = model.predict(x_val)
        # Вычисление Precision и Recall для различных порогов
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_probs)
        # Вычисление F1-score для каждого порога
        f1_scores = 2 * (precisions * recalls) / (
                    precisions + recalls + 1e-10)  # добавляем маленькое число для избежания деления на ноль
        # Найти порог, который максимизирует F1-score
        opt_idx = np.argmax(f1_scores)
        opt_threshold = thresholds[opt_idx]
        opt_f1_score = f1_scores[opt_idx]

        print("Распределение классов:", class_distribution)
        print(f"Test Accuracy: {scores[1] * 100:.2f}%")
        print(f"Test Precision: {scores[2]:.2f}")
        print(f"Test Recall: {scores[3]:.2f}")
        print(f"Test AUC: {scores[4]:.2f}")
        print(f"Test F1-Score: {scores[-1]:.2f}")

        print("Оптимальный порог:", opt_threshold)
        print("Максимальный F1-Score:", opt_f1_score)

        # Получение предсказаний на тестовом наборе
        y_test_probs = model.predict(x_test)
        y_test_pred = (y_test_probs >= opt_threshold).astype(int)


        # Оценка производительности
        print(classification_report(y_test, y_test_pred))
        conf_matrx_test = confusion_matrix(y_test, y_test_pred)
        print("Confusion Matrix:\n", conf_matrx_test)

        if (scores[1] * 100 >= 80 and
                scores[2] >= 0.60 and
                scores[3] >= 0.60 and
                scores[4] >= 0.70 and
                scores[-1] >= 0.60):

            # Подготовка данных для записи в файл
            data = {
                "Test Accuracy": f"{scores[1] * 100:.2f}%",
                "Test Precision": f"{scores[2]:.2f}",
                "Test Recall": f"{scores[3]:.2f}",
                "Test AUC": f"{scores[4]:.2f}",
                "Test F1-Score": f"{scores[-1]:.2f}",
                "Optimal Threshold": opt_threshold,
                "Maximum F1-Score": opt_f1_score,
                "Confusion Matrix": conf_matrx_test,
                "current_period": f"{current_period}",
                "current_window": f"{current_window}",
                "current_threshold": f"{current_threshold}",
                "current_neiron": f"{current_neiron}",
                "current_dropout": f"{current_dropout}",
                "class_distribution": class_distribution,
            }

            directory_save = f'keras_model/lstm/{coin}/{current_period}/{current_window}/{current_threshold}/{current_neiron}/{current_dropout}/'
            if not os.path.exists(directory_save):
                os.makedirs(directory_save)

            # Запись данных в файл
            with open(f"{directory_save}/results.txt", "w") as file:
                for key, value in data.items():
                    file.write(f"{key}={value}\n")

            joblib.dump(scaler, f'{directory_save}/{current_period}.gz')
            # Сохранение модели
            model.save(f'{directory_save}/{current_period}.h5')

            # Вывод структуры модели
            model.summary()

        else:
            print('Model not good')
    except Exception as e:
        logging.error(f"Error in task: {current_period, current_window, current_threshold, current_neiron, current_dropout}, Exception: {str(e)}")
        return None



def create_rolling_windows(df_not_scaled, df, current_threshold ,current_window): # work BTC and TON and VOLUME
    num_samples = len(df) - current_window
    feature_columns = numeric
    num_features = len(feature_columns)

    # Создание memmap файлов
    uuid_mmap = generate_uuid()
    x_mmap = np.memmap(f'temp/{uuid_mmap}_x.dat', dtype=np.float32, mode='w+', shape=(num_samples, current_window, num_features))
    y_mmap = np.memmap(f'temp/{uuid_mmap}_y.dat', dtype=np.int8, mode='w+', shape=(num_samples,))

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
    return f'temp/{uuid_mmap}_x.dat', f'temp/{uuid_mmap}_y.dat'

def f1_score(y_true, y_pred):
    # Первое, что нам нужно сделать, это убедиться, что обе переменные имеют тип float32
    y_true = K.cast(y_true, 'float32')
    y_pred = K.round(y_pred)  # y_pred округляется до ближайшего целого и также приводится к float32

    # Вычисляем количество истинно положительных срабатываний
    tp = K.sum(K.cast(y_true * y_pred, 'float32'), axis=0)
    # Ложноположительные срабатывания
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float32'), axis=0)
    # Ложноотрицательные срабатывания
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float32'), axis=0)

    # Precision
    precision = tp / (tp + fp + K.epsilon())
    # Recall
    recall = tp / (tp + fn + K.epsilon())

    # Вычисляем F1 score
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    # Обработка случая NaN в f1
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    logging.basicConfig(
        filename='training.log',  # Имя файла логирования
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    window_size = [10,15,20,25,30,35,40,45,50,55,60]
    threshold = [0.005, 0.01, 0.02, 0.03, 0.04]
    period = ["5m",]
    neiron = [50,100,150,200,250,300,350,400,450]
    dropout = [0.1, 0.2, 0.3, 0.4]


    # Создание списка всех возможных комбинаций параметров
    all_tasks = [(p, w, t, n, d) for p in period for w in window_size for t in threshold for n in neiron for d in dropout]
    max_workers = min(4, len(all_tasks))
    # Использование ProcessPoolExecutor для параллельного выполнения
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Запуск процессов
        futures = [executor.submit(goLSTM, *task) for task in all_tasks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                print(result)
            else:
                print("A task failed. See logs for more details.")
            # Логирование использования памяти
            memory_info = psutil.virtual_memory()
            logging.info(f"Memory usage: {memory_info.percent}%")



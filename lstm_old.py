import stat
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, Activation, LeakyReLU, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam
from itertools import product

import os

import tensorflow as tf
from tensorflow.keras import backend as K

import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

import psutil

#from test import goTest
from utils import create_dataframe, delete_folder, generate_uuid, path_exist, read_x_y_ns_path, clear_folder

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import joblib
date_df = ['2024-03','2024-04','2024-05'] # 1m
coin = 'TONUSDT'
#layer = '75day-2layer250-Dropout02'
numeric = ['open', 'high', 'low', 'close', 'bullish_volume', 'bearish_volume']
memmap = True

metric_thresholds = {
    'val_accuracy': 0.75,
    'val_precision': 0.80,
    'val_recall': 0.70,
    'val_f1_score': 0.75,
    'val_auc': 0.85
}

class ModelCheckpointWithMetricThreshold(Callback):
    def __init__(self, filepath, thresholds, verbose=1):
        super(ModelCheckpointWithMetricThreshold, self).__init__()
        self.filepath = filepath  # Путь для сохранения лучшей модели
        self.thresholds = thresholds  # Словарь с пороговыми значениями для метрик
        self.best_metrics = {key: 0 for key in thresholds.keys()}  # Лучшие метрики
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        current_metrics = {key: logs.get(key) for key in self.thresholds.keys()}
        if all(current_metrics[key] >= self.thresholds[key] for key in self.thresholds.keys()):
            improved = False
            # Проверяем, улучшилась ли хотя бы одна метрика
            for key in self.thresholds.keys():
                if current_metrics[key] > self.best_metrics[key]:
                    self.best_metrics[key] = current_metrics[key]
                    improved = True
            if improved:
                if self.verbose > 0:
                    print("\nMetrics at the end of epoch {}:".format(epoch + 1))
                    print(f"Test Accuracy: {logs.get('val_accuracy') * 100:.2f}%")
                    print(f"Test Precision: {logs.get('val_precision'):.2f}")
                    print(f"Test Recall: {logs.get('val_recall'):.2f}")
                    print(f"Test AUC: {logs.get('val_auc'):.2f}")
                    print(f"Test F1-Score: {logs.get('val_f1_score'):.2f}")
                    print(f'\nMetrics improved. Saving model to {self.filepath}')
                self.model.save(self.filepath, overwrite=True)

def goLSTM(current_period: str, current_window: int, current_threshold: float, current_neiron: int, current_dropout: float,
           current_batch_size: int, current_epochs: int, current_activation: str):
    #df = pd.read_pickle(f'temp/df/prd-{current_period}_win-{current_window}.pkl')

    x_path, y_path, num_samples = read_x_y_ns_path(f'temp/roll_win/roll_path_ct-{current_threshold}_cw-{current_window}.txt')

    # Загрузка memmap массивов
    num_features = len(numeric)
    x = np.memmap(f'{x_path}', dtype=np.float32, mode='r', shape=(int(num_samples), current_window, num_features))
    y = np.memmap(f'{y_path}', dtype=np.int8, mode='r', shape=(int(num_samples),))

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
    model.add(Input(shape=(x.shape[1], x.shape[2])))
    model.add(LSTM(current_neiron, return_sequences=True))
    model.add(Dropout(current_dropout))
    model.add(LSTM(current_neiron, return_sequences=True))
    model.add(Dropout(current_dropout))  # Добавление слоя Dropout
    model.add(LSTM(current_neiron, return_sequences=False))
    model.add(Dropout(current_dropout))  # Добавление слоя Dropout
    model.add(Dense(25))
    if current_activation.lower() == 'leakyrelu':
        model.add(Dense(1))
        model.add(LeakyReLU())
    else:
        model.add(Dense(1, activation=current_activation))  # Используем сигмоидальную функцию активации для бинарной классификации
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=[
                      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                      tf.keras.metrics.Precision(name='precision'),
                      tf.keras.metrics.Recall(name='recall'),
                      tf.keras.metrics.AUC(name='auc'),
                      f1_score])
    # Обучение модели
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))

    checkpoint = ModelCheckpointWithMetricThreshold('best_model.h5', metric_thresholds)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

    #csv_logger = CSVLogger(f'temp/lstm_log/{generate_uuid()}_.csv', append=True)
    history = model.fit(x_train, y_train, batch_size=current_batch_size, epochs=current_epochs, validation_data=(x_val, y_val),
                        class_weight=class_weights_dict, callbacks=[checkpoint, early_stopping])


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
    conf_matrx_test = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:\n", conf_matrx_test)

    # Подготовка данных для записи в файл
    data = {
        "Test Accuracy": f"{scores[1] * 100:.2f}%",
        "Test Precision": f"{scores[2]:.2f}",
        "Test Recall": f"{scores[3]:.2f}",
        "Test AUC": f"{scores[4]:.2f}",
        "Test F1-Score": f"{scores[-1]:.2f}",
        "Optimal Threshold": opt_threshold,
        "Maximum F1-Score": opt_f1_score,
        "batch_sizes": current_batch_size,
        "epoch": current_epochs,
        "activation": current_activation,
        "Confusion Matrix": conf_matrx_test,
        "current_period": f"{current_period}",
        "current_window": f"{current_window}",
        "current_threshold": f"{current_threshold}",
        "current_neiron": f"{current_neiron}",
        "current_dropout": f"{current_dropout}",
        "class_distribution": class_distribution,
    }

    if (scores[1] * 100 >= 70 and
            scores[2] >= 0.58 and
            scores[3] >= 0.58 and
            scores[4] >= 0.65 and
            scores[-1] >= 0.58):



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
        del x
        del y
        return 'Good model'
    else:
        if (scores[1] * 100 >= 60 and
                scores[2] >= 0.60 and
                scores[3] >= 0.60 and
                scores[4] >= 0.60 and
                scores[-1] >= 0.60):
            directory_save = f'keras_model/lstm/NotGood/{coin}/'
            if not os.path.exists(directory_save):
                os.makedirs(directory_save)
            # Запись данных в файл
            with open(f"{directory_save}/{generate_uuid()}.txt", "w") as file:
                for key, value in data.items():
                    file.write(f"{key}={value}\n")
    del x
    del y
    return 'Bad model'


def create_rolling_windows(df_not_scaled, df, current_threshold ,current_window): # work BTC and TON and VOLUME
    num_samples = len(df) - current_window
    feature_columns = numeric
    num_features = len(feature_columns)

    # Создание memmap файлов
    uuid_mmap = generate_uuid()
    if not os.path.exists('temp'):
        os.makedirs('temp')
    x_mmap = np.memmap(f'temp/{uuid_mmap}_x.dat', dtype=np.float32, mode='w+', shape=(num_samples, current_window, num_features))
    y_mmap = np.memmap(f'temp/{uuid_mmap}_y.dat', dtype=np.int8, mode='w+', shape=(num_samples,))

    for i in range(num_samples):
        if i % 20000 == 0:
            print(f'create window {i} from {len(df)}')
        change = df_not_scaled['pct_change'].iloc[i + current_window]

        # Запись в memmap
        x_mmap[i] = df[feature_columns].iloc[i:i + current_window].values
        y_mmap[i] = 1 if abs(change) >= current_threshold else 0

    # Синхронизация данных с диском и закрытие файлов
    x_mmap.flush()
    y_mmap.flush()
    x_mmap._mmap.close()
    y_mmap._mmap.close()
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

def process_data(df, current_window, current_threshold, numeric):
    df.ffill(inplace=True)
    df['pct_change'] = df['close'].pct_change(periods=current_window)
    df['bullish_volume'] = df['volume'] * (df['close'] > df['open'])
    df['bearish_volume'] = df['volume'] * (df['close'] < df['open'])

    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[numeric] = scaler.fit_transform(df_scaled[numeric])
    num_samples = len(df_scaled) - current_window
    x_path, y_path = create_rolling_windows(df, df_scaled, current_threshold, current_window)
    try:
        with open(f'temp/roll_win/roll_path_ct-{current_threshold}_cw-{current_window}.txt', 'w') as f:
            f.write(x_path + '\n')
            f.write(y_path + '\n')
            f.write(str(num_samples) + '\n')
    except Exception as e:
        print(f"Failed to write paths to file: {e}")
    return x_path, y_path

def run_multiprocessing_rolling_window():
    with ProcessPoolExecutor() as executor:
        futures = []
        for current_period in period:
            df = create_dataframe(coin=coin, period=current_period, data=date_df)
            for current_window in window_size:
                for current_threshold in threshold:
                    # Запускаем процесс
                    futures.append(executor.submit(process_data, df.copy(), current_window, current_threshold, numeric))

        # Обработка завершенных задач
        for future in as_completed(futures):
            x_path, y_path = future.result()
            print(x_path, y_path)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.config.threading.set_intra_op_parallelism_threads(20)
    tf.config.threading.set_inter_op_parallelism_threads(20)

    tf.get_logger().setLevel('ERROR')
    log_file = os.path.expanduser('~/training.log')

    logging.basicConfig(
        filename=log_file,  # Имя файла логирования
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # period = ["5m", ]
    # window_size = [5,7,14,19,24,32,48,53,59]
    # threshold = [0.005, 0.006, 0.007, 0.008, 0.009]
    # neiron = [60,90,130,160,190,210]
    # dropout = [0.15, 0.25, 0.35]
    # batch_sizes = [96, 128]
    # epochs_list = [6,7,8,9,10]
    # activations = ['sigmoid', 'relu','tanh','LeakyReLU','elu']

    period = ["5m", ]
    window_size = [5,]
    threshold = [0.009,]
    neiron = [50,100,150,200]
    dropout = [0.15, 0.25, 0.35]
    batch_sizes = [32, 64]
    epochs_list = [4,6,8,10,12]
    activations = ['sigmoid', 'relu','tanh','LeakyReLU','elu']

    # создание данных для ии, которые могут загружаться повторно
    clear_folder('temp/')
    path_exist('temp/roll_win/')
    run_multiprocessing_rolling_window()

    '''
    for current_period in period:
        df = create_dataframe(coin=coin, period=current_period, data=date_df)
        for current_window in window_size:
            df_copy = df.copy()
            df_copy['pct_change'] = df_copy['close'].pct_change(periods=current_window)
            # Рассчитываем Bullish и Bearish объемы
            df_copy['bullish_volume'] = df_copy.apply(lambda row: row['volume'] if row['close'] > row['open'] else 0, axis=1)
            df_copy['bearish_volume'] = df_copy.apply(lambda row: row['volume'] if row['close'] < row['open'] else 0, axis=1)
            # Удаление строк с NaN, образовавшихся после pct_change
            df_copy.dropna(subset=['pct_change'], inplace=True)
            #pkl_df_copy.to_pickle(f'temp/df/prd-{pkl_period}_win-{pkl_window}.pkl')
            #print(f'pkl create prd-{pkl_period}_win-{pkl_window}.pkl')

            scaler = MinMaxScaler()
            # Нормализация данных
            df_scaled = df_copy.copy()
            df_scaled[numeric] = scaler.fit_transform(df_scaled[numeric])
            #df_scaled = pd.DataFrame(pkl_df_copy[numeric], columns=numeric)
            print(df_copy.tail(1))
            print(df_scaled.tail(1))
            for current_threshold in threshold:
                x_path, y_path = create_rolling_windows(df_copy, df_scaled, current_threshold, current_window)
                with open(f'temp/roll_win/roll_path_ct-{current_threshold}_cw-{current_window}.txt', 'w') as f:
                    f.write(x_path + '\n')
                    f.write(y_path + '\n')
    '''
    task_count = list(product(period, window_size, threshold, neiron, dropout, batch_sizes, epochs_list, activations))
    total_iterations = len(task_count)

    all_tasks = [(p, w, t, n, d, b, e, a) for p in period for w in window_size for t in threshold for n in neiron for d
                 in dropout for b in batch_sizes for e in epochs_list for a in activations]

    max_workers = min(4, len(all_tasks)) #max_workers=max_workers

    start_time = time.perf_counter()


    # Использование ProcessPoolExecutor для параллельного выполнения
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Запуск процессов
        futures = [executor.submit(goLSTM, *task) for task in all_tasks]
        for iteration, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            iteration_start_time = time.perf_counter()
            result = future.result()
            iteration_end_time = time.perf_counter()
            iteration_time = iteration_end_time - iteration_start_time
            print(f'Iteration {iteration}: iteration_time {iteration_time}')

            if result is not None:
                print(result)
            else:
                print("A task failed. See logs for more details.")
            # Логирование использования памяти
            memory_info = psutil.virtual_memory()
            logging.info(f"Memory usage: {memory_info.percent}%")

    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Total time taken for all iterations: {total_time:.2f} seconds")
    logging.info(f"Total time taken for all iterations: {total_time:.2f} seconds")

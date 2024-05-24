import shutil
import stat
import time
import gc

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, Activation, LeakyReLU, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall, AUC, CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from keras.callbacks import Callback
from itertools import product

import os

import tensorflow as tf
from tensorflow.keras import backend as K

import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

import psutil

#from test import goTest
from utils import create_dataframe, delete_folder, generate_uuid, path_exist, clear_folder, read_temp_path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import joblib
date_df = ['2024-03',] # 1m
coin = 'TONUSDT'
#layer = '75day-2layer250-Dropout02'
numeric = ['open', 'high', 'low', 'close', 'bullish_volume', 'bearish_volume']
memmap = True



metric_thresholds = {
    'val_accuracy': 0.60,
    'val_precision': 0.60,
    'val_recall': 0.60,
    'val_f1_score': 0.10,
    'val_auc': 0.60
}

def goLSTM(current_period: str, current_window: int, current_threshold: float, current_neiron: int, current_dropout: float,
           current_batch_size: int, current_epochs: int):
    #df = pd.read_pickle(f'temp/df/prd-{current_period}_win-{current_window}.pkl')

    x_path, y_path, num_samples = read_temp_path(f'temp/roll_win/roll_path_ct-{current_threshold}_cw-{current_window}.txt')

    # Загрузка memmap массивов
    num_features = len(numeric)

    #test_memory(int(num_samples), current_window, num_features)
    #check_memory()

    x = np.memmap(f'{x_path}', dtype=np.float32, mode='r', shape=(int(num_samples), current_window, num_features))
    y = np.memmap(f'{y_path}', dtype=np.int8, mode='r', shape=(int(num_samples), current_window))
    y = to_categorical(y, num_classes=3)

    y_last = y[:, -1, :]
    print(f"Shape of y_last: {y_last.shape}")

    # X теперь содержит входные данные для сети, y - целевые значения
    #print("Shape of X:", x.shape)
    #print("Shape of y:", y.shape)


    # Сначала разделите данные на обучающий+валидационный и тестовый наборы
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y_last, test_size=0.2, random_state=42)

    # Затем разделите обучающий+валидационный набор на обучающий и валидационный наборы
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25,
                                                      random_state=42)  # 0.25 x 0.8 = 0.2
    print(f"Shape of x_train: {x_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of x_val: {x_val.shape}")
    print(f"Shape of y_val: {y_val.shape}")

    # Создание модели LSTM
    model = Sequential()
    model.add(Input(shape=(current_window, num_features)))
    model.add(LSTM(current_neiron, return_sequences=True))
    model.add(Dropout(current_dropout))
    model.add(LSTM(current_neiron, return_sequences=True))
    model.add(Dropout(current_dropout))  # Добавление слоя Dropout
    model.add(LSTM(current_neiron, return_sequences=False))
    model.add(Dropout(current_dropout))  # Добавление слоя Dropout
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=[
                      CategoricalAccuracy(name='accuracy'),
                      MacroPrecision(),
                      MacroRecall(),
                      tf.keras.metrics.AUC(name='auc', multi_label=True),
                      macro_f1
                      ])
    # Обучение модели
    # Проверка корректности y_train_labels и class_labels
    y_train_labels = np.argmax(y_train, axis=-1).flatten()
    class_labels = np.unique(y_train_labels)
    class_weights = compute_class_weight('balanced', classes=class_labels, y=y_train_labels)
    class_weights_dict = dict(enumerate(class_weights))


    data = {
        "batch_sizes": current_batch_size,
        "epoch": current_epochs,
        "current_period": f"{current_period}",
        "current_window": f"{current_window}",
        "current_threshold": f"{current_threshold}",
        "current_neiron": f"{current_neiron}",
        "current_dropout": f"{current_dropout}",
    }

    directory_save = f'keras_model/lstm/{coin}/{current_period}/{current_window}/{current_threshold}/{current_neiron}/{current_dropout}/' \
                     f'{current_epochs}/{current_batch_size}/'

    checkpoint = ModelCheckpointWithMetricThreshold(thresholds=metric_thresholds, filedata=data, directory_save=directory_save, current_threshold=current_threshold,
                                                    current_window=current_window, model=model)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

    #csv_logger = CSVLogger(f'temp/lstm_log/{generate_uuid()}_.csv', append=True)
    history = model.fit(x_train, y_train, batch_size=current_batch_size, epochs=current_epochs, validation_data=(x_val, y_val),
                        class_weight=class_weights_dict, callbacks=[checkpoint, early_stopping])

    # Подсчёт количества примеров каждого класса и вывод информации
    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique, counts))

    # Оценка модели
    test_results = model.evaluate(x_test, y_test, verbose=1)
    #
    # подбора порога классификации
    #
    y_val_probs = model.predict(x_val)
    y_val_labels = np.argmax(y_val, axis=1)
    precisions, recalls, thresholds = calculate_metrics(y_val_labels, y_val_probs[:, 1])
    f1_scores = 2 * (precisions * recalls) / (
                precisions + recalls + 1e-10)  # добавляем маленькое число для избежания деления на ноль
    opt_idx = np.argmax(f1_scores)
    opt_threshold = thresholds[opt_idx]
    opt_f1_score = f1_scores[opt_idx]

    print("Распределение классов:", class_distribution)
    print("Оптимальный порог:", opt_threshold)
    print("Максимальный F1-Score:", opt_f1_score)

    # Получение предсказаний на тестовом наборе
    y_test_probs = model.predict(x_test)
    y_test_pred = (y_test_probs >= opt_threshold).astype(int)
    y_test_labels = np.argmax(y_test, axis=1)
    y_test_pred_labels = np.argmax(y_test_pred, axis=1)
    print(classification_report(y_test_labels, y_test_pred_labels))
    conf_matrx_test = confusion_matrix(y_test_labels, y_test_pred_labels)
    print("Confusion Matrix:\n", conf_matrx_test)
    del x_train, x_val, y_train, y_val, x, y
    gc.collect()
    return 'End epoch'


def create_rolling_windows(df_not_scaled, df, current_threshold, input_window): # work BTC and TON and VOLUME
    output_window = input_window  # Предсказываем на столько же периодов вперед, сколько и входных данных
    num_samples = len(df) - input_window - output_window
    feature_columns = numeric
    num_features = len(feature_columns)

    # Создание memmap файлов
    uuid_mmap = generate_uuid()
    if not os.path.exists('temp'):
        os.makedirs('temp')
    x_mmap = np.memmap(f'temp/{uuid_mmap}_x.dat', dtype=np.float32, mode='w+', shape=(num_samples, input_window, num_features))
    y_mmap = np.memmap(f'temp/{uuid_mmap}_y.dat', dtype=np.int8, mode='w+', shape=(num_samples, output_window))

    for i in range(num_samples):
        if i % 20000 == 0:
            print(f'create window {i} from {len(df)}')
        x_mmap[i] = df[feature_columns].iloc[i:(i + input_window)].values

        # Здесь мы берем следующие output_window значений после окончания входного окна
        future_prices = df['close'].iloc[(i + input_window):(i + input_window + output_window)]
        closing_price = df['close'].iloc[i + input_window - 1]
        changes = (future_prices - closing_price) / closing_price
        y_mmap[i] = [1 if change >= current_threshold else 2 if change <= -current_threshold else 0 for change in changes] #0,1, 2(-1)

    # Синхронизация данных с диском и закрытие файлов
    x_mmap.flush()
    y_mmap.flush()
    x_mmap._mmap.close()
    y_mmap._mmap.close()
    del x_mmap, y_mmap
    return f'temp/{uuid_mmap}_x.dat', f'temp/{uuid_mmap}_y.dat', num_samples


def calculate_metrics(y_true, y_probs):
    """
    Вычисление точности (precision), полноты (recall) и порогов для меток.

    :param y_true: Истинные метки.
    :param y_probs: Вероятности предсказаний.
    :return: Точность, полнота и пороги.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs, pos_label=1)
    return precisions, recalls, thresholds

def macro_f1(y_true, y_pred, num_classes=3):
    # Преобразуем предсказания и метки в one-hot формат
    y_pred = K.one_hot(K.argmax(y_pred, axis=-1), num_classes=num_classes)
    y_true = K.cast(y_true, K.floatx())

    # Вычисляем точность и полноту для каждого класса
    tp = K.sum(y_true * y_pred, axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    # Вычисляем F1-счет для каждого класса
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

    # Возвращаем среднее значение по всем классам
    return K.mean(f1)

def process_data(df, current_window, current_threshold, numeric):
    df.ffill(inplace=True)
    df['pct_change'] = df['close'].pct_change(periods=current_window)
    df['bullish_volume'] = df['volume'] * (df['close'] > df['open'])
    df['bearish_volume'] = df['volume'] * (df['close'] < df['open'])
    df[numeric] = df[numeric].astype(np.float32)
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[numeric] = scaler.fit_transform(df_scaled[numeric])
    joblib.dump(scaler, f'temp/scaler/scaler_ct-{current_threshold}_cw-{current_window}.gz')
    x_path, y_path, num_samples = create_rolling_windows(df, df_scaled, current_threshold, current_window)
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

def compute_memory_requirements(dtype, shape):
    itemsize = np.dtype(dtype).itemsize
    total_size = itemsize * np.prod(shape)
    return total_size

def check_memory():
    memory_info = psutil.virtual_memory()
    print(f"Total memory: {memory_info.total / (1024**2):.2f} MB")
    print(f"Available memory: {memory_info.available / (1024**2):.2f} MB")
    print(f"Used memory: {memory_info.used / (1024**2):.2f} MB")
    print(f"Memory percent: {memory_info.percent}%")

# Пример использования
def test_memory(num_samples, current_window, num_features):
    total_memory_required = num_samples * current_window * num_features * 4 / (1024**2)  # 4 байта на float32
    available_memory = psutil.virtual_memory().available / (1024**2)
    print(f"Memory required: {total_memory_required:.2f} MB")
    print(f"Available memory: {available_memory:.2f} MB")
    if total_memory_required > available_memory:
        raise MemoryError("Not enough memory available for the operation.")


class ModelCheckpointWithMetricThreshold(Callback):
    def __init__(self, thresholds, filedata, directory_save, current_threshold, current_window, model, verbose=1):
        super(ModelCheckpointWithMetricThreshold, self).__init__()
        self.thresholds = thresholds  # Словарь с пороговыми значениями для метрик
        self.best_metrics = {key: 0 for key in thresholds.keys()}  # Лучшие метрики
        self.verbose = verbose
        self.filedata = filedata
        self.directory_save = directory_save
        self.current_threshold = current_threshold
        self.current_window = current_window
        self.mymodel = model

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
                self.filedata.update(
                    {
                        "Test Accuracy": f"{logs.get('val_accuracy') * 100:.2f}%",
                        "Test Precision": f"{logs.get('val_precision'):.2f}",
                        "Test Recall": f"{logs.get('val_recall'):.2f}",
                        "Test AUC": f"{logs.get('val_auc'):.2f}",
                        "Test F1-Score": f"{logs.get('val_f1_score'):.2f}"
                    }
                )
                path_exist(self.directory_save)
                with open(f"{self.directory_save}/results.txt", "w") as file:
                    for key, value in self.filedata.items():
                        file.write(f"{key}={value}\n")
                source_path = f'temp/scaler/scaler_ct-{self.current_threshold}_cw-{self.current_window}.gz'
                destination_path = f'{self.directory_save}/scaler.gz'
                try:
                    shutil.copy(source_path, destination_path)
                except Exception as e:
                    print(f'exc {e}')
                self.mymodel.save(f'{self.directory_save}/model.keras', overwrite=True)
                self.mymodel.summary()
                print(f'Metrics improved. Saving model')

class MacroPrecision(tf.keras.metrics.Metric):
    def __init__(self, name='macro_precision', **kwargs):
        super(MacroPrecision, self).__init__(name=name, **kwargs)
        self.precision = self.add_weight(name='precision', initializer='zeros')
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
        fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32))
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.precision.assign(tp / (tp + fp + tf.keras.backend.epsilon()))

    def result(self):
        return self.precision

    def reset_states(self):
        self.precision.assign(0)
        self.true_positives.assign(0)
        self.false_positives.assign(0)

class MacroRecall(tf.keras.metrics.Metric):
    def __init__(self, name='macro_recall', **kwargs):
        super(MacroRecall, self).__init__(name=name, **kwargs)
        self.recall = self.add_weight(name='recall', initializer='zeros')
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
        fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32))
        self.true_positives.assign_add(tp)
        self.false_negatives.assign_add(fn)
        self.recall.assign(tp / (tp + fn + tf.keras.backend.epsilon()))

    def result(self):
        return self.recall

    def reset_states(self):
        self.recall.assign(0)
        self.true_positives.assign(0)
        self.false_negatives.assign(0)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.config.threading.set_intra_op_parallelism_threads(10)
    tf.config.threading.set_inter_op_parallelism_threads(10)

    tf.get_logger().setLevel('ERROR')
    log_file = os.path.expanduser('~/training.log')

    logging.basicConfig(
        filename=log_file,  # Имя файла логирования
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # period = ["3m", ]
    # window_size = [5,7,14,19,24,32,48,53,59]
    # threshold = [0.005, 0.008, 0.01, 0.012, 0.015]
    # neiron = [60,90,130,160,190,210]
    # dropout = [0.15, 0.25, 0.35]
    # batch_sizes = [1,6,12,24,32]
    # epochs_list = [10,30,60,80,100]


    period = ["5m",]
    window_size = [5,]
    threshold = [0.01,]
    neiron = [50,]
    dropout = [0.15,]
    batch_sizes = [1,]
    epochs_list = [1,]

    # создание данных для ии, которые могут загружаться повторно
    clear_folder('temp/')
    path_exist('temp/roll_win/')
    path_exist('temp/scaler/')

    run_multiprocessing_rolling_window()

    task_count = list(product(period, window_size, threshold, neiron, dropout, batch_sizes, epochs_list))
    total_iterations = len(task_count)

    all_tasks = [(p, w, t, n, d, b, e) for p in period for w in window_size for t in threshold for n in neiron for d
                 in dropout for b in batch_sizes for e in epochs_list]

    max_workers = min(2, len(all_tasks)) #max_workers=max_workers

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

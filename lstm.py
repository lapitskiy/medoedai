import os
import warnings
from utils.rolling import run_multiprocessing_rolling_window
warnings.filterwarnings('ignore', category=FutureWarning)
import shutil
import stat
import time
import gc
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

tfGPU = False
if not tfGPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf

tf.get_logger().setLevel('ERROR')
if tfGPU:
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    tf.config.experimental.list_physical_devices('GPU')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            print("Доступные GPU: ", gpus)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print("Произошла ошибка:", e)


from tensorflow.keras.models import Sequential
import scikeras
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall, AUC, CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Metric
from tensorflow.keras.callbacks import Callback
from itertools import product
from tensorflow.keras.layers import LSTM, Dense, Dropout, LeakyReLU, Input, Bidirectional, BatchNormalization, \
    Conv1D, Attention, GRU, InputLayer, MultiHeadAttention
from tensorflow.keras import backend as K
import logging
import psutil
from utils.path import generate_uuid, path_exist, clear_folder, read_temp_path, save_grid_checkpoint, \
    file_exist
import matplotlib.pyplot as plt
import matplotlib
from utils.models_list import ModelLSTM_2Class, create_model
import joblib
matplotlib.use('Agg')

CPU_COUNT = 3
date_df = ['2024-03','2024-04','2024-05'] # 1m
coin = 'TONUSDT'
numeric = ['open', 'high', 'low', 'close', 'bullish_volume', 'bearish_volume']
checkpoint_file = 'temp/checkpoint/grid_search_checkpoint.txt'
goLSTM = True

metric_thresholds = {
    'val_accuracy': 0.60,
    'val_precision': 0.60,
    'val_recall': 0.60,
    'val_f1_score': 0.10,
    'val_auc': 0.60
}
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

def goLSTM(task):
    list_ = read_temp_path(f'temp/roll_win/roll_path_ct-_cw-{current_window}_cp{period}.txt')
    #x_path, y_path, num_samples
    num_features = len(numeric)


    x = np.memmap(f'{x_path}', dtype=np.float32, mode='r', shape=(int(num_samples), current_window, num_features))
    y = np.memmap(f'{y_path}', dtype=np.int8, mode='r', shape=(int(num_samples),))

    print("Распределение классов в исходных данных:")
    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print(class_distribution)


    # Сначала разделите данные на обучающий+валидационный и тестовый наборы
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Затем разделите обучающий+валидационный набор на обучающий и валидационный наборы
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25,
                                                      random_state=42)  # 0.25 x 0.8 = 0.2

    # После разделения на train и test
    print("Распределение классов в обучающем наборе:")
    unique_train, counts_train = np.unique(y_train_val, return_counts=True)
    print(dict(zip(unique_train, counts_train)))

    print(f"Shape of x_train: {x_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of x_val: {x_val.shape}")
    print(f"Shape of y_val: {y_val.shape}")

    # Создание модели LSTM
    print(f'cur model {current_model}')

    model = ModelLSTM_2Class(model_number=current_model, current_window=current_window, num_features=num_features,
                             current_neiron=current_neiron, current_dropout=current_dropout)
    model.build_model()
    model = model.model

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=[
                      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                      tf.keras.metrics.Precision(name='precision'),
                      tf.keras.metrics.Recall(name='recall'),
                      tf.keras.metrics.AUC(name='auc'),
                      f1_score])
    # Обучение модели
    # Проверка корректности y_train_labels и class_labels
    # Обучение модели
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
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

    history = model.fit(x_train, y_train, batch_size=current_batch_size, epochs=current_epochs, validation_data=(x_val, y_val), class_weight=class_weights_dict, callbacks=[checkpoint, early_stopping])

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
        "Confusion Matrix": conf_matrx_test,
        "current_period": f"{current_period}",
        "current_window": f"{current_window}",
        "current_threshold": f"{current_threshold}",
        "current_neiron": f"{current_neiron}",
        "current_dropout": f"{current_dropout}",
        "class_distribution": class_distribution,
    }

    if (scores[1] * 100 >= 75 and
            scores[2] >= 0.75 and
            scores[3] >= 0.75 and
            scores[4] >= 0.75 and
            scores[-1] >= 0.10):

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
        if (scores[1] * 100 >= 60 and
                scores[2] >= 0.60 and
                scores[3] >= 0.60 and
                scores[4] >= 0.60 and
                scores[-1] >= 0.10):
            directory_save = f'keras_model/lstm/NotGood/{coin}/'
            if not os.path.exists(directory_save):
                os.makedirs(directory_save)
            # Запись данных в файл
            with open(f"{directory_save}/{generate_uuid()}.txt", "w") as file:
                for key, value in data.items():
                    file.write(f"{key}={value}\n")
    del x_train, x_val, y_train, y_val, x, y
    return 'End current epoch'

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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.config.threading.set_intra_op_parallelism_threads(CPU_COUNT)
    tf.config.threading.set_inter_op_parallelism_threads(CPU_COUNT)

    tf.get_logger().setLevel('ERROR')
    log_file = os.path.expanduser('~/training.log')

    logging.basicConfig(
        filename=log_file,  # Имя файла логирования
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    period = ["5m",]
    window_size = [3, 5, 8, 13, 21, 34]
    threshold = [0.005, 0.007, 0.01, 0.02]
    neiron = [50, 100, 150, 200]
    dropout = [0.10, 0.15, 0.20, 0.25, 0.30]
    model_count = ModelLSTM_2Class.model_count

    # создание данных для ии, которые могут загружаться повторно
    path_exist('temp/')
    path_exist('temp/checkpoint/')
    path_exist('temp/best_params/')
    path_exist('temp/roll_win/')
    path_exist('temp/scaler/')
    path_exist('temp/mmap/')
    clear_folder('temp/roll_win/')
    clear_folder('temp/scaler/')
    clear_folder('temp/mmap/')

    run_multiprocessing_rolling_window(coin=coin, period=period, date_df=date_df, window_size=window_size,
                                       threshold=threshold, numeric=numeric)
    #goLSTM
    if goLSTM:
        all_task = []
        for filename in os.listdir(f'keras_model/best_params/'):
            if filename.endswith('.csv'):
                # Составление полного пути к файлу
                filepath = os.path.join(directory_path, filename)
                uuid_name, extension = os.path.splitext(filename)
                # Здесь можно выполнить операции с файлом
                data = pd.read_csv(f'{filepath}')
                for i in range(len(data)):
                    date_df_check = data.at[i, 'date_df'].split(',')
                    if date_df_check != date_df:
                        print(f'Ошибка входных данных по дням свечей \ndate_df {date_df}\n date ib csv {date_df_check}')
                        exit()
                    row_slice = data.iloc[i]

                    # Сборка задачи с именами переменных и их значениями
                    task = tuple((column_name, value) for column_name, value in row_slice.items())
                    all_tasks.append(task)

        total_iterations = len(all_tasks)
        print(f'Total number of iterations: {total_iterations}')

        max_workers = min(1, len(all_tasks)) #max_workers=max_workers

        start_time = time.perf_counter()


        # Использование ProcessPoolExecutor для параллельного выполнения
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Запуск процессов
            futures = [executor.submit(goLSTM, task) for task in all_tasks]
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

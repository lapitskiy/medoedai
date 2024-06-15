import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import os
import shutil
import stat
import time
import gc
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.get_logger().setLevel('ERROR')
# Установка уровня журналирования среды выполнения TensorFlow


#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
tf.config.experimental.list_physical_devices('GPU')
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        print("Доступные GPU: ", gpus)
        # Пример: ограничение использования GPU памяти до 5GB на GPU
        memory_limit = 5012  # Укажите здесь значение в мегабайтах
        virtual_devices = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit) for _ in gpus]
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(gpu, virtual_devices)
    except RuntimeError as e:
        print("Произошла ошибка:", e)
#export TF_FORCE_GPU_ALLOW_GROWTH='true'

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
K.clear_session()
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import psutil
from utils.path import create_dataframe, generate_uuid, path_exist, clear_folder, read_temp_path, save_grid_checkpoint, \
    file_exist
import matplotlib.pyplot as plt
import matplotlib
from utils.models_list import ModelLSTM_2Class, create_model
import joblib
matplotlib.use('Agg')
# Проверка доступности GPU

# Динамическое выделение памяти на GPU
gc.collect()
tf.keras.backend.clear_session()
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
CPU_COUNT = 7
date_df = ['2024-03','2024-04','2024-05'] # 1m
coin = 'TONUSDT'
#layer = '75day-2layer250-Dropout02'
numeric = ['open', 'high', 'low', 'close', 'bullish_volume', 'bearish_volume']
checkpoint_file = 'temp/checkpoint/grid_search_checkpoint.txt'
goKeras = True

metric_thresholds = {
    'val_accuracy': 0.60,
    'val_precision': 0.60,
    'val_recall': 0.60,
    'val_f1_score': 0.10,
    'val_auc': 0.60
}
def goKerasRegressor(windows_size, thresholds, periods, dropouts, neirons):
    model_count = ModelLSTM_2Class.model_count
    start_index_model = 5
    start_index_win = 0
    start_index_thr = 0
    start_index_per = 0
    start_index_drop = 0
    start_index_neiron = 0
    if file_exist(checkpoint_file):
            list_ = read_temp_path(f'{checkpoint_file}', 6)
            # model_number, window_size, threshold, period, dropout, neiron
            start_index_model = int(list_[0])
            start_index_win = windows_size.index(int(list_[1]))
            start_index_thr = thresholds.index(float(list_[2]))
            start_index_per = periods.index(str(list_[3]))
            start_index_drop = dropouts.index(float(list_[4]))
            start_index_neiron = neirons.index(float(list_[5]))
    for model_number in range(start_index_model, model_count + 1):
        for period in periods[start_index_per:]:
            for window_size in windows_size[start_index_win:]:
                for threshold in thresholds[start_index_thr:]:
                    for dropout in dropouts[start_index_drop:]:
                        for neiron in neirons[start_index_neiron:]:
                            save_grid_checkpoint(model_number=model_number, window_size=window_size, threshold=threshold,
                                                 period=period, dropout=dropout, neiron=neiron, file_path=checkpoint_file)
                            print(f"Progress saved to {checkpoint_file}")
                            iteration_start_time = time.perf_counter()
                            list_ = read_temp_path(f'temp/roll_win/roll_path_ct{threshold}_cw{window_size}_cp{period}.txt', 3)
                            x_path = list_[0]
                            y_path = list_[1]
                            num_samples = list_[2]
                            num_features = len(numeric)
                            x = np.memmap(f'{x_path}', dtype=np.float32, mode='r', shape=(int(num_samples), window_size, num_features))
                            y = np.memmap(f'{y_path}', dtype=np.int8, mode='r', shape=(int(num_samples),))
                            # Сначала разделите данные на обучающий+валидационный и тестовый наборы
                            x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
                            # Затем разделите обучающий+валидационный набор на обучающий и валидационный наборы
                            x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25,
                                                                              random_state=42)  # 0.25 x 0.8 = 0.2

                            model = KerasRegressor(model=create_model, verbose=0)
                            # Словарь гиперпараметров для поиска
                            param_grid = {
                                'model__model_number': [model_number, ],  # Note the prefix "model__"
                                'model__current_window': [window_size, ],  # Note the prefix "model__"
                                'model__num_features': [num_features, ],  # Note the prefix "model__"
                                'model__current_dropout': [dropout, ],  # Note the prefix "model__"
                                'model__current_neiron': [neiron, ],  # Note the prefix "model__"
                                'batch_size': [32, 64, 96, 128],
                                'epochs': [10, 30, 60, 80, 100]
                            }


                            grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3,
                                                    verbose=2, n_jobs=CPU_COUNT)

                            grid.fit(x_train, y_train)
                            tf.keras.backend.clear_session()

                            iteration_end_time = time.perf_counter()
                            iteration_time = iteration_end_time - iteration_start_time
                            print(f'Iteration_time {iteration_time}; CPU use {CPU_COUNT}')
                            # Результаты поиска
                            print("Лучший результат: %f используя %s" % (grid.best_score_, grid.best_params_))
                            file_name = f'temp/best_params/best_params_{coin}.csv'
                            best_score = grid.best_score_
                            best_params = grid.best_params_
                            results_df = pd.DataFrame([best_params])
                            results_df['threshold'] = threshold
                            results_df['num_samples'] = num_samples
                            results_df['period'] = period
                            date_str = ','.join(date_df)
                            results_df['date_df'] = date_str
                            results_df['coin'] = coin
                            results_df['time'] = f'time {iteration_time} - cpu {CPU_COUNT}'
                            results_df['best_score'] = best_score
                            try:
                                # Проверяем, существует ли файл
                                with open(file_name, 'r') as f:
                                    existing_df = pd.read_csv(file_name)
                                    # Проверяем, есть ли столбец 'threshold' в существующем файле
                                    if 'threshold' not in existing_df.columns or 'num_samples' not in existing_df.columns \
                                            or 'period' not in existing_df.columns or 'date_df' not in existing_df.columns \
                                            or 'coin' not in existing_df.columns or 'time' not in existing_df.columns:
                                        # Если нет, добавляем столбец с заголовком
                                        results_df.to_csv(file_name, mode='a', header=True, index=False)
                                    else:
                                        # Если есть, добавляем данные без заголовка
                                        results_df.to_csv(file_name, mode='a', header=False, index=False)
                            except FileNotFoundError:
                                # Если файл не существует, создаем его и записываем данные с заголовком
                                results_df.to_csv(file_name, mode='w', header=True, index=False)
    # Если завершено успешно, удаляем файл чекпойнта
    shutil.move(file_name, f'keras_model/best_params/{generate_uuid()}.csv')
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

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
    # task_dict = dict(task)
    # batch_size =
    # epoch =
    # current_dropout
    # current_neiron
    # current_window
    # model_number
    # num_features
    # threshold
    # num_samples
    # period

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

def create_rolling_windows(df, df_scaled, current_threshold, input_window): # work BTC and TON and VOLUME
    output_window = input_window  # Предсказываем на столько же периодов вперед, сколько и входных данных
    num_samples = len(df) - input_window - output_window
    feature_columns = numeric
    num_features = len(feature_columns)

    # Создание memmap файлов
    uuid_mmap = generate_uuid()
    if not os.path.exists('temp'):
        os.makedirs('temp')
    x_mmap = np.memmap(f'temp/mmap/{uuid_mmap}_x.dat', dtype=np.float32, mode='w+', shape=(num_samples, input_window, num_features))
    y_mmap = np.memmap(f'temp/mmap/{uuid_mmap}_y.dat', dtype=np.int8, mode='w+', shape=(num_samples, output_window))

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
    return f'temp/mmap/{uuid_mmap}_x.dat', f'temp/mmap/{uuid_mmap}_y.dat', num_samples

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

def process_data(df, current_window, current_threshold, current_period, numeric):
    df.ffill(inplace=True)
    df['pct_change'] = df['close'].pct_change(periods=current_window)
    df['bullish_volume'] = df['volume'] * (df['close'] > df['open'])
    df['bearish_volume'] = df['volume'] * (df['close'] < df['open'])
    df[numeric] = df[numeric].astype(np.float32)
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[numeric] = scaler.fit_transform(df_scaled[numeric])
    joblib.dump(scaler, f'temp/scaler/scaler_ct{current_threshold}_cw{current_window}_cp{current_period}.gz')
    x_path, y_path, num_samples = create_rolling_windows(df, df_scaled, current_threshold, current_window)
    try:
        with open(f'temp/roll_win/roll_path_ct{current_threshold}_cw{current_window}_cp{current_period}.txt', 'w') as f:
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
                    futures.append(executor.submit(process_data, df.copy(), current_window, current_threshold, current_period, numeric))

        # Обработка завершенных задач
        for future in as_completed(futures):
            x_path, y_path = future.result()
            print(x_path, y_path)

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


    run_multiprocessing_rolling_window()
    #goKerasRegress
    if goKeras:
        goKerasRegressor(windows_size=window_size, thresholds=threshold, periods=period, dropouts=dropout, neirons=neiron)
    else:
        # goLSTM
        for filename in os.listdir(f'keras_model/best_params/'):
            if filename.endswith('.csv'):
                # Составление полного пути к файлу
                filepath = os.path.join(directory_path, filename)
                uuid_name, extension = os.path.splitext(filename)
                # Здесь можно выполнить операции с файлом
                data = pd.read_csv(f'{filepath}')
                for i in range(len(data)):
                    df['date_df'].apply(lambda x: x.split(','))
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

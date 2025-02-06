from utils.env import lstmcfg
from utils.rolling import run_multiprocessing_rolling_window
from utils.path import generate_uuid, path_exist, clear_folder, read_temp_path, save_grid_checkpoint, \
    file_exist
from utils.models_list import create_model

import os
import warnings
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
import concurrent.futures
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
if lstmcfg.tfGPU:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            print("Доступные GPU: ", gpus)
            for gpu in gpus:
                tf.config.set_visible_devices(gpu, 'GPU')
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
else:
    tf.config.set_visible_devices([], 'GPU')

    #os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    #os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    # tf.config.experimental.list_physical_devices('GPU')
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         print("Доступные GPU: ", gpus)
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #     except RuntimeError as e:
    #         print("Произошла ошибка:", e)


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
import matplotlib.pyplot as plt
import matplotlib

import joblib
matplotlib.use('Agg')

def goLSTM(task):
    task_dict = dict(task)
    # Извлечение нужных значений
    current_threshold = task_dict.get('threshold')
    current_window = task_dict.get('model__current_window')
    current_period = task_dict.get('period')
    current_batch_size = task_dict.get('batch_size')
    current_epochs = task_dict.get('epochs')
    current_dropout = task_dict.get('model__current_dropout')
    current_neiron = task_dict.get('model__current_neiron')
    num_features = task_dict.get('model__num_features')
    model_number = task_dict.get('model__model_number')
    num_samples = task_dict.get('num_samples')
    coin = task_dict.get('coin')
    date_df = task_dict.get('date_df').split(',')
    best_score = task_dict.get('best_score')

    list_ = read_temp_path(f'temp/lstm/roll_win/roll_path_ct{current_threshold}_cw{current_window}_cp{current_period}.txt', 4)
    x_path = list_[0]
    y_path = list_[1]
    num_samples = list_[2]
    hash_value = list_[3]
    if num_features != len(lstmcfg.numeric):
        print(f'num_features не соотвествует config.numeric')
        exit()

    x = np.memmap(f'{x_path}', dtype=np.float32, mode='r', shape=(int(num_samples), current_window, num_features))
    y = np.memmap(f'{y_path}', dtype=np.int8, mode='r', shape=(int(num_samples),))

    print("Распределение классов в исходных данных:")
    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print(class_distribution)

    # Сначала разделите данные на обучающий+валидационный и тестовый наборы
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2,
                                                                random_state=42)

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
    print(f'model_number {model_number}')

    model = create_model(current_dropout=current_dropout, current_neiron=current_neiron, current_window=current_window,
                         num_features=num_features, model_number=model_number, type='lstm')
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

    #lstmcfg.uuid = generate_uuid()
    #lstmcfg.directory_save = f'model/lstm/{coin}/{current_period}/'

    checkpoint = ModelCheckpointWithMetricThreshold(filedata=data, model=model)
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
    print("Оптимальный порог:", opt_threshold)
    print("Максимальный F1-Score:", opt_f1_score)

    # Получение предсказаний на тестовом наборе
    y_test_probs = model.predict(x_test)
    y_test_pred = (y_test_probs >= opt_threshold).astype(int)

    # Оценка производительности
    conf_matrx_test = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:\n", conf_matrx_test)

    # Подготовка данных для записи в файл
    data2 = {
        "hash": hash_value,
        "coin": lstmcfg.coin,
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
        "date_df": ','.join(date_df)
    }

    data.update(data2)
    print(f"Test Accuracy: {scores[1] * 100:.2f}%")
    print(f"Test Precision: {scores[2]:.2f}")
    print(f"Test Recall: {scores[3]:.2f}")
    print(f"Test AUC: {scores[4]:.2f}")
    print(f"Test F1-Score: {scores[-1]:.2f}")
    print('ЗАПИСЬ ФАЙЛА?')
    if (scores[1] * 100 >= 60 and
            scores[2] >= 0.10 and
            scores[3] >= 0.40 and
            scores[4] >= 0.60 and
            scores[-1] >= 0.04):

        result_filename = f'{lstmcfg.directory_save}/metric/{lstmcfg.coin}/result.csv'
        directory_save = f'{lstmcfg.directory_save}work_model/{coin}/{current_period}/'
        path_exist(f'{lstmcfg.directory_save}/metric/{lstmcfg.coin}/')
        path_exist(lstmcfg.directory_save)
        path_exist(directory_save)
        # Запись данных в файл
        results_df = pd.DataFrame([data])
        print('ЗАПИСЬ ФАЙЛА')
        try:
            # Проверяем, существует ли файл
            with open(result_filename, 'r') as f:
                existing_df = pd.read_csv(result_filename, delimiter=';')
                # Проверяем, есть ли столбец 'threshold' в существующем файле
                if 'batch_sizes' not in existing_df.columns or 'epoch' not in existing_df.columns \
                        or 'hash' not in existing_df.columns or 'coin' not in existing_df.columns:
                    # Если нет, добавляем столбец с заголовком
                    results_df.to_csv(result_filename, mode='a', header=True, index=False, sep=';')
                else:
                    # Если есть, добавляем данные без заголовка
                    results_df.to_csv(result_filename, mode='a', header=False, index=False, sep=';')
        except FileNotFoundError:
            # Если файл не существует, создаем его и записываем данные с заголовком
            results_df.to_csv(result_filename, mode='w', header=True, index=False, sep=';')

        source_path = f'temp/{lstmcfg.ii_path}/scaler/scaler_ct{current_threshold}_cw{current_window}_cp{current_period}.gz'
        target_path = f'{directory_save}{hash_value}.gz'
        shutil.copy2(source_path, target_path)
        #joblib.dump(scaler, f"{directory_save}{hash_value}.gz")
        # Сохранение модели
        model.save(f"{directory_save}{hash_value}.h5")
        # Вывод структуры модели
        model.summary()
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
    def __init__(self, filedata, model, verbose=1):
        super(ModelCheckpointWithMetricThreshold, self).__init__()
        self.thresholds = lstmcfg.metric_thresholds  # Словарь с пороговыми значениями для метрик
        self.best_metrics = {key: 0 for key in self.thresholds.keys()}  # Лучшие метрики
        self.verbose = verbose
        self.filedata = filedata
        self.directory_save = lstmcfg.directory_save + 'metric/'
        self.current_threshold = filedata['current_threshold']
        self.current_window = filedata['current_window']
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
                with open(f"{config.directory_save + config.uuid}/callback.txt", "w") as file:
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
    tf.config.threading.set_intra_op_parallelism_threads(lstmcfg.CPU_COUNT)
    tf.config.threading.set_inter_op_parallelism_threads(lstmcfg.CPU_COUNT)

    tf.get_logger().setLevel('ERROR')
    log_file = os.path.expanduser('~/training.log')

    logging.basicConfig(
        filename=log_file,  # Имя файла логирования
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # создание данных для ии, которые могут загружаться повторно
    path_exist('temp/')
    path_exist('temp/lstm/checkpoint/')
    path_exist('temp/lstm/best_params/')
    path_exist('temp/lstm/roll_win/')
    path_exist('temp/lstm/scaler/')
    path_exist('temp/lstm/mmap/')
    clear_folder('temp/lstm/roll_win/')
    clear_folder('temp/lstm/scaler/')
    #clear_folder('temp/lstm/mmap/')

    run_multiprocessing_rolling_window(config=lstmcfg)
    #goLSTM
    if lstmcfg.goLSTM:
        all_task = []
        directory_path = f'model/{lstmcfg.ii_path}/best_params/'
        for filename in os.listdir(f'{directory_path}'):
            if filename.endswith('.csv'):
                # Составление полного пути к файлу
                filepath = os.path.join(directory_path, filename)
                uuid_name, extension = os.path.splitext(filename)
                # Здесь можно выполнить операции с файлом
                data = pd.read_csv(f'{filepath}', delimiter=';')
                data_fields = data.columns.tolist()

                # Проверка наличия всех необходимых полей в DataFrame
                missing_fields = [field for field in lstmcfg.required_params if field not in data.columns]
                if missing_fields:
                    print(f"Error: Missing fields {', '.join(missing_fields)} in the file {filename}.")
                    exit()

                for index, row in data.iterrows():
                    if all(row[field] for field in lstmcfg.required_params):
                        #print(f'head {data.head()}')
                        #print(f'tail {data.tail()}')
                        #
                        date_df_check = row['date_df'].split(',')
                        #print(f'Row {index} dates:', date_df_check)
                        #print(f' date_df_check {date_df_check} != lstmcfg.date_df {lstmcfg.date_df}')
                        if date_df_check != lstmcfg.date_df:
                            print(
                                f'Ошибка входных данных по дням свечей \ndate_df {lstmcfg.date_df}\n date ib csv {date_df_check}')
                            exit()
                    else:
                        print(f"Error: Missing data in required fields at row {index}.")
                        exit()
                    # Сборка задачи с именами переменных и их значениями
                    task = tuple((column_name, value) for column_name, value in row.items())
                    all_task.append(task)
                else:
                    print(f'Skipping row {index} in file {filename} due to missing values in required fields.')


        total_iterations = len(all_task)
        print(f'Total number of iterations: {total_iterations}')

        max_workers = min(lstmcfg.CPU_COUNT, len(all_task)) #max_workers=max_workers

        start_time = time.perf_counter()

        # Использование ProcessPoolExecutor для параллельного выполнения
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            # Запуск процессов
            futures = [executor.submit(goLSTM, task) for task in all_task]
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

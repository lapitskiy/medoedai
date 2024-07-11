from utils.env import config
from utils.rolling import run_multiprocessing_rolling_window
from utils.path import generate_uuid, path_exist, clear_folder, read_temp_path, save_grid_checkpoint, \
    file_exist
from utils.models_list import ModelLSTM_2Class, create_model

import os
#if not tfGPU:
#    os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_intra_op_parallelism_threads(config.CPU_COUNT)
tf.config.threading.set_inter_op_parallelism_threads(config.CPU_COUNT)
tf.get_logger().setLevel('ERROR')
if config.tfGPU:
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

def goKerasRegressor(windows_size, thresholds, periods, dropouts, neirons):
    model_count = config.model_count
    start_index_model = 2
    start_index_win = 0
    start_index_thr = 0
    start_index_per = 0
    start_index_drop = 0
    start_index_neiron = 0
    if file_exist(config.checkpoint_file):
            list_ = read_temp_path(f'{config.checkpoint_file}', 6)
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
                            tf.keras.backend.clear_session()
                            gc.collect()
                            K.clear_session()
                            save_grid_checkpoint(model_number=model_number, window_size=window_size, threshold=threshold,
                                                 period=period, dropout=dropout, neiron=neiron, file_path=config.checkpoint_file)
                            print(f"Progress saved to {config.checkpoint_file}")
                            iteration_start_time = time.perf_counter()
                            list_ = read_temp_path(f'temp/{config.ii_path}/roll_win/roll_path_ct{threshold}_cw{window_size}_cp{period}.txt', 3)
                            x_path = list_[0]
                            y_path = list_[1]
                            num_samples = list_[2]
                            num_features = len(config.numeric)
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
                                                    verbose=2, n_jobs=config.CPU_COUNT)

                            grid.fit(x_train, y_train)
                            tf.keras.backend.clear_session()

                            iteration_end_time = time.perf_counter()
                            iteration_time = iteration_end_time - iteration_start_time
                            print(f'Iteration_time {iteration_time}s; CPU use {config.CPU_COUNT}')
                            # Результаты поиска
                            print("Лучший результат: %f используя %s" % (grid.best_score_, grid.best_params_))
                            file_name = f'temp/best_params/best_params_{config.coin}.csv'
                            best_score = grid.best_score_
                            best_params = grid.best_params_
                            results_df = pd.DataFrame([best_params])
                            results_df['threshold'] = threshold
                            results_df['num_samples'] = num_samples
                            results_df['period'] = period
                            date_str = ','.join(config.date_df)
                            results_df['date_df'] = date_str
                            results_df['coin'] = config.coin
                            results_df['time'] = f'time {iteration_time:.2f} - cpu {config.CPU_COUNT}'
                            results_df['best_score'] = best_score

                            try:
                                with open(file_name, 'r') as f:
                                    first_line = f.readline()
                                    headers = first_line.strip().split(';')  # Адаптируйте разделитель к вашему файлу

                                required_columns = {'threshold', 'num_samples', 'period', 'date_df', 'coin', 'time',
                                                    'batch_size', 'epochs', 'model__current_dropout',
                                                    'model__current_neiron', 'model__current_window',
                                                    'model__model_number', 'model__num_features', 'best_score'}
                                existing_columns = set(headers)

                                # Проверяем, содержит ли первая строка все необходимые заголовки
                                if required_columns.issubset(existing_columns):
                                    # Если заголовки на месте, добавляем данные без заголовков
                                    results_df.to_csv(file_name, mode='a', header=False, index=False, sep=';')
                                else:
                                    # Если заголовков не хватает, пересоздаем файл с правильными заголовками
                                    results_df.to_csv(file_name, mode='w', header=True, index=False, sep=';')
                            except FileNotFoundError:
                                # Если файл не существует, создаем его и записываем данные с заголовком
                                results_df.to_csv(file_name, mode='w', header=True, index=False, sep=';')

    # Если завершено успешно, удаляем файл чекпойнта
    shutil.move(file_name, f'keras_model/lstm/best_params/{generate_uuid()}.csv')
    if os.path.exists(config.checkpoint_file):
        os.remove(config.checkpoint_file)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    log_file = os.path.expanduser('~/training.log')

    logging.basicConfig(
        filename=log_file,  # Имя файла логирования
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # создание данных для ии, которые могут загружаться повторно
    path_exist('temp/')
    path_exist(f'temp/{config.ii_path}/')
    path_exist(f'temp/{config.ii_path}/checkpoint/')
    path_exist(f'temp/{config.ii_path}/best_params/')
    path_exist(f'temp/{config.ii_path}/roll_win/')
    path_exist(f'temp/{config.ii_path}/scaler/')
    path_exist(f'temp/{config.ii_path}/mmap/')
    path_exist(f'keras_model/lstm/best_params/')
    clear_folder(f'temp/{config.ii_path}/roll_win/')
    clear_folder(f'temp/{config.ii_path}/scaler/')
    clear_folder(f'temp/{config.ii_path}/mmap/')


    run_multiprocessing_rolling_window(coin=config.coin, period=config.period, date_df=config.date_df, window_size=config.window_size,
                                       threshold=config.threshold, numeric=config.numeric, ii_path=config.ii_path)
    #goKerasRegress
    if config.goKeras:
        goKerasRegressor(windows_size=config.window_size, thresholds=config.threshold, periods=config.period, dropouts=config.dropout, neirons=config.neiron)
    else:
        print(f'goKeras False')
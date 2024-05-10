import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tensorflow.keras.models import load_model
import joblib

# Загрузка модели



class goTest():
    directory = 'history_csv/'  # Укажите путь к вашей директории с CSV файлами
    coin = 'BTCUSDT'
    window_size = 10
    predict_percent = 0.3
    df_scaled: None
    close_prices: None
    threshold = 0.01
    date_test = ['2024-05', ]
    day_test = 4
    period = '5m'

    def __init__(self):
        self.keras_model = load_model('medoed_model.keras')
        self.df = self.create_dataframe()
        self.scaler = joblib.load('scaler.gz')


    def run(self):
        x_new, self.close_prices = self.prepare_new_data()
        new_predictions = self.keras_model.predict(x_new)
        new_predicted_classes = (new_predictions > self.predict_percent).astype(int)  # ton 0.6

        # Вызов функции отрисовки
        self.plot_predictions(new_predicted_classes.flatten())

        print("Predicted classes:", len(new_predicted_classes))

        unique, counts = np.unique(new_predicted_classes, return_counts=True)

        # Вывод предсказанных классов и соответствующих цен закрытия
        print("Predicted classes and closing prices:")
        for predicted_class, close_price in zip(new_predicted_classes.flatten(), self.close_prices):
            if predicted_class == 1:
                print(f"Class: {predicted_class}, Close Price: {close_price}")

        print("Unique predicted classes and their counts:", dict(zip(unique, counts)))

    def prepare_new_data(self):
        self.df['pct_change'] = self.df['close'].pct_change(periods=self.window_size)
        # Предполагается, что new_data уже содержит нужные столбцы и очищен от недостающих значений
        numeric_features = ['open', 'high', 'low', 'close', 'volume']
        new_data_scaled = self.scaler.transform(self.df[numeric_features])
        self.df_scaled = pd.DataFrame(new_data_scaled, columns=numeric_features)
        x_new, _, close_prices = self.create_rolling_windows()
        return x_new, close_prices

    def create_rolling_windows(self): # with VOLUME
        x = []
        y = []
        close_prices = []
        for i in range(len(self.df_scaled) - self.window_size):
            # Получение цены закрытия на последнем шаге окна
            close_price = self.df['close'].iloc[i + self.window_size - 1]
            close_prices.append(close_price)

            change = self.df['pct_change'].iloc[i + self.window_size]  # Использование ранее рассчитанного изменения
            x.append(self.df_scaled[['open', 'high', 'low', 'close', 'volume']].iloc[i:i + self.window_size].values) # , 'volume', 'count', 'taker_buy_volume'
            # Создание бинарной целевой переменной
            y.append(1 if abs(change) >= self.threshold else 0)
        return np.array(x), np.array(y), np.array(close_prices)

    # создание датафрейм из csv
    def create_dataframe(self, data, period):

        # Создаем пустой DataFrame
        df = pd.DataFrame()
        try:
            for item in data:
                directory = f'history_csv/{self.coin}/{period}/{item}/'
                for file_name in os.listdir(directory):
                    if file_name.endswith('.csv'):
                        file_path = os.path.join(directory, file_name)

                        # Определяем количество столбцов в CSV файле, исключая последний
                        use_cols = pd.read_csv(file_path, nrows=1).columns.difference(
                            ['open_time', 'close_time', 'ignore'])

                        # Считываем данные из CSV, исключая последний столбец
                        data = pd.read_csv(file_path, usecols=use_cols)

                        # Преобразуем поля с временными метками в datetime
                        # data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
                        # data['close_time'] = pd.to_datetime(data['close_time'], unit='ms')

                        # Добавляем считанные данные в DataFrame

                        df = pd.concat([df, data], ignore_index=True)
        except Exception as e:
            print(f'error os load {e}')
        return df

    def plot_predictions(self, predictions):
        # Создание фигуры и оси
        plt.figure(figsize=(14, 7))
        #np.array(close_prices)
        plt.plot(self.df_scaled['close'], label='Close Price', color='blue')  # Рисуем цену закрытия
        #plt.plot(self.close_prices, label='Close Price', color='blue')  # Рисуем цену закрытия

        # Расчет индексов, на которых были получены предсказания
        prediction_indexes = np.arange(self.window_size, len(predictions) + self.window_size)

        last_prediction_index = None
        # Отметка предсказаний модели зелеными метками
        for i, predicted in enumerate(predictions):
            if predicted == 1:  # Если модель предсказала движение на 1% или более
                plt.scatter(prediction_indexes[i], self.df_scaled['close'].iloc[prediction_indexes[i]], color='red',
                            label='Predicted >1% Change' if i == 0 else "")

        # Добавление легенды и заголовка
        plt.title('Model Predictions on Price Data')
        plt.xlabel('Time')
        plt.ylabel('Close Price')
        plt.legend()
        plt.savefig('predict.png')
        plt.close()



# Использование функции prepare_new_data с новыми данными
#new_data = create_new_data()  # Здесь вам нужно определить, как получать новые данные
#x_new = prepare_new_data(new_data)
go = goTest()
go.run()


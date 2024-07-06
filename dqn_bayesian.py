import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from skopt import gp_minimize

# Загрузка и подготовка данных
df = pd.read_csv('your_data.csv')
# Предположим, что у вас есть данные в виде numpy массивов X и y
X = df[['open', 'high', 'low', 'close', 'volume']].values
y = df['target'].values

# Разделение данных на обучающий и валидационный наборы
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


def objective(params):
    # params = [number_of_neurons, learning_rate]
    number_of_neurons, learning_rate = params

    model = Sequential()
    model.add(Dense(number_of_neurons, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(number_of_neurons, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

    model.fit(X_train, y_train, epochs=5, verbose=0)
    loss = model.evaluate(X_val, y_val, verbose=0)
    return loss

res = gp_minimize(objective,                  # Функция для минимизации
                  [(10, 100), (1e-6, 1e-2)],  # Диапазоны гиперпараметров
                  n_calls=50,                 # Количество итераций
                  random_state=0)

print("Лучшие параметры:", res.x)
best_neurons, best_learning_rate = res.x

# комплексную функцию потерь, которая одновременно максимизировать прибыль и минимизировать риск.
# компоненты как профит-базированных потерь, так и элементы, связанные с риском:
import os

import pandas as pd


def combined_loss(y_true, y_pred, beta=0.5):
    # Расчет профит-базированных потерь
    actions = torch.sign(y_pred[1:] - y_pred[:-1])
    price_differences = y_true[1:] - y_true[:-1]
    profit = torch.sum(actions * price_differences)

    # Расчет потерь с учетом риска (например, на основе коэффициента Шарпа)
    returns = actions * price_differences
    risk_adjusted_return = returns.mean() / returns.std()

    # Комбинирование потерь: максимизация прибыли и максимизация риск-адаптированной доходности
    loss = -beta * profit + (1 - beta) * (-risk_adjusted_return)  # beta контролирует баланс между прибылью и риском
    return loss

# создание датафрейм из csv
def create_dataframe(coin, data, period):
    # Путь к директории с CSV файлами
    directory = f'history_csv/{coin}/{period}/{data}/'  # Укажите путь к вашей директории с CSV файлами
    # Создаем пустой DataFrame
    df = pd.DataFrame()

    try:
        for file_name in os.listdir(directory):
            if file_name.endswith('.csv'):
                file_path = os.path.join(directory, file_name)

                # Определяем количество столбцов в CSV файле, исключая последний
                use_cols = pd.read_csv(file_path, nrows=1).columns.difference(['open_time', 'close_time', 'ignore'])

                # Считываем данные из CSV, исключая последний столбец
                data = pd.read_csv(file_path, usecols=use_cols)


                # Преобразуем поля с временными метками в datetime
                #data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
                #data['close_time'] = pd.to_datetime(data['close_time'], unit='ms')

                # Добавляем считанные данные в DataFrame
                df = pd.concat([df, data], ignore_index=True)
    except Exception as e:
        print(f'error os load {e}')
    return df
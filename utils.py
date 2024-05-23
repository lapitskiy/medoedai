
# комплексную функцию потерь, которая одновременно максимизировать прибыль и минимизировать риск.
# компоненты как профит-базированных потерь, так и элементы, связанные с риском:
import os
import shutil

import pandas as pd

import uuid

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

    # Создаем пустой DataFrame
    df = pd.DataFrame()
    if period == '1m':
        date_ = []
        date_.append(data[0])
        data = date_
    try:
        i = 0
        for item in data:
            directory = f'history_csv/{coin}/{period}/{item}/'
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

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Папка '{folder_path}' и все её содержимое успешно удалены.")
    else:
        print(f"Папка '{folder_path}' не существует.")


def delete_empty_folders(path):
    # Рекурсивная функция для удаления пустых папок
    def remove_empty_folders(directory):
        # Проверяем содержимое текущей директории
        for foldername in os.listdir(directory):
            folderpath = os.path.join(directory, foldername)
            if os.path.isdir(folderpath):
                # Рекурсивно удаляем пустые папки внутри текущей папки
                remove_empty_folders(folderpath)
                # Если текущая папка теперь пустая, удаляем её
                if not os.listdir(folderpath):
                    os.rmdir(folderpath)
                    print(f"Пустая папка '{folderpath}' успешно удалена.")

    # Запускаем рекурсивное удаление пустых папок с указанного пути
    remove_empty_folders(path)

def generate_uuid():
    short_uuid = str(uuid.uuid4())[:10]  # Берем первые 8 символов UUID
    return f"{short_uuid}"

def path_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_x_y_ns_path(file_path):
    with open(file_path, 'r') as file:
        x_path = file.readline().strip()  # Читаем первую строку и удаляем лишние символы
        y_path = file.readline().strip()  # Читаем вторую строку и удаляем лишние символы
        num_samples = file.readline().strip()  # Читаем вторую строку и удаляем лишние символы
    return x_path, y_path, num_samples

def clear_folder(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)  # Удаление файла или ссылки
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Удаление директории

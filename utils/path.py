import os
import shutil
import pandas as pd
import uuid
import psutil

# создание датафрейм из csv
def save_grid_checkpoint(model_number, window_size, threshold, period, dropout, neiron, file_path):
    try:
        with open(f'{file_path}', 'w') as f:
            f.write(str(model_number) + '\n')
            f.write(str(window_size) + '\n')
            f.write(str(threshold) + '\n')
            f.write(str(period) + '\n')
            f.write(str(dropout) + '\n')
            f.write(str(neiron) + '\n')
    except Exception as e:
        print(f"Failed to write save_grid_checkpoint paths to file: {e}")

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

def file_exist(path):
    if os.path.isfile(path):
        return True
    return False


def read_temp_path(file_path, count):
    # x_path, y_path, num_samples
    # model_number, window_size, threshold
    list_ = []
    with open(file_path, 'r') as file:
        for x in range(1,count+1):
            list_.append(file.readline().strip())
    return list_

def clear_folder(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            try:
                os.unlink(item_path)  # Удаление файла или ссылки
            except PermissionError:
                print(f"Файл или ссылка {item_path} занят(а) другим процессом. Попытка завершить процесс...")
                kill_process_using_file(item_path)
                try:
                    os.unlink(item_path)  # Повторная попытка удаления файла после завершения процесса
                    print(f"Файл или ссылка {item_path} успешно удалён(а) после завершения процесса.")
                except Exception as e:
                    print(f"Не удалось удалить файл или ссылку {item_path} после завершения процесса: {e}")

        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Удаление директории


def is_hashfile_in_folder(folder_path, hash_value):
    # Проходим по всем файлам в папке
    for file_name in os.listdir(folder_path):
        # Проверяем, содержится ли хеш в имени файла
        if hash_value in file_name:
            return True  # Если найден, возвращаем True
    return False  # Если хеш не найден ни в одном файле

def kill_process_using_file(file_path):
    try:
        # Получаем список всех запущенных процессов
        for proc in psutil.process_iter(['pid', 'name', 'open_files']):
            try:
                # Проверяем, какие файлы открыты этим процессом
                open_files = proc.info.get('open_files')
                if open_files:
                    for f in open_files:
                        if f.path == file_path:
                            # Завершаем процесс, который использует файл
                            print(f"Процесс {proc.info['name']} (PID: {proc.info['pid']}) использует файл {file_path}.")
                            proc.terminate()  # Отправляем сигнал на завершение процесса
                            proc.wait()  # Ожидаем завершения процесса
                            print(f"Процесс {proc.info['name']} (PID: {proc.info['pid']}) был завершен.")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Игнорируем ошибки, если процесс уже завершён или доступ к нему ограничен
                pass
    except Exception as e:
        print(f"Ошибка при попытке завершить процесс, использующий файл {file_path}: {e}")

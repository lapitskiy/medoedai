import pandas as pd
import requests
import zipfile
import os
from sklearn.preprocessing import MinMaxScaler






def download_and_extract(base_url, month, year, days, coin, period):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Referer": "https://data.binance.vision/"

    }

    for day in range(1, days + 1):
        day_str = f"{day:02}"
        archive_name = f"{coin}-{period}-{year}-{month}-{day_str}.zip"
        url = f"{base_url}{archive_name}"

        # Скачивание файла
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            print(f"Download {url}")
            with open(archive_name, 'wb') as f:
                f.write(response.content)

            output_directory = f"history_csv/{coin}/{period}/{year}-{month}/"

            # Распаковка архива
            with zipfile.ZipFile(archive_name, 'r') as zip_ref:
                zip_ref.extractall(output_directory)  # Распаковка в текущий каталог

            # Удаление .zip файла
            os.remove(archive_name)
        else:
            print(f"Failed to download {url}")


# Установка параметров

# https://data.binance.vision/?prefix=data/futures/um/daily/klines/
# https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/5m/BTCUSDT-5m-2024-05-01.zip
# base_url = "https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/5m/" #btc


#period = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
period = ["1m",]
coin = 'TONUSDT'

for time in period:
    base_url = f"https://data.binance.vision/data/futures/um/daily/klines/{coin}/{time}/" #ton
    download_and_extract(base_url, "04", "2024", 31, f'{coin}', period=f'{time}')

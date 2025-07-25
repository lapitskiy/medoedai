import requests
from flask import Flask, request, jsonify, render_template
from flask import redirect, url_for

import redis

from tasks.celery_tasks import celery
from celery.result import AsyncResult

import os
from tasks.celery_tasks import search_lstm_task, train_dqn, trade_step
from utils.db_utils import clean_ohlcv_data, delete_ohlcv_for_symbol_timeframe, load_latest_candles_from_csv_to_db
from utils.parser import parser_download_and_combine_with_library

app = Flask(__name__, template_folder="templates")

# Подключение к Redis
redis_client = redis.Redis(host="redis", port=6379, db=0)

import logging
from flask import Response
import json

logging.basicConfig(level=logging.INFO)

@app.before_request
def log_request_info():
    logging.info(f"Request from {request.remote_addr}: {request.method} {request.path}")
    logging.info("Headers: %s", dict(request.headers))  # Логируем все заголовки

@app.route("/")
def index():
    """Возвращает список всех задач Celery и их состояние"""
    task_ids = redis_client.keys("celery-task-meta-*")  # Ищем все задачи
    print(f"task_ids {print(task_ids)}")
    tasks = []

    for task_id in task_ids:
        task_id = task_id.decode("utf-8").replace("celery-task-meta-", "")
        task = AsyncResult(task_id, app=celery)

        tasks.append({
            "task_id": task_id,
            "state": task.state,
            "result": task.result if task.successful() else None
        })

    return render_template("index.html", tasks=tasks)


@app.route("/task-start-search", methods=["POST"])
def start_task():
    """Запускает задачу Celery и делает редирект на главную страницу"""
    query = request.form.get("query", "")  # Берём query из формы

    if not query:
        return redirect(url_for("index"))  # Перенаправление на главную

    task = search_lstm_task.apply_async(args=[query])  # Запускаем в Celery

    return redirect(url_for("index"))  # Перенаправление на главную


@app.route("/task-status/<task_id>", methods=["GET"])
def get_task_status(task_id):
    """Проверяет статус задачи по task_id"""
    task = long_running_task.AsyncResult(task_id)

    if task.state == "PENDING":
        response = {"state": "PENDING", "status": "Задача в очереди"}
    elif task.state == "IN_PROGRESS":
        response = {"state": "IN_PROGRESS", "status": "Задача выполняется", "progress": task.info}
    elif task.state == "SUCCESS":
        response = {"state": "SUCCESS", "status": "Задача завершена", "result": task.result}
    elif task.state == "FAILURE":
        response = {"state": "FAILURE", "status": "Ошибка", "error": str(task.info)}
    else:
        response = {"state": task.state, "status": "Неизвестное состояние"}

    return jsonify(response)


@app.route("/start-search", methods=["POST"])
def start_parameter_search():
    try:
        response = requests.post(
            "http://parameter-search:5052/run-search",
            json={"query": "some_value"},  # Здесь передаём нужный query
            headers={"Content-Type": "application/json"}
        )
        return response.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/train_dqn', methods=['POST'])
def train():
    task = train_dqn.apply_async(queue="train")
    return redirect(url_for("index"))

@app.route('/trade_dqn', methods=['POST'])
def trade():
    task = trade_step.apply_async()
    return redirect(url_for("index"))

# Новый маршрут для запуска очистки данных
@app.route('/clean_data', methods=['POST'])
def clean_data():
    timeframes_to_clean = ['5m', '15m', '1h']
    symbol_name ='BTC/USDT'
    
    max_close_change_percent = 15.0
    max_hl_range_percent = 20.0
    volume_multiplier = 10.0

    results = []
    for tf in timeframes_to_clean:
        try:
            # Вызываем функцию очистки напрямую (она больше не Celery-задача)
            result = clean_ohlcv_data(tf, symbol_name,
                                      max_close_change_percent,
                                      max_hl_range_percent,
                                      volume_multiplier)
            results.append(result)
        except Exception as e:
            results.append({"status": "error", "message": f"Ошибка при очистке {tf} для {symbol_name}: {str(e)}"})

    return jsonify({'status': 'Очистка данных завершена для всех указанных таймфреймов.', 'results': results})


# Новый маршрут для запуска очистки данных
@app.route('/clean_db', methods=['POST'])
def clean_db():
    timeframes_to_clean = '5m'
    symbol_name ='BTC/USDT'            

    results = []
    try:
        # Вызываем функцию очистки напрямую (она больше не Celery-задача)
        delete_ohlcv_for_symbol_timeframe('BTC/USDT', timeframes_to_clean)
        delete_ohlcv_for_symbol_timeframe('BTCUSDT', timeframes_to_clean)
    except Exception as e:
        results.append({"status": "error", "message": f"Ошибка при очистке для {symbol_name}: {str(e)}"})


    return jsonify({'status': 'Очистка базы от всех свечей завершена указанных таймфреймов.', 'results': results})

# Новый маршрут для запуска очистки данных
@app.route('/parser', methods=['POST'])
def parser():
    results = []
    symbol = 'BTCUSDT'
    interval = '5m'
    desired_candles = 100000
    csv_file_path = None

    try:
        # 1. Вызываем внешнюю функцию, которая создает CSV с 100 000 свечами
        csv_file_path = parser_download_and_combine_with_library(
            symbol='BTCUSDT',
            interval=interval,
            months_to_fetch=12, # Это параметр для parser_download_and_combine_with_library
            desired_candles=desired_candles
        )
        results.append({"status": "success", "message": f"CSV файл создан: {csv_file_path}"})

        # 2. Загружаем эти 100 000 свечей из CSV в базу данных
        # Эта функция теперь отвечает за перезапись/обновление данных в БД
        loaded_count = load_latest_candles_from_csv_to_db(
            file_path=csv_file_path,
            symbol_name=symbol,
            timeframe=interval
        )
        if loaded_count > 0:
            results.append({"status": "success", "message": f"Загружено {loaded_count} свечей из CSV в БД."})
        else:
            results.append({"status": "warning", "message": "Не удалось загрузить свечи из CSV в БД."})

    except Exception as e:
        results.append({"status": "error", "message": f"Ошибка в процессе парсинга: {str(e)}"})

    response = {'status': 'Парсинг завершен', 'results': results}
    return Response(json.dumps(response, ensure_ascii=False), mimetype='application/json')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))  # Получаем порт из переменной окружения
    app.run(host="0.0.0.0", port=port, debug=True)    



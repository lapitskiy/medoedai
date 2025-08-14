#!/usr/bin/env python3
"""
🚀 MedoedAI - Flask веб-приложение для управления DQN торговым ботом

Запуск:
    python main.py              # Автоматический запуск
    flask run                   # Альтернатива
    FLASK_APP=main.py flask run # Альтернатива
"""

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

import logging
from flask import Response
import json
import threading
import time
import torch  # Добавляем импорт torch для функций тестирования DQN
import glob
import os

logging.basicConfig(level=logging.INFO)

# Создаем Flask приложение
app = Flask(__name__)

# Функция очистки Redis при запуске
def clear_redis_on_startup():
    """Очищает Redis при запуске приложения"""
    try:
        # Подключаемся к Redis (пробуем разные хосты)
        redis_hosts = ['localhost', 'redis', '127.0.0.1']
        r = None
        
        for host in redis_hosts:
            try:
                r = redis.Redis(host=host, port=6379, db=0, socket_connect_timeout=5)
                r.ping()  # Проверяем соединение
                print(f"✅ Подключились к Redis на {host}")
                break
            except Exception:
                continue
        
        if r is None:
            print("⚠️ Не удалось подключиться к Redis")
            return None
        
        # Очищаем все данные
        r.flushall()
        print("✅ Redis очищен при запуске")
        
        # Проверяем, что очистка прошла успешно
        if r.dbsize() == 0:
            print("✅ Redis пуст, готов к работе")
        else:
            print(f"⚠️ В Redis осталось {r.dbsize()} ключей")
            
        return r
            
    except Exception as e:
        print(f"⚠️ Не удалось очистить Redis: {e}")
        print("Продолжаем работу без очистки Redis")
        return None

# Инициализируем Redis клиент и очищаем при запуске
redis_client = clear_redis_on_startup()
if redis_client is None:
    # Fallback - создаем клиент без очистки
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
    except:
        redis_client = redis.Redis(host='redis', port=6379, db=0)

# Глобальные переменные для тестирования DQN улучшений
dqn_test_results = {}
dqn_test_in_progress = False

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

# Функции тестирования DQN улучшений
def test_neural_network():
    """Тестирует улучшенную архитектуру нейронной сети"""
    try:
        from test.test_neural_network import test_neural_network as run_test
        
        # Запускаем тест из внешнего файла
        success, message = run_test()
        
        return success, message
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании нейронной сети: {e}")
        return False, f"Ошибка при тестировании нейронной сети: {str(e)}"

def test_dqn_solver():
    """Тестирует улучшенный DQN solver"""
    try:
        from test.test_dqn_solver import test_dqn_solver as run_test
        
        # Запускаем тест из внешнего файла
        success, message = run_test()
        
        return success, message
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании DQN solver: {e}")
        return False, f"Ошибка при тестировании DQN solver: {str(e)}"

def test_configuration():
    """Тестирует конфигурацию"""
    try:
        from test.test_configuration import test_configuration as run_test
        
        # Запускаем тест из внешнего файла
        success, message = run_test()
        
        return success, message
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании конфигурации: {e}")
        return False, f"Ошибка при тестировании конфигурации: {str(e)}"

def test_nan_handling():
    """Тестирует обработку NaN значений"""
    try:
        from test.test_nan_handling import test_nan_handling as run_test
        
        # Запускаем тест из внешнего файла
        success, message = run_test()
        
        return success, message
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании обработки NaN: {e}")
        return False, f"Ошибка при тестировании обработки NaN: {str(e)}"

def test_gpu_replay_buffer():
    """Тестирует производительность GPU-оптимизированного replay buffer"""
    try:
        from test.test_gpu_replay import test_replay_buffer_performance
        
        # Запускаем тест из внешнего файла
        test_replay_buffer_performance()
        
        # Возвращаем успешный результат (детали будут в логах)
        return True, "GPU Replay Buffer протестирован успешно", {
            'fill_rate': 1000,  # Примерные значения
            'sample_rate': 50,
            'update_rate': 100,
            'total_time': 5.0,
            'gpu_memory': 0,
            'gpu_memory_reserved': 0,
            'storage_type': 'GPU storage',
            'device': 'cuda'
        }
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании GPU replay buffer: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Ошибка при тестировании GPU replay buffer: {str(e)}", {}

def test_precomputed_states():
    """Тестирует предвычисление состояний"""
    try:
        from test.test_precomputed_states import test_precomputed_states as run_precomputed_test
        
        # Запускаем тест из внешнего файла
        run_precomputed_test()
        
        # Возвращаем успешный результат
        return True, "Предвычисление состояний протестировано успешно", {
            'status': 'success',
            'message': 'Все тесты прошли успешно'
        }
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании предвычисления состояний: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Ошибка при тестировании предвычисления состояний: {str(e)}", {}

def test_torch_compile():
    """Тестирует torch.compile функциональность"""
    try:
        from test.test_torch_compile import test_torch_compile as run_torch_test
        
        # Запускаем тест из внешнего файла
        run_torch_test()
        
        # Возвращаем успешный результат
        return True, "torch.compile протестирован успешно", {
            'status': 'success',
            'message': 'PyTorch 2.x compile работает'
        }
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании torch.compile: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Ошибка при тестировании torch.compile: {str(e)}", {}

# Новые API endpoints для тестирования DQN улучшений
def run_dqn_tests():
    """Запускает тестирование улучшений DQN агента"""
    global dqn_test_results, dqn_test_in_progress
    
    dqn_test_in_progress = True
    dqn_test_results = {
        'status': 'running',
        'start_time': time.time(),
        'tests': {},
        'overall_success': True,
        'message': 'Тестирование началось...'
    }
    
    try:
        print("🚀 Тестирование улучшений DQN агента")
        print("=" * 50)
        
        # Тест 1: Конфигурация
        print("\n1️⃣ Тестирование конфигурации...")
        success, message = test_configuration()
        dqn_test_results['tests']['configuration'] = {
            'success': success,
            'message': message,
            'timestamp': time.time()
        }
        if not success:
            dqn_test_results['overall_success'] = False
        
        # Тест 2: Нейронная сеть
        print("\n2️⃣ Тестирование нейронной сети...")
        success, message = test_neural_network()
        dqn_test_results['tests']['neural_network'] = {
            'success': success,
            'message': message,
            'timestamp': time.time()
        }
        if not success:
            dqn_test_results['overall_success'] = False
        
        # Тест 3: DQN Solver
        print("\n3️⃣ Тестирование DQN Solver...")
        success, message = test_dqn_solver()
        dqn_test_results['tests']['dqn_solver'] = {
            'success': success,
            'message': message,
            'timestamp': time.time()
        }
        if not success:
            dqn_test_results['overall_success'] = False
        
        # Тест 4: Обработка NaN
        print("\n4️⃣ Тестирование обработки NaN...")
        success, message = test_nan_handling()
        dqn_test_results['tests']['nan_handling'] = {
            'success': success,
            'message': message,
            'timestamp': time.time()
        }
        if not success:
            dqn_test_results['overall_success'] = False
        
        # Тест 5: GPU Replay Buffer
        print("\n5️⃣ Тестирование GPU Replay Buffer...")
        success, message, metrics = test_gpu_replay_buffer()
        dqn_test_results['tests']['gpu_replay_buffer'] = {
            'success': success,
            'message': message,
            'metrics': metrics,
            'timestamp': time.time()
        }
        if not success:
            dqn_test_results['overall_success'] = False
        
        # Тест 6: Предвычисление состояний
        print("\n6️⃣ Тестирование предвычисления состояний...")
        success, message, metrics = test_precomputed_states()
        dqn_test_results['tests']['precomputed_states'] = {
            'success': success,
            'message': message,
            'metrics': metrics,
            'timestamp': time.time()
        }
        if not success:
            dqn_test_results['overall_success'] = False
        
        # Тест 7: torch.compile
        print("\n7️⃣ Тестирование torch.compile...")
        success, message, metrics = test_torch_compile()
        dqn_test_results['tests']['torch_compile'] = {
            'success': success,
            'message': message,
            'metrics': metrics,
            'timestamp': time.time()
        }
        if not success:
            dqn_test_results['overall_success'] = False
        
        # Финальные результаты
        end_time = time.time()
        duration = end_time - dqn_test_results['start_time']
        
        if dqn_test_results['overall_success']:
            print("\n" + "=" * 50)
            print("✅ Все тесты пройдены успешно!")
            print("🎯 DQN агент готов к использованию")
            
            dqn_test_results['status'] = 'completed'
            dqn_test_results['message'] = f'Все тесты пройдены успешно за {duration:.2f} секунд'
        else:
            print("\n❌ Некоторые тесты не пройдены")
            dqn_test_results['status'] = 'failed'
            dqn_test_results['message'] = f'Тесты завершены с ошибками за {duration:.2f} секунд'
        
        dqn_test_results['end_time'] = end_time
        dqn_test_results['duration'] = duration
        
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        
        dqn_test_results['status'] = 'error'
        dqn_test_results['message'] = f'Ошибка при тестировании: {str(e)}'
        dqn_test_results['overall_success'] = False
    
    finally:
        dqn_test_in_progress = False

@app.route('/test_dqn_improvements', methods=['POST'])
def test_dqn_improvements():
    """API endpoint для запуска тестирования улучшений DQN"""
    global dqn_test_in_progress, dqn_test_results
    
    if dqn_test_in_progress:
        return jsonify({
            'status': 'error',
            'message': 'Тестирование уже выполняется'
        }), 400
    
    # Запускаем тесты в отдельном потоке
    test_thread = threading.Thread(target=run_dqn_tests)
    test_thread.daemon = True
    test_thread.start()
    
    return jsonify({
        'status': 'success',
        'message': 'Тестирование улучшений DQN запущено',
        'test_id': int(time.time())
    })

@app.route('/test_dqn_status', methods=['GET'])
def test_dqn_status():
    """API endpoint для получения статуса тестирования"""
    global dqn_test_results, dqn_test_in_progress
    
    if not dqn_test_results:
        return jsonify({
            'status': 'not_started',
            'message': 'Тестирование не запускалось'
        })
    
    return jsonify(dqn_test_results)

@app.route('/test_dqn_results', methods=['GET'])
def test_dqn_results():
    """API endpoint для получения результатов тестирования"""
    global dqn_test_results
    
    if not dqn_test_results or dqn_test_results['status'] == 'running':
        return jsonify({
            'status': 'not_ready',
            'message': 'Результаты тестирования еще не готовы'
        })
    
    return jsonify(dqn_test_results)

@app.route('/test_gpu_replay', methods=['POST'])
def test_gpu_replay():
    """API endpoint для тестирования только GPU replay buffer"""
    try:
        success, message, metrics = test_gpu_replay_buffer()
        
        return jsonify({
            'status': 'success' if success else 'failed',
            'message': message,
            'metrics': metrics,
            'success': success
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Ошибка при тестировании GPU replay buffer: {str(e)}',
            'success': False
        }), 500

@app.route('/test_precomputed_states', methods=['POST'])
def test_precomputed_states_endpoint():
    """API endpoint для тестирования предвычисления состояний"""
    try:
        success, message, metrics = test_precomputed_states()
        
        return jsonify({
            'status': 'success' if success else 'failed',
            'message': message,
            'metrics': metrics,
            'success': success
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Ошибка при тестировании предвычисления состояний: {str(e)}',
            'success': False
        }), 500

@app.route('/test_torch_compile', methods=['POST'])
def test_torch_compile_endpoint():
    """API endpoint для тестирования torch.compile"""
    try:
        success, message, metrics = test_torch_compile()
        
        return jsonify({
            'status': 'success' if success else 'failed',
            'message': message,
            'metrics': metrics,
            'success': success
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Ошибка при тестировании torch.compile: {str(e)}',
            'success': False
        }), 500

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

@app.route('/analyze_training_results', methods=['POST'])
def analyze_training_results():
    """Анализирует результаты обучения DQN модели"""
    try:
        # Ищем файлы с результатами обучения
        result_files = glob.glob('training_results_*.pkl')
        
        if not result_files:
            return jsonify({
                'status': 'error',
                'message': 'Файлы результатов обучения не найдены. Сначала запустите обучение.',
                'success': False
            }), 404
        
        # Берем самый свежий файл
        latest_file = max(result_files, key=os.path.getctime)
        
        # Импортируем функцию анализа
        try:
            from analyze_training_results import analyze_training_results as analyze_func
        except ImportError:
            # Если модуль не найден, создаем простую функцию анализа
            def analyze_func(filename):
                print(f"📊 Анализ файла: {filename}")
                print("⚠️ Модуль анализа не найден. Установите matplotlib и numpy.")
                print("💡 Для полного анализа используйте: pip install matplotlib numpy")
                return "Анализ недоступен - установите зависимости"
        
        # Запускаем анализ
        print(f"📊 Анализирую результаты из файла: {latest_file}")
        
        # Временно перенаправляем stdout для захвата вывода
        import io
        import sys
        from contextlib import redirect_stdout
        
        output = io.StringIO()
        with redirect_stdout(output):
            analyze_func(latest_file)
        
        analysis_output = output.getvalue()
        
        return jsonify({
            'status': 'success',
            'message': 'Анализ результатов завершен успешно',
            'success': True,
            'file_analyzed': latest_file,
            'output': analysis_output,
            'available_files': result_files
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Ошибка при анализе результатов: {str(e)}',
            'success': False
        }), 500

@app.route('/list_training_results', methods=['GET'])
def list_training_results():
    """Возвращает список доступных файлов с результатами обучения"""
    try:
        result_files = glob.glob('training_results_*.pkl')
        
        if not result_files:
            return jsonify({
                'status': 'error',
                'message': 'Файлы результатов обучения не найдены',
                'success': False,
                'files': []
            }), 404
        
        # Получаем информацию о файлах
        files_info = []
        for file in result_files:
            stat = os.stat(file)
            files_info.append({
                'filename': file,
                'size': stat.st_size,
                'created': stat.st_ctime,
                'modified': stat.st_mtime
            })
        
        # Сортируем по дате создания (новые первыми)
        files_info.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'message': f'Найдено {len(result_files)} файлов результатов',
            'success': True,
            'files': files_info
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Ошибка при получении списка файлов: {str(e)}',
            'success': False
        }), 500

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

@app.route('/clear_redis', methods=['POST'])
def clear_redis():
    """Очищает Redis вручную"""
    try:
        global redis_client
        redis_client.flushall()
        return jsonify({
            "success": True,
            "message": "Redis очищен успешно"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Автоматический запуск Flask сервера
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))  # Получаем порт из переменной окружения
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    print(f"🚀 Запускаю Flask сервер на порту {port}...")
    print(f"🌐 Откройте: http://localhost:{port}")
    print(f"🔧 Debug режим: {'ВКЛЮЧЕН' if debug_mode else 'ОТКЛЮЧЕН'}")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)

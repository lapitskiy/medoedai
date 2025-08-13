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
import threading
import time
import torch  # Добавляем импорт torch для функций тестирования DQN

logging.basicConfig(level=logging.INFO)

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
    print("🧠 Тестирование улучшенной архитектуры нейронной сети...")
    
    try:
        from agents.vdqn.cfg.vconfig import vDqnConfig
        from agents.vdqn.dqnn import DQNN
        
        cfg = vDqnConfig()
        
        # Тестируем Dueling DQN
        obs_dim = 100
        act_dim = 3
        hidden_sizes = (512, 256, 128)
        
        model = DQNN(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            dropout_rate=cfg.dropout_rate,
            layer_norm=cfg.layer_norm,
            dueling=cfg.dueling_dqn
        )
        
        print(f"✅ Модель создана успешно")
        print(f"   - Архитектура: {hidden_sizes}")
        print(f"   - Dropout: {cfg.dropout_rate}")
        print(f"   - Layer Norm: {cfg.layer_norm}")
        print(f"   - Dueling: {cfg.dueling_dqn}")
        
        # Тестируем forward pass
        test_input = torch.randn(1, obs_dim)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ Forward pass успешен")
        print(f"   - Вход: {test_input.shape}")
        print(f"   - Выход: {output.shape}")
        print(f"   - Q-значения: {output.squeeze().tolist()}")
        
        # Проверяем на NaN
        if torch.isnan(output).any():
            print("❌ Обнаружены NaN значения в выходе!")
            return False, "Обнаружены NaN значения в выходе"
        else:
            print("✅ NaN значения не обнаружены")
            return True, "Модель создана и протестирована успешно"
            
    except Exception as e:
        print(f"❌ Ошибка при тестировании нейронной сети: {e}")
        return False, f"Ошибка при тестировании нейронной сети: {str(e)}"

def test_dqn_solver():
    """Тестирует улучшенный DQN solver"""
    print("\n🔧 Тестирование улучшенного DQN solver...")
    
    try:
        from agents.vdqn.cfg.vconfig import vDqnConfig
        from agents.vdqn.dqnsolver import DQNSolver
        import numpy as np
        
        cfg = vDqnConfig()
        
        # Создаем solver
        observation_space = 100
        action_space = 3
        
        solver = DQNSolver(observation_space, action_space, load=False)
        
        print(f"✅ DQN Solver создан успешно")
        print(f"   - Prioritized Replay: {cfg.prioritized}")
        print(f"   - Memory Size: {cfg.memory_size}")
        print(f"   - Batch Size: {cfg.batch_size}")
        print(f"   - Learning Rate: {cfg.lr}")
        print(f"   - Gamma: {cfg.gamma}")
        
        # Тестируем добавление переходов
        test_state = np.random.randn(100)
        test_action = 1
        test_reward = 0.5
        test_next_state = np.random.randn(100)
        test_done = False
        
        solver.store_transition(test_state, test_action, test_reward, test_next_state, test_done)
        print(f"✅ Переход добавлен в replay buffer")
        print(f"   - Размер буфера: {len(solver.memory)}")
        
        # Тестируем выбор действия
        action = solver.act(test_state)
        print(f"✅ Действие выбрано: {action}")
        print(f"   - Epsilon: {solver.epsilon:.4f}")
        
        return True, "DQN Solver протестирован успешно"
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании DQN solver: {e}")
        return False, f"Ошибка при тестировании DQN solver: {str(e)}"

def test_configuration():
    """Тестирует конфигурацию"""
    print("\n⚙️ Тестирование конфигурации...")
    
    try:
        from agents.vdqn.cfg.vconfig import vDqnConfig
        
        cfg = vDqnConfig()
        
        print("✅ Конфигурация загружена:")
        print(f"   - Epsilon: {cfg.eps_start} → {cfg.eps_final} за {cfg.eps_decay_steps} шагов")
        print(f"   - Архитектура: {cfg.hidden_sizes}")
        print(f"   - Обучение: lr={cfg.lr}, gamma={cfg.gamma}")
        print(f"   - Replay: size={cfg.memory_size}, batch={cfg.batch_size}")
        print(f"   - PER: {cfg.prioritized}, alpha={cfg.alpha}, beta={cfg.beta}")
        print(f"   - Улучшения: dropout={cfg.dropout_rate}, layer_norm={cfg.layer_norm}")
        print(f"   - DQN: double={cfg.double_dqn}, dueling={cfg.dueling_dqn}")
        
        # Проверяем совместимость параметров
        if cfg.batch_size > cfg.memory_size:
            print("❌ Batch size больше memory size!")
            return False, "Batch size больше memory size"
        else:
            print("✅ Параметры совместимы")
        
        if cfg.eps_final >= cfg.eps_start:
            print("❌ Epsilon final должен быть меньше eps start!")
            return False, "Epsilon final должен быть меньше eps start"
        else:
            print("✅ Epsilon параметры корректны")
        
        return True, "Конфигурация протестирована успешно"
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании конфигурации: {e}")
        return False, f"Ошибка при тестировании конфигурации: {str(e)}"

def test_nan_handling():
    """Тестирует обработку NaN значений"""
    print("\n🛡️ Тестирование обработки NaN значений...")
    
    try:
        from agents.vdqn.cfg.vconfig import vDqnConfig
        from agents.vdqn.dqnn import DQNN
        from agents.vdqn.dqnsolver import DQNSolver
        import numpy as np
        
        cfg = vDqnConfig()
        
        # Создаем модель
        model = DQNN(100, 3, (512, 256, 128))
        
        # Тестируем с NaN входом
        test_input = np.random.randn(100)
        test_input[0] = np.nan  # Добавляем NaN
        
        print(f"   - Вход содержит NaN: {np.isnan(test_input).any()}")
        
        # Тестируем обработку в solver
        solver = DQNSolver(100, 3, load=False)
        
        # Должно автоматически заменить NaN на нули
        action = solver.act(test_input)
        print(f"✅ Действие выбрано даже с NaN входом: {action}")
        
        # Проверяем, что NaN заменены
        cleaned_input = np.nan_to_num(test_input, nan=0.0)
        print(f"   - NaN заменены на нули: {np.isnan(cleaned_input).any()}")
        
        return True, "Обработка NaN значений протестирована успешно"
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании обработки NaN: {e}")
        return False, f"Ошибка при тестировании обработки NaN: {str(e)}"

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



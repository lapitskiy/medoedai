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
from tasks.celery_tasks import search_lstm_task, train_dqn, train_dqn_multi_crypto, trade_step, start_trading_task, train_dqn_symbol
from utils.db_utils import clean_ohlcv_data, delete_ohlcv_for_symbol_timeframe, load_latest_candles_from_csv_to_db
from utils.parser import parser_download_and_combine_with_library
from utils.trade_utils import (
    get_recent_trades, 
    get_trade_statistics, 
    get_trades_by_symbol,
    get_model_predictions, 
    get_prediction_statistics
)

import logging
from flask import Response
import json

import time

import glob
import os

import docker

from tasks.celery_tasks import start_trading_task

# Убираем импорт TradingAgent и глобальный экземпляр
# from trading_agent.trading_agent import TradingAgent
# trading_agent = None

# Убираем функцию init_trading_agent
# def init_trading_agent():
#     """Инициализация торгового агента"""
#     global trading_agent
#     try:
#         trading_agent = TradingAgent()
#         app.logger.info("Торговый агент инициализирован")
#     except Exception as e:
#         app.logger.error(f"Ошибка инициализации торгового агента: {e}")

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



def ensure_symbol_worker(queue_name: str) -> dict:
    """Гарантирует наличие отдельного Celery-воркера для указанной очереди.

    - Проверяет Redis-лок `celery:worker:<queue>`
    - Проверяет процесс внутри контейнера `celery-worker`
    - При отсутствии — запускает новый процесс воркера для очереди
    """
    try:
        lock_key = f"celery:worker:{queue_name}"
        # Быстрая проверка по Redis-метке
        try:
            if redis_client.get(lock_key):
                return {"started": False, "reason": "already_marked_running"}
        except Exception:
            pass

        import docker
        client = docker.from_env()
        container = client.containers.get('celery-worker')
        app.logger.info(f"[ensure_worker] container 'celery-worker' status={container.status}")

        # Проверяем по процессам внутри контейнера
        check_cmd = f"sh -lc 'pgrep -af \"celery.*-Q {queue_name}\" >/dev/null 2>&1'"
        exec_res = container.exec_run(check_cmd, tty=True)
        already_running = (exec_res.exit_code == 0)
        app.logger.info(f"[ensure_worker] check queue={queue_name} exit={exec_res.exit_code}")

        if already_running:
            # Ставим метку в Redis и выходим
            try:
                redis_client.setex(lock_key, 24 * 3600, 'running')
            except Exception:
                pass
            return {"started": False, "reason": "process_exists"}

        # Запускаем новый воркер-процесс в фоне
        start_cmd = (
            "sh -lc '"
            "mkdir -p /workspace/logs && "
            f"(OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4 nohup celery -A tasks.celery_tasks worker -Q {queue_name} -P solo -c 1 --loglevel=info "
            f"> /workspace/logs/{queue_name}.log 2>&1 & echo $! > /workspace/logs/{queue_name}.pid) "
            "'"
        )
        res = container.exec_run(start_cmd, tty=True)
        app.logger.info(f"[ensure_worker] start queue={queue_name} exit={res.exit_code}")

        # Повторная проверка
        exec_res2 = container.exec_run(check_cmd, tty=True)
        app.logger.info(f"[ensure_worker] recheck queue={queue_name} exit={exec_res2.exit_code}")

        # Отмечаем в Redis
        try:
            redis_client.setex(lock_key, 24 * 3600, 'running')
        except Exception:
            pass

        return {"started": True}
    except Exception as e:
        app.logger.error(f"[ensure_worker] error: {e}")
        return {"started": False, "error": str(e)}

@app.before_request
def log_request_info():
    logging.info(f"Request from {request.remote_addr}: {request.method} {request.path}")
    logging.info("Headers: %s", dict(request.headers))  # Логируем все заголовки

@app.route("/")
def index():
    """Возвращает список всех задач Celery и их состояние"""
    tasks = []

    # 1) Показываем последние отправленные задачи из списка ui:tasks
    try:
        recent_ids = redis_client.lrange("ui:tasks", 0, 49) or []
        for raw_id in recent_ids:
            try:
                task_id = raw_id.decode("utf-8")
            except Exception:
                task_id = str(raw_id)
            task = AsyncResult(task_id, app=celery)
            tasks.append({
                "task_id": task_id,
                "state": task.state,
                "result": task.result if task.successful() else None
            })
    except Exception as _e:
        app.logger.error(f"/ index: ошибка чтения ui:tasks: {_e}")

    # 2) Fallback: сканируем backend по ключам (может быть пусто для PENDING)
    if not tasks:
        try:
            task_keys = redis_client.keys("celery-task-meta-*") or []
            for key in task_keys:
                task_id = key.decode("utf-8").replace("celery-task-meta-", "")
                task = AsyncResult(task_id, app=celery)
                tasks.append({
                    "task_id": task_id,
                    "state": task.state,
                    "result": task.result if task.successful() else None
                })
        except Exception as _e:
            app.logger.error(f"/ index: ошибка сканирования backend: {_e}")

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
    task = AsyncResult(task_id, app=celery)

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

@app.route('/train_dqn_multi_crypto', methods=['POST'])
def train_multi_crypto():
    """Запускает мультивалютное обучение DQN"""
    task = train_dqn_multi_crypto.apply_async(queue="train")
    return redirect(url_for("index"))

@app.route('/train_dqn_symbol', methods=['POST'])
def train_dqn_symbol_route():
    data = request.get_json(silent=True) or {}
    symbol = data.get('symbol') or request.form.get('symbol') or 'BTCUSDT'
    # Эпизоды: из формы/JSON, fallback None (чтоб Celery взял из ENV)
    episodes_str = data.get('episodes') or request.form.get('episodes')
    episodes = None
    try:
        if episodes_str is not None and str(episodes_str).strip() != '':
            episodes = int(episodes_str)
    except Exception:
        episodes = None
    # Очередь per-symbol
    queue_name = f"train_{symbol.lower()}"

    # Проверка активной задачи per-symbol в Redis + Celery
    running_key = f"celery:train:task:{symbol.upper()}"
    try:
        existing_task_id = redis_client.get(running_key)
        if existing_task_id:
            try:
                existing_task_id = existing_task_id.decode('utf-8')
            except Exception:
                existing_task_id = str(existing_task_id)
            ar = AsyncResult(existing_task_id, app=celery)
            if ar.state in ("PENDING", "STARTED", "RETRY", "IN_PROGRESS"):
                app.logger.info(f"train for {symbol} already running: {existing_task_id} state={ar.state}")
                return redirect(url_for("index"))
    except Exception as _e:
        app.logger.error(f"Не удалось проверить активную задачу для {symbol}: {_e}")

    # Временно отправляем в общую очередь 'train' (слушается базовым воркером)
    task = train_dqn_symbol.apply_async(args=[symbol, episodes], queue="train")
    app.logger.info(f"/train_dqn_symbol queued symbol={symbol} queue=train task_id={task.id}")
    # Сохраняем task_id для отображения на главной и отметку per-symbol
    try:
        redis_client.lrem("ui:tasks", 0, task.id)  # убираем дубликаты
        redis_client.lpush("ui:tasks", task.id)
        redis_client.ltrim("ui:tasks", 0, 49)     # ограничиваем список
        redis_client.setex(running_key, 24 * 3600, task.id)
    except Exception as _e:
        app.logger.error(f"/train_dqn_symbol: не удалось записать ui:tasks: {_e}")
    return redirect(url_for("index"))


@app.route('/continue_training', methods=['POST'])
def continue_training_route():
    """Продолжает обучение из существующих файлов в result (dqn_model_*.pth, replay_buffer_*.pkl).

    Body: { file: 'result/train_result_<code>.pkl', episodes?: int }
    По имени train_result_* извлекается <code>, затем берутся dqn_model_<code>.pth и replay_buffer_<code>.pkl.
    """
    try:
        data = request.get_json(silent=True) or {}
        requested_file = (data.get('file') or data.get('model') or '').strip()
        episodes_str = data.get('episodes') or request.form.get('episodes')
        episodes = None
        try:
            if episodes_str is not None and str(episodes_str).strip() != '':
                episodes = int(episodes_str)
        except Exception:
            episodes = None

        from pathlib import Path as _Path
        result_dir = _Path('result')
        if not result_dir.exists():
            return jsonify({"success": False, "error": "Папка result не найдена"}), 400

        if not requested_file:
            return jsonify({"success": False, "error": "Не указан путь к файлу весов dqn_model_*.pth"}), 400

        # Нормализуем и проверяем путь
        req_norm = requested_file.replace('\\', '/')
        safe_path = _Path(req_norm)
        if not safe_path.is_absolute():
            if not (safe_path.parts and safe_path.parts[0].lower() == result_dir.name.lower()):
                safe_path = result_dir / safe_path.name
        try:
            cand_resolved = safe_path.resolve()
        except Exception:
            cand_resolved = safe_path
        inside_result = str(cand_resolved).lower().startswith(str(result_dir.resolve()).lower())

        if not (inside_result and cand_resolved.exists() and cand_resolved.is_file()):
            return jsonify({"success": False, "error": "Неверный путь к файлу или файл не существует"}), 400

        model_file = None
        replay_file = None
        code = None

        # Поддержка двух форматов: train_result_* и dqn_model_*
        if cand_resolved.name.startswith('train_result_') and cand_resolved.suffix == '.pkl':
            fname = cand_resolved.stem  # train_result_<code>
            parts = fname.split('_', 2)
            if len(parts) < 3:
                return jsonify({"success": False, "error": "Не удалось извлечь код модели из имени файла"}), 400
            code = parts[2]
            model_file = result_dir / f'dqn_model_{code}.pth'
            replay_file = result_dir / f'replay_buffer_{code}.pkl'
            if not model_file.exists():
                return jsonify({"success": False, "error": f"Файл {model_file.name} не найден"}), 400
            # replay_file может отсутствовать – это допустимо
        elif cand_resolved.name.startswith('dqn_model_') and cand_resolved.suffix == '.pth':
            fname = cand_resolved.stem  # dqn_model_<code>
            parts = fname.split('_', 2)
            if len(parts) < 3:
                return jsonify({"success": False, "error": "Не удалось извлечь код модели из имени весов"}), 400
            code = parts[2]
            model_file = cand_resolved
            replay_file = result_dir / f'replay_buffer_{code}.pkl'  # может не существовать
        else:
            return jsonify({"success": False, "error": "Ожидался файл dqn_model_*.pth или train_result_*.pkl"}), 400

        # Запускаем фоновой таск обучения, передав пути в ENV (упрощение без изменения сигнатуры Celery)
        # Используем общую очередь 'train'
        from celery import group
        # Сохраним параметры в Redis на час
        try:
            redis_client.setex('continue:model_path', 3600, str(model_file))
            if replay_file and os.path.exists(str(replay_file)):
                redis_client.setex('continue:buffer_path', 3600, str(replay_file))
        except Exception:
            pass

        # Таск train_dqn_symbol не принимает пути, поэтому временно применим символ из code для данных,
        # а загрузка модели произойдет в v_train_model_optimized по этим путям через ENV.
        symbol_guess = (code.split('_', 1)[0] + 'USDT').upper()
        task = train_dqn_symbol.apply_async(args=[symbol_guess, episodes], queue='train')

        # Прокинем пути через ENV переменные для этого процесса воркера (если он читает их)
        # Если воркер в другом процессе, используем Redis как источник — уже сохранено выше.
        try:
            os.environ['CONTINUE_MODEL_PATH'] = str(model_file)
            if replay_file and os.path.exists(str(replay_file)):
                os.environ['CONTINUE_BUFFER_PATH'] = str(replay_file)
        except Exception:
            pass

        # Добавим задачу в UI список
        try:
            redis_client.lrem("ui:tasks", 0, task.id)
            redis_client.lpush("ui:tasks", task.id)
            redis_client.ltrim("ui:tasks", 0, 49)
        except Exception:
            pass

        return jsonify({"success": True, "task_id": task.id})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500





@app.route('/trade_dqn', methods=['POST'])
def trade():
    task = trade_step.apply_async()
    return redirect(url_for("index"))

# Новый маршрут для запуска очистки данных
@app.route('/clean_data', methods=['POST'])
def clean_data():
    timeframes_to_clean = ['5m', '15m', '1h']
    symbol_name ='BTCUSDT'
    
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
    symbol_name ='BTCUSDT'            

    results = []
    try:
        # Вызываем функцию очистки напрямую (она больше не Celery-задача)
        delete_ohlcv_for_symbol_timeframe('BTCUSDT', timeframes_to_clean)
    except Exception as e:
        results.append({"status": "error", "message": f"Ошибка при очистке для {symbol_name}: {str(e)}"})


    return jsonify({'status': 'Очистка базы от всех свечей завершена указанных таймфреймов.', 'results': results})

@app.route('/analyze_training_results', methods=['POST'])
def analyze_training_results():
    """Анализирует результаты обучения DQN модели"""
    try:
        # Ищем файлы с результатами обучения в папке result
        results_dir = "result"
        if not os.path.exists(results_dir):
            return jsonify({
                'status': 'error',
                'message': f'Папка {results_dir} не найдена. Сначала запустите обучение.',
                'success': False
            }), 404
        
        result_files = glob.glob(os.path.join(results_dir, 'train_result_*.pkl'))
        
        if not result_files:
            return jsonify({
                'status': 'error',
                'message': 'Файлы результатов обучения не найдены. Сначала запустите обучение.',
                'success': False
            }), 404
        
        # Опционально берём конкретный файл из тела запроса
        data = request.get_json(silent=True) or {}
        requested_file = data.get('file')
        selected_file = None
        if requested_file:
            # Простая валидация: путь внутри result/
            safe_path = os.path.abspath(requested_file)
            base_path = os.path.abspath(results_dir)
            if safe_path.startswith(base_path) and os.path.exists(safe_path):
                selected_file = safe_path
        # Если не указан/невалиден — берём самый свежий
        if not selected_file:
            selected_file = max(result_files, key=os.path.getctime)
        
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
        
        # Загружаем результаты для дополнительного анализа
        try:
            import pickle
            with open(selected_file, 'rb') as f:
                results = pickle.load(f)
            
            # Добавляем информацию об actual_episodes
            if 'actual_episodes' in results:
                actual_episodes = results['actual_episodes']
                planned_episodes = results['episodes']
                
                if actual_episodes < planned_episodes:
                    print(f"⚠️ Early Stopping сработал! Обучение остановлено на {actual_episodes} эпизоде из {planned_episodes}")
                else:
                    print(f"✅ Обучение завершено полностью: {actual_episodes} эпизодов")                    
        except Exception as e:
            print(f"⚠️ Не удалось загрузить детали результатов: {e}")
        
        # Запускаем анализ
        print(f"📊 Анализирую результаты из файла: {selected_file}")
        
        # Временно перенаправляем stdout для захвата вывода
        import io
        import sys
        from contextlib import redirect_stdout
        
        output = io.StringIO()
        with redirect_stdout(output):
            analyze_func(selected_file)
        
        analysis_output = output.getvalue()
        
        # Добавляем информацию об actual_episodes в ответ
        response_data = {
            'status': 'success',
            'message': 'Анализ результатов завершен успешно',
            'success': True,
            'file_analyzed': selected_file,
            'output': analysis_output,
            'available_files': result_files
        }
        
        # Добавляем информацию об эпизодах если доступна
        try:
            # Добавляем episode_winrates_count для правильного определения early stopping
            if 'episode_winrates' in results:
                response_data['episode_winrates_count'] = len(results['episode_winrates'])
                print(f"🔍 episode_winrates_count: {response_data['episode_winrates_count']}")
            
            if 'actual_episodes' in results:
                response_data['actual_episodes'] = results['actual_episodes']
                response_data['episodes'] = results['episodes']
                
                # Проверяем несоответствие actual_episodes и episode_winrates_count
                if 'episode_winrates_count' in response_data:
                    if response_data['actual_episodes'] != response_data['episode_winrates_count']:
                        print(f"⚠️ НЕСООТВЕТСТВИЕ: actual_episodes={response_data['actual_episodes']}, episode_winrates_count={response_data['episode_winrates_count']}")
                        # Исправляем actual_episodes на правильное значение
                        response_data['actual_episodes'] = response_data['episode_winrates_count']
                        print(f"🔧 Исправлено: actual_episodes = {response_data['actual_episodes']}")
            else:
                # Если actual_episodes не найден, пытаемся извлечь из логов
                if 'output' in response_data:
                    output_text = response_data['output']
                    # Ищем Early stopping в логах
                    if 'Early stopping triggered after' in output_text:
                        import re
                        early_stopping_match = re.search(r'Early stopping triggered after (\d+) episodes', output_text)
                        if early_stopping_match:
                            actual_episodes = int(early_stopping_match.group(1))
                            # Ищем планируемое количество эпизодов
                            episodes_match = re.search(r'Количество эпизодов: (\d+)', output_text)
                            if episodes_match:
                                planned_episodes = int(episodes_match.group(1))
                                response_data['actual_episodes'] = actual_episodes
                                response_data['episodes'] = planned_episodes
                                print(f"🔍 Извлечено из логов: actual_episodes={actual_episodes}, episodes={planned_episodes}")
        except Exception as e:
            print(f"⚠️ Ошибка при извлечении actual_episodes: {e}")
        
        return jsonify(response_data)
        
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
        # Ищем файлы с результатами обучения в папке result
        results_dir = "result"
        if not os.path.exists(results_dir):
            return jsonify({
                'status': 'error',
                'message': f'Папка {results_dir} не найдена',
                'success': False,
                'files': []
            }), 404
        
        result_files = glob.glob(os.path.join(results_dir, 'train_result_*.pkl'))
        
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
@app.route('/list_result_models', methods=['GET'])
def list_result_models():
    """Возвращает список доступных весов моделей из result (dqn_model_*.pth)."""
    try:
        results_dir = "result"
        if not os.path.exists(results_dir):
            return jsonify({
                'status': 'error',
                'message': f'Папка {results_dir} не найдена',
                'success': False,
                'files': []
            }), 404

        model_files = glob.glob(os.path.join(results_dir, 'dqn_model_*.pth'))
        if not model_files:
            return jsonify({
                'status': 'error',
                'message': 'Файлы весов не найдены',
                'success': False,
                'files': []
            }), 404

        files_info = []
        for file in model_files:
            stat = os.stat(file)
            files_info.append({
                'filename': file,
                'size': stat.st_size,
                'created': stat.st_ctime,
                'modified': stat.st_mtime
            })

        files_info.sort(key=lambda x: x['modified'], reverse=True)

        return jsonify({
            'status': 'success',
            'message': f'Найдено {len(files_info)} файлов весов',
            'success': True,
            'files': files_info
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Ошибка при получении списка весов: {str(e)}',
            'success': False
        }), 500

@app.route('/get_result_model_info', methods=['POST'])
def get_result_model_info():
    """Возвращает краткую информацию по выбранному файлу весов из result/ (dqn_model_*.pth):
    символ/код, наличие replay/train_result, базовая статистика (winrate, trades_count, episodes).
    Body: { filename: 'result/dqn_model_XXXX.pth' }
    """
    try:
        data = request.get_json(silent=True) or {}
        filename = (data.get('filename') or '').strip()
        if not filename:
            return jsonify({'success': False, 'error': 'Не указан filename'}), 400

        from pathlib import Path as _Path
        results_dir = _Path('result')
        # Нормализация пути
        req_norm = filename.replace('\\', '/')
        p = _Path(req_norm)
        if not p.is_absolute():
            if not (p.parts and p.parts[0].lower() == results_dir.name.lower()):
                p = results_dir / p.name
        try:
            p = p.resolve()
        except Exception:
            pass
        # Безопасность: только внутри result/
        if not str(p).lower().startswith(str(results_dir.resolve()).lower()):
            return jsonify({'success': False, 'error': 'Файл вне папки result'}), 400
        if not p.exists() or not p.is_file() or not p.name.startswith('dqn_model_') or p.suffix != '.pth':
            return jsonify({'success': False, 'error': 'Ожидается существующий файл dqn_model_*.pth'}), 400

        # Извлекаем код
        code = p.stem.replace('dqn_model_', '')
        replay_file = results_dir / f'replay_buffer_{code}.pkl'
        train_file = results_dir / f'train_result_{code}.pkl'

        info = {
            'success': True,
            'model_file': str(p),
            'model_size_bytes': p.stat().st_size if p.exists() else 0,
            'code': code,
            'replay_exists': replay_file.exists(),
            'train_result_exists': train_file.exists(),
            'replay_file': str(replay_file) if replay_file.exists() else None,
            'train_result_file': str(train_file) if train_file.exists() else None,
            'stats': {},
            'episodes': None
        }

        # Загружаем статистику из train_result_*.pkl если есть
        if train_file.exists():
            try:
                import pickle
                with open(train_file, 'rb') as f:
                    results = pickle.load(f)
                stats = results.get('final_stats', {}) or {}
                info['stats'] = {
                    'winrate': stats.get('winrate'),
                    'pl_ratio': stats.get('pl_ratio'),
                    'trades_count': stats.get('trades_count')
                }
                info['episodes'] = results.get('actual_episodes', results.get('episodes'))
            except Exception as _e:
                info['stats_error'] = str(_e)

        return jsonify(info)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/analyze_bad_trades', methods=['POST'])
def analyze_bad_trades():
    """Анализирует плохие сделки из результатов обучения DQN модели"""
    try:
        # Ищем файлы с результатами обучения в папке result
        results_dir = "result"
        if not os.path.exists(results_dir):
            return jsonify({
                'status': 'error',
                'message': f'Папка {results_dir} не найдена. Сначала запустите обучение.',
                'success': False
            }), 404
        
        result_files = glob.glob(os.path.join(results_dir, 'train_result_*.pkl'))
        
        if not result_files:
            return jsonify({
                'status': 'error',
                'message': 'Файлы результатов обучения не найдены. Сначала запустите обучение.',
                'success': False
            }), 404
        
        # Выбор конкретного файла из тела запроса (если указан)
        data = request.get_json(silent=True) or {}
        requested_file = data.get('file')
        selected_file = None
        if requested_file:
            safe_path = os.path.abspath(requested_file)
            base_path = os.path.abspath(results_dir)
            if safe_path.startswith(base_path) and os.path.exists(safe_path):
                selected_file = safe_path
        if not selected_file:
            selected_file = max(result_files, key=os.path.getctime)
        
        # Импортируем функцию анализа плохих сделок (с fallback без зависимостей)
        try:
            from analyze_bad_trades import analyze_bad_trades_detailed, print_bad_trades_analysis, print_detailed_recommendations
        except ImportError as e:
            app.logger.warning(f"Fallback analyze_bad_trades (ImportError: {e})")
            import numpy as _np
            def analyze_bad_trades_detailed(trades):
                if not trades:
                    return {
                        'bad_trades': [], 'bad_trades_count': 0,
                        'bad_trades_percentage': 0.0, 'avg_bad_roi': 0.0,
                        'avg_bad_duration': 0.0, 'loss_distribution': {},
                    }
                total = len(trades)
                bad = [t for t in trades if float(t.get('roi', 0.0)) < 0.0]
                bad_rois = [float(t.get('roi', 0.0)) for t in bad]
                bad_durs = [float(t.get('duration', 0.0)) for t in bad if t.get('duration') is not None]
                return {
                    'bad_trades': bad,
                    'bad_trades_count': len(bad),
                    'bad_trades_percentage': (len(bad)/total*100.0) if total else 0.0,
                    'avg_bad_roi': float(_np.mean(bad_rois)) if bad_rois else 0.0,
                    'avg_bad_duration': float(_np.mean(bad_durs)) if bad_durs else 0.0,
                    'loss_distribution': {
                        'very_small_losses': sum(1 for r in bad_rois if -0.002 <= r < 0),
                        'small_losses':      sum(1 for r in bad_rois if -0.01  <= r < -0.002),
                        'medium_losses':     sum(1 for r in bad_rois if -0.03  <= r < -0.01),
                        'large_losses':      sum(1 for r in bad_rois if r < -0.03),
                    }
                }
            def print_bad_trades_analysis(analysis):
                print("📉 АНАЛИЗ ПЛОХИХ СДЕЛОК")
                print(f"Всего плохих сделок: {analysis.get('bad_trades_count', 0)}")
                print(f"Процент плохих сделок: {analysis.get('bad_trades_percentage', 0):.2f}%")
                print(f"Средний ROI плохих сделок: {analysis.get('avg_bad_roi', 0.0)*100:.4f}%")
                print(f"Средняя длительность плохих сделок: {analysis.get('avg_bad_duration', 0.0):.1f} мин")
            def print_detailed_recommendations(analysis):
                print("🧠 РЕКОМЕНДАЦИИ: ")
        
        # Загружаем результаты для анализа
        try:
            import pickle
            with open(selected_file, 'rb') as f:
                results = pickle.load(f)
            
            # Проверяем наличие сделок
            if 'all_trades' not in results:
                return jsonify({
                    'status': 'error',
                    'message': 'В файле нет данных о сделках',
                    'success': False
                }), 404
            
            trades = results['all_trades']
            
            # Анализируем плохие сделки
            bad_trades_analysis = analyze_bad_trades_detailed(trades)
            
            # Добавляем все сделки для сравнения
            bad_trades_analysis['all_trades'] = trades
            
            # Временно перенаправляем stdout для захвата вывода
            import io
            import sys
            from contextlib import redirect_stdout
            
            output = io.StringIO()
            with redirect_stdout(output):
                print_bad_trades_analysis(bad_trades_analysis)
                print_detailed_recommendations(bad_trades_analysis)
            
            analysis_output = output.getvalue()
            
            # Подготавливаем ответ
            response_data = {
                'status': 'success',
                'message': 'Анализ плохих сделок завершен успешно',
                'success': True,
                'file_analyzed': selected_file,
                'output': analysis_output,
                'bad_trades_count': bad_trades_analysis.get('bad_trades_count', 0),
                'bad_trades_percentage': bad_trades_analysis.get('bad_trades_percentage', 0),
                'analysis_summary': {
                    'total_trades': len(trades),
                    'bad_trades': bad_trades_analysis.get('bad_trades_count', 0),
                    'avg_bad_roi': bad_trades_analysis.get('avg_bad_roi', 0),
                    'avg_bad_duration': bad_trades_analysis.get('avg_bad_duration', 0),
                    'loss_distribution': bad_trades_analysis.get('loss_distribution', {})
                }
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Ошибка при анализе файла: {str(e)}',
                'success': False
            }), 500
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Ошибка при анализе плохих сделок: {str(e)}',
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
        # Если файл не создан — не пытаемся читать/загружать
        try:
            import os as _os
            if not csv_file_path or not _os.path.exists(csv_file_path):
                results.append({"status": "warning", "message": "Не найдено CSV-файлов (библиотека не дала данных)."})
                response = {'status': 'Парсинг завершен', 'results': results}
                return Response(json.dumps(response, ensure_ascii=False), mimetype='application/json')
        except Exception:
            pass
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

# Новая страница управления данными
@app.route('/data', methods=['GET'])
def data_page():
    return render_template('data.html')

# Отдельный маршрут для XMRUSDT
@app.route('/parser_xmr', methods=['POST'])
def parser_xmr():
    results = []
    symbol = 'XMRUSDT'
    interval = '5m'
    desired_candles = 100000
    csv_file_path = None

    try:
        csv_file_path = parser_download_and_combine_with_library(
            symbol=symbol,
            interval=interval,
            months_to_fetch=12,
            desired_candles=desired_candles
        )
        # Проверяем наличие файла; если нет — возвращаем понятное предупреждение
        try:
            import os as _os
            if not csv_file_path or not _os.path.exists(csv_file_path):
                results.append({"status": "warning", "message": "Не найдено CSV-файлов для XMRUSDT (в библиотеке нет данных)."})
                response = {'status': 'Парсинг завершен', 'results': results}
                return Response(json.dumps(response, ensure_ascii=False), mimetype='application/json')
        except Exception:
            pass
        results.append({"status": "success", "message": f"CSV файл создан: {csv_file_path}"})

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

@app.route('/parser_xrp', methods=['POST'])
def parser_xrp():
    results = []
    symbol = 'XRPUSDT'
    interval = '5m'
    desired_candles = 100000
    csv_file_path = None

    try:
        csv_file_path = parser_download_and_combine_with_library(
            symbol=symbol,
            interval=interval,
            months_to_fetch=12,
            desired_candles=desired_candles
        )
        # Проверяем наличие файла; если нет — возвращаем понятное предупреждение
        try:
            import os as _os
            if not csv_file_path or not _os.path.exists(csv_file_path):
                results.append({"status": "warning", "message": "Не найдено CSV-файлов для XRPUSDT (в библиотеке нет данных)."})
                response = {'status': 'Парсинг завершен', 'results': results}
                return Response(json.dumps(response, ensure_ascii=False), mimetype='application/json')
        except Exception:
            pass
        results.append({"status": "success", "message": f"CSV файл создан: {csv_file_path}"})

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

# Новый маршрут для мультивалютного скачивания свечей
@app.route('/parser_multi_crypto', methods=['POST'])
def parser_multi_crypto():
    """Скачивает свечи для всех криптовалют одновременно"""
    results = []
    interval = '5m'
    desired_candles = 100000
    
    # Список всех криптовалют для скачивания
    crypto_symbols = [
        'BTCUSDT',  # Биткоин
        'TONUSDT',  # TON
        'ETHUSDT',  # Эфириум
        'SOLUSDT',  # Solana
        'ADAUSDT',  # Cardano
        'BNBUSDT'   # Binance Coin
    ]
    
    print(f"🚀 Начинаю скачивание свечей для {len(crypto_symbols)} криптовалют...")
    print(f"📊 Таймфрейм: {interval}, Целевое количество: {desired_candles}")
    
    for i, symbol in enumerate(crypto_symbols, 1):
        try:
            print(f"\n📥 [{i}/{len(crypto_symbols)}] Скачиваю {symbol}...")
            
            # 1. Скачиваем свечи для текущей криптовалюты
            csv_file_path = parser_download_and_combine_with_library(
                symbol=symbol,
                interval=interval,
                months_to_fetch=12,
                desired_candles=desired_candles
            )
            
            if csv_file_path:
                results.append({
                    "status": "success", 
                    "symbol": symbol,
                    "message": f"CSV файл создан: {csv_file_path}"
                })
                
                # 2. Загружаем свечи в базу данных
                try:
                    loaded_count = load_latest_candles_from_csv_to_db(
                        file_path=csv_file_path,
                        symbol_name=symbol,
                        timeframe=interval
                    )
                    
                    if loaded_count > 0:
                        results.append({
                            "status": "success",
                            "symbol": symbol,
                            "message": f"Загружено {loaded_count} свечей в БД"
                        })
                        print(f"  ✅ {symbol}: {loaded_count} свечей загружено в БД")
                    else:
                        results.append({
                            "status": "warning",
                            "symbol": symbol,
                            "message": "Не удалось загрузить свечи в БД"
                        })
                        print(f"  ⚠️ {symbol}: ошибка загрузки в БД")
                        
                except Exception as db_error:
                    results.append({
                        "status": "error",
                        "symbol": symbol,
                        "message": f"Ошибка загрузки в БД: {str(db_error)}"
                    })
                    print(f"  ❌ {symbol}: ошибка БД - {db_error}")
                    
            else:
                results.append({
                    "status": "error",
                    "symbol": symbol,
                    "message": "Не удалось создать CSV файл"
                })
                print(f"  ❌ {symbol}: не удалось создать CSV")
                
        except Exception as e:
            error_msg = f"Ошибка при скачивании {symbol}: {str(e)}"
            results.append({
                "status": "error",
                "symbol": symbol,
                "message": error_msg
            })
            print(f"  ❌ {symbol}: {error_msg}")
            continue
    
    # Сводка по всем криптовалютам
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'error')
    warnings = sum(1 for r in results if r['status'] == 'warning')
    
    print(f"\n{'='*60}")
    print(f"📊 СВОДКА СКАЧИВАНИЯ")
    print(f"{'='*60}")
    print(f"✅ Успешно: {successful}")
    print(f"⚠️ Предупреждения: {warnings}")
    print(f"❌ Ошибки: {failed}")
    print(f"📈 Всего криптовалют: {len(crypto_symbols)}")
    
    response = {
        'status': 'Мультивалютное скачивание завершено',
        'summary': {
            'total_cryptos': len(crypto_symbols),
            'successful': successful,
            'warnings': warnings,
            'failed': failed
        },
        'results': results
    }
    
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

@app.route('/models')
def models_page():
    """Страница управления моделями"""
    return render_template('models.html')

@app.route('/trading_agent')
def trading_agent_page():
    """Страница торгового агента"""
    return render_template('trading_agent.html')

@app.route('/create_model_version', methods=['POST'])
def create_model_version():
    """Создает новую версию модели с уникальным ID"""
    import shutil
    import uuid
    from datetime import datetime
    from pathlib import Path
    
    try:
        # Читаем символ из запроса (опционально)
        try:
            data = request.get_json(silent=True) or {}
        except Exception:
            data = {}
        requested_symbol = (data.get('symbol') or '').strip()
        requested_file = (data.get('file') or '').strip()
        
        # Генерируем уникальный ID (4 символа)
        model_id = str(uuid.uuid4())[:4].upper()
        
        # Новая логика источников: используем папку result/ и символ (если задан)
        result_dir = Path('result')
        if not result_dir.exists():
            return jsonify({
                "success": False,
                "error": "Папка result не найдена. Сначала запустите обучение."
            })
        
        # Определяем base_code (символьный префикс)
        def normalize_symbol(sym: str) -> str:
            if not sym:
                return ''
            s = sym.upper().replace('/', '')
            for suffix in ["USDT", "USD", "USDC", "BUSD", "USDP"]:
                if s.endswith(suffix):
                    s = s[:-len(suffix)]
                    break
            s = s.lower()
            if s in ("мультивалюта", "multi", "multicrypto"):
                s = "multi"
            return s
        
        base_code = normalize_symbol(requested_symbol)
        
        selected_result_file = None
        # 1) Если фронт прислал явный файл — используем его
        if requested_file:
            from pathlib import Path as _Path
            # Нормализуем слеши (Windows/Unix)
            req_norm = requested_file.replace('\\', '/')
            safe_path = _Path(req_norm)
            result_dir_abs = result_dir.resolve()

            # Определяем кандидат с учётом разных вариантов формата пути
            if safe_path.is_absolute():
                candidate = safe_path
            else:
                # Если путь уже начинается с 'result/'
                if safe_path.parts and safe_path.parts[0].lower() == result_dir.name.lower():
                    candidate = _Path(req_norm)
                else:
                    # Используем имя файла внутри result/
                    candidate = result_dir / safe_path.name

            try:
                cand_resolved = candidate.resolve()
            except Exception:
                cand_resolved = candidate

            # Безопасность: файл должен быть внутри result/
            try:
                inside_result = str(cand_resolved).lower().startswith(str(result_dir_abs).lower())
            except Exception:
                inside_result = False

            if inside_result and cand_resolved.exists() and cand_resolved.is_file() and cand_resolved.suffix == '.pkl' and cand_resolved.name.startswith('train_result_'):
                selected_result_file = cand_resolved
                # Пытаемся извлечь base_code из имени файла
                try:
                    fname = cand_resolved.stem  # train_result_<code>
                    parts = fname.split('_', 2)
                    if len(parts) >= 3:
                        base_code = parts[2].lower()
                except Exception:
                    pass
            else:
                return jsonify({
                    "success": False,
                    "error": "Неверный путь к файлу результата или файл не существует"
                })
        elif base_code:
            # Ищем точный train_result_<base_code>.pkl
            candidate = result_dir / f"train_result_{base_code}.pkl"
            if candidate.exists():
                selected_result_file = candidate
            else:
                return jsonify({
                    "success": False,
                    "error": f"Файл результатов для символа {base_code} не найден в result/"
                })
        else:
            # Fallback: берём самый свежий train_result_*.pkl
            result_files = list(result_dir.glob('train_result_*.pkl'))
            if not result_files:
                return jsonify({
                    "success": False,
                    "error": "Файлы результатов обучения не найдены в result/"
                })
            selected_result_file = max(result_files, key=lambda x: x.stat().st_mtime)
            # Пытаемся извлечь символ из имени
            try:
                fname = selected_result_file.stem  # train_result_<code>
                parts = fname.split('_', 2)
                if len(parts) >= 3:
                    base_code = parts[2].lower()
            except Exception:
                pass

        if not base_code:
            base_code = "model"
        
        # Определяем пути источников в result/
        model_file = result_dir / f'dqn_model_{base_code}.pth'
        replay_file = result_dir / f'replay_buffer_{base_code}.pkl'
        
        if not model_file.exists():
            return jsonify({
                "success": False,
                "error": f"Файл {model_file.name} не найден в result/"
            })
        
        if not replay_file.exists():
            return jsonify({
                "success": False,
                "error": f"Файл {replay_file.name} не найден в result/"
            })
        
        # Сохраняем в структуру models/<symbol>/<ensemble>/vN
        named_id = f"{base_code}_{model_id}"
        try:
            from datetime import datetime as _dt
            from pathlib import Path as _Path
            import json as _json

            models_root = _Path('models')
            models_root.mkdir(exist_ok=True)

            # Символ папки как в примере (btc, bnb, ton)
            symbol_dir = models_root / base_code.lower()
            symbol_dir.mkdir(exist_ok=True)

            # Ensemble можно передать в payload, иначе по умолчанию 'ensemble-a'
            ensemble_name = (data.get('ensemble') or 'ensemble-a').strip() or 'ensemble-a'
            ensemble_dir = symbol_dir / ensemble_name
            ensemble_dir.mkdir(exist_ok=True)

            # Определяем следующий номер версии vN
            existing_versions = []
            for p in ensemble_dir.iterdir():
                try:
                    if p.is_dir() and p.name.startswith('v'):
                        n = int(p.name[1:])
                        existing_versions.append(n)
                except Exception:
                    pass
            next_num = (max(existing_versions) + 1) if existing_versions else 1
            version_name = f'v{next_num}'
            version_dir = ensemble_dir / version_name
            version_dir.mkdir(exist_ok=False)

            # Копируем артефакты версии
            ver_model = version_dir / f'dqn_model_{named_id}.pth'
            ver_replay = version_dir / f'replay_buffer_{named_id}.pkl'
            ver_result = version_dir / f'train_result_{named_id}.pkl'
            shutil.copy2(model_file, ver_model)
            shutil.copy2(replay_file, ver_replay)
            shutil.copy2(selected_result_file, ver_result)

            # Пишем manifest.yaml (без зависимости от PyYAML)
            manifest_path = version_dir / 'manifest.yaml'
            try:
                # Попытаемся извлечь краткую статистику
                stats_brief = {}
                try:
                    import pickle as _pickle
                    with open(new_result_file, 'rb') as _f:
                        _res = _pickle.load(_f)
                        if isinstance(_res, dict) and 'final_stats' in _res:
                            stats_brief = _res['final_stats'] or {}
                except Exception:
                    pass
                created_ts = _dt.utcnow().isoformat()
                yaml_text = (
                    'id: "' + named_id + '"\n'
                    'symbol: "' + base_code.lower() + '"\n'
                    'ensemble: "' + ensemble_name + '"\n'
                    'version: "' + version_name + '"\n'
                    'created_at: "' + created_ts + '"\n'
                    'files:\n'
                    '  model: "' + ver_model.name + '"\n'
                    '  replay: "' + ver_replay.name + '"\n'
                    '  results: "' + ver_result.name + '"\n'
                    'stats:\n' + ''.join([
                        f"  {k}: {v}\n" for k, v in (stats_brief.items() if isinstance(stats_brief, dict) else [])
                    ])
                )
                with open(manifest_path, 'w', encoding='utf-8') as mf:
                    mf.write(yaml_text)
            except Exception:
                pass

            # Обновляем симлинк current -> vN (если не удаётся — создаём файл-указатель)
            current_link = ensemble_dir / 'current'
            try:
                if current_link.exists() or current_link.is_symlink():
                    try:
                        if current_link.is_symlink() or current_link.is_file():
                            current_link.unlink()
                        elif current_link.is_dir():
                            shutil.rmtree(current_link)
                    except Exception:
                        pass
                os.symlink(version_name, current_link)
            except Exception:
                # Фоллбек: записываем имя версии в файл
                try:
                    with open(current_link, 'w', encoding='utf-8') as fcur:
                        fcur.write(version_name)
                except Exception:
                    pass
        except Exception:
            # Не валим основной сценарий, если models/ недоступна
            pass
        
        return jsonify({
            "success": True,
            "model_id": named_id,
            "files": [
                f'dqn_model_{named_id}.pth',
                f'replay_buffer_{named_id}.pkl',
                f'train_result_{named_id}.pkl'
            ]
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/get_models_list')
def get_models_list():
    """Возвращает список всех сохраненных моделей"""
    from pathlib import Path
    import pickle
    from datetime import datetime
    
    try:
        models: list = []

        # Читаем только новую структуру models/

        # 2) Новая структура: models/<symbol>/<ensemble>/vN
        models_root = Path('models')
        if models_root.exists():
            for symbol_dir in models_root.iterdir():
                if not symbol_dir.is_dir():
                    continue
                for ensemble_dir in symbol_dir.iterdir():
                    if not ensemble_dir.is_dir():
                        continue
                    for version_dir in ensemble_dir.iterdir():
                        if not version_dir.is_dir() or not version_dir.name.startswith('v'):
                            continue
                        # Пытаемся найти артефакты
                        model_files = list(version_dir.glob('dqn_model_*.pth'))
                        replay_files = list(version_dir.glob('replay_buffer_*.pkl'))
                        result_files = list(version_dir.glob('train_result_*.pkl'))
                        if not model_files or not replay_files or not result_files:
                            continue
                        model_file = model_files[0]
                        replay_file = replay_files[0]
                        result_file = result_files[0]
                        model_id = model_file.stem.replace('dqn_model_', '')
                        model_size = f"{model_file.stat().st_size / 1024 / 1024:.1f} MB"
                        replay_size = f"{replay_file.stat().st_size / 1024 / 1024:.1f} MB"
                        result_size = f"{result_file.stat().st_size / 1024:.1f} KB"
                        creation_time = datetime.fromtimestamp(model_file.stat().st_ctime)
                        date_str = creation_time.strftime('%d.%m.%Y %H:%M')
                        stats = {}
                        try:
                            with open(result_file, 'rb') as f:
                                results = pickle.load(f)
                                if isinstance(results, dict) and 'final_stats' in results:
                                    stats = results['final_stats']
                        except Exception:
                            pass
                        models.append({
                            "id": model_id,
                            "date": date_str,
                            "files": {
                                "model": model_file.name,
                                "model_size": model_size,
                                "replay": replay_file.name,
                                "replay_size": replay_size,
                                "results": result_file.name,
                                "results_size": result_size
                            },
                            "stats": stats
                        })

        # Сортируем по дате создания (новые сначала)
        models.sort(key=lambda x: x['date'], reverse=True)

        return jsonify({
            "success": True,
            "models": models
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/analyze_model', methods=['POST'])
def analyze_model():
    """Анализирует конкретную модель"""
    import pickle
    from pathlib import Path
    
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        
        if not model_id:
            return jsonify({
                "success": False,
                "error": "ID модели не указан"
            })
        
        # Ищем файл результатов в новой структуре models/
        result_file = None
        try:
            models_root = Path('models')
            for symbol_dir in models_root.iterdir():
                if not symbol_dir.is_dir():
                    continue
                for ensemble_dir in symbol_dir.iterdir():
                    if not ensemble_dir.is_dir():
                        continue
                    for version_dir in ensemble_dir.iterdir():
                        if not version_dir.is_dir():
                            continue
                        cand = version_dir / f'train_result_{model_id}.pkl'
                        if cand.exists():
                            result_file = cand
                            raise StopIteration
        except StopIteration:
            pass
        except Exception:
            pass
        if result_file is None:
            return jsonify({
                "success": False,
                "error": f"Файл результатов для модели {model_id} не найден"
            })
        if not result_file.exists():
            return jsonify({
                "success": False,
                "error": f"Файл результатов для модели {model_id} не найден"
            })
        
        # Загружаем результаты
        with open(result_file, 'rb') as f:
            results = pickle.load(f)
        
        # Формируем анализ
        analysis = f"""📊 АНАЛИЗ МОДЕЛИ {model_id}
{'='*50}

📅 Дата обучения: {results.get('training_date', 'Неизвестно')}
⏱️ Время обучения: {results.get('total_training_time', 0) / 3600:.1f} часов
🎯 Эпизодов: {results.get('actual_episodes', results.get('episodes', 'Неизвестно'))}

📈 СТАТИСТИКА:
"""
        
        if 'final_stats' in results:
            stats = results['final_stats']
            analysis += f"""
• Winrate: {stats.get('winrate', 0) * 100:.1f}%
• P/L Ratio: {stats.get('pl_ratio', 0):.2f}
• Сделок: {stats.get('trades_count', 0)}
• Средняя прибыль: {stats.get('avg_profit', 0) * 100:.2f}%
• Средний убыток: {stats.get('avg_loss', 0) * 100:.2f}%
• Плохих сделок: {stats.get('bad_trades_count', 0)}
"""
        
        if 'all_trades' in results:
            trades = results['all_trades']
            analysis += f"""
📊 ДЕТАЛЬНАЯ СТАТИСТИКА СДЕЛОК:
• Всего сделок: {len(trades)}
• Прибыльных: {sum(1 for t in trades if t.get('roi', 0) > 0)}
• Убыточных: {sum(1 for t in trades if t.get('roi', 0) < 0)}
• Нейтральных: {sum(1 for t in trades if abs(t.get('roi', 0)) < 0.001)}
"""
        
        return jsonify({
            "success": True,
            "analysis": analysis
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/delete_model', methods=['POST'])
def delete_model():
    """Удаляет модель и все связанные файлы"""
    from pathlib import Path
    import os
    
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        
        if not model_id:
            return jsonify({
                "success": False,
                "error": "ID модели не указан"
            })
        
        # Удаляем модель из новой структуры models/
        models_root = Path('models')
        deleted_files = []
        if models_root.exists():
            for symbol_dir in models_root.iterdir():
                if not symbol_dir.is_dir():
                    continue
                for ensemble_dir in symbol_dir.iterdir():
                    if not ensemble_dir.is_dir():
                        continue
                    for version_dir in list(ensemble_dir.iterdir()):
                        if not version_dir.is_dir():
                            continue
                        # Проверяем совпадение по файлам модели
                        for name in [f'dqn_model_{model_id}.pth', f'replay_buffer_{model_id}.pkl', f'train_result_{model_id}.pkl']:
                            fp = version_dir / name
                            if fp.exists():
                                try:
                                    os.remove(fp)
                                    deleted_files.append(str(fp))
                                except Exception:
                                    pass
                        # Удаляем пустую версию
                        try:
                            if version_dir.is_dir() and not any(version_dir.iterdir()):
                                import shutil as _sh
                                _sh.rmtree(version_dir, ignore_errors=True)
                        except Exception:
                            pass

        if not deleted_files:
            return jsonify({
                "success": False,
                "error": f"Файлы модели {model_id} не найдены"
            })

        return jsonify({
            "success": True,
            "message": f"Модель {model_id} удалена",
            "deleted_files": deleted_files
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ==================== ТОРГОВЫЕ ENDPOINT'Ы ====================

@app.route('/api/trading/start', methods=['POST'])
def start_trading():
    """
    Запуск торговли в контейнере trading_agent через Celery задачу
    """
    try:
        data = request.get_json() or {}
        symbols = data.get('symbols', ['BTCUSDT'])
        model_path = data.get('model_path', '/workspace/good_model/dqn_model.pth')
        
        # Сохраняем выбранные параметры в Redis для последующих вызовов (status/stop/balance/history)
        try:
            from redis import Redis as _Redis
            import json as _json
            _rc = _Redis(host='redis', port=6379, db=0, decode_responses=True)
            _rc.set('trading:model_path', model_path)
            _rc.set('trading:symbols', _json.dumps(symbols, ensure_ascii=False))
            # Пишем мгновенный «активный» статус, чтобы UI сразу показывал Активна до первого RESULT
            initial_status = {
                'success': True,
                'is_trading': True,
                'trading_status': 'Активна',
                'trading_status_emoji': '🟢',
                'trading_status_full': '🟢 Активна',
                'symbol': symbols[0] if symbols else None,
                'symbol_display': symbols[0] if symbols else 'Не указана',
                'amount': None,
                'amount_display': 'Не указано',
                'amount_usdt': 0.0,
                'position': None,
                'trades_count': 0,
                'balance': {},
                'current_price': 0.0,
                'last_model_prediction': None,
            }
            _rc.set('trading:current_status', _json.dumps(initial_status, ensure_ascii=False))
            from datetime import datetime as _dt
            _rc.set('trading:current_status_ts', _dt.utcnow().isoformat())
        except Exception as _e:
            app.logger.error(f"Не удалось сохранить параметры торговли в Redis: {_e}")

        # Запускаем Celery задачу для старта торговли в очереди 'celery'
        task = start_trading_task.apply_async(args=[symbols, model_path], countdown=0, expires=300, queue='celery')
        
        return jsonify({
            'success': True,
            'message': 'Торговля запущена через Celery задачу',
            'task_id': task.id
        }), 200
    except Exception as e:
        app.logger.error(f"Ошибка запуска торговли: {e}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

@app.route('/api/trading/stop', methods=['POST'])
def stop_trading():
    """Остановка торговли в контейнере trading_agent"""
    try:
        # Подключаемся к Docker
        client = docker.from_env()
        
        try:
            # Получаем контейнер medoedai
            container = client.containers.get('medoedai')
            
            # Проверяем что контейнер запущен
            if container.status != 'running':
                return jsonify({
                    'success': False, 
                    'error': f'Контейнер medoedai не запущен. Статус: {container.status}'
                }), 500
            
            # Получаем ранее выбранный путь к модели (если есть)
            model_path = None
            try:
                mp = redis_client.get('trading:model_path')
                if mp:
                    model_path = mp.decode('utf-8')
            except Exception:
                pass

            # Останавливаем торговлю через exec
            if model_path:
                cmd = f'python -c "import json; from trading_agent.trading_agent import TradingAgent; agent = TradingAgent(model_path=\\"{model_path}\\"); result = agent.stop_trading(); print(\\"RESULT: \\" + json.dumps(result))"'
            else:
                cmd = 'python -c "import json; from trading_agent.trading_agent import TradingAgent; agent = TradingAgent(); result = agent.stop_trading(); print(\\"RESULT: \\" + json.dumps(result))"'
            
            exec_result = container.exec_run(cmd, tty=True)
            
            # Логируем результат выполнения команды
            app.logger.info(f"Stop trading - Exit code: {exec_result.exit_code}")
            if exec_result.output:
                output_str = exec_result.output.decode('utf-8')
                app.logger.info(f"Stop trading - Output: {output_str}")
            
            if exec_result.exit_code == 0:
                output = exec_result.output.decode('utf-8') if exec_result.output else ""
                # Ищем результат в выводе
                if 'RESULT:' in output:
                    result_str = output.split('RESULT:')[1].strip()
                    try:
                        import json
                        result = json.loads(result_str)
                        return jsonify(result), 200
                    except Exception as parse_error:
                        app.logger.error(f"Ошибка парсинга результата: {parse_error}")
                        return jsonify({
                            'success': True,
                            'message': 'Торговля остановлена',
                            'output': output
                        }), 200
                else:
                    return jsonify({
                        'success': True,
                        'message': 'Торговля остановлена',
                        'output': output
                    }), 200
            else:
                error_output = exec_result.output.decode('utf-8') if exec_result.output else "No error output"
                app.logger.error(f"Ошибка выполнения команды остановки торговли: {error_output}")
                return jsonify({
                    'success': False,
                    'error': f'Ошибка выполнения команды: {error_output}'
                }), 500
                
        except docker.errors.NotFound:
            return jsonify({
                'success': False, 
                'error': 'Контейнер medoedai не найден. Запустите docker-compose up medoedai'
            }), 500
        except Exception as e:
            return jsonify({
                'success': False, 
                'error': f'Ошибка Docker: {str(e)}'
            }), 500
            
    except Exception as e:
        app.logger.error(f"Ошибка остановки торговли: {e}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

@app.route('/api/trading/status', methods=['GET'])
def trading_status():
    """Статус торговли из контейнера trading_agent"""
    try:
        # Сначала пробуем быстрый статус из Redis (обновляется периодическим таском)
        try:
            from redis import Redis as _Redis
            _rc = _Redis(host='redis', port=6379, db=0, decode_responses=True)
            cached = _rc.get('trading:current_status')
            cached_ts = _rc.get('trading:current_status_ts')
            if cached:
                import json as _json
                status_obj = _json.loads(cached)
                # Проверим свежесть (не старее 6 минут, > интервала beat)
                from datetime import datetime, timedelta
                is_fresh = True
                try:
                    if cached_ts:
                        from datetime import datetime as _dt
                        ts = _dt.fromisoformat(cached_ts)
                        is_fresh = _dt.utcnow() <= (ts + timedelta(minutes=6))
                except Exception:
                    is_fresh = True
                if is_fresh:
                    # Возвращаем плоскую структуру для совместимости с фронтендом
                    flat = {'success': True, 'agent_status': status_obj}
                    if isinstance(status_obj, dict):
                        flat.update(status_obj)
                    return jsonify(flat), 200
        except Exception:
            pass

        # Fallback: получаем статус через Docker exec
        # Подключаемся к Docker
        client = docker.from_env()
        
        try:
            # Получаем контейнер medoedai
            container = client.containers.get('medoedai')
            
            # Проверяем что контейнер запущен
            if container.status != 'running':
                return jsonify({
                    'success': False, 
                    'error': f'Контейнер medoedai не запущен. Статус: {container.status}'
                }), 500
            
            # Получаем ранее выбранный путь к модели (если есть)
            model_path = None
            try:
                mp = redis_client.get('trading:model_path')
                if mp:
                    model_path = mp.decode('utf-8')
            except Exception:
                pass

            # Получаем статус через exec (прокидываем API ключи в окружение процесса)
            api_key = os.environ.get('BYBIT_API_KEY', '') or ''
            secret_key = os.environ.get('BYBIT_SECRET_KEY', '') or ''
            api_key_esc = api_key.replace("'", "\\'")
            secret_key_esc = secret_key.replace("'", "\\'")
            if model_path:
                cmd = (
                    f"python -c \"import sys, json, os; print(sys.version); "
                    f"os.environ['BYBIT_API_KEY']='{api_key_esc}'; os.environ['BYBIT_SECRET_KEY']='{secret_key_esc}'; "
                    f"from trading_agent.trading_agent import TradingAgent; agent = TradingAgent(model_path=\\\"{model_path}\\\"); "
                    f"result = agent.get_trading_status(); print(\\\"RESULT: \\\" + json.dumps(result))\""
                )
            else:
                cmd = (
                    "python -c \"import sys, json, os; print(sys.version); "
                    f"os.environ['BYBIT_API_KEY']='{api_key_esc}'; os.environ['BYBIT_SECRET_KEY']='{secret_key_esc}'; "
                    "from trading_agent.trading_agent import TradingAgent; agent = TradingAgent(); result = agent.get_trading_status(); print(\\\"RESULT: \\\" + json.dumps(result))\""
                )
            
            exec_result = container.exec_run(cmd, tty=True)
            
            # Логируем результат выполнения команды
            app.logger.info(f"Get status - Exit code: {exec_result.exit_code}")
            if exec_result.output:
                output_str = exec_result.output.decode('utf-8')
                app.logger.info(f"Get status - Output: {output_str}")
            else:
                app.logger.info("Get status - No output received")
            
            if exec_result.exit_code == 0:
                output = exec_result.output.decode('utf-8') if exec_result.output else ""
                # Ищем результат в выводе
                if 'RESULT:' in output:
                    result_str = output.split('RESULT:')[1].strip()
                    try:
                        import json
                        result = json.loads(result_str)
                        # Возвращаем плоскую структуру для фронтенда (как и при кэше)
                        flat = {'success': True, 'agent_status': result}
                        if isinstance(result, dict):
                            flat.update(result)
                        return jsonify(flat), 200
                    except Exception as parse_error:
                        app.logger.error(f"Ошибка парсинга статуса: {parse_error}")
                        return jsonify({
                            'success': False,
                            'error': f'Ошибка парсинга JSON: {parse_error}',
                            'raw_output': output
                        }), 500
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Не найден RESULT в выводе команды',
                        'raw_output': output
                    }), 500
            else:
                error_output = exec_result.output.decode('utf-8') if exec_result.output else "No error output"
                app.logger.error(f"Ошибка получения статуса: {error_output}")
                return jsonify({
                    'success': False,
                    'error': f'Ошибка выполнения команды: {error_output}'
                }), 500
                
        except docker.errors.NotFound:
            return jsonify({
                'success': False, 
                'error': 'Контейнер medoedai не найден. Запустите docker-compose up medoedai'
            }), 500
        except Exception as e:
            return jsonify({
                'success': False, 
                'error': f'Ошибка Docker: {str(e)}'
            }), 500
            
    except Exception as e:
        app.logger.error(f"Ошибка получения статуса: {e}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

@app.route('/api/trading/latest_results', methods=['GET'])
def trading_latest_results():
    """Получение последних результатов торговли из Celery"""
    try:
        # Получаем последние результаты из Redis (ключи с таймстампом)
        latest_results = []
        try:
            keys = redis_client.keys('trading:latest_result_*') or []
            # Загружаем все и сортируем по времени
            for k in keys:
                try:
                    raw = redis_client.get(k)
                    if not raw:
                        continue
                    try:
                        result = json.loads(raw.decode('utf-8'))
                    except Exception:
                        result = json.loads(raw)
                    latest_results.append(result)
                except Exception as e:
                    app.logger.warning(f"Ошибка чтения {k}: {e}")
        except Exception as e:
            app.logger.warning(f"Ошибка получения ключей trading:latest_result_*: {e}")

        # Сортируем по времени (новые сначала)
        try:
            latest_results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        except Exception:
            pass
        
        return jsonify({
            'success': True,
            'latest_results': latest_results,
            'total_results': len(latest_results)
        }), 200
        
    except Exception as e:
        app.logger.error(f"Ошибка получения последних результатов: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/trading/balance', methods=['GET'])
def trading_balance():
    """Баланс на бирже из контейнера trading_agent"""
    try:
        # Подключаемся к Docker
        client = docker.from_env()
        
        try:
            # Получаем контейнер medoedai
            container = client.containers.get('medoedai')
            
            # Проверяем что контейнер запущен
            if container.status != 'running':
                return jsonify({
                    'success': False, 
                    'error': f'Контейнер medoedai не запущен. Статус: {container.status}'
                }), 500
            
            # Получаем ранее выбранный путь к модели (если есть)
            model_path = None
            try:
                mp = redis_client.get('trading:model_path')
                if mp:
                    model_path = mp.decode('utf-8')
            except Exception:
                pass

            # Получаем баланс через exec (прокидываем API ключи)
            api_key = os.environ.get('BYBIT_API_KEY', '') or ''
            secret_key = os.environ.get('BYBIT_SECRET_KEY', '') or ''
            api_key_esc = api_key.replace("'", "\\'")
            secret_key_esc = secret_key.replace("'", "\\'")
            if model_path:
                cmd = (
                    f"python -c \"import json, os; os.environ['BYBIT_API_KEY']='{api_key_esc}'; os.environ['BYBIT_SECRET_KEY']='{secret_key_esc}'; "
                    f"from trading_agent.trading_agent import TradingAgent; agent = TradingAgent(model_path=\\\"{model_path}\\\"); result = agent.get_balance(); print(\\\"RESULT: \\\" + json.dumps(result))\""
                )
            else:
                cmd = (
                    "python -c \"import json, os; os.environ['BYBIT_API_KEY']='{api_key_esc}'; os.environ['BYBIT_SECRET_KEY']='{secret_key_esc}'; "
                    "from trading_agent.trading_agent import TradingAgent; agent = TradingAgent(); result = agent.get_balance(); print(\\\"RESULT: \\\" + json.dumps(result))\""
                )
            
            exec_result = container.exec_run(cmd, tty=True)
            
            if exec_result.exit_code == 0:
                output = exec_result.output.decode('utf-8')
                # Ищем результат в выводе
                if 'RESULT:' in output:
                    result_str = output.split('RESULT:')[1].strip()
                    try:
                        import json
                        result = json.loads(result_str)
                        return jsonify(result), 200
                    except:
                        return jsonify({
                            'success': True,
                            'message': 'Баланс получен',
                            'output': output
                        }), 200
                else:
                    return jsonify({
                        'success': True,
                        'message': 'Баланс получен',
                        'output': output
                    }), 200
            else:
                return jsonify({
                    'success': False,
                    'error': f'Ошибка выполнения команды: {exec_result.output.decode("utf-8")}'
                }), 500
        except docker.errors.NotFound:
            return jsonify({
                'success': False, 
                'error': 'Контейнер medoedai не найден'
            }), 500
        except Exception as e:
            return jsonify({
                'success': False, 
                'error': f'Ошибка Docker: {str(e)}'
            }), 500
            
    except Exception as e:
        app.logger.error(f"Ошибка получения баланса: {e}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500


@app.route('/api/trading/test_order', methods=['POST'])
def trading_test_order():
    """Мгновенное размещение РЕАЛЬНОГО рыночного ордера BUY/SELL в обход предсказаний.
    Ничего не пишет в БД/мониторинг. Выполняется внутри контейнера medoedai через TradingAgent.execute_direct_order.
    Body: { action: 'buy'|'sell', symbol?: 'BTCUSDT', quantity?: float }
    """
    try:
        data = request.get_json() or {}
        action = (data.get('action') or '').lower()
        symbol = data.get('symbol')
        quantity = data.get('quantity')

        if action not in ('buy', 'sell'):
            return jsonify({'success': False, 'error': "action должен быть 'buy' или 'sell'"}), 400

        # Подключаемся к Docker
        client = docker.from_env()
        try:
            container = client.containers.get('medoedai')
            if container.status != 'running':
                return jsonify({'success': False, 'error': f'Контейнер medoedai не запущен. Статус: {container.status}'}), 500

            # Получаем ранее выбранный путь к модели (если есть), чтобы TradingAgent корректно инициализировал биржу
            model_path = None
            try:
                mp = redis_client.get('trading:model_path')
                if mp:
                    model_path = mp.decode('utf-8')
            except Exception:
                pass

            # Формируем python-команду без вложенного JSON, чтобы избежать проблем с кавычками
            def _py_str_literal(s):
                if s is None:
                    return 'None'
                return "'" + str(s).replace("'", "\\'") + "'"

            py_action = _py_str_literal(action)
            py_symbol = _py_str_literal(symbol)
            py_quantity = 'None' if quantity is None else str(float(quantity))

            api_key = os.environ.get('BYBIT_API_KEY', '') or ''
            secret_key = os.environ.get('BYBIT_SECRET_KEY', '') or ''
            api_key_esc = api_key.replace("'", "\\'")
            secret_key_esc = secret_key.replace("'", "\\'")
            if model_path:
                cmd = (
                    f"python -c \"import json, os; os.environ['BYBIT_API_KEY']='{api_key_esc}'; os.environ['BYBIT_SECRET_KEY']='{secret_key_esc}'; "
                    f"from trading_agent.trading_agent import TradingAgent; agent = TradingAgent(model_path=\\\"{model_path}\\\"); "
                    f"action = {py_action}; symbol = {py_symbol}; quantity = {py_quantity}; "
                    f"result = agent.execute_direct_order(action, symbol, quantity); print(\\\"RESULT: \\\" + json.dumps(result))\""
                )
            else:
                cmd = (
                    "python -c \"import json, os; os.environ['BYBIT_API_KEY']='{api_key_esc}'; os.environ['BYBIT_SECRET_KEY']='{secret_key_esc}'; "
                    "from trading_agent.trading_agent import TradingAgent; agent = TradingAgent(); "
                    f"action = {py_action}; symbol = {py_symbol}; quantity = {py_quantity}; "
                    "result = agent.execute_direct_order(action, symbol, quantity); print(\\\"RESULT: \\\" + json.dumps(result))\""
                )

            exec_result = container.exec_run(cmd, tty=True)
            output = exec_result.output.decode('utf-8') if exec_result.output else ''
            app.logger.info(f"test_order exit={exec_result.exit_code} output={output}")

            if exec_result.exit_code == 0 and 'RESULT:' in output:
                result_str = output.split('RESULT:')[1].strip()
                try:
                    import json
                    result = json.loads(result_str)
                except Exception as parse_err:
                    return jsonify({'success': False, 'error': f'Ошибка парсинга результата: {parse_err}', 'raw_output': output}), 500
                return jsonify(result), 200
            else:
                return jsonify({'success': False, 'error': 'Команда завершилась с ошибкой', 'raw_output': output}), 500

        except docker.errors.NotFound:
            return jsonify({'success': False, 'error': 'Контейнер medoedai не найден'}), 500
        except Exception as e:
            return jsonify({'success': False, 'error': f'Ошибка Docker: {str(e)}'}), 500

    except Exception as e:
        app.logger.error(f"Ошибка test_order: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/trades/recent', methods=['GET'])
def get_recent_trades_api():
    """Получение последних сделок из базы данных"""
    try:

        
        limit = request.args.get('limit', 50, type=int)
        trades = get_recent_trades(limit=limit)
        
        # Преобразуем в JSON-совместимый формат
        trades_data = []
        for trade in trades:
            trades_data.append({
                'trade_number': trade.trade_number,
                'symbol': trade.symbol.name if trade.symbol else 'Unknown',
                'action': trade.action,
                'status': trade.status,
                'quantity': trade.quantity,
                'price': trade.price,
                'total_value': trade.total_value,
                'model_prediction': trade.model_prediction,
                'current_balance': trade.current_balance,
                'position_pnl': trade.position_pnl,
                'created_at': trade.created_at.isoformat() if trade.created_at else None,
                'executed_at': trade.executed_at.isoformat() if trade.executed_at else None,
                'is_successful': trade.is_successful,
                'error_message': trade.error_message
            })
        
        return jsonify({
            'success': True,
            'trades': trades_data,
            'total_trades': len(trades_data)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/trades/statistics', methods=['GET'])
def get_trade_statistics_api():
    """Получение статистики по сделкам"""
    try:

        
        symbol = request.args.get('symbol', None)
        stats = get_trade_statistics(symbol_name=symbol)
        
        return jsonify({
            'success': True,
            'statistics': stats
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/trades/by_symbol/<symbol_name>', methods=['GET'])
def get_trades_by_symbol_api(symbol_name):
    """Получение сделок по символу"""
    try:

        
        limit = request.args.get('limit', 100, type=int)
        trades = get_trades_by_symbol(symbol_name, limit=limit)
        
        # Преобразуем в JSON-совместимый формат
        trades_data = []
        for trade in trades:
            trades_data.append({
                'trade_number': trade.trade_number,
                'symbol': trade.symbol.name if trade.symbol else 'Unknown',
                'action': trade.action,
                'status': trade.status,
                'quantity': trade.quantity,
                'price': trade.price,
                'total_value': trade.total_value,
                'model_prediction': trade.model_prediction,
                'current_balance': trade.current_balance,
                'position_pnl': trade.position_pnl,
                'created_at': trade.created_at.isoformat() if trade.created_at else None,
                'executed_at': trade.executed_at.isoformat() if trade.executed_at else None,
                'is_successful': trade.is_successful,
                'error_message': trade.error_message
            })
        
        return jsonify({
            'success': True,
            'trades': trades_data,
            'total_trades': len(trades_data),
            'symbol': symbol_name
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/trading/history', methods=['GET'])
def trading_history():
    """История торговли из контейнера trading_agent"""
    try:
        # Подключаемся к Docker
        client = docker.from_env()
        
        try:
            # Получаем контейнер medoedai
            container = client.containers.get('medoedai')
            
            # Проверяем что контейнер запущен
            if container.status != 'running':
                return jsonify({
                    'success': False, 
                    'error': f'Контейнер medoedai не запущен. Статус: {container.status}'
                }), 500
            
            # Получаем ранее выбранный путь к модели (если есть)
            model_path = None
            try:
                mp = redis_client.get('trading:model_path')
                if mp:
                    model_path = mp.decode('utf-8')
            except Exception:
                pass

            # Получаем историю через exec (прокидываем API ключи)
            api_key = os.environ.get('BYBIT_API_KEY', '') or ''
            secret_key = os.environ.get('BYBIT_SECRET_KEY', '') or ''
            api_key_esc = api_key.replace("'", "\\'")
            secret_key_esc = secret_key.replace("'", "\\'")
            if model_path:
                cmd = (
                    f"python -c \"import json, os; os.environ['BYBIT_API_KEY']='{api_key_esc}'; os.environ['BYBIT_SECRET_KEY']='{secret_key_esc}'; "
                    f"from trading_agent.trading_agent import TradingAgent; agent = TradingAgent(model_path=\\\"{model_path}\\\"); result = agent.get_trading_history(); print(\\\"RESULT: \\\" + json.dumps(result))\""
                )
            else:
                cmd = (
                    "python -c \"import json, os; os.environ['BYBIT_API_KEY']='{api_key_esc}'; os.environ['BYBIT_SECRET_KEY']='{secret_key_esc}'; "
                    "from trading_agent.trading_agent import TradingAgent; agent = TradingAgent(); result = agent.get_trading_history(); print(\\\"RESULT: \\\" + json.dumps(result))\""
                )
            
            exec_result = container.exec_run(cmd, tty=True)
            
            if exec_result.exit_code == 0:
                output = exec_result.output.decode('utf-8')
                # Ищем результат в выводе
                if 'RESULT:' in output:
                    result_str = output.split('RESULT:')[1].strip()
                    try:
                        import json
                        result = json.loads(result_str)
                        return jsonify(result), 200
                    except:
                        return jsonify({
                            'success': True,
                            'message': 'История получена',
                            'output': output
                        }), 200
                else:
                    return jsonify({
                        'success': True,
                        'message': 'История получена',
                        'output': output
                    }), 200
            else:
                return jsonify({
                    'success': False,
                    'error': f'Ошибка выполнения команды: {exec_result.output.decode("utf-8")}'
                }), 500
                
        except docker.errors.NotFound:
            return jsonify({
                'success': False, 
                'error': 'Контейнер medoedai не найден. Запустите docker-compose up medoedai'
            }), 500
        except Exception as e:
            return jsonify({
                'success': False, 
                'error': f'Ошибка Docker: {str(e)}'
            }), 500
            
    except Exception as e:
        app.logger.error(f"Ошибка получения истории: {e}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500


@app.route('/api/predictions/recent')
def get_recent_predictions():
    """API для получения последних предсказаний модели"""
    try:
        symbol = request.args.get('symbol')
        action = request.args.get('action')
        limit = int(request.args.get('limit', 50))
        
        predictions = get_model_predictions(symbol=symbol, action=action, limit=limit)
        
        # Преобразуем в JSON-совместимый формат
        predictions_data = []
        for prediction in predictions:
            try:
                q_values = json.loads(prediction.q_values) if prediction.q_values else []
                market_conditions = json.loads(prediction.market_conditions) if prediction.market_conditions else {}
            except:
                q_values = []
                market_conditions = {}
            
            prediction_data = {
                'id': prediction.id,
                'timestamp': prediction.timestamp.isoformat() if prediction.timestamp else None,
                'symbol': prediction.symbol,
                'action': prediction.action,
                'q_values': q_values,
                'current_price': prediction.current_price,
                'position_status': prediction.position_status,
                'confidence': prediction.confidence,
                'model_path': prediction.model_path,
                'market_conditions': market_conditions,
                'created_at': prediction.created_at.isoformat() if prediction.created_at else None
            }
            predictions_data.append(prediction_data)
        
        return jsonify({
            'success': True,
            'predictions': predictions_data,
            'total_predictions': len(predictions_data)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predictions/statistics')
def get_prediction_statistics_api():
    """API для получения статистики предсказаний модели"""
    try:
        symbol = request.args.get('symbol')
        stats = get_prediction_statistics(symbol=symbol)
        
        return jsonify({
            'success': True,
            'statistics': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# =============================================================

# Автоматический запуск Flask сервера
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))  # Получаем порт из переменной окружения
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    print(f"🚀 Запускаю Flask сервер на порту {port}...")
    print(f"🌐 Откройте: http://localhost:{port}")
    print(f"🔧 Debug режим: {'ВКЛЮЧЕН' if debug_mode else 'ОТКЛЮЧЕН'}")
    
    # Убираем инициализацию торгового агента
    # init_trading_agent()
    
    app.run(host="0.0.0.0", port=port, debug=debug_mode)

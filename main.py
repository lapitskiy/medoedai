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
from routes.bybit import bybit_bp
from routes.trading import trading_bp
from routes.models_admin import models_admin_bp
from utils.redis_utils import get_redis_client, clear_redis_on_startup

"""get_redis_client берём из utils.redis_utils"""

 
from celery.result import AsyncResult

import os
import re
from tasks.celery_tasks import search_lstm_task, train_dqn, train_dqn_multi_crypto, trade_step, start_trading_task, train_dqn_symbol
from utils.db_utils import load_latest_candles_from_csv_to_db
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
from utils.celery_utils import ensure_symbol_worker

logging.basicConfig(level=logging.INFO)

# Создаем Flask приложение
app = Flask(__name__)
app.register_blueprint(bybit_bp)
app.register_blueprint(trading_bp)
app.register_blueprint(models_admin_bp)
from routes.clean import clean_bp
app.register_blueprint(clean_bp)

# Функция очистки Redis вынесена в utils.redis_utils.clear_redis_on_startup

# Инициализируем Redis клиент и очищаем при запуске
redis_client = clear_redis_on_startup()
if redis_client is None:
    # Fallback - создаем клиент без очистки
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
    except:
        redis_client = redis.Redis(host='redis', port=6379, db=0)

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
    # Если запрос из fetch/XHR — отвечаем JSON, иначе редиректим на главную
    try:
        wants_json = request.is_json or 'application/json' in (request.headers.get('Accept') or '')
    except Exception:
        wants_json = False
    if wants_json:
        return jsonify({"success": True, "task_id": task.id})
    return redirect(url_for("index"))

@app.route('/train_dqn_multi_crypto', methods=['POST'])
def train_multi_crypto():
    """Запускает мультивалютное обучение DQN"""
    task = train_dqn_multi_crypto.apply_async(queue="train")
    try:
        wants_json = request.is_json or 'application/json' in (request.headers.get('Accept') or '')
    except Exception:
        wants_json = False
    if wants_json:
        return jsonify({"success": True, "task_id": task.id})
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
    # Seed: из формы/JSON; если пусто — сгенерируем случайный на бэкенде
    seed_raw = (data.get('seed') or request.form.get('seed') or '').strip()
    seed = None
    try:
        if seed_raw != '':
            seed = int(seed_raw)
    except Exception:
        seed = None
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
                # Если это fetch — вернём JSON без редиректа
                wants_json = request.is_json or 'application/json' in (request.headers.get('Accept') or '')
                if wants_json:
                    return jsonify({
                        "success": False,
                        "error": f"training for {symbol} already running",
                        "task_id": existing_task_id,
                        "state": ar.state
                    })
                return redirect(url_for("index"))
    except Exception as _e:
        app.logger.error(f"Не удалось проверить активную задачу для {symbol}: {_e}")

    # Временно отправляем в общую очередь 'train' (слушается базовым воркером)
    # Если seed не указан — сгенерируем случайный на стороне web и передадим в Celery
    if seed is None:
        import random as _rnd
        seed = _rnd.randint(1, 2**31 - 1)
        app.logger.info(f"[train_dqn_symbol] generated random seed={seed}")

    task = train_dqn_symbol.apply_async(args=[symbol, episodes, seed], queue="train")
    app.logger.info(f"/train_dqn_symbol queued symbol={symbol} queue=train task_id={task.id}")
    # Сохраняем task_id для отображения на главной и отметку per-symbol
    try:
        redis_client.lrem("ui:tasks", 0, task.id)  # убираем дубликаты
        redis_client.lpush("ui:tasks", task.id)
        redis_client.ltrim("ui:tasks", 0, 49)     # ограничиваем список
        redis_client.setex(running_key, 24 * 3600, task.id)
    except Exception as _e:
        app.logger.error(f"/train_dqn_symbol: не удалось записать ui:tasks: {_e}")
    # Ответ для fetch/XHR — JSON; для форм — редирект
    wants_json = request.is_json or 'application/json' in (request.headers.get('Accept') or '')
    if wants_json:
        return jsonify({
            "success": True,
            "task_id": task.id,
            "symbol": symbol,
            "episodes": episodes,
            "seed": seed
        })
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
        # Разобрать seed; если не указан — сгенерировать позже
        seed_raw = (data.get('seed') or request.form.get('seed') or '').strip()
        seed = None
        try:
            if seed_raw != '':
                seed = int(seed_raw)
        except Exception:
            seed = None

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
        symbol_from_path = None

        # Поддержка трёх форматов: train_result_*, dqn_model_* и runs/<run_id>/model.pth
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
        elif cand_resolved.name == 'model.pth' and 'runs' in cand_resolved.parts:
            # Новый формат: result/<SYMBOL>/runs/<run_id>/model.pth
            try:
                parts = list(cand_resolved.parts)
                runs_idx = parts.index('runs')
                run_id = parts[runs_idx + 1] if runs_idx + 1 < len(parts) else None
                symbol_from_path = parts[runs_idx - 1] if runs_idx - 1 >= 0 else None
                if not run_id:
                    return jsonify({"success": False, "error": "Некорректный путь к run"}), 400
                code = run_id
                model_file = cand_resolved
                run_dir = cand_resolved.parent
                replay_file = run_dir / 'replay.pkl'
                # train_result не обязателен для запуска дообучения
            except Exception:
                return jsonify({"success": False, "error": "Не удалось распознать путь run"}), 400
        else:
            return jsonify({"success": False, "error": "Ожидался файл dqn_model_*.pth, train_result_*.pkl или runs/.../model.pth"}), 400

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
        symbol_guess = None
        try:
            if symbol_from_path:
                symbol_guess = (str(symbol_from_path).upper())
            else:
                symbol_guess = (code.split('_', 1)[0] + 'USDT').upper()
        except Exception:
            symbol_guess = None
        if not symbol_guess:
            symbol_guess = 'BTCUSDT'
        # Если seed не указан — сгенерируем случайный
        if seed is None:
            import random as _rnd
            seed = _rnd.randint(1, 2**31 - 1)
            app.logger.info(f"[continue_training] generated random seed={seed}")

        task = train_dqn_symbol.apply_async(args=[symbol_guess, episodes, seed], queue='train')

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

        return jsonify({"success": True, "task_id": task.id, "seed": seed})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500




# Новый маршрут для запуска очистки данных
 

@app.route('/analyze_training_results', methods=['POST'])
def analyze_training_results():
    """Анализирует результаты обучения DQN модели"""
    try:
        # Новая логика: сначала пробуем конкретный файл из запроса (поддержка runs/*/train_result.pkl)
        results_dir = "result"
        data = request.get_json(silent=True) or {}
        requested_file = (data.get('file') or '').strip()
        selected_file = None
        if requested_file:
            # Нормализуем путь; поддерживаем Windows-разделители
            req = requested_file.replace('\\', '/')
            if not os.path.isabs(req):
                # Если уже начинается с result/ — трактуем как относительный к корню проекта
                if req.startswith('result/'):
                    cand = os.path.abspath(req)
                else:
                    # Иначе считаем относительным к каталогу result/
                    cand = os.path.abspath(os.path.join(results_dir, req))
            else:
                cand = os.path.abspath(req)
            # Принимаем только пути внутри result/
            base_path = os.path.abspath(results_dir)
            if cand.startswith(base_path) and os.path.exists(cand):
                selected_file = cand
        # Если файл не указан/не найден — ищем новый формат result/<SYMBOL>/runs/*/train_result.pkl
        if not selected_file:
            run_results = []
            if os.path.exists(results_dir):
                for sym in os.listdir(results_dir):
                    rdir = os.path.join(results_dir, sym, 'runs')
                    if not os.path.isdir(rdir):
                        continue
                    for run_id in os.listdir(rdir):
                        p = os.path.join(rdir, run_id, 'train_result.pkl')
                        if os.path.exists(p):
                            run_results.append(p)
            # Фоллбек на старый плоский формат
            flat_results = glob.glob(os.path.join(results_dir, 'train_result_*.pkl'))
            all_candidates = (run_results or []) + (flat_results or [])
            if not all_candidates:
                return jsonify({'status': 'error','message': 'Файлы результатов обучения не найдены. Сначала запустите обучение.','success': False}), 404
            selected_file = max(all_candidates, key=os.path.getctime)
        
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
            'available_files': []
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
    """Возвращает список доступных результатов в новой структуре result/<SYMBOL>/runs/*/{train_result.pkl, manifest.json}"""
    try:
        base = os.path.join('result')
        if not os.path.exists(base):
            return jsonify({'status': 'error','message': f'Папка {base} не найдена','success': False,'files': []}), 404
        items = []
        for sym in sorted(os.listdir(base)):
            sym_dir = os.path.join(base, sym, 'runs')
            if not os.path.isdir(sym_dir):
                continue
            for run_id in os.listdir(sym_dir):
                run_dir = os.path.join(sym_dir, run_id)
                if not os.path.isdir(run_dir):
                    continue
                tr = os.path.join(run_dir, 'train_result.pkl')
                mf = os.path.join(run_dir, 'manifest.json')
                mp = os.path.join(run_dir, 'model.pth')
                if os.path.exists(tr):
                    try:
                        stat = os.stat(tr)
                        manifest = {}
                        try:
                            if os.path.exists(mf):
                                import json as _json
                                manifest = _json.loads(open(mf,'r',encoding='utf-8').read())
                        except Exception:
                            manifest = {}
                        items.append({
                            'symbol': sym,
                            'run_id': run_id,
                            'train_result': tr,
                            'model_path': mp if os.path.exists(mp) else None,
                            'manifest': manifest,
                            'size': stat.st_size,
                            'created': stat.st_ctime,
                            'modified': stat.st_mtime
                        })
                    except Exception:
                        continue
        if not items:
            return jsonify({'status': 'error','message': 'Файлы результатов не найдены','success': False,'files': []}), 404
        items.sort(key=lambda x: (x.get('created') or 0), reverse=True)
        return jsonify({'status': 'success','message': f'Найдено {len(items)} результатов','success': True,'files': items})
    except Exception as e:
        return jsonify({'status': 'error','message': f'Ошибка при получении списка: {str(e)}','success': False}), 500

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
        if not p.exists() or not p.is_file() or p.suffix != '.pth':
            return jsonify({'success': False, 'error': 'Ожидается существующий файл модели (.pth) внутри result'}), 400

        # Поддержка двух форматов путей:
        # 1) Плоский: result/dqn_model_<code>.pth
        # 2) Структурированный run: result/<SYMBOL>/runs/<run_id>/model.pth
        code = None
        symbol_str = None
        replay_file = None
        train_file = None
        if p.name.startswith('dqn_model_'):
            code = p.stem.replace('dqn_model_', '')
            replay_file = results_dir / f'replay_buffer_{code}.pkl'
            train_file = results_dir / f'train_result_{code}.pkl'
            # Попытаемся извлечь символ из префикса кода
            try:
                base = code.split('_')[0].upper()
                if base and len(base) <= 6:
                    symbol_str = base + 'USDT'
            except Exception:
                symbol_str = None
        else:
            # Пытаемся распознать путь run: .../result/<SYMBOL>/runs/<run_id>/model.pth
            try:
                parts = [x for x in p.parts]
                # Найдём индекс 'runs'
                if 'runs' in parts:
                    idx = parts.index('runs')
                    if idx + 1 < len(parts):
                        run_id = parts[idx + 1]
                        code = run_id
                        run_dir = p.parent
                        replay_file = run_dir / 'replay.pkl'
                        train_file = run_dir / 'train_result.pkl'
                        # Символ берём из каталога перед 'runs'
                        if idx - 1 >= 0:
                            symbol_str = parts[idx - 1]
            except Exception:
                pass
            if not code:
                return jsonify({'success': False, 'error': 'Не удалось распознать модель: ожидается dqn_model_*.pth или runs/.../model.pth'}), 400

        info = {
            'success': True,
            'model_file': str(p),
            'model_size_bytes': p.stat().st_size if p.exists() else 0,
            'code': code,
            'symbol': symbol_str,
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
                # Пробрасываем seed из метаданных, если сохранён
                try:
                    meta = results.get('train_metadata') or {}
                    if isinstance(meta, dict) and 'seed' in meta:
                        info['seed'] = meta.get('seed')
                    # Устройство обучения
                    if isinstance(meta, dict):
                        info['cuda_available'] = bool(meta.get('cuda_available')) if ('cuda_available' in meta) else None
                        info['gpu_name'] = meta.get('gpu_name')
                except Exception:
                    pass
                # Добавим суммарное время и среднее время на эпизод (если есть)
                total_training_time = results.get('total_training_time')
                if isinstance(total_training_time, (int, float)):
                    info['total_training_time'] = float(total_training_time)
                    try:
                        if info.get('episodes'):
                            avg_sec = float(total_training_time) / float(info['episodes'])
                            info['avg_time_per_episode_sec'] = avg_sec
                    except Exception:
                        pass
            except Exception as _e:
                info['stats_error'] = str(_e)

        # Fallback: если train_result.pkl отсутствует, но есть manifest.json — достаём seed/эпизоды/дату
        try:
            if (not train_file.exists()) and p.name == 'model.pth' and 'runs' in str(p):
                run_dir = p.parent
                mf = run_dir / 'manifest.json'
                if mf.exists():
                    import json as _json
                    try:
                        with open(mf, 'r', encoding='utf-8') as _f:
                            _m = _json.load(_f)
                        if isinstance(_m, dict):
                            if 'seed' in _m and ('seed' not in info or info.get('seed') in (None, '—')):
                                info['seed'] = _m.get('seed')
                            if 'episodes_end' in _m and (info.get('episodes') is None):
                                info['episodes'] = _m.get('episodes_end')
                            if not info.get('symbol') and _m.get('symbol'):
                                info['symbol'] = _m.get('symbol')
                            # Отметим дату создания
                            if _m.get('created_at'):
                                info['created_at'] = _m.get('created_at')
                    except Exception:
                        pass
        except Exception:
            pass

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

@app.route('/agent/<symbol>')
def agent_symbol_page(symbol: str):
    """Страница агента, отфильтрованная по конкретному символу (BTCUSDT, TONUSDT и т.д.)"""
    try:
        sym = (symbol or '').upper().strip()
        # Простейшая валидация: только латиница и 'USDT' в конце
        import re
        if not re.match(r'^[A-Z]{2,10}USDT$', sym):
            # дефолт на BTCUSDT
            sym = 'BTCUSDT'
        return render_template('agent_symbol.html', symbol=sym)
    except Exception:
        return render_template('agent_symbol.html', symbol='BTCUSDT')

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
            # Извлекаем символ из кода (префикс до первого '_')
            try:
                symbol_base = (base_code.split('_', 1)[0] or base_code).lower()
            except Exception:
                symbol_base = base_code.lower()
            symbol_dir = models_root / symbol_base
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
        
@app.route('/list_ensembles')
def list_ensembles():
    """Возвращает структуру ансамблей и версий для указанного символа.
    Query params: symbol=BTCUSDT (или btc)
    """
    try:
        symbol = (request.args.get('symbol') or '').strip()
        if not symbol:
            return jsonify({"success": False, "error": "symbol обязателен"}), 400
        s = symbol.upper().replace('/', '')
        if s.endswith('USDT'):
            s = s[:-4]
        base_code = s.lower()

        from pathlib import Path
        import pickle as _pickle
        root = Path('models') / base_code
        if not root.exists():
            return jsonify({"success": True, "ensembles": {}})

        ensembles = {}
        for ens_dir in root.iterdir():
            if not ens_dir.is_dir():
                continue
            versions = []
            current = None
            canary = None
            # Собираем vN
            for vdir in ens_dir.iterdir():
                if vdir.is_dir() and vdir.name.startswith('v'):
                    # Ищем файлы и manifest
                    files = { 'model': None, 'replay': None, 'results': None }
                    manifest = None
                    stats = {}
                    for f in vdir.iterdir():
                        if f.name.startswith('dqn_model_') and f.suffix == '.pth':
                            files['model'] = f.name
                        elif f.name.startswith('replay_buffer_') and f.suffix == '.pkl':
                            files['replay'] = f.name
                        elif f.name.startswith('train_result_') and f.suffix == '.pkl':
                            files['results'] = f.name
                        elif f.name == 'manifest.yaml':
                            manifest = f.name
                    # Пытаемся вытащить краткую статистику из train_result_*.pkl
                    try:
                        if files.get('results'):
                            _res_path = vdir / files['results']
                            if _res_path.exists():
                                with open(_res_path, 'rb') as _f:
                                    _res = _pickle.load(_f)
                                    if isinstance(_res, dict) and 'final_stats' in _res:
                                        stats = _res['final_stats'] or {}
                    except Exception:
                        stats = {}
                    versions.append({
                        'version': vdir.name,
                        'files': files,
                        'manifest': manifest,
                        'stats': stats,
                        'path': str(vdir).replace('\\','/')
                    })
                elif vdir.name == 'current':
                    current = str(vdir).replace('\\','/')
                elif vdir.name == 'canary':
                    canary = str(vdir).replace('\\','/')
            ensembles[ens_dir.name] = {
                'versions': sorted(versions, key=lambda x: int(x['version'][1:]) if x['version'].startswith('v') and x['version'][1:].isdigit() else 0),
                'current': current,
                'canary': canary,
                'path': str(ens_dir).replace('\\','/')
            }

        return jsonify({"success": True, "ensembles": ensembles})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

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

@app.route('/api/trading/save_config', methods=['POST'])
def save_trading_config():
    """Автосохранение выбора моделей и консенсуса без запуска торговли."""
    try:
        data = request.get_json() or {}
        symbols = data.get('symbols') or []
        model_paths = data.get('model_paths') or []
        consensus = data.get('consensus') or None
        import json as _json
        rc = get_redis_client()
        if symbols:
            rc.set('trading:symbols', _json.dumps(symbols, ensure_ascii=False))
        # Не перезаписываем model_paths пустым списком — чтобы не ломать следующий тик
        if isinstance(model_paths, list) and len(model_paths) > 0:
            rc.set('trading:model_paths', _json.dumps(model_paths, ensure_ascii=False))
            rc.set('trading:last_model_paths', _json.dumps(model_paths, ensure_ascii=False))
        if consensus is not None:
            rc.set('trading:consensus', _json.dumps(consensus, ensure_ascii=False))
            rc.set('trading:last_consensus', _json.dumps(consensus, ensure_ascii=False))
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/trading/start', methods=['POST'])
def start_trading():
    """
    Запуск торговли в контейнере trading_agent через Celery задачу
    """
    try:
        data = request.get_json() or {}
        symbols = data.get('symbols', ['BTCUSDT'])
        account_id = str(data.get('account_id') or '').strip()
        # Поддержка многомодельного запуска: model_paths (список) + совместимость с model_path
        model_paths = data.get('model_paths') or []
        model_path = data.get('model_path')
        if (not model_path) and isinstance(model_paths, list) and len(model_paths) > 0:
            model_path = model_paths[0]
        if not model_path:
            model_path = '/workspace/models/btc/ensemble-a/current/dqn_model.pth'
        
        # Сохраняем выбранные параметры в Redis для последующих вызовов (status/stop/balance/history)
        try:
            import json as _json
            _rc = get_redis_client()
            _rc.set('trading:model_path', model_path)
            try:
                if isinstance(model_paths, list):
                    _rc.set('trading:model_paths', _json.dumps(model_paths, ensure_ascii=False))
            except Exception:
                pass
            _rc.set('trading:symbols', _json.dumps(symbols, ensure_ascii=False))
            if account_id:
                _rc.set('trading:account_id', account_id)
            # Консенсус (counts/percents) — запоминаем выбор пользователя
            try:
                consensus = data.get('consensus')
                if consensus is not None:
                    _rc.set('trading:consensus', _json.dumps(consensus, ensure_ascii=False))
                    # Сохраняем дубликат для фолбэка тиков
                    _rc.set('trading:last_consensus', _json.dumps(consensus, ensure_ascii=False))
            except Exception:
                pass
            # Обновим last_model_paths для фолбэка тиков
            try:
                if isinstance(model_paths, list) and model_paths:
                    _rc.set('trading:last_model_paths', _json.dumps(model_paths, ensure_ascii=False))
            except Exception:
                pass
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

        # Redis-лок: если уже идёт торговый шаг, не стартуем второй параллельно
        try:
            _rc_lock = get_redis_client()
            if _rc_lock.get('trading:agent_lock'):
                return jsonify({
                    'success': False,
                    'error': 'Торговый шаг уже выполняется (agent_lock_active)'
                }), 429
        except Exception:
            pass

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
        # Входная точка: фиксируем вызов эндпоинта
        try:
            app.logger.info("[trading_status] ▶ request received")
        except Exception:
            pass
        # Сначала пробуем быстрый статус из Redis (обновляется периодическим таском)
        try:
            _rc = get_redis_client()
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
                    try:
                        app.logger.info(f"[trading_status] ✓ using cached status | keys={list(flat.keys())}")
                        # Краткий обзор важных полей
                        app.logger.info("[trading_status] summary: is_trading=%s, position=%s, trades_count=%s",
                                        flat.get('is_trading'), bool(flat.get('position') or flat.get('current_position')), flat.get('trades_count'))
                    except Exception:
                        pass
                    return jsonify(flat), 200
        except Exception:
            pass

        # Нет свежего статуса в Redis — возвращаем понятный OFF статус для UI
        try:
            default_status = {
                'success': True,
                'is_trading': False,
                'trading_status': 'Остановлена',
                'trading_status_emoji': '🔴',
                'trading_status_full': '🔴 Остановлена (агент не запущен)',
                'symbol': None,
                'symbol_display': 'Не указана',
                'amount': None,
                'amount_display': 'Не указано',
                'amount_usdt': 0.0,
                'position': None,
                'trades_count': 0,
                'balance': {},
                'current_price': 0.0,
                'last_model_prediction': None,
                'is_fresh': False,
                'reason': 'status not available in redis'
            }
            flat = {'success': True, 'agent_status': default_status}
            flat.update(default_status)
            try:
                app.logger.info("[trading_status] ⚠ no redis status, returning OFF state")
            except Exception:
                pass
            return jsonify(flat), 200
        except Exception:
            return jsonify({'success': False, 'error': 'status not available in redis', 'is_fresh': False}), 200
            
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
        requested_symbol = (request.args.get('symbol') or '').upper().strip()
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

        # Фильтрация по символу, если задан
        if requested_symbol:
            try:
                latest_results = [r for r in latest_results if isinstance(r.get('symbols'), list) and requested_symbol in r.get('symbols')]
            except Exception:
                pass
        
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
    """Баланс берём из Redis-кэша trading:current_status. Без Docker exec."""
    try:
        _rc = get_redis_client()
        cached = _rc.get('trading:current_status') if _rc else None
        resp_obj = None
        if cached:
            import json as _json
            try:
                st = _json.loads(cached)
                if isinstance(st, dict) and st.get('balance'):
                    resp_obj = {
                        'success': True,
                        'balance': st.get('balance'),
                        'is_trading': st.get('is_trading', False),
                        'is_fresh': st.get('is_fresh', False)
                    }
            except Exception:
                resp_obj = None

        if resp_obj is None:
            resp_obj = {
                'success': True,
                'balance': {},
                'message': 'balance not available (agent not running)'
            }

        return jsonify(resp_obj), 200
    except Exception as e:
        app.logger.error(f"Ошибка получения баланса: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

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

            def _disc():
                ak = os.environ.get('BYBIT_1_API_KEY')
                sk = os.environ.get('BYBIT_1_SECRET_KEY')
                if ak and sk:
                    return ak, sk
                # Поиск первого BYBIT_<ID>_*
                c = []
                for k, v in os.environ.items():
                    if not k.startswith('BYBIT_') or not k.endswith('_API_KEY'):
                        continue
                    idx = k[len('BYBIT_'):-len('_API_KEY')]
                    sec = f'BYBIT_{idx}_SECRET_KEY'
                    sv = os.environ.get(sec)
                    if v and sv:
                        c.append((k, v, sec, sv))
                if c:
                    c.sort(key=lambda x: x[0])
                    return c[0][1], c[0][3]
                return '', ''
            api_key, secret_key = _disc()
            api_key_esc = api_key.replace("'", "\\'")
            secret_key_esc = secret_key.replace("'", "\\'")
            if model_path:
                cmd = (
                    f"python -c \"import json, os; os.environ['BYBIT_1_API_KEY']='{api_key_esc}'; os.environ['BYBIT_1_SECRET_KEY']='{secret_key_esc}'; "
                    f"from trading_agent.trading_agent import TradingAgent; agent = TradingAgent(model_path=\\\"{model_path}\\\"); "
                    f"action = {py_action}; symbol = {py_symbol}; quantity = {py_quantity}; "
                    f"result = agent.execute_direct_order(action, symbol, quantity); print(\\\"RESULT: \\\" + json.dumps(result))\""
                )
            else:
                cmd = (
                    "python -c \"import json, os; os.environ['BYBIT_1_API_KEY']='{api_key_esc}'; os.environ['BYBIT_1_SECRET_KEY']='{secret_key_esc}'; "
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

@app.route('/api/trades/matched_full', methods=['GET'])
def get_matched_full_trades():
    """Сопоставляет BUY→SELL сделки с предсказаниями в той же 5м свече (допуск ±1 свеча)."""
    try:
        symbol = request.args.get('symbol')
        limit_trades = request.args.get('limit_trades', 200, type=int)
        limit_predictions = request.args.get('limit_predictions', 1000, type=int)
        tolerance_buckets = request.args.get('tolerance_buckets', 1, type=int)

        # Загружаем сделки
        if symbol:
            trades = get_trades_by_symbol(symbol_name=symbol, limit=limit_trades)
        else:
            trades = get_recent_trades(limit=limit_trades)

        # Загружаем предсказания
        preds = get_model_predictions(symbol=symbol, action=None, limit=limit_predictions)

        # Хелперы
        def unify_symbol(s: str) -> str:
            try:
                return (s or '').upper().replace('/', '')
            except Exception:
                return ''

        def to_ms(v):
            if not v:
                return None
            try:
                if isinstance(v, (int, float)):
                    return int(v)
                s = str(v)
                # ISO без TZ трактуем как UTC
                has_tz = ('Z' in s) or ('z' in s) or ('+' in s) or ('-' in s and 'T' in s and s.rfind('-') > s.find('T'))
                if not has_tz and len(s) >= 19 and s[10] == 'T':
                    s = s + 'Z'
                from datetime import datetime
                dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
                return int(dt.timestamp() * 1000)
            except Exception:
                return None

        def bucket_5m(ms: int):
            return None if ms is None else (ms // 300000)

        # Индексация предсказаний: bucket → symbol → {buy, sell, hold}
        preds_by_bucket = {}
        for p in preds:
            ts = to_ms(p.timestamp.isoformat() if getattr(p, 'timestamp', None) else None)
            b = bucket_5m(ts)
            if b is None:
                continue
            act = (p.action or '').lower()
            sym = unify_symbol(getattr(p, 'symbol', ''))
            if b not in preds_by_bucket:
                preds_by_bucket[b] = {}
            if sym not in preds_by_bucket[b]:
                preds_by_bucket[b][sym] = { 'buy': [], 'sell': [], 'hold': [] }
            if act in preds_by_bucket[b][sym]:
                try:
                    q_vals = json.loads(p.q_values) if p.q_values else []
                except Exception:
                    q_vals = []
                preds_by_bucket[b][sym][act].append({
                    'timestamp': getattr(p.timestamp, 'isoformat', lambda: None)(),
                    'symbol': getattr(p, 'symbol', None),
                    'action': p.action,
                    'q_values': q_vals,
                    'current_price': getattr(p, 'current_price', None),
                    'position_status': getattr(p, 'position_status', None),
                    'confidence': getattr(p, 'confidence', None),
                    'model_path': getattr(p, 'model_path', None),
                })

        def pick_pred(bkt: int, sym_u: str, typ: str):
            for delta in [0] + [d for d in range(-tolerance_buckets, tolerance_buckets+1) if d != 0]:
                bb = bkt + delta
                bucket = preds_by_bucket.get(bb)
                if not bucket:
                    continue
                by_sym = bucket.get(sym_u)
                if not by_sym:
                    continue
                arr = by_sym.get(typ) or []
                if arr:
                    return arr[0]
            return None

        # Нормализуем сделки и собираем пары BUY→SELL (предсказания НЕ обязательны)
        norm_trades = []
        for t in trades:
            ms = to_ms((t.executed_at or t.created_at or None).isoformat() if getattr(t, 'executed_at', None) or getattr(t, 'created_at', None) else None)
            act_raw = (t.action or '').lower()
            # Нормализуем действие: считаем sell_partial как sell
            if 'buy' in act_raw:
                act = 'buy'
            elif 'sell' in act_raw:
                act = 'sell'
            else:
                continue
            sym = unify_symbol(t.symbol.name if getattr(t, 'symbol', None) else getattr(t, 'symbol', '') or '')
            price = float(getattr(t, 'price', 0.0) or 0.0)
            if not (price and price > 0):
                # Игнорируем нулевые/отсутствующие цены как шум
                continue
            qty = float(getattr(t, 'quantity', None) or getattr(t, 'total_value', 0.0) / price if price else (getattr(t, 'quantity', 0.0) or 0.0))
            if ms is None or act not in ('buy', 'sell'):
                continue
            norm_trades.append({ 'ms': ms, 'action': act, 'symbol': sym, 'price': price, 'qty': qty })

        norm_trades.sort(key=lambda x: x['ms'])

        # Подсчёт для отладки
        num_buys = sum(1 for x in norm_trades if x['action'] == 'buy')
        num_sells = sum(1 for x in norm_trades if x['action'] == 'sell')

        used_sell_idx = set()
        pairs = []
        for i, tb in enumerate(norm_trades):
            if tb['action'] != 'buy':
                continue
            b_buy = bucket_5m(tb['ms'])
            pred_buy = pick_pred(b_buy, tb['symbol'], 'buy')  # опционально
            # Найти последнюю продажу до следующего BUY того же символа
            sell_j = None
            next_buy_idx = None
            for j in range(i+1, len(norm_trades)):
                ts = norm_trades[j]
                if ts['symbol'] != tb['symbol']:
                    continue
                if ts['action'] == 'buy':
                    next_buy_idx = j
                    break
            # диапазон для поиска sell: (i, next_buy_idx) или до конца
            end_idx = next_buy_idx if next_buy_idx is not None else len(norm_trades)
            for j in range(i+1, end_idx):
                if j in used_sell_idx:
                    continue
                ts = norm_trades[j]
                if ts['symbol'] != tb['symbol']:
                    continue
                if ts['action'] == 'sell':
                    sell_j = j  # запоминаем последнюю продажу в диапазоне
            if sell_j is None:
                continue
            if sell_j is None:
                continue
            ts = norm_trades[sell_j]
            b_sell = bucket_5m(ts['ms'])
            pred_sell = pick_pred(b_sell, ts['symbol'], 'sell')  # опционально

            used_sell_idx.add(sell_j)
            qty = tb['qty'] or ts['qty'] or 0.0
            pnl_abs = (ts['price'] - tb['price']) * qty
            pnl_pct = ((ts['price'] - tb['price']) / tb['price'] * 100.0) if tb['price'] else None

            from datetime import datetime, timezone
            entry_iso = datetime.fromtimestamp(tb['ms']/1000.0, tz=timezone.utc).isoformat()
            exit_iso = datetime.fromtimestamp(ts['ms']/1000.0, tz=timezone.utc).isoformat()

            pairs.append({
                'symbol': tb['symbol'],
                'entry_time': entry_iso,
                'exit_time': exit_iso,
                'entry_price': tb['price'],
                'exit_price': ts['price'],
                'qty': qty,
                'pnl_abs': pnl_abs,
                'pnl_pct': pnl_pct,
                'pred_buy': pred_buy,
                'pred_sell': pred_sell,
            })

        # Доп. отладка по запросу
        debug_payload = None
        try:
            if (request.args.get('debug') in ('1','true','yes')):
                app.logger.info(
                    f"[matched_full] trades={len(trades)} norm={len(norm_trades)} (buy={num_buys}/sell={num_sells}) pairs={len(pairs)} preds={len(preds)} symbol={symbol or 'ALL'} tol={tolerance_buckets}"
                )
                from datetime import datetime, timezone
                sample = []
                for it in norm_trades[:10]:
                    sample.append({
                        'time': datetime.fromtimestamp(it['ms']/1000.0, tz=timezone.utc).isoformat(),
                        'action': it['action'],
                        'symbol': it['symbol'],
                        'price': it['price'],
                        'qty': it['qty']
                    })
                debug_payload = {
                    'trades_total': len(trades),
                    'norm_total': len(norm_trades),
                    'norm_buys': num_buys,
                    'norm_sells': num_sells,
                    'pairs_total': len(pairs),
                    'norm_sample': sample,
                }
        except Exception:
            debug_payload = None

        resp = {'success': True, 'pairs': pairs, 'total_pairs': len(pairs)}
        if debug_payload is not None:
            resp['debug'] = debug_payload
        return jsonify(resp), 200

    except Exception as e:
        return jsonify({ 'success': False, 'error': str(e) }), 500

@app.route('/api/analysis/qvalues_vs_pnl', methods=['GET'])
def analysis_qvalues_vs_pnl():
    """Возвращает датасет BUY→SELL пар с P&L и q_values для анализа порогов.

    Params:
      symbol: опционально, фильтр по символу
      limit_trades: максимум сделок (по умолчанию 400)
      limit_predictions: максимум предсказаний (по умолчанию 2000)
      tolerance_buckets: допуск по времени в 5м свечах (по умолчанию 1)
    """
    try:
        symbol = request.args.get('symbol')
        limit_trades = request.args.get('limit_trades', 400, type=int)
        limit_predictions = request.args.get('limit_predictions', 2000, type=int)
        tolerance_buckets = request.args.get('tolerance_buckets', 1, type=int)

        # Переиспользуем логику сопоставления
        with app.test_request_context(
            f"/api/trades/matched_full?symbol={symbol or ''}&limit_trades={limit_trades}&limit_predictions={limit_predictions}&tolerance_buckets={tolerance_buckets}"
        ):
            resp_raw = get_matched_full_trades()
        # Нормализуем (Response | (Response, status))
        if isinstance(resp_raw, tuple):
            resp_obj, status_code = resp_raw
        else:
            resp_obj, status_code = resp_raw, getattr(resp_raw, 'status_code', 200)
        if status_code != 200:
            return resp_obj, status_code
        payload = resp_obj.get_json() or {}
        if not payload.get('success'):
            return jsonify(payload), 200

        rows = []
        counters = {
            'pairs_total': 0,
            'pairs_with_buy_q': 0,
            'pairs_with_sell_q': 0
        }
        for p in payload.get('pairs', []):
            counters['pairs_total'] += 1
            pred_buy = p.get('pred_buy') or {}
            pred_sell = p.get('pred_sell') or {}
            q_vals_buy = pred_buy.get('q_values') or []
            q_vals_sell = pred_sell.get('q_values') or []
            # Явно берем Q(BUY) и разрыв относительно max(HOLD, SELL)
            q_buy = None
            q_buy_gap = None
            try:
                if isinstance(q_vals_buy, list) and len(q_vals_buy) >= 2:
                    q_buy = float(q_vals_buy[1])  # [hold, buy, sell]
                    other = []
                    if len(q_vals_buy) >= 1:
                        other.append(float(q_vals_buy[0]))
                    if len(q_vals_buy) >= 3:
                        other.append(float(q_vals_buy[2]))
                    if other:
                        q_buy_gap = q_buy - max(other)
                    counters['pairs_with_buy_q'] += 1
            except Exception:
                q_buy = None
                q_buy_gap = None

            # Для SELL: Q(SELL) и разрыв относительно max(HOLD, BUY)
            q_sell = None
            q_sell_gap = None
            try:
                if isinstance(q_vals_sell, list) and len(q_vals_sell) >= 3:
                    q_sell = float(q_vals_sell[2])
                    other_s = []
                    if len(q_vals_sell) >= 1:
                        other_s.append(float(q_vals_sell[0]))
                    if len(q_vals_sell) >= 2:
                        other_s.append(float(q_vals_sell[1]))
                    if other_s:
                        q_sell_gap = q_sell - max(other_s)
                    counters['pairs_with_sell_q'] += 1
            except Exception:
                q_sell = None
                q_sell_gap = None

            rows.append({
                'symbol': p.get('symbol'),
                'entry_time': p.get('entry_time'),
                'exit_time': p.get('exit_time'),
                'pnl_abs': p.get('pnl_abs'),
                'pnl_pct': p.get('pnl_pct'),
                'buy_confidence': pred_buy.get('confidence'),
                'sell_confidence': pred_sell.get('confidence'),
                'buy_q_values': q_vals_buy,
                'sell_q_values': pred_sell.get('q_values') or [],
                'buy_max_q': q_buy,
                'buy_gap_q': q_buy_gap,
                'sell_max_q': q_sell,
                'sell_gap_q': q_sell_gap,
            })

        return jsonify({ 'success': True, 'rows': rows, 'total': len(rows), 'counters': counters }), 200

    except Exception as e:
        return jsonify({ 'success': False, 'error': str(e) }), 500

@app.route('/api/analysis/qgate_suggest', methods=['GET'])
def analysis_qgate_suggest():
    """Подбирает пороги T1/T2 (maxQ/gapQ) по сетке квантилей без внешних зависимостей.

    Params: такие же, как у /api/analysis/qvalues_vs_pnl
    Возвращает: { T1, T2, hit_rate, n, score }
    """
    try:
        symbol = request.args.get('symbol')
        limit_trades = request.args.get('limit_trades', 400, type=int)
        limit_predictions = request.args.get('limit_predictions', 2000, type=int)
        tolerance_buckets = request.args.get('tolerance_buckets', 1, type=int)
        metric = request.args.get('metric', 'hit_rate')  # hit_rate | pnl_sum | pnl_per_trade
        min_n = request.args.get('min_n', 20, type=int)
        grid_points = request.args.get('grid_points', 15, type=int)
        side = request.args.get('side', 'buy')  # buy | sell

        # Получаем датасет
        with app.test_request_context(
            f"/api/analysis/qvalues_vs_pnl?symbol={symbol or ''}&limit_trades={limit_trades}&limit_predictions={limit_predictions}&tolerance_buckets={tolerance_buckets}"
        ):
            resp_raw = analysis_qvalues_vs_pnl()
        # Нормализуем (Response | (Response, status))
        if isinstance(resp_raw, tuple):
            resp_obj, status_code = resp_raw
        else:
            resp_obj, status_code = resp_raw, getattr(resp_raw, 'status_code', 200)
        if status_code != 200:
            return resp_obj, status_code
        js = resp_obj.get_json() or {}
        if not js.get('success'):
            return jsonify(js), 200

        rows = js.get('rows', [])
        # Преобразуем в простые массивы
        maxqs = []
        gapqs = []
        wins = []
        for r in rows:
            if side == 'sell':
                max_q = r.get('sell_max_q')
                gap_q = r.get('sell_gap_q')
            else:
                max_q = r.get('buy_max_q')
                gap_q = r.get('buy_gap_q')
            pnl_abs = r.get('pnl_abs')
            if max_q is None:
                continue
            maxqs.append(float(max_q))
            gapqs.append(float(gap_q) if gap_q is not None else float('nan'))
            wins.append(1.0 if (pnl_abs is not None and float(pnl_abs) > 0.0) else 0.0)

        if not maxqs:
            return jsonify({ 'success': False, 'error': 'Недостаточно данных' }), 200

        # Квантильные сетки
        def quantiles(arr, qs):
            a = sorted([x for x in arr if not (x is None or (isinstance(x, float) and (x != x)))])
            if not a:
                return []
            res = []
            n = len(a)
            for q in qs:
                if q <= 0:
                    res.append(a[0])
                elif q >= 1:
                    res.append(a[-1])
                else:
                    idx = int(q * (n - 1))
                    res.append(a[idx])
            return res

        gp = max(5, grid_points)
        qs = [0.2 + i*(0.7/(gp-1)) for i in range(gp)]  # 0.2..0.9, gp точек
        maxq_vals = quantiles(maxqs, qs)
        has_gap = any(not (g != g) for g in gapqs)  # есть хотя бы одно не-NaN
        gapq_clean = [g for g in gapqs if not (isinstance(g, float) and (g != g))]
        gapq_vals = quantiles(gapq_clean, qs) if gapq_clean else [0.0]

        best = None
        total = len(maxqs)
        for t1 in maxq_vals:
            for t2 in gapq_vals:
                selected = 0
                wins_sel = 0
                pnl_sum = 0.0
                for i in range(total):
                    ok = (maxqs[i] >= t1)
                    if has_gap and i < len(gapqs) and not (isinstance(gapqs[i], float) and (gapqs[i] != gapqs[i])):
                        ok = ok and (gapqs[i] >= t2)
                    if ok:
                        selected += 1
                        wins_sel += wins[i]
                        # добавляем pnl, если доступен
                        try:
                            pnl_val = float(rows[i].get('pnl_abs') or 0.0)
                        except Exception:
                            pnl_val = 0.0
                        pnl_sum += pnl_val
                if selected < min_n:
                    continue
                hit = wins_sel / selected if selected > 0 else 0.0
                if metric == 'hit_rate':
                    score = hit * (selected / total)
                elif metric == 'pnl_sum':
                    score = pnl_sum
                else:  # pnl_per_trade
                    score = (pnl_sum / selected) if selected else -1e9
                if (best is None) or (score > best['score']):
                    best = { 'T1': float(t1), 'T2': float(t2), 'hit_rate': float(hit), 'n': int(selected), 'score': float(score), 'pnl_sum': float(pnl_sum), 'pnl_per_trade': float((pnl_sum/selected) if selected else 0.0) }

        if not best:
            # Эвристический фолбэк по квантилям, чтобы вернуть разумные пороги даже на малой выборке
            def quantile_one(arr, q):
                a = sorted([x for x in arr if not (isinstance(x, float) and (x != x))])
                if not a:
                    return None
                if q <= 0:
                    return a[0]
                if q >= 1:
                    return a[-1]
                idx = int(q * (len(a) - 1))
                return a[idx]

            t1_fb = quantile_one(maxqs, 0.7) or maxqs[0]
            t2_fb = quantile_one(gapq_clean if gapq_clean else [0.0], 0.6) if has_gap else 0.0

            # Оценим метрики для фолбэка
            selected = 0
            wins_sel = 0
            pnl_sum = 0.0
            for i in range(total):
                ok = (maxqs[i] >= t1_fb)
                if has_gap and i < len(gapqs) and not (isinstance(gapqs[i], float) and (gapqs[i] != gapqs[i])):
                    ok = ok and (gapqs[i] >= t2_fb)
                if ok:
                    selected += 1
                    wins_sel += wins[i]
                    try:
                        pnl_val = float(rows[i].get('pnl_abs') or 0.0)
                    except Exception:
                        pnl_val = 0.0
                    pnl_sum += pnl_val
            hit = wins_sel / selected if selected else 0.0
            best = { 'T1': float(t1_fb), 'T2': float(t2_fb), 'hit_rate': float(hit), 'n': int(selected), 'score': float(hit * (selected/total) if total else 0.0), 'pnl_sum': float(pnl_sum), 'pnl_per_trade': float((pnl_sum/selected) if selected else 0.0) }
            approx = True
        else:
            approx = False

        # Сводки распределений по win/loss
        def summarize(arr, mask):
            vals = [float(arr[i]) for i in range(len(arr)) if mask[i] and not (isinstance(arr[i], float) and (arr[i] != arr[i]))]
            if not vals:
                return { 'n': 0 }
            vals_sorted = sorted(vals)
            def q(p):
                if not vals_sorted:
                    return None
                idx = int(p * (len(vals_sorted)-1))
                return vals_sorted[idx]
            return {
                'n': len(vals_sorted),
                'mean': sum(vals_sorted)/len(vals_sorted),
                'q10': q(0.1), 'q50': q(0.5), 'q90': q(0.9)
            }

        mask_win = [w == 1.0 for w in wins]
        mask_loss = [w == 0.0 for w in wins]
        summary = {
            'q_buy_win': summarize(maxqs, mask_win),
            'q_buy_loss': summarize(maxqs, mask_loss),
            'gap_win': summarize(gapqs, mask_win),
            'gap_loss': summarize(gapqs, mask_loss)
        }

        env_str = (f"QGATE_{'SELL_' if side=='sell' else ''}MAXQ={best['T1']:.6f} "
                   f"QGATE_{'SELL_' if side=='sell' else ''}GAPQ={best['T2']:.6f}")
        # Дополнительные счетчики доступности q-значений
        counters = None
        try:
            with app.test_request_context(
                f"/api/analysis/qvalues_vs_pnl?symbol={symbol or ''}&limit_trades={limit_trades}&limit_predictions={limit_predictions}&tolerance_buckets={tolerance_buckets}"
            ):
                resp_raw2 = analysis_qvalues_vs_pnl()
            if isinstance(resp_raw2, tuple):
                resp_obj2, status_code2 = resp_raw2
            else:
                resp_obj2, status_code2 = resp_raw2, getattr(resp_raw2, 'status_code', 200)
            if status_code2 == 200:
                js2 = resp_obj2.get_json() or {}
                counters = js2.get('counters')
        except Exception:
            counters = None

        return jsonify({ 'success': True, 'suggestion': best, 'env': env_str, 'summary': summary, 'side': side, 'approx': approx, 'counters': counters }), 200

    except Exception as e:
        return jsonify({ 'success': False, 'error': str(e) }), 500


@app.route('/api/trading/history', methods=['GET'])
def trading_history():
    """История торговли без Docker exec: читаем из Redis последние результаты Celery."""
    try:
        # Читаем последние результаты из Redis (как в /api/trading/latest_results)
        latest_results = []
        try:
            keys = redis_client.keys('trading:latest_result_*') or []
            for k in keys:
                try:
                    raw = redis_client.get(k)
                    if not raw:
                        continue
                    try:
                        item = json.loads(raw.decode('utf-8'))
                    except Exception:
                        item = json.loads(raw)
                    latest_results.append(item)
                except Exception as e:
                    app.logger.warning(f"Ошибка чтения {k}: {e}")
        except Exception as e:
            app.logger.warning(f"Ошибка получения ключей trading:latest_result_*: {e}")

        # Преобразуем результаты в упрощённые торговые записи
        trades = []
        for r in sorted(latest_results, key=lambda x: x.get('timestamp', ''), reverse=True):
            try:
                tr = r.get('trade_result') or {}
                decision = r.get('decision') or (r.get('parsed_result') or {}).get('action')
                action = None
                if isinstance(tr, dict):
                    # Явная метка из trade_result (hold или факт сделки)
                    action = tr.get('action') or tr.get('trade_executed')
                if not action:
                    action = decision
                if not action:
                    action = 'hold'
                action = str(action).lower()

                # Пропускаем чистые HOLD, чтобы история была осмысленной
                if action == 'hold':
                    continue

                # Время
                tstamp = r.get('timestamp')
                # Цена
                price = None
                order = tr.get('order') if isinstance(tr, dict) else None
                if isinstance(order, dict):
                    for key in ('price', 'average'):
                        v = order.get(key)
                        if v is not None:
                            try:
                                price = float(v)
                                break
                            except Exception:
                                pass
                    if price is None:
                        info = order.get('info') or {}
                        for key in ('avgPrice', 'lastPrice', 'orderPrice'):
                            v = info.get(key)
                            if v is not None:
                                try:
                                    price = float(v)
                                    break
                                except Exception:
                                    pass
                if price is None:
                    # Фолбэк: цена из parsed_result или отсутствует
                    parsed = r.get('parsed_result') or {}
                    v = parsed.get('price')
                    try:
                        price = float(v) if v is not None else None
                    except Exception:
                        price = None

                # Количество
                amount = None
                if action.startswith('sell') and isinstance(tr, dict):
                    amount = tr.get('sold_amount') or (tr.get('closed_position') or {}).get('amount')
                if amount is None and isinstance(tr, dict):
                    pos = tr.get('position') or tr.get('remaining_position') or tr.get('closed_position')
                    if isinstance(pos, dict):
                        amount = pos.get('amount')
                # Фолбэк
                if amount is None:
                    amount = r.get('trade_amount')

                # Сбор записи
                trades.append({
                    'time': tstamp,
                    'action': 'sell' if action.startswith('sell') else 'buy',
                    'price': price,
                    'amount': amount
                })
            except Exception:
                continue

        return jsonify({
            'success': True,
            'trades': trades,
            'total_trades': len(trades)
        }), 200

    except Exception as e:
        app.logger.error(f"Ошибка получения истории: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==== Конфигурация режима рынка (multi-window) через Redis ====
@app.route('/api/trading/regime_config', methods=['GET', 'POST'])
def regime_config():
    """GET — получить текущую конфигурацию мульти-окон/порогов, POST — сохранить.

    JSON структура (пример по умолчанию):
    {
      "windows": [60, 180, 300],
      "weights": [1, 1, 1],
      "voting": "majority",            # majority | weighted
      "tie_break": "flat",             # flat | longest
      "drift_threshold": 0.002,
      "flat_vol_threshold": 0.0025,
      "use_regression": false,
      "reg_slope_min": 0.0,
      "reg_r2_min": 0.2,
      "use_adx": false,
      "adx_period": 14,
      "adx_trend_min": 25
    }
    """
    try:
        import json as _json
        rc = get_redis_client()
        key = 'trading:regime_config'
        if request.method == 'POST':
            cfg = request.get_json(silent=True) or {}
            rc.set(key, _json.dumps(cfg, ensure_ascii=False))
            return jsonify({ 'success': True, 'saved': True, 'config': cfg })
        else:
            raw = rc.get(key)
            if raw:
                try:
                    js = _json.loads(raw)
                except Exception:
                    js = None
            else:
                js = None
            # Значения по умолчанию
            if not isinstance(js, dict):
                js = {
                    'windows': [60, 180, 300],
                    'weights': [1, 1, 1],
                    'voting': 'majority',
                    'tie_break': 'flat',
                    'drift_threshold': 0.002,
                    'flat_vol_threshold': 0.0025,
                    'use_regression': False,
                    'reg_slope_min': 0.0,
                    'reg_r2_min': 0.2,
                    'use_adx': False,
                    'adx_period': 14,
                    'adx_trend_min': 25
                }
            return jsonify({ 'success': True, 'config': js })
    except Exception as e:
        return jsonify({ 'success': False, 'error': str(e) }), 500


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

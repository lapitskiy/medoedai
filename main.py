#!/usr/bin/env python3
"""
🚀 MedoedAI - Flask веб-приложение для управления DQN торговым ботом

Запуск:
    python main.py              # Автоматический запуск
    flask run                   # Альтернатива
    FLASK_APP=main.py flask run # Альтернатива
"""

import requests # type: ignore
from flask import Flask, request, jsonify, render_template, current_app # type: ignore
from flask import redirect, url_for
from pathlib import Path

import redis # type: ignore

from tasks.celery_tasks import celery
from routes.bybit import bybit_bp
from routes.trading import trading_bp, get_matched_full_trades
from routes.models_admin import models_admin_bp
from routes.oos import oos_bp
from routes.training import training_bp
from routes.analysis_page import analytics_bp
from routes.sac import sac_bp
from utils.redis_utils import get_redis_client, clear_redis_on_startup

"""get_redis_client берём из utils.redis_utils"""

 
from celery.result import AsyncResult

import os
from utils.config_loader import get_config_value
import re
from tasks.celery_tasks import search_lstm_task, train_dqn, train_dqn_multi_crypto, train_dqn_symbol
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
import pickle

import docker # type: ignore

from tasks.celery_task_trade import start_trading_task
from utils.celery_utils import ensure_symbol_worker

logging.basicConfig(level=logging.INFO)

# Создаем Flask приложение
app = Flask(__name__)

# Регистрируем маршруты аналитики (анализ DQN)
from routes.analysis import analysis_api_bp
app.register_blueprint(analysis_api_bp)
app.register_blueprint(bybit_bp)
app.register_blueprint(trading_bp)
app.register_blueprint(models_admin_bp)
app.register_blueprint(training_bp, url_prefix='/training')
from routes.clean import clean_bp
app.register_blueprint(clean_bp)
app.register_blueprint(oos_bp)
app.register_blueprint(sac_bp)
app.register_blueprint(analytics_bp)

# Функция очистки Redis вынесена в utils.redis_utils.clear_redis_on_startup

# Инициализируем Redis клиент без очистки (decode_responses=True)
redis_client = get_redis_client()

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
                                manifest = json.loads(open(mf,'r',encoding='utf-8').read())
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

        results_dir = Path('result')
        # Нормализация пути
        req_norm = filename.replace('\\', '/')
        p = Path(req_norm)
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
                with open(train_file, 'rb') as f:
                    results = pickle.load(f)
                stats = results.get('final_stats', {}) or {}
                info['stats'] = {
                    'winrate': stats.get('winrate'),
                    'pl_ratio': stats.get('pl_ratio'),
                    'trades_count': stats.get('trades_count')
                }

                planned_episodes = results.get('episodes')
                actual_episodes = results.get('actual_episodes')
                info['episodes'] = actual_episodes if actual_episodes is not None else planned_episodes
                info['episodes_planned'] = planned_episodes
                info['actual_episodes'] = actual_episodes

                train_metadata = results.get('train_metadata') or {}
                info['seed'] = train_metadata.get('seed')
                info['cuda_available'] = train_metadata.get('cuda_available', False)
                info['gpu_name'] = train_metadata.get('gpu_name')
                info['total_training_time'] = results.get('total_training_time')
                if info['total_training_time'] and info['episodes']:
                    try:
                        info['avg_time_per_episode_sec'] = info['total_training_time'] / max(info['episodes'], 1)
                    except Exception:
                        info['avg_time_per_episode_sec'] = None
                else:
                    info['avg_time_per_episode_sec'] = None

                info['train_metadata'] = {
                    'created_at_utc': train_metadata.get('created_at_utc'),
                    'hostname': train_metadata.get('hostname'),
                    'platform': train_metadata.get('platform'),
                    'python_version': train_metadata.get('python_version'),
                    'torch_version': train_metadata.get('torch_version'),
                    'cuda_available': train_metadata.get('cuda_available'),
                    'gpu_name': train_metadata.get('gpu_name'),
                    'git_commit': train_metadata.get('git_commit'),
                    'seed': train_metadata.get('seed'),
                }

                # Добавляем длину эпизода, если она сохранена в gym_snapshot
                gym_snapshot = results.get('gym_snapshot', {}) or {}
                if 'episode_length' in gym_snapshot:
                    info['episode_length'] = gym_snapshot['episode_length']
                elif 'cfg_snapshot' in results and isinstance(results['cfg_snapshot'], dict):
                    info['episode_length'] = results['cfg_snapshot'].get('episode_length')

                # Пути к артефактам
                info['model_path'] = results.get('model_path') or info.get('model_file')
                info['buffer_path'] = results.get('buffer_path')

                weights = results.get('weights') or {}
                info['weights'] = {
                    'model_path': weights.get('model_path'),
                    'model_sha256': weights.get('model_sha256'),
                    'buffer_path': weights.get('buffer_path'),
                    'buffer_sha256': weights.get('buffer_sha256'),
                }

                # Сводная статистика валидации и winrate по эпизодам
                def _sanitize_sequence(seq):
                    if not isinstance(seq, (list, tuple)):
                        return []
                    clean = []
                    for value in seq:
                        if isinstance(value, (int, float)) and value == value:
                            clean.append(float(value))
                    return clean

                episode_winrates = _sanitize_sequence(results.get('episode_winrates'))
                best_winrate = results.get('best_winrate')
                if best_winrate is None and episode_winrates:
                    best_winrate = max(episode_winrates)
                info['best_winrate'] = best_winrate
                if episode_winrates:
                    info['episode_winrate_summary'] = {
                        'count': len(episode_winrates),
                        'last': episode_winrates[-1],
                        'best': max(episode_winrates),
                        'avg': sum(episode_winrates) / len(episode_winrates),
                    }

                validation_rewards = _sanitize_sequence(results.get('validation_rewards'))
                if validation_rewards:
                    info['validation_summary'] = {
                        'count': len(validation_rewards),
                        'last': validation_rewards[-1],
                        'best': max(validation_rewards),
                        'avg': sum(validation_rewards) / len(validation_rewards),
                    }

            except Exception as e:
                app.logger.warning(f"get_result_model_info: не удалось загрузить train_file {train_file}: {e}")
                app.logger.warning(f"get_result_model_info: Путь к train_file: {train_file.resolve()}")

        return jsonify(info)
    except Exception as e:
        app.logger.error(f"Ошибка в get_result_model_info: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

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


@app.route('/models')
def models_page():
    """Страница управления моделями"""
    return render_template('models.html')

@app.route('/create_model_version', methods=['POST'])
def create_model_version():
    """Создает новую версию модели с уникальным ID"""
    import shutil
    import uuid
    from datetime import datetime
    from pathlib import Path
    import traceback
    
    try:
        # Читаем символ из запроса (опционально)
        try:
            data = request.get_json(silent=True) or {}
        except Exception:
            data = {}
        requested_symbol = (data.get('symbol') or '').strip()
        requested_file = (data.get('file') or '').strip()
        requested_ensemble = (data.get('ensemble') or 'ensemble-a').strip() or 'ensemble-a'
        
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
        selected_model_file = None
        selected_replay_file = None
        run_dir_path = None
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

            if inside_result and cand_resolved.exists() and cand_resolved.is_file():
                run_dir = cand_resolved.parent
                run_dir_path = run_dir
                name_low = cand_resolved.name.lower()
                # 1) Если выбрали модель .pth из папки run — ищем рядом replay и results
                if cand_resolved.suffix == '.pth':
                    # принимаем любой *.pth как модель
                    selected_model_file = cand_resolved
                    for f in run_dir.iterdir():
                        if not f.is_file():
                            continue
                        n = f.name.lower()
                        if n.endswith('.pkl') and (n.startswith('replay_buffer') or 'replay' in n):
                            selected_replay_file = selected_replay_file or f
                        elif n.endswith('.pkl') and (n.startswith('train_result') or 'result' in n):
                            selected_result_file = selected_result_file or f
                    # Если каких-то файлов нет — продолжаем, но предупредим
                    if not (selected_replay_file and selected_result_file):
                        pass
                    try:
                        if selected_result_file is not None:
                            fname = selected_result_file.stem
                            parts = fname.split('_', 2)
                            if len(parts) >= 3:
                                base_code = parts[2].lower()
                    except Exception:
                        pass
                # 2) Если выбрали train_result_*.pkl — валидируем и ищем рядом .pth и replay
                elif cand_resolved.suffix == '.pkl' and name_low.startswith('train_result_'):
                    selected_result_file = cand_resolved
                    for f in run_dir.iterdir():
                        if not f.is_file():
                            continue
                        n = f.name.lower()
                        if n.endswith('.pth'):
                            selected_model_file = selected_model_file or f
                        elif n.endswith('.pkl') and (n.startswith('replay_buffer') or 'replay' in n):
                            selected_replay_file = selected_replay_file or f
                    if not selected_model_file:
                        return jsonify({
                            "success": False,
                            "error": "Не найдена модель *.pth рядом с файлом результатов"
                        })
                    try:
                        fname = selected_result_file.stem
                        parts = fname.split('_', 2)
                        if len(parts) >= 3:
                            base_code = parts[2].lower()
                    except Exception:
                        pass
                else:
                    return jsonify({
                        "success": False,
                        "error": "Ожидался .pth или train_result_*.pkl внутри result/"
                    })
            else:
                return jsonify({
                    "success": False,
                    "error": "Неверный путь к файлу внутри result/"
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
        
        # Определяем пути источников
        if selected_model_file and selected_replay_file and selected_result_file:
            model_file = selected_model_file
            replay_file = selected_replay_file
        else:
            model_file = result_dir / f'dqn_model_{base_code}.pth'
            replay_file = result_dir / f'replay_buffer_{base_code}.pkl'
        
        if not model_file.exists():
            return jsonify({
                "success": False,
                "error": f"Файл {model_file.name} не найден в result/"
            })
        # replay и results могут отсутствовать — это не критично
        
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
            
            # Копируем артефакты версии (сохраняем оригинальные имена файлов)
            ver_model = version_dir / model_file.name
            ver_replay = version_dir / (replay_file.name if replay_file.exists() else 'replay_buffer.pkl')
            ver_result = version_dir / (selected_result_file.name if (selected_result_file and selected_result_file.exists()) else 'train_result.pkl')
            shutil.copy2(model_file, ver_model)
            try:
                if replay_file.exists():
                    shutil.copy2(replay_file, ver_replay)
            except Exception:
                pass
            try:
                if selected_result_file and selected_result_file.exists():
                    shutil.copy2(selected_result_file, ver_result)
            except Exception:
                pass
            
            # Пишем manifest.yaml (без зависимости от PyYAML)
            manifest_path = version_dir / 'manifest.yaml'
            try:
                # Попытаемся извлечь краткую статистику
                stats_brief = {}
                if ver_result.exists():
                    try:
                        import pickle as _pickle
                        with open(ver_result, 'rb') as _f:
                            _res = _pickle.load(_f)
                            if isinstance(_res, dict) and 'final_stats' in _res:
                                stats_brief = _res['final_stats'] or {}
                    except Exception:
                        pass
                created_ts = _dt.utcnow().isoformat()
                # Используем run_id из исходной папки, если она известна
                manifest_id = (str(run_dir_path.name) if run_dir_path is not None else named_id)
                yaml_text = (
                    'id: "' + manifest_id + '"\n'
                    'symbol: "' + base_code.lower() + '"\n'
                    'ensemble: "' + ensemble_name + '"\n'
                    'version: "' + version_name + '"\n'
                    'created_at: "' + created_ts + '"\n'
                    'run_id: "' + manifest_id + '"\n'
                    + (('source_run_path: "' + str(run_dir_path).replace('\\','/') + '"\n') if run_dir_path is not None else '')
                    + 'files:\n'
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
            "model_id": (run_dir_path.name if run_dir_path is not None else named_id),
            "files": [
                ver_model.name,
                ver_replay.name if ver_replay.exists() else None,
                ver_result.name if ver_result.exists() else None
            ]
        })
        
    except Exception as e:
        print("[create_model_version] ERROR:\n" + traceback.format_exc())
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
                    # Фолбэки, если имена не по шаблону
                    fallback_model = None
                    fallback_replay = None
                    fallback_results = None
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
                        # Сохраняем фолбэки
                        elif f.suffix == '.pth' and fallback_model is None:
                            fallback_model = f.name
                        elif f.suffix == '.pkl':
                            n = f.name.lower()
                            if ('replay' in n) and fallback_replay is None:
                                fallback_replay = f.name
                            if (('train_result' in n) or ('result' in n)) and fallback_results is None:
                                fallback_results = f.name
                    # Пытаемся вытащить краткую статистику из train_result_*.pkl
                    try:
                        # Выбор файла результатов с учетом фолбэков
                        res_name = files.get('results') or fallback_results
                        if res_name:
                            _res_path = vdir / res_name
                            if _res_path.exists():
                                with open(_res_path, 'rb') as _f:
                                    _res = _pickle.load(_f)
                                    if isinstance(_res, dict) and 'final_stats' in _res:
                                        stats = _res['final_stats'] or {}
                    except Exception:
                        stats = {}
                    # Применяем фолбэки, если строгие имена не найдены
                    if files['model'] is None and fallback_model is not None:
                        files['model'] = fallback_model
                    if files['replay'] is None and fallback_replay is not None:
                        files['replay'] = fallback_replay
                    if files['results'] is None and fallback_results is not None:
                        files['results'] = fallback_results
                    
                    # Читаем ID из манифеста, если он есть
                    manifest_id = None
                    if manifest:
                        try:
                            manifest_path = vdir / manifest
                            if manifest_path.exists():
                                with open(manifest_path, 'r', encoding='utf-8') as mf:
                                    manifest_content = mf.read()
                                    # Ищем строку с id: "значение"
                                    for line in manifest_content.split('\n'):
                                        if line.strip().startswith('id:'):
                                            manifest_id = line.split(':', 1)[1].strip().strip('"\'')
                                            break
                        except Exception:
                            pass
                    
                    versions.append({
                        'version': vdir.name,
                        'files': files,
                        'manifest': manifest,
                        'manifest_id': manifest_id,
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

@app.route('/api/predictions/recent')
def get_recent_predictions():
    """API для получения последних предсказаний модели"""
    try:
        symbol = request.args.get('symbol')
        action = request.args.get('action')
        limit = int(request.args.get('limit', 50))
        
        # 1) Основной путь: берём из БД
        predictions = get_model_predictions(symbol=symbol, action=action, limit=limit)

        # 2) Нормализация символа (fallback): BTCUSDT <-> BTC/USDT
        if (not predictions) and symbol:
            try:
                alt = None
                if '/' in symbol:
                    alt = symbol.replace('/', '')
                else:
                    # вставим слэш перед USDT/USDC/FDUSD/DAI и т.п. (по умолчанию USDT)
                    if symbol.upper().endswith('USDT'):
                        alt = symbol[:-4] + '/' + symbol[-4:]
                if alt and alt != symbol:
                    alt_predictions = get_model_predictions(symbol=alt, action=action, limit=limit)
                    if alt_predictions:
                        predictions = alt_predictions
                        symbol = alt  # сообщим в ответе фактический символ
            except Exception:
                pass
        
        # Преобразуем в JSON-совместимый формат
        predictions_data = []
        for prediction in predictions or []:
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
        
        # 3) Если БД пуста — отдаём последний снимок из Redis (результаты celery-trade)
        if len(predictions_data) == 0:
            try:
                from utils.redis_utils import get_redis_client as _get_rc
                _rc = _get_rc()
                keys = sorted(_rc.keys('trading:latest_result_*') or [], reverse=True)
                for k in keys:
                    try:
                        raw = _rc.get(k)
                        if not raw:
                            continue
                        snap = json.loads(raw)
                        preds = snap.get('predictions') or []
                        if not preds:
                            continue
                        ts = snap.get('timestamp')
                        # В результирующих предсказаниях символ не хранится — подставим запрошенный
                        sym_for_resp = symbol or (snap.get('symbols') or [None])[0]
                        for p in preds[:limit]:
                            predictions_data.append({
                                'id': None,
                                'timestamp': ts,
                                'symbol': sym_for_resp,
                                'action': p.get('action'),
                                'q_values': p.get('q_values') or [],
                                'current_price': None,
                                'position_status': None,
                                'confidence': p.get('confidence'),
                                'model_path': p.get('model_path'),
                                'market_conditions': {},
                                'created_at': ts
                            })
                        if predictions_data:
                            break
                    except Exception:
                        continue
            except Exception:
                pass

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

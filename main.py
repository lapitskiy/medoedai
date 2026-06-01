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
from datetime import datetime, timedelta

import redis # type: ignore

from tasks.celery_tasks import celery
from routes.bybit import bybit_bp
from routes.trading import trading_bp, get_matched_full_trades
from routes.settings import settings_bp
from routes.models_admin import models_admin_bp
from routes.oos import oos_bp
from routes.xgb_oos import xgb_oos_bp
from routes.training import training_bp
from routes.analysis_page import analytics_bp
from routes.sac import sac_bp
from routes.xgb import xgb_bp
from routes.atr import atr_bp
from routes.llm import llm_bp
from utils.redis_utils import get_redis_client, clear_redis_on_startup
from routes.system_models import system_models_bp
from routes.stock_models import stock_models_bp

"""get_redis_client берём из utils.redis_utils"""

 
from celery.result import AsyncResult

import os
from utils.config_loader import get_config_value
import re
from tasks.celery_tasks import search_lstm_task, train_dqn, train_dqn_multi_crypto, train_dqn_symbol
from utils.db_utils import load_latest_candles_from_csv_to_db
from utils.parser import parser_download_and_combine_with_library
from utils.time_log import msk_tag
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
import math

import time

import glob
import os
import pickle

import docker # type: ignore

from tasks.celery_task_trade import start_trading_task
from utils.celery_utils import ensure_symbol_worker

def _msk_time_converter(ts: float):
    # Force app log timestamps to MSK regardless of container TZ.
    return (datetime.utcfromtimestamp(ts) + timedelta(hours=3)).timetuple()


logging.Formatter.converter = staticmethod(_msk_time_converter)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

# Создаем Flask приложение
app = Flask(__name__)

# Регистрируем маршруты аналитики (анализ DQN)
from routes.analysis import analysis_api_bp
app.register_blueprint(analysis_api_bp)
app.register_blueprint(bybit_bp)
app.register_blueprint(trading_bp)
app.register_blueprint(settings_bp)
app.register_blueprint(models_admin_bp)
app.register_blueprint(training_bp, url_prefix='/training')
from routes.clean import clean_bp
app.register_blueprint(clean_bp)
app.register_blueprint(system_models_bp)
app.register_blueprint(oos_bp)
app.register_blueprint(xgb_oos_bp)
app.register_blueprint(sac_bp)
app.register_blueprint(xgb_bp)
app.register_blueprint(analytics_bp)
app.register_blueprint(atr_bp)
app.register_blueprint(llm_bp)
app.register_blueprint(stock_models_bp)
from routes.trade_optimizer import trade_opt_bp
app.register_blueprint(trade_opt_bp)

# Функция очистки Redis вынесена в utils.redis_utils.clear_redis_on_startup

# Инициализируем Redis клиент без очистки (decode_responses=True)
redis_client = get_redis_client()

@app.before_request
def log_request_info():
    # По умолчанию шумные логи отключены. Включить: UI_LOG_REQUESTS=1 (и опционально UI_LOG_HEADERS=1)
    try:
        if str(os.environ.get('UI_LOG_REQUESTS', '0')).lower() not in ('1', 'true', 'yes', 'on'):
            return
        path = request.path or ''
        # Фильтруем статические и шумные частые запросы UI
        noisy_prefixes = (
            '/static/',
            '/api/runs/',
            '/api/encoders',
        )
        noisy_exact = {
            '/api/trading/status_all',
            '/list_result_models',
            '/get_result_model_info',
        }
        if path.startswith(noisy_prefixes) or path in noisy_exact:
            return
        logging.info(f"Request from {request.remote_addr}: {request.method} {path}")
        if str(os.environ.get('UI_LOG_HEADERS', '0')).lower() in ('1', 'true', 'yes', 'on'):
            logging.info("Headers: %s", dict(request.headers))
    except Exception:
        pass

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
        run_dir = None
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

        # Пробуем прочитать направление (long/short) из manifest.json рядом с моделью
        try:
            manifest_path = None
            if run_dir is not None:
                manifest_path = run_dir / 'manifest.json'
            else:
                # Для плоских файлов dqn_model_*.pth найти manifest проблематично — пропускаем
                manifest_path = None
            if manifest_path and manifest_path.exists():
                with open(manifest_path, 'r', encoding='utf-8') as mf:
                    mf_data = json.load(mf)
                # Пытаемся взять явное поле direction, иначе trained_as
                direction = (mf_data.get('direction') or mf_data.get('trained_as') or '').strip().lower()
                if direction in ('long', 'short'):
                    info['direction'] = direction
                # Тип обучения: с нуля или continue (по parent_run_id)
                parent_run_id = mf_data.get('parent_run_id')
                info['parent_run_id'] = parent_run_id
                info['model_training_type'] = 'continued' if parent_run_id else 'from_scratch'
        except Exception:
            pass

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

                # GPU профиль (из agents/vdqn/cfg/gpu_configs.py) — чтобы показать в /analitika
                try:
                    gpu_name = str(train_metadata.get('gpu_name') or '').strip().lower()
                    gpu_key = None
                    if 'v100' in gpu_name:
                        gpu_key = 'tesla_v100'
                    elif 'p100' in gpu_name:
                        gpu_key = 'tesla_p100'
                    elif 'gtx 1660' in gpu_name or '1660 super' in gpu_name:
                        gpu_key = 'gtx_1660_super'
                    elif 'rtx 3080' in gpu_name:
                        gpu_key = 'rtx_3080'
                    elif 'rtx 4090' in gpu_name:
                        gpu_key = 'rtx_4090'
                    else:
                        gpu_key = None

                    if gpu_key:
                        from agents.vdqn.cfg.gpu_configs import GPU_CONFIGS  # type: ignore
                        cfgp = GPU_CONFIGS.get(gpu_key)
                        if cfgp:
                            info['gpu_config_profile'] = {
                                'key': gpu_key,
                                'name': getattr(cfgp, 'name', None),
                                'vram_gb': getattr(cfgp, 'vram_gb', None),
                                'batch_size': getattr(cfgp, 'batch_size', None),
                                'memory_size': getattr(cfgp, 'memory_size', None),
                                'hidden_sizes': list(getattr(cfgp, 'hidden_sizes', []) or []),
                                'train_repeats': getattr(cfgp, 'train_repeats', None),
                                'use_amp': getattr(cfgp, 'use_amp', None),
                                'use_gpu_storage': getattr(cfgp, 'use_gpu_storage', None),
                                'learning_rate': getattr(cfgp, 'learning_rate', None),
                                'use_torch_compile': getattr(cfgp, 'use_torch_compile', None),
                                'eps_decay_steps': getattr(cfgp, 'eps_decay_steps', None),
                                'dropout_rate': getattr(cfgp, 'dropout_rate', None),
                            }
                except Exception:
                    pass

                # Добавляем длину эпизода, если она сохранена в gym_snapshot
                gym_snapshot = results.get('gym_snapshot', {}) or {}
                if 'episode_length' in gym_snapshot:
                    info['episode_length'] = gym_snapshot['episode_length']

                # Risk-management параметры из gym_snapshot (TP/SL/min_hold/volume_threshold)
                risk_snap = gym_snapshot.get('risk_management') if isinstance(gym_snapshot.get('risk_management'), dict) else {}
                if risk_snap:
                    info['risk_management'] = risk_snap
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

                # Информация об энкодере (для UI /analitika)
                try:
                    cfg_snapshot = results.get('cfg_snapshot') or {}
                    encoder_path = cfg_snapshot.get('encoder_path') or weights.get('encoder_path') or results.get('encoder_path')
                    freeze_encoder = cfg_snapshot.get('freeze_encoder', None)
                    enc_version = cfg_snapshot.get('encoder_version') or results.get('encoder_version')
                    enc_type = cfg_snapshot.get('encoder_type') or results.get('encoder_type')
                    if encoder_path:
                        enc_info = {
                            'path': str(encoder_path),
                            'frozen': bool(freeze_encoder) if freeze_encoder is not None else None,
                            'type': str(enc_type) if enc_type is not None else None,
                            'version': enc_version,
                        }
                        try:
                            import re
                            m = re.search(r'/v(\d+)/', str(encoder_path).replace('\\', '/'))
                            if m:
                                enc_info['version_str'] = f"v{m.group(1)}"
                        except Exception:
                            pass
                        info['encoder_info'] = enc_info
                except Exception:
                    pass

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
                # Fallback: если полный список winrate не сохранён, используем агрегаты из train_result.pkl
                if not episode_winrates:
                    tail = _sanitize_sequence(results.get('episode_winrates_tail'))
                    if tail:
                        episode_winrates = tail
                best_winrate = results.get('best_winrate')
                if best_winrate is None:
                    bw = results.get('episode_winrates_best')
                    if isinstance(bw, (int, float)) and bw == bw:
                        best_winrate = float(bw)
                    elif episode_winrates:
                        best_winrate = max(episode_winrates)
                info['best_winrate'] = best_winrate
                count_val = results.get('episode_winrates_count')
                if episode_winrates or isinstance(count_val, int):
                    info['episode_winrate_summary'] = {
                        'count': int(count_val) if isinstance(count_val, int) else len(episode_winrates),
                        'last': (results.get('episode_winrates_last') if not episode_winrates else episode_winrates[-1]),
                        'best': (results.get('episode_winrates_best') if results.get('episode_winrates_best') is not None else (max(episode_winrates) if episode_winrates else None)),
                        'avg': (results.get('episode_winrates_avg') if results.get('episode_winrates_avg') is not None else ((sum(episode_winrates) / len(episode_winrates)) if episode_winrates else None)),
                    }
                # Полный/частичный ряд winrate по эпизодам для графика на /analitika
                try:
                    MAX_POINTS = 2000
                    if episode_winrates:
                        if len(episode_winrates) > MAX_POINTS:
                            info['episode_winrates_offset'] = int(len(episode_winrates) - MAX_POINTS)
                            info['episode_winrates'] = episode_winrates[-MAX_POINTS:]
                        else:
                            info['episode_winrates_offset'] = 0
                            info['episode_winrates'] = episode_winrates
                except Exception:
                    pass

                # Компактный тренд winrate (снэпшоты/EMA/квантили), если он есть в train_result.pkl
                try:
                    wt = results.get('winrate_trend')
                    if isinstance(wt, dict) and wt:
                        info['winrate_trend'] = wt
                except Exception:
                    pass

                # Epsilon по эпизодам (для графика exploration→exploitation)
                try:
                    eps_seq = _sanitize_sequence(results.get('episode_epsilons'))
                    if eps_seq:
                        MAX_EPS = 5000
                        if len(eps_seq) > MAX_EPS:
                            info['episode_epsilons_offset'] = int(len(eps_seq) - MAX_EPS)
                            info['episode_epsilons'] = eps_seq[-MAX_EPS:]
                        else:
                            info['episode_epsilons_offset'] = 0
                            info['episode_epsilons'] = eps_seq
                except Exception:
                    pass

                # Статистика действий (сколько было HOLD/BUY/SELL) — как "state counts"
                try:
                    raw_actions = results.get('action_counts_total') or {}
                    actions_norm = {}
                    if isinstance(raw_actions, dict):
                        for k, v in raw_actions.items():
                            try:
                                kk = str(k)
                                actions_norm[kk] = int(v)
                            except Exception:
                                continue
                    if actions_norm:
                        info['action_counts_total'] = actions_norm
                except Exception:
                    pass

                # BUY/HOLD фильтры и churn-метрики (для /analitika)
                try:
                    def _sanitize_int_dict(d):
                        out = {}
                        if not isinstance(d, dict):
                            return out
                        for k, v in d.items():
                            try:
                                out[str(k)] = int(v)
                            except Exception:
                                continue
                        return out

                    buy_stats_total = _sanitize_int_dict(results.get('buy_stats_total') or {})
                    hold_stats_total = _sanitize_int_dict(results.get('hold_stats_total') or {})
                    if buy_stats_total:
                        info['buy_stats_total'] = buy_stats_total
                    if hold_stats_total:
                        info['hold_stats_total'] = hold_stats_total

                    buy_accept_rate = results.get('buy_accept_rate')
                    if isinstance(buy_accept_rate, (int, float)) and buy_accept_rate == buy_accept_rate:
                        info['buy_accept_rate'] = float(buy_accept_rate)
                    avg_minutes_between_buys = results.get('avg_minutes_between_buys')
                    if isinstance(avg_minutes_between_buys, (int, float)) and avg_minutes_between_buys == avg_minutes_between_buys:
                        info['avg_minutes_between_buys'] = float(avg_minutes_between_buys)

                    # churn: trades / episode (если episodes известны)
                    try:
                        ep_n = info.get('episodes')
                        tr_n = info.get('stats', {}).get('trades_count')
                        if isinstance(ep_n, (int, float)) and isinstance(tr_n, (int, float)) and float(ep_n) > 0:
                            info['trades_per_episode'] = float(tr_n) / float(ep_n)
                    except Exception:
                        pass
                except Exception:
                    pass

                # Market state (NORMAL/HIGH_VOL/PANIC/DRAWDOWN) — распределение состояний в обучении
                try:
                    raw_ms = results.get('market_state_counts_total') or {}
                    ms_norm = {}
                    if isinstance(raw_ms, dict):
                        for k, v in raw_ms.items():
                            try:
                                ms_norm[str(k)] = int(v)
                            except Exception:
                                continue
                    if ms_norm:
                        info['market_state_counts_total'] = ms_norm
                except Exception:
                    pass

                # Market state за последний эпизод (если есть)
                try:
                    raw_ms_ep = results.get('market_state_counts_episode') or {}
                    ms_ep_norm = {}
                    if isinstance(raw_ms_ep, dict):
                        for k, v in raw_ms_ep.items():
                            try:
                                ms_ep_norm[str(k)] = int(v)
                            except Exception:
                                continue
                    if ms_ep_norm:
                        info['market_state_counts_episode'] = ms_ep_norm
                except Exception:
                    pass

                # Validation rewards (если сохранялись) — ряд для графика
                try:
                    validation_rewards = _sanitize_sequence(results.get('validation_rewards'))
                    if validation_rewards:
                        MAX_VAL = 5000
                        if len(validation_rewards) > MAX_VAL:
                            info['validation_rewards_offset'] = int(len(validation_rewards) - MAX_VAL)
                            info['validation_rewards'] = validation_rewards[-MAX_VAL:]
                        else:
                            info['validation_rewards_offset'] = 0
                            info['validation_rewards'] = validation_rewards
                except Exception:
                    validation_rewards = []

                if validation_rewards:
                    info['validation_summary'] = {
                        'count': len(validation_rewards),
                        'last': validation_rewards[-1],
                        'best': max(validation_rewards),
                        'avg': sum(validation_rewards) / len(validation_rewards),
                    }

                # --- Equity curve + histogram по сделкам (all_trades) ---
                try:
                    trades = results.get('all_trades') or []
                    if (not trades) and isinstance(results.get('all_trades_path'), str):
                        p = results.get('all_trades_path')
                        if p and os.path.exists(p):
                            trades = json.loads(open(p, 'r', encoding='utf-8').read()) or []
                    # fallback: all_trades.json рядом с train_result.pkl (для старых/битых версий)
                    if (not trades):
                        try:
                            sib = train_file.parent / 'all_trades.json'
                            if sib.exists():
                                trades = json.loads(sib.read_text(encoding='utf-8')) or []
                        except Exception:
                            pass
                    rois = []
                    # New compact series written by trainer (preferred for /analitika)
                    try:
                        _roi = results.get('trades_roi')
                        if isinstance(_roi, list) and _roi:
                            for v in _roi:
                                try:
                                    fv = float(v)
                                    if fv == fv:
                                        rois.append(fv)
                                except Exception:
                                    continue
                    except Exception:
                        pass
                    if isinstance(trades, list) and trades:
                        for t in trades:
                            if not isinstance(t, dict):
                                continue
                            v = t.get('roi', None)
                            try:
                                fv = float(v)
                                if fv == fv:  # not NaN
                                    rois.append(fv)
                            except Exception:
                                continue
                    if rois:
                        # equity as cumulative ROI (percent)
                        eq = []
                        underwater = []
                        c = 0.0
                        peak = -1e9
                        for i, r in enumerate(rois):
                            c += float(r)
                            v = c * 100.0
                            if v > peak:
                                peak = v
                            dd = v - peak  # <= 0
                            eq.append((i + 1, v))
                            underwater.append((i + 1, dd))
                        # downsample equity to max points
                        MAX_EQ = 2000
                        if len(eq) > MAX_EQ:
                            step = max(1, int(len(eq) / MAX_EQ))
                            idxs = list(range(0, len(eq), step))
                            if idxs and idxs[-1] != (len(eq) - 1):
                                idxs.append(len(eq) - 1)
                            eq = [eq[i] for i in idxs][: MAX_EQ + 1]
                            underwater = [underwater[i] for i in idxs][: MAX_EQ + 1]
                        info['equity_curve'] = [{'i': int(i), 'v': float(v)} for (i, v) in eq]
                        info['underwater_curve'] = [{'i': int(i), 'v': float(v)} for (i, v) in underwater]

                        # Rolling metrics (last N trades): avg ROI% + profit factor
                        try:
                            N = int(results.get('rolling_window_trades') or 50)
                            N = max(10, min(500, N))
                        except Exception:
                            N = 50
                        try:
                            roll_avg = []
                            roll_pf = []
                            pos_sum = 0.0
                            neg_sum = 0.0
                            # sliding window sums
                            for i in range(len(rois)):
                                r = float(rois[i])
                                if r >= 0:
                                    pos_sum += r
                                else:
                                    neg_sum += -r
                                if i >= N:
                                    old = float(rois[i - N])
                                    if old >= 0:
                                        pos_sum -= old
                                    else:
                                        neg_sum -= -old
                                if i + 1 >= N:
                                    window = rois[i - N + 1:i + 1]
                                    avg = (float(sum(window)) / float(N)) * 100.0
                                    pf = (pos_sum / neg_sum) if neg_sum > 1e-12 else None
                                    roll_avg.append((i + 1, avg))
                                    if pf is not None and pf == pf:
                                        roll_pf.append((i + 1, float(pf)))
                            # downsample rolling
                            MAX_R = 2000
                            if len(roll_avg) > MAX_R:
                                step = max(1, int(len(roll_avg) / MAX_R))
                                roll_avg = [roll_avg[i] for i in range(0, len(roll_avg), step)]
                            if len(roll_pf) > MAX_R:
                                step = max(1, int(len(roll_pf) / MAX_R))
                                roll_pf = [roll_pf[i] for i in range(0, len(roll_pf), step)]
                            if roll_avg:
                                info['rolling_window_trades'] = int(N)
                                info['rolling_avg_roi'] = [{'i': int(i), 'v': float(v)} for (i, v) in roll_avg]
                            if roll_pf:
                                info['rolling_profit_factor'] = [{'i': int(i), 'v': float(v)} for (i, v) in roll_pf]
                        except Exception:
                            pass

                        # histogram of ROI (percent) with fixed bins
                        import numpy as _np  # type: ignore
                        arr = _np.asarray([float(x) * 100.0 for x in rois], dtype=_np.float32)
                        # robust range (clip extreme tails)
                        lo = float(_np.quantile(arr, 0.01))
                        hi = float(_np.quantile(arr, 0.99))
                        if hi <= lo:
                            lo = float(arr.min())
                            hi = float(arr.max())
                        # ensure some width
                        if hi - lo < 1e-6:
                            lo -= 0.1
                            hi += 0.1
                        bins = 60
                        counts, edges = _np.histogram(arr, bins=bins, range=(lo, hi))
                        centers = (edges[:-1] + edges[1:]) / 2.0
                        info['roi_histogram'] = {
                            'lo': float(lo),
                            'hi': float(hi),
                            'bins': int(bins),
                            'centers': [float(x) for x in centers.tolist()],
                            'counts': [int(x) for x in counts.tolist()],
                        }
                except Exception:
                    pass

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
                elif cand_resolved.suffix == '.pkl' and (name_low.startswith('train_result_') or name_low == 'train_result.pkl'):
                    selected_result_file = cand_resolved
                    for f in run_dir.iterdir():
                        if not f.is_file():
                            continue
                        n = f.name.lower()
                        # Новая структура: веса называются model.pth; также поддержим любые *.pth
                        if n == 'model.pth' or n.endswith('.pth'):
                            selected_model_file = selected_model_file or f
                        # Новая структура: буфер называется replay.pkl; также поддержим старые имена
                        elif n == 'replay.pkl' or (n.endswith('.pkl') and (n.startswith('replay_buffer') or 'replay' in n)):
                            selected_replay_file = selected_replay_file or f
                    if not selected_model_file:
                        return jsonify({
                            "success": False,
                            "error": "Не найдена модель *.pth рядом с файлом результатов"
                        })
                    try:
                        # Попробуем извлечь base_code из родительского символа, если файл без суффикса
                        if name_low == 'train_result.pkl':
                            # Ожидаем путь вида result/<agent>/<SYMBOL>/runs/<run_id>/train_result.pkl
                            parts = list(run_dir.parts)
                            # Найдём индекс 'runs' и возьмём символ перед ним
                            if 'runs' in parts:
                                idx = parts.index('runs')
                                if idx >= 1:
                                    base_code = parts[idx-1].lower()
                        else:
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
        
        # Определяем пути источников (гибкая логика под новую структуру runs)
        # 1) Модель: сначала выбранная, затем model.pth в run, затем любой *.pth в run, затем старый dqn_model_<code>.pth в result/
        model_file = None
        candidates_model = []
        if 'selected_model_file' in locals() and selected_model_file is not None:
            candidates_model.append(selected_model_file)
        if run_dir_path is not None:
            candidates_model.append(run_dir_path / 'model.pth')
            try:
                for f in run_dir_path.iterdir():
                    if f.is_file() and f.suffix.lower() == '.pth' and f not in candidates_model:
                        candidates_model.append(f)
            except Exception:
                pass
        candidates_model.append(result_dir / f'dqn_model_{base_code}.pth')
        for c in candidates_model:
            try:
                if c.exists() and c.is_file():
                    model_file = c
                    break
            except Exception:
                continue
        if model_file is None:
            return jsonify({
                "success": False,
                "error": f"Не найден файл модели (.pth) ни в {run_dir_path or 'result/runs'}, ни в result/",
            })

        # 2) Буфер: выбранный, затем replay.pkl в run, затем старый replay_buffer_<code>.pkl в result/
        replay_file = None
        if 'selected_replay_file' in locals() and selected_replay_file is not None and selected_replay_file.exists():
            replay_file = selected_replay_file
        elif run_dir_path is not None and (run_dir_path / 'replay.pkl').exists():
            replay_file = run_dir_path / 'replay.pkl'
        else:
            replay_file = result_dir / f'replay_buffer_{base_code}.pkl'
        # replay и results могут отсутствовать — это не критично
        try:
            print(f"[create_model_version] SOURCES | base_code={base_code} | model={model_file} | replay={replay_file if 'replay_file' in locals() else None} | result={selected_result_file}")
        except Exception:
            pass
        
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
            
            # Копируем артефакты версии с унифицированными именами, чтобы список моделей корректно находил файлы
            try:
                print(f"[create_model_version] DEST | version_dir={version_dir} | files: model=dqn_model_{base_code}.pth, replay=replay_buffer_{base_code}.pkl, result=train_result_{base_code}.pkl")
            except Exception:
                pass
            ver_model = version_dir / f'dqn_model_{base_code}.pth'
            ver_replay = version_dir / f'replay_buffer_{base_code}.pkl'
            ver_result = version_dir / f'train_result_{base_code}.pkl'
            try:
                shutil.copy2(model_file, ver_model)
                try:
                    print(f"[create_model_version] COPY OK model: {model_file} -> {ver_model}")
                except Exception:
                    pass
            except Exception as _copy_err:
                try:
                    print(f"[create_model_version] COPY FAIL model: {model_file} -> {ver_model} | err={_copy_err}")
                except Exception:
                    pass
                return jsonify({
                    "success": False,
                    "error": f"Не удалось скопировать модель: {str(_copy_err)}",
                    "src": str(model_file),
                    "dst": str(ver_model)
                }), 500
            try:
                if replay_file.exists():
                    shutil.copy2(replay_file, ver_replay)
                    try:
                        print(f"[create_model_version] COPY OK replay: {replay_file} -> {ver_replay}")
                    except Exception:
                        pass
            except Exception:
                try:
                    print(f"[create_model_version] COPY SKIP/ERR replay: {replay_file} -> {ver_replay}")
                except Exception:
                    pass
            try:
                if selected_result_file and selected_result_file.exists():
                    shutil.copy2(selected_result_file, ver_result)
                    try:
                        print(f"[create_model_version] COPY OK result: {selected_result_file} -> {ver_result}")
                    except Exception:
                        pass
            except Exception:
                try:
                    print(f"[create_model_version] COPY SKIP/ERR result: {selected_result_file} -> {ver_result}")
                except Exception:
                    pass

            # Дополнительно копируем ВСЕ остальные артефакты из run_dir (новый состав папки: all_trades.json, encoder_selection.json, last_*.pth, train.log, ...)
            # Важно: не дублируем тяжёлые/канонические файлы, т.к. они уже скопированы под унифицированными именами выше.
            try:
                if run_dir_path is not None and run_dir_path.exists() and run_dir_path.is_dir():
                    excluded_names = {'model.pth', 'replay.pkl', 'train_result.pkl'}
                    copied_extra = 0
                    for f in run_dir_path.iterdir():
                        try:
                            if not f.is_file():
                                continue
                            if f.name in excluded_names:
                                continue
                            dst = version_dir / f.name
                            if dst.exists():
                                # не перезаписываем (на случай если имя совпало)
                                continue
                            shutil.copy2(f, dst)
                            copied_extra += 1
                        except Exception:
                            continue
                    try:
                        print(f"[create_model_version] COPY EXTRA | from={run_dir_path} -> {version_dir} | files={copied_extra}")
                    except Exception:
                        pass
            except Exception:
                # extra-copy best-effort, не валим основной сценарий
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
                # Direction (long/short) — если есть manifest.json в run_dir_path
                trained_dir = None
                try:
                    if run_dir_path is not None:
                        mfj = run_dir_path / 'manifest.json'
                        if mfj.exists():
                            with open(mfj, 'r', encoding='utf-8') as _mf:
                                _mj = _json.load(_mf) or {}
                                trained_dir = (_mj.get('direction') or _mj.get('trained_as') or None)
                except Exception:
                    trained_dir = None
                try:
                    trained_dir = str(trained_dir).strip().lower() if trained_dir is not None else None
                except Exception:
                    trained_dir = None
                if trained_dir not in ('long', 'short'):
                    trained_dir = None
                yaml_text = (
                    'id: "' + manifest_id + '"\n'
                    'symbol: "' + base_code.lower() + '"\n'
                    'ensemble: "' + ensemble_name + '"\n'
                    'version: "' + version_name + '"\n'
                    'created_at: "' + created_ts + '"\n'
                    'run_id: "' + manifest_id + '"\n'
                    + (('direction: "' + trained_dir + '"\ntrained_as: "' + trained_dir + '"\n') if trained_dir else '')
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
        
        try:
            print(f"[create_model_version] DONE | version={version_name} | out={{'model': '{ver_model.name}', 'replay': '{ver_replay.name if ver_replay.exists() else None}', 'result': '{ver_result.name if ver_result.exists() else None}'}}")
        except Exception:
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
                    manifest_direction = None
                    source_run_path = None
                    if manifest:
                        try:
                            manifest_path = vdir / manifest
                            if manifest_path.exists():
                                with open(manifest_path, 'r', encoding='utf-8') as mf:
                                    manifest_content = mf.read()
                                    # Ищем простые поля (id/direction/source_run_path) без YAML-парсера
                                    for line in manifest_content.split('\n'):
                                        sline = line.strip()
                                        if sline.startswith('id:') and not manifest_id:
                                            manifest_id = sline.split(':', 1)[1].strip().strip('"\'')
                                        elif sline.startswith('direction:') and not manifest_direction:
                                            manifest_direction = sline.split(':', 1)[1].strip().strip('"\'').lower()
                                        elif sline.startswith('trained_as:') and not manifest_direction:
                                            manifest_direction = sline.split(':', 1)[1].strip().strip('"\'').lower()
                                        elif sline.startswith('source_run_path:') and not source_run_path:
                                            source_run_path = sline.split(':', 1)[1].strip().strip('"\'')
                        except Exception:
                            pass
                    # Fallback: manifest.json рядом с моделью
                    if manifest_direction not in ('long', 'short'):
                        manifest_direction = None
                        try:
                            import json as _json
                            mfj = vdir / 'manifest.json'
                            if mfj.exists():
                                with open(mfj, 'r', encoding='utf-8') as _mf:
                                    _mj = _json.load(_mf) or {}
                                md = (_mj.get('direction') or _mj.get('trained_as') or None)
                                md = str(md).strip().lower() if md is not None else None
                                if md in ('long', 'short'):
                                    manifest_direction = md
                        except Exception:
                            pass
                    # Fallback: manifest.json в source_run_path (если сохранён в manifest.yaml)
                    if manifest_direction not in ('long', 'short') and source_run_path:
                        try:
                            from pathlib import Path as _Path
                            import json as _json
                            sp = _Path(str(source_run_path).replace('\\','/'))
                            mfj2 = sp / 'manifest.json'
                            if mfj2.exists():
                                with open(mfj2, 'r', encoding='utf-8') as _mf:
                                    _mj = _json.load(_mf) or {}
                                md = (_mj.get('direction') or _mj.get('trained_as') or None)
                                md = str(md).strip().lower() if md is not None else None
                                if md in ('long', 'short'):
                                    manifest_direction = md
                        except Exception:
                            pass
                    if manifest_direction not in ('long', 'short'):
                        manifest_direction = 'long'
                    
                    versions.append({
                        'version': vdir.name,
                        'files': files,
                        'manifest': manifest,
                        'manifest_id': manifest_id,
                        'direction': manifest_direction,
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
        # Helper: sanitize NaN/Infinity → None for strict JSON
        def _sanitize_for_json(value):
            try:
                if isinstance(value, float):
                    return value if math.isfinite(value) else None
                if isinstance(value, (list, tuple)):
                    return [ _sanitize_for_json(v) for v in value ]
                if isinstance(value, dict):
                    return { str(k): _sanitize_for_json(v) for k, v in value.items() }
                return value
            except Exception:
                return None

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
        # Enrich: добавим long-short контекст из Redis даже для "старых" предсказаний, где этих полей ещё нет
        try:
            from utils.redis_utils import get_redis_client as _get_rc2
        except Exception:
            _get_rc2 = None
        _rc2 = None
        _sym_cfg = {}
        if _get_rc2 is not None:
            try:
                _rc2 = _get_rc2()
            except Exception:
                _rc2 = None
        if _rc2 is not None and symbol:
            try:
                tm = _rc2.get(f'trading:trade_mode:{symbol}')
                direction = _rc2.get(f'trading:direction:{symbol}')
                roles_raw = _rc2.get(f'trading:model_roles:{symbol}')
                tm = str(tm).strip() if tm else None
                direction = str(direction).strip().lower() if direction else None
                roles = {}
                try:
                    parsed = json.loads(roles_raw) if roles_raw else None
                    if isinstance(parsed, dict):
                        for k, v in parsed.items():
                            try:
                                kp = str(k).replace('\\', '/')
                                rv = str(v).strip().lower()
                                roles[kp] = ('short' if rv == 'short' else 'long')
                                # также добавим вариант с /workspace префиксом, если ключ относительный
                                if not kp.startswith('/') and kp:
                                    roles['/workspace/' + kp.lstrip('/')] = roles[kp]
                            except Exception:
                                continue
                except Exception:
                    roles = {}
                _sym_cfg = {'trade_mode': tm, 'direction': direction, 'roles': roles}
            except Exception:
                _sym_cfg = {}

        predictions_data = []
        for prediction in predictions or []:
            try:
                q_values = json.loads(prediction.q_values) if prediction.q_values else []
                market_conditions = json.loads(prediction.market_conditions) if prediction.market_conditions else {}
            except:
                q_values = []
                market_conditions = {}

            # Ретро-enrichment long-short полей (чтобы UI показывал роль модели и соответствие режиму)
            try:
                if isinstance(market_conditions, dict) and _sym_cfg:
                    if 'trade_mode' not in market_conditions and _sym_cfg.get('trade_mode'):
                        market_conditions['trade_mode'] = _sym_cfg.get('trade_mode')
                    # active_role/trade_direction считаем по режиму рынка, если long-short
                    tm0 = market_conditions.get('trade_mode')
                    reg0 = market_conditions.get('ensemble_regime') or market_conditions.get('market_regime')
                    if str(tm0).strip() == 'long-short':
                        if 'active_role' not in market_conditions and str(reg0).strip().lower() in ('uptrend', 'downtrend'):
                            market_conditions['active_role'] = ('long' if str(reg0).strip().lower() == 'uptrend' else 'short')
                        if 'trade_direction' not in market_conditions and market_conditions.get('active_role'):
                            market_conditions['trade_direction'] = market_conditions.get('active_role')
                    else:
                        if 'trade_direction' not in market_conditions and _sym_cfg.get('direction'):
                            market_conditions['trade_direction'] = _sym_cfg.get('direction')
                    # model_role по пути модели из Redis roles map
                    if 'model_role' not in market_conditions and _sym_cfg.get('roles'):
                        mp = str(getattr(prediction, 'model_path', '') or '').replace('\\', '/')
                        role = _sym_cfg['roles'].get(mp)
                        if role:
                            market_conditions['model_role'] = role
            except Exception:
                pass
            
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

        safe_payload = _sanitize_for_json({
            'success': True,
            'predictions': predictions_data,
            'total_predictions': len(predictions_data)
        })
        return jsonify(safe_payload)
        
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
    print(msk_tag(f"🚀 Запускаю Flask сервер на порту {port}..."))
    print(msk_tag(f"🌐 Откройте: http://localhost:{port}"))
    print(msk_tag(f"🔧 Debug режим: {'ВКЛЮЧЕН' if debug_mode else 'ОТКЛЮЧЕН'}"))
    if os.environ.get("ENABLE_TELEGRAM_BOT_POLLING", "1").lower() in ("1", "true", "yes", "on"):
        from utils.telegram_bot_poller import start_telegram_bot_poller
        start_telegram_bot_poller()
    from utils.max_bot_poller import start_max_bot_poller
    start_max_bot_poller()
    
    # Убираем инициализацию торгового агента
    # init_trading_agent()
    
    # threaded=True: OOS batch UI polls /xgb_oos_test_status for many Celery tasks in parallel;
    # default threaded=False cannot drain ~80 short polls/s and progress appears stuck at 0/N.
    app.run(host="0.0.0.0", port=port, debug=debug_mode, threaded=True)

from flask import Blueprint, render_template, jsonify, request, current_app # type: ignore
from pathlib import Path
import os
import json
import pickle as _pkl
import torch # type: ignore
import shutil
import uuid
from datetime import datetime
import traceback
import random
import io
from contextlib import redirect_stdout

from celery.result import AsyncResult
from typing import Dict, Any

from envs.dqn_model.gym.gconfig import GymConfig
from utils.redis_utils import get_redis_client
from tasks.celery_tasks import celery

sac_bp = Blueprint('sac', __name__)

@sac_bp.route('/sac_models')
def sac_models_page():
    """Страница управления моделями (SAC)"""
    return render_template('sac_models.html')

@sac_bp.get('/api/sac_runs/symbols')
def api_sac_runs_symbols():
    try:
        base = Path('result') / 'sac'
        if not base.exists():
            return jsonify({'success': True, 'symbols': []})
        symbols = []
        for d in base.iterdir():
            if d.is_dir():
                symbols.append(d.name)
        symbols.sort()
        return jsonify({'success': True, 'symbols': symbols})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@sac_bp.get('/api/sac_runs/list')
def api_sac_runs_list():
    try:
        run_name = (request.args.get('run_name') or '').strip()
        if not run_name:
            return jsonify({'success': False, 'error': 'run_name required'}), 400

        base_dir = Path('result') / 'sac' / run_name
        if not base_dir.exists():
            return jsonify({'success': True, 'runs': []})

        runs_dir = base_dir / 'runs' if (base_dir / 'runs').exists() else base_dir

        runs = []
        for rd in runs_dir.iterdir():
            if not rd.is_dir():
                continue

            run_id = rd.name

            manifest: dict[str, Any] = {}
            manifest_file = rd / 'manifest.json'
            if manifest_file.exists():
                try:
                    manifest_data = json.loads(manifest_file.read_text(encoding='utf-8'))
                    if isinstance(manifest_data, dict):
                        manifest.update(manifest_data)
                        metrics_embedded = manifest_data.get('metrics')
                        if isinstance(metrics_embedded, dict):
                            for key in ('winrate', 'avg_roi', 'roi', 'max_drawdown', 'actual_episodes', 'episodes', 'created_at', 'seed'):
                                if manifest.get(key) is None and metrics_embedded.get(key) is not None:
                                    manifest[key] = metrics_embedded.get(key)
                except Exception:
                    pass
            metrics_file = rd / 'metrics.json'
            if metrics_file.exists():
                try:
                    metrics_data = json.loads(metrics_file.read_text(encoding='utf-8'))
                    if isinstance(metrics_data, dict):
                        manifest.update(metrics_data)
                except Exception:
                    manifest = {}

            model_path = str(rd / 'model.pth') if (rd / 'model.pth').exists() else None
            replay_path = str(rd / 'replay.pkl') if (rd / 'replay.pkl').exists() else None
            result_path = str(rd / 'train_result.pkl') if (rd / 'train_result.pkl').exists() else None

            winrate = manifest.get('winrate')
            roi_value = manifest.get('roi')
            if roi_value is None:
                roi_value = manifest.get('avg_roi')
            max_dd = manifest.get('max_drawdown')
            episodes = manifest.get('actual_episodes', manifest.get('episodes'))

            runs.append({
                'run_id': run_id,
                'parent_run_id': manifest.get('parent_run_id'),
                'root_id': manifest.get('root_id'),
                'seed': manifest.get('seed'),
                'episodes_end': manifest.get('episodes_end'),
                'created_at': manifest.get('created_at'),
                'model_path': model_path,
                'replay_path': replay_path,
                'result_path': result_path,
                'winrate': winrate,
                'pl_ratio': roi_value,
                'roi': roi_value,
                'max_dd': max_dd,
                'episodes': episodes,
                'agent_type': manifest.get('agent_type', 'sac'),
                'version': manifest.get('version', 'N/A'),
            })
        try:
            runs.sort(key=lambda r: (r.get('created_at') or '', r['run_id']))
        except Exception:
            runs.sort(key=lambda r: r['run_id'])
        return jsonify({'success': True, 'runs': runs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@sac_bp.post('/api/sac/analyze_training_results')
def sac_analyze_training_results():
    """Выполняет анализ результата обучения SAC и возвращает stdout отчета."""
    try:
        data = request.get_json(silent=True) or {}
        requested_file = (data.get('file') or '').strip()
        run_name = (data.get('run_name') or data.get('symbol') or '').strip()
        run_id = (data.get('run_id') or '').strip()

        base_dir = Path('result') / 'sac'
        if not base_dir.exists():
            return jsonify({
                'success': False,
                'message': 'Папка result/sac не найдена. Сначала запустите обучение SAC.',
                'status': 'error'
            }), 404

        selected_file: Path | None = None

        def _normalize_candidate(raw: str) -> Path:
            norm = raw.replace('\\', '/')
            candidate = Path(norm)
            if not candidate.is_absolute():
                parts = candidate.parts
                if len(parts) >= 2 and parts[0].lower() == 'result' and parts[1].lower() == 'sac':
                    candidate = Path(norm)
                else:
                    candidate = base_dir / candidate
            return candidate

        if requested_file:
            cand = _normalize_candidate(requested_file)
            try:
                cand_resolved = cand.resolve()
            except Exception:
                cand_resolved = cand.absolute()
            if str(cand_resolved).lower().startswith(str(base_dir.resolve()).lower()) and cand_resolved.exists():
                selected_file = cand_resolved

        if selected_file is None and run_name and run_id:
            candidate = base_dir / run_name.lower() / 'runs' / run_id / 'train_result.pkl'
            if candidate.exists():
                selected_file = candidate.resolve()

        if selected_file is None:
            candidates = list(base_dir.rglob('train_result.pkl'))
            if not candidates:
                return jsonify({
                    'success': False,
                    'message': 'Файлы train_result.pkl для SAC не найдены.',
                    'status': 'error'
                }), 404
            selected_file = max(candidates, key=lambda p: p.stat().st_ctime)

        if selected_file.suffix.lower() != '.pkl' or not selected_file.exists():
            return jsonify({
                'success': False,
                'message': 'Указан некорректный файл результатов.',
                'status': 'error'
            }), 400

        try:
            from analyze_training_results import analyze_training_results as analyze_func
        except ImportError:
            def analyze_func(filename: str):
                print(f"📊 Анализ файла: {filename}")
                print("⚠️ Модуль analyze_training_results не найден. Установите numpy/matplotlib.")
                return 'Анализ недоступен'

        run_dir = selected_file.parent
        metrics_data = {}
        metrics_file = run_dir / 'metrics.json'
        if metrics_file.exists():
            try:
                metrics_data = json.loads(metrics_file.read_text(encoding='utf-8'))
            except Exception as exc:
                current_app.logger.warning(f"[SAC analyze] Не удалось прочитать metrics.json: {exc}")

        results_snapshot = {}
        try:
            with open(selected_file, 'rb') as f:
                results_snapshot = _pkl.load(f) or {}
        except Exception as exc:
            current_app.logger.warning(f"[SAC analyze] Не удалось загрузить {selected_file}: {exc}")

        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer):
            analyze_func(str(selected_file))
        analysis_output = output_buffer.getvalue()

        response_data = {
            'success': True,
            'status': 'success',
            'message': 'Анализ результатов SAC завершен успешно',
            'file_analyzed': str(selected_file).replace('\\', '/'),
            'output': analysis_output,
            'metrics': metrics_data,
            'model_type': 'sac'
        }

        if results_snapshot:
            if isinstance(results_snapshot, dict):
                episode_winrates = results_snapshot.get('episode_winrates')
                if isinstance(episode_winrates, (list, tuple)):
                    response_data['episode_winrates_count'] = len(episode_winrates)
                elif isinstance(results_snapshot.get('episode_winrates_count'), int):
                    response_data['episode_winrates_count'] = int(results_snapshot.get('episode_winrates_count'))
                if 'actual_episodes' in results_snapshot:
                    response_data['actual_episodes'] = results_snapshot.get('actual_episodes')
                if 'episodes' in results_snapshot:
                    response_data['episodes'] = results_snapshot.get('episodes')
        return jsonify(response_data)

    except Exception as e:
        current_app.logger.error(f"[SAC analyze] Ошибка: {e}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'status': 'error',
            'message': f'Ошибка при анализе результатов SAC: {e}'
        }), 500

@sac_bp.post('/api/sac_runs/model_info')
def get_sac_result_model_info():
    """Возвращает краткую информацию по выбранному файлу весов из result/sac/<run_name>/model.pth"""
    try:
        data = request.get_json(silent=True) or {}
        filename = (data.get('filename') or '').strip()
        if not filename:
            return jsonify({'success': False, 'error': 'Не указан filename'}), 400

        results_dir = Path('result') / 'sac'
        req_norm = filename.replace('\\', '/')
        p = Path(req_norm)
        
        if not p.is_absolute():
            if not (p.parts and p.parts[0].lower() == results_dir.parent.name.lower() and p.parts[1].lower() == results_dir.name.lower()):
                if len(p.parts) >= 2 and p.parts[-1].lower() == 'model.pth':
                    run_name_from_path = p.parts[-2]
                    p = results_dir / run_name_from_path / p.name
                else:
                    return jsonify({'success': False, 'error': 'Неверный путь к файлу SAC модели'}), 400
        
        try:
            p = p.resolve()
        except Exception:
            pass
        
        if not str(p).lower().startswith(str(results_dir.resolve()).lower()):
            return jsonify({'success': False, 'error': 'Файл вне папки result/sac/'}), 400
        if not p.exists() or not p.is_file() or p.name != 'model.pth':
            return jsonify({'success': False, 'error': 'Ожидается существующий файл model.pth внутри result/sac/<run_name>/'}), 400

        run_dir = p.parent
        run_name = run_dir.name

        replay_file = run_dir / 'replay.pkl'
        metrics_file = run_dir / 'metrics.json'
        train_result_file = run_dir / 'train_result.pkl' 

        info = {
            'success': True,
            'model_file': str(p),
            'model_size_bytes': p.stat().st_size if p.exists() else 0,
            'run_name': run_name,
            'replay_exists': replay_file.exists(),
            'metrics_exists': metrics_file.exists(),
            'train_result_exists': train_result_file.exists(),
            'replay_file': str(replay_file) if replay_file.exists() else None,
            'metrics_file': str(metrics_file) if metrics_file.exists() else None,
            'train_result_file': str(train_result_file) if train_result_file.exists() else None,
            'stats': {},
            'episodes': None,
            'agent_type': 'sac',
            'version': 'N/A',
        }

        if metrics_file.exists():
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics_data = json.load(f)
                info['stats'] = {
                    'winrate': metrics_data.get('winrate'),
                    'roi': metrics_data.get('roi'),
                    'max_drawdown': metrics_data.get('max_drawdown'),
                    'trades_count': metrics_data.get('trades_count'),
                }
                info['episodes'] = metrics_data.get('total_episodes_planned')
                info['seed'] = metrics_data.get('seed')
                info['cuda_available'] = metrics_data.get('cuda_available', False)
                info['gpu_name'] = metrics_data.get('gpu_name')
                info['total_training_time'] = metrics_data.get('current_total_training_time')
                if info['total_training_time'] and info['episodes']:
                    info['avg_time_per_episode_sec'] = info['total_training_time'] / info['episodes']
                else:
                    info['avg_time_per_episode_sec'] = None
                info['agent_type'] = metrics_data.get('agent_type', 'sac')
                info['version'] = metrics_data.get('version', 'N/A')

            except Exception as e:
                current_app.logger.warning(f"get_sac_result_model_info: не удалось загрузить metrics.json {metrics_file}: {e}")

        try:
            if os.path.exists(str(p)):
                ckpt = torch.load(str(p), map_location='cpu')
                if isinstance(ckpt, dict) and 'normalization_stats' in ckpt:
                    info['normalization_stats'] = ckpt.get('normalization_stats')
                    current_app.logger.info(f"[SAC OOS] Loaded normalization_stats from checkpoint.")
                else:
                    current_app.logger.info(f"[SAC OOS] normalization_stats not found in checkpoint.")
        except Exception as e:
            current_app.logger.warning(f"[SAC OOS] Failed to load normalization_stats from checkpoint {p}: {e}")

        return jsonify(info)
    except Exception as e:
        current_app.logger.error(f"Ошибка в get_sac_result_model_info: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@sac_bp.post('/api/sac_runs/create_version')
def create_sac_model_version():
    """Создает новую версию SAC модели с уникальным ID"""
    try:
        data = request.get_json(silent=True) or {}
        requested_run_name = (data.get('run_name') or '').strip()
        requested_ensemble = (data.get('ensemble') or 'ensemble-a').strip() or 'ensemble-a'
        
        if not requested_run_name:
            return jsonify({
                "success": False,
                "error": "Не указано имя запуска (run_name)."
            }), 400

        print(f"[create_sac_model_version] payload: run_name='{requested_run_name}', ensemble='{requested_ensemble}'")
        
        model_id = str(uuid.uuid4())[:4].upper() 
        
        results_sac_dir = Path('result') / 'sac'
        if not results_sac_dir.exists():
            return jsonify({
                "success": False,
                "error": "Папка result/sac не найдена. Сначала запустите обучение SAC."
            }), 404

        run_dir_path = results_sac_dir / requested_run_name
        if not run_dir_path.exists() or not run_dir_path.is_dir():
            return jsonify({
                "success": False,
                "error": f"Папка запуска SAC '{requested_run_name}' не найдена в result/sac."
            }), 404

        model_file_source = run_dir_path / 'model.pth'
        replay_file_source = run_dir_path / 'replay.pkl'
        metrics_file_source = run_dir_path / 'metrics.json'
        train_result_file_source = run_dir_path / 'train_result.pkl' 
        
        if not model_file_source.exists():
            return jsonify({
                "success": False,
                "error": f"Файл модели model.pth не найден в {run_dir_path}"
            }), 404

        models_root = Path('models') / 'sac'
        models_root.mkdir(parents=True, exist_ok=True)

        symbol_dir = models_root / requested_run_name 
        symbol_dir.mkdir(exist_ok=True)

        ensemble_dir = symbol_dir / requested_ensemble
        ensemble_dir.mkdir(exist_ok=True)

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
        print(f"[create_sac_model_version] create version_dir='{version_dir}'")

        shutil.copy2(model_file_source, version_dir / 'model.pth')
        if replay_file_source.exists():
            shutil.copy2(replay_file_source, version_dir / 'replay.pkl')
        if metrics_file_source.exists():
            shutil.copy2(metrics_file_source, version_dir / 'metrics.json')
        if train_result_file_source.exists(): 
            shutil.copy2(train_result_file_source, version_dir / 'train_result.pkl')
        print(f"[create_sac_model_version] copied core files to '{version_dir.name}'")

        manifest_path = version_dir / 'manifest.yaml'
        stats_brief = {}
        if metrics_file_source.exists():
            try:
                with open(metrics_file_source, 'r', encoding='utf-8') as f:
                    stats_brief = json.load(f)
            except Exception:
                print("[create_sac_model_version] WARN: cannot read stats from metrics.json")

        created_ts = datetime.utcnow().isoformat()
        manifest_id = requested_run_name

        yaml_lines = [
            f'id: "{manifest_id}"',
            f'symbol: "{requested_run_name.lower()}"',
            f'ensemble: "{requested_ensemble}"',
            f'version: "{version_name}"',
            f'created_at: "{created_ts}"',
            'agent_type: "sac"',
            f'run_id: "{manifest_id}"',
            f'source_run_path: "{str(run_dir_path).replace(os.sep, "/" )}"',
            'files:',
            '  model: "model.pth"',
            f'  replay: "{replay_file_source.name if replay_file_source.exists() else None}"',
            f'  metrics: "{metrics_file_source.name if metrics_file_source.exists() else None}"',
            f'  results: "{train_result_file_source.name if train_result_file_source.exists() else None}"',
            'stats:'
        ]

        if isinstance(stats_brief, dict):
            yaml_lines.extend([f'  {k}: {v}' for k, v in stats_brief.items()])

        yaml_text = "\n".join(yaml_lines) + "\n"

        with open(manifest_path, 'w', encoding='utf-8') as mf:
            mf.write(yaml_text)
        print(f"[create_sac_model_version] manifest written to '{manifest_path}'")

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
            print(f"[create_sac_model_version] symlink 'current' -> '{version_name}' created.")
        except Exception:
            try:
                with open(current_link, 'w', encoding='utf-8') as fcur:
                    fcur.write(version_name)
                print(f"[create_sac_model_version] fallback: wrote '{version_name}' to 'current' file.")
            except Exception:
                print("[create_sac_model_version] WARN: failed to create 'current' link/file.")
        
        return jsonify({
            "success": True,
            "model_id": requested_run_name,
            "version": version_name,
            "files": [
                "model.pth",
                replay_file_source.name if replay_file_source.exists() else None,
                metrics_file_source.name if metrics_file_source.exists() else None,
                train_result_file_source.name if train_result_file_source.exists() else None
            ]
        })
        
    except Exception as e:
        print("[create_sac_model_version] ERROR:\n" + traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@sac_bp.get('/api/sac_models/list')
def get_sac_models_list():
    """Возвращает список всех сохраненных SAC моделей"""
    try:
        models: list = []

        models_root = Path('models') / 'sac'
        if not models_root.exists():
            return jsonify({
                "success": True,
                "models": []
            })

        for symbol_dir in models_root.iterdir():
            if not symbol_dir.is_dir():
                continue
            for ensemble_dir in symbol_dir.iterdir():
                if not ensemble_dir.is_dir():
                    continue
                for version_dir in ensemble_dir.iterdir():
                    if not version_dir.is_dir() or not version_dir.name.startswith('v'):
                        continue
                    
                    model_file = version_dir / 'model.pth'
                    replay_file = version_dir / 'replay.pkl'
                    metrics_file = version_dir / 'metrics.json'
                    manifest_file = version_dir / 'manifest.yaml'
                    
                    if not model_file.exists():
                        continue

                    model_id = symbol_dir.name 
                    model_size = f"{model_file.stat().st_size / (1024 * 1024):.1f} MB"
                    replay_size = f"{replay_file.stat().st_size / (1024 * 1024):.1f} MB" if replay_file.exists() else "N/A"
                    metrics_size = f"{metrics_file.stat().st_size / 1024:.1f} KB" if metrics_file.exists() else "N/A"
                    
                    creation_time = datetime.fromtimestamp(model_file.stat().st_ctime)
                    date_str = creation_time.strftime('%d.%m.%Y %H:%M')
                    
                    stats = {}
                    if metrics_file.exists():
                        try:
                            with open(metrics_file, 'r', encoding='utf-8') as f:
                                metrics_data = json.load(f)
                            stats = {
                                'winrate': metrics_data.get('winrate'),
                                'roi': metrics_data.get('roi'),
                                'max_drawdown': metrics_data.get('max_drawdown'),
                                'trades_count': metrics_data.get('trades_count'),
                            }
                        except Exception:
                            pass

                    manifest_data = {}
                    if manifest_file.exists():
                        try:
                            with open(manifest_file, 'r', encoding='utf-8') as f:
                                for line in f:
                                    if ':' in line:
                                        key, value = line.split(':', 1)
                                        manifest_data[key.strip()] = value.strip().strip('"\'')
                        except Exception:
                            pass

                    models.append({
                        "id": model_id,
                        "date": date_str,
                        "ensemble": ensemble_dir.name,
                        "version": version_dir.name,
                        "files": {
                            "model": model_file.name,
                            "model_size": model_size,
                            "replay": replay_file.name if replay_file.exists() else None,
                            "replay_size": replay_size,
                            "metrics": metrics_file.name if metrics_file.exists() else None,
                            "metrics_size": metrics_size,
                            "manifest": manifest_file.name if manifest_file.exists() else None,
                        },
                        "stats": stats,
                        "agent_type": manifest_data.get('agent_type', 'sac'),
                        "full_path": str(version_dir).replace('\\','/')
                    })
        
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

@sac_bp.post('/api/sac_runs/train')
def api_sac_runs_train():
    """Запускает обучение SAC для указанного символа через Celery."""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        episodes = int(data.get('episodes', 1000))
        episode_length = int(data.get('episode_length', 2000))
        seed = data.get('seed')

        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol is required.'}), 400

        # Инициализируем Redis клиент в начале функции
        redis_client = get_redis_client()

        # Глобальная блокировка: не запускать обучение если уже запущено любое SAC обучение
        global_lock_key = "sac:training:global_lock"
        if redis_client.exists(global_lock_key):
            return jsonify({'success': False, 'error': 'Другое обучение SAC уже запущено. Дождитесь завершения.'}), 409

        # Проверяем, не запущено ли уже обучение для этого символа
        running_key = f"celery:train:sac:task:{symbol}"
        if redis_client.exists(running_key):
            return jsonify({'success': False, 'error': f'Обучение для {symbol} уже запущено.'}), 409

        # Дополнительная проверка: проверяем статус последней задачи для этого символа
        # чтобы избежать повторных запусков после перезагрузки страницы
        last_run_key = f"sac:last_run:{symbol}"
        last_run_time = redis_client.get(last_run_key)
        if last_run_time:
            import time
            current_time = time.time()
            # Если прошло менее 10 секунд с момента последнего запуска, блокируем
            if current_time - float(last_run_time) < 10:  # Уменьшено до 10 секунд для более строгой защиты
                return jsonify({'success': False, 'error': f'Обучение для {symbol} запущено слишком недавно ({current_time - float(last_run_time):.1f} сек назад). Подождите немного.'}), 429

        # Дополнительная проверка: не запускать обучение чаще чем раз в 5 минут для одного символа
        last_run_key = f"sac:last_run:{symbol}"
        last_run_time = redis_client.get(last_run_key)
        if last_run_time:
            import time
            current_time = time.time()
            if current_time - float(last_run_time) < 300:  # 5 минут
                return jsonify({'success': False, 'error': f'Обучение для {symbol} можно запускать не чаще раза в 5 минут.'}), 429

        from tasks.sac_tasks import train_sac_symbol
        task = train_sac_symbol.apply_async(args=[symbol, episodes, episode_length, seed])

        redis_client.lpush('ui:tasks', task.id)

        return jsonify({'success': True, 'task_id': task.id, 'status': 'Training started.'})
    except Exception as e:
        current_app.logger.error(f"Error starting SAC training task: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@sac_bp.get('/api/sac_runs/task_status/<task_id>')
def api_sac_runs_task_status(task_id):
    """Возвращает статус и логи задачи обучения SAC по ID."""
    try:
        task = AsyncResult(task_id, app=celery)
        response: Dict[str, Any] = {
            'success': True,
            'task_id': task.id,
            'state': task.state,
            'status': 'UNKNOWN',
        }

        info = task.info if isinstance(task.info, dict) else {}
        logs = info.get('logs') if isinstance(info, dict) else None
        if logs:
            response['logs'] = logs

        if task.state == 'PENDING':
            response['status'] = 'Задача в очереди'
        elif task.state == 'STARTED':
            response['status'] = 'Задача запущена'
        elif task.state in {'PROGRESS', 'IN_PROGRESS'}:
            response['status'] = 'Выполнение...'
            response['logs'] = logs or []
        elif task.state == 'SUCCESS':
            response['status'] = 'Задача завершена'
            response['result'] = task.result
            if isinstance(task.result, dict):
                response['result_summary'] = {
                    'symbol': task.result.get('symbol'),
                    'run_name': task.result.get('run_name'),
                    'result_dir': task.result.get('result_dir'),
                }
        elif task.state == 'FAILURE':
            response['success'] = False
            response['status'] = 'Ошибка выполнения'
            response['error'] = str(task.info)
            response['traceback'] = task.traceback
        else:
            response['status'] = 'Неизвестное состояние'

        return jsonify(response)
    except Exception as e:
        current_app.logger.error(f"Error getting SAC task status for {task_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@sac_bp.get('/list_sac_result_models')
def list_sac_result_models():
    """Возвращает список всех файлов model.pth в result/sac/ для выбора для дообучения."""
    try:
        base_path = Path('result') / 'sac'
        if not base_path.exists():
            return jsonify({'success': True, 'files': []})

        found_files = []
        for root, _, files in os.walk(base_path):
            for file in files:
                if file == 'model.pth':
                    full_path = Path(root) / file
                    found_files.append({'filename': str(full_path).replace('\\', '/')})
        
        return jsonify({'success': True, 'files': found_files})

    except Exception as e:
        current_app.logger.error(f"Ошибка в list_sac_result_models: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@sac_bp.get('/api/sac_ensembles/list')
def list_sac_ensembles():
    """Возвращает структуру ансамблей и версий для указанного run_name SAC."""
    """Query params: run_name=my_sac_run_001"""
    try:
        run_name = (request.args.get('run_name') or '').strip()
        if not run_name:
            return jsonify({"success": False, "error": "run_name обязателен"}), 400

        root = Path('models') / 'sac' / run_name
        if not root.exists():
            return jsonify({"success": True, "ensembles": {}})

        ensembles = {}
        for ens_dir in root.iterdir():
            if not ens_dir.is_dir():
                continue
            versions = []
            current = None
            canary = None
            
            for vdir in ens_dir.iterdir():
                if vdir.is_dir() and vdir.name.startswith('v'):
                    files = { 'model': None, 'replay': None, 'metrics': None, 'results': None }
                    fallback_model = None
                    fallback_replay = None
                    fallback_metrics = None
                    fallback_results = None
                    manifest = None
                    stats = {}

                    for f in vdir.iterdir():
                        if f.name == 'model.pth':
                            files['model'] = f.name
                        elif f.name == 'replay.pkl':
                            files['replay'] = f.name
                        elif f.name == 'metrics.json':
                            files['metrics'] = f.name
                        elif f.name == 'train_result.pkl':
                            files['results'] = f.name
                        elif f.name == 'manifest.yaml':
                            manifest = f.name
                        elif f.suffix == '.pth' and fallback_model is None: # Corrected line
                            fallback_model = f.name
                        elif f.suffix == '.pkl':
                            n = f.name.lower()
                            if ('replay' in n) and fallback_replay is None:
                                fallback_replay = f.name
                            if ('metrics' in n) and fallback_metrics is None:
                                fallback_metrics = f.name
                            if (('train_result' in n) or ('result' in n)) and fallback_results is None:
                                fallback_results = f.name
                    
                    if files['model'] is None and fallback_model is not None:
                        files['model'] = fallback_model
                    if files['replay'] is None and fallback_replay is not None:
                        files['replay'] = fallback_replay
                    if files['metrics'] is None and fallback_metrics is not None:
                        files['metrics'] = fallback_metrics
                    if files['results'] is None and fallback_results is not None:
                        files['results'] = fallback_results

                    manifest_id = None
                    if manifest:
                        try:
                            manifest_path = vdir / manifest
                            if manifest_path.exists():
                                with open(manifest_path, 'r', encoding='utf-8') as mf:
                                    manifest_content = mf.read()
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
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500

@sac_bp.post('/api/sac_models/delete')
def delete_sac_model():
    """Удаляет SAC модель и все связанные файлы"""
    try:
        data = request.get_json()
        run_name = (data.get('run_name') or '').strip()
        ensemble_name = (data.get('ensemble') or '').strip()
        version_name = (data.get('version') or '').strip()

        if not run_name or not ensemble_name or not version_name:
            return jsonify({
                "success": False,
                "error": "Требуются run_name, ensemble и version для удаления SAC модели"
            }), 400

        models_root = Path('models') / 'sac'
        target_dir = models_root / run_name / ensemble_name / version_name
        ensemble_dir = models_root / run_name / ensemble_name # Определяем ensemble_dir здесь

        if not target_dir.exists() or not target_dir.is_dir():
            return jsonify({
                "success": False,
                "error": f"Директория модели SAC '{target_dir}' не найдена"
            }), 404

        deleted_files = []
        try:
            for item in target_dir.iterdir():
                if item.is_file():
                    os.remove(item)
                    deleted_files.append(str(item))
                elif item.is_dir():
                    shutil.rmtree(item)
                    deleted_files.append(str(item))

            shutil.rmtree(target_dir)
            deleted_files.append(str(target_dir))

            if not any(ensemble_dir.iterdir()):
                shutil.rmtree(ensemble_dir)
                deleted_files.append(str(ensemble_dir))
                parent_run_name_dir = ensemble_dir.parent
                if not any(parent_run_name_dir.iterdir()):
                    shutil.rmtree(parent_run_name_dir)
                    deleted_files.append(str(parent_run_name_dir))

        except Exception as e:
            print(f"[delete_sac_model] ERROR during deletion: {e}\n" + traceback.format_exc())
            return jsonify({
                "success": False,
                "error": f"Ошибка при удалении файлов SAC модели: {e}"
            }), 500
        
        return jsonify({
            "success": True,
            "message": f"SAC модель {run_name}/{ensemble_name}/{version_name} удалена",
            "deleted_files": deleted_files
        })
        
    except Exception as e:
        print("[delete_sac_model] ERROR:\n" + traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@sac_bp.get('/api/sac_runs/oos_results')
def api_sac_runs_oos_results():
    try:
        run_id = (request.args.get('run_id') or '').strip()
        symbol = (request.args.get('symbol') or '').strip()
        if not run_id:
            return jsonify({'success': False, 'error': 'run_id required'}), 400

        run_dir = Path('result') / 'sac'
        if symbol:
            run_dir = run_dir / symbol.lower() / 'runs' / run_id
        else:
            run_dir = run_dir / run_id
        if not run_dir.exists():
            return jsonify({'success': False, 'error': f'Run directory not found: {run_dir}'}), 404

        results_file = run_dir / 'oos_results.json'
        oos_data = {'counters': {'good': 0, 'bad': 0, 'neutral': 0}, 'history': []}

        if results_file.exists():
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    oos_data = json.load(f)
            except Exception as e:
                current_app.logger.error(f"Error reading oos_results.json for SAC run {run_id}: {e}")

        return jsonify({'success': True, 'data': oos_data})
    except Exception as e:
        current_app.logger.error(f"Error in api_sac_runs_oos_results: {e}")
        import traceback
        current_app.logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

@sac_bp.get('/api/sac_runs/trades')
def api_sac_runs_trades():
    try:
        run_id = (request.args.get('run_id') or '').strip()
        trades_file = (request.args.get('trades_file') or '').strip()
        symbol = (request.args.get('symbol') or '').strip()

        if not run_id or not trades_file:
            return jsonify({'success': False, 'error': 'run_id and trades_file required'}), 400

        trades_path = Path('result') / 'sac'
        if symbol:
            trades_path = trades_path / symbol.lower() / 'runs' / run_id / trades_file
        else:
            trades_path = trades_path / run_id / trades_file

        if not trades_path.exists():
            return jsonify({'success': False, 'error': f'Trades file not found: {trades_path}'}), 404

        with open(trades_path, 'r', encoding='utf-8') as f:
            trades_data = json.load(f)

        return jsonify({'success': True, 'trades': trades_data})
    except Exception as e:
        current_app.logger.error(f"Error in api_sac_runs_trades: {e}")
        import traceback
        current_app.logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

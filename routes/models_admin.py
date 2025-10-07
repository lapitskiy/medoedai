from flask import Blueprint, jsonify, request, current_app, render_template # type: ignore
from celery.result import AsyncResult # type: ignore
from tasks import celery # type: ignore
from pathlib import Path
import os
from utils.config_loader import get_config_value
import json
from envs.dqn_model.gym.crypto_trading_env_optimized import CryptoTradingEnvOptimized # type: ignore
from envs.dqn_model.gym.indicators_optimized import preprocess_dataframes # type: ignore
from agents.vdqn.cfg.vconfig import vDqnConfig # type: ignore
import pickle
import numpy as np # type: ignore
import pandas as pd # type: ignore
import requests # type: ignore
from redis import Redis # type: ignore
from datetime import datetime
import torch # type: ignore

from utils.path import (
    resolve_run_dir,
    resolve_symbol_dir,
    list_symbol_dirs,
)

models_admin_bp = Blueprint('models_admin', __name__)

@models_admin_bp.route('/models')
def models_page():
    """Страница управления моделями (DQN)"""
    return render_template('models.html')


@models_admin_bp.get('/api/runs/symbols')
def api_runs_symbols():
    try:
        model_type = (request.args.get('model_type') or 'dqn').strip().lower()
        if model_type not in {'dqn', 'sac'}:
            model_type = 'dqn'

        symbols = []
        for path in list_symbol_dirs(model_type):
            try:
                if not path.is_dir():
                    continue
                runs_dir = path / 'runs'
                if not runs_dir.exists():
                    continue
                if model_type == 'sac':
                    symbol_value = path.name
                else:
                    symbol_value = path.name.upper()
                if symbol_value not in symbols:
                    symbols.append(symbol_value)
            except Exception:
                continue

        symbols.sort()
        return jsonify({'success': True, 'symbols': symbols})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@models_admin_bp.post('/api/strategies/save')
def api_strategy_save():
    try:
        data = request.get_json(silent=True) or {}
        filename = (data.get('filename') or '').strip()
        code = (data.get('code') or '').strip()
        label = (data.get('label') or '').strip()
        params = data.get('params') or {}
        metrics = data.get('metrics') or {}
        set_current = bool(data.get('set_current') or False)

        # Определяем run_dir из filename или code
        run_dir = None
        symbol = None
        run_id = None
        if filename:
            p = Path(filename.replace('\\','/')).resolve()
            if p.name == 'model.pth' and 'runs' in p.parts:
                runs_idx = list(p.parts).index('runs')
                run_id = p.parts[runs_idx+1] if runs_idx+1 < len(p.parts) else None
                symbol = p.parts[runs_idx-1] if runs_idx-1 >= 0 else None
                run_dir = Path('result')/str(symbol)/'runs'/str(run_id)
        if run_dir is None or not run_dir.exists():
            return jsonify({'success': False, 'error': 'cannot resolve run directory'}), 400

        # Совместим сохранение стратегий с oos_results.json: складываем в раздел 'strategies'
        stg_file = run_dir / 'oos_results.json'
        payload = {
            'label': label,
            'created_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'params': params,
            'metrics': metrics,
            'symbol': str(symbol),
            'run_id': str(run_id),
        }
        stg_id = f"stg_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        # Читаем/пишем файл oos_results.json
        doc = {'version': 1, 'counters': {'good':0,'bad':0,'neutral':0}, 'history': [], 'strategies': {'current': None, 'items': {}}}
        if stg_file.exists():
            try:
                doc = json.loads(stg_file.read_text(encoding='utf-8'))
                if not isinstance(doc, dict):
                    doc = {'version': 1, 'counters': {'good':0,'bad':0,'neutral':0}, 'history': [], 'strategies': {'current': None, 'items': {}}}
            except Exception:
                doc = {'version': 1, 'counters': {'good':0,'bad':0,'neutral':0}, 'history': [], 'strategies': {'current': None, 'items': {}}}
        if 'strategies' not in doc or not isinstance(doc['strategies'], dict):
            doc['strategies'] = {'current': None, 'items': {}}
        items = doc['strategies'].get('items') or {}
        items[stg_id] = payload
        doc['strategies']['items'] = items
        if set_current:
            doc['strategies']['current'] = stg_id
        stg_file.parent.mkdir(parents=True, exist_ok=True)
        stg_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')
        return jsonify({'success': True, 'stg_id': stg_id, 'file': str(stg_file)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@models_admin_bp.post('/api/strategies/apply')
def api_strategy_apply():
    try:
        data = request.get_json(silent=True) or {}
        filename = (data.get('filename') or '').strip()
        stg_id = (data.get('stg_id') or '').strip()
        if not filename or not stg_id:
            return jsonify({'success': False, 'error': 'filename and stg_id required'}), 400
        p = Path(filename.replace('\\','/')).resolve()
        if not (p.name == 'model.pth' and 'runs' in p.parts):
            return jsonify({'success': False, 'error': 'bad run path'}), 400
        runs_idx = list(p.parts).index('runs')
        run_id = p.parts[runs_idx+1] if runs_idx+1 < len(p.parts) else None
        symbol = p.parts[runs_idx-1] if runs_idx-1 >= 0 else None
        run_dir = Path('result')/str(symbol)/'runs'/str(run_id)
        stg_file = run_dir / 'oos_results.json'
        if not stg_file.exists():
            return jsonify({'success': False, 'error': 'oos_results.json not found'}), 404
        doc = json.loads(stg_file.read_text(encoding='utf-8'))
        strategies = (doc.get('strategies') or {})
        items = (strategies.get('items') or {})
        item = items.get(stg_id)
        if not item:
            return jsonify({'success': False, 'error': 'strategy not found'}), 404
        # Обновим current
        if 'strategies' not in doc or not isinstance(doc['strategies'], dict):
            doc['strategies'] = {'current': None, 'items': {}}
        doc['strategies']['current'] = stg_id
        stg_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')
        # Применим qgate в Redis (если доступен)
        try:
            rc = Redis(host='redis', port=6379, db=0, decode_responses=True)
            params = item.get('params') or {}
            t1 = float(params.get('t1_base') or 0)
            t2 = float(params.get('t2_base') or 0)
            # gate_scale/gate_disable можно учесть на стороне трейдинга, пока сохранём базовые
            rc.set('trading:qgate', json.dumps({'T1': t1, 'T2': t2}, ensure_ascii=False))
        except Exception:
            pass
        return jsonify({'success': True, 'current': stg_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@models_admin_bp.get('/api/runs/list')
def api_runs_list():
    try:
        model_type = (request.args.get('model_type') or 'dqn').strip().lower()
        if model_type not in {'dqn', 'sac'}:
            model_type = 'dqn'

        symbol_raw = (request.args.get('symbol') or '').strip()
        if not symbol_raw:
            return jsonify({'success': False, 'error': 'symbol required'}), 400

        base = Path('result')
        symbol_candidates = []
        if model_type == 'sac':
            base = base / 'sac'
            core_symbol = symbol_raw.split(':',1)[-1]
            symbol_candidates = [core_symbol.lower(), core_symbol.upper(), core_symbol]
        else:
            core_symbol = symbol_raw
            symbol_candidates = [core_symbol.upper(), core_symbol.lower(), core_symbol]
        requested_value = core_symbol

        runs = []
        symbol_dir = resolve_symbol_dir(model_type, symbol_raw, create=False)
        if symbol_dir is None:
            return jsonify({'success': True, 'runs': []})

        runs_root = symbol_dir / 'runs'
        if not runs_root.exists():
            return jsonify({'success': True, 'runs': []})

        for rd in runs_root.iterdir():
            if not rd.is_dir():
                continue
            run_id = rd.name
            manifest = {}
            mf = rd / 'manifest.json'
            if mf.exists():
                try:
                    manifest = json.loads(mf.read_text(encoding='utf-8'))
                except Exception:
                    manifest = {}
            model_path = (rd / 'model.pth') if (rd / 'model.pth').exists() else None
            replay_path = (rd / 'replay.pkl') if (rd / 'replay.pkl').exists() else None
            result_path = (rd / 'train_result.pkl') if (rd / 'train_result.pkl').exists() else None
            # Попытаемся извлечь краткую статистику
            winrate = None; pl_ratio = None; episodes = None
            episodes_planned = None
            best_winrate = None
            episode_winrate_avg = None
            episode_winrate_last = None
            validation_avg = None
            validation_last = None
            seed_value = manifest.get('seed')
            if result_path:
                try:
                    import pickle as _pkl
                    with open(result_path,'rb') as _f:
                        _res = _pkl.load(_f)
                    _fs = _res.get('final_stats') or {}
                    winrate = _fs.get('winrate')
                    pl_ratio = _fs.get('pl_ratio')
                    episodes_planned = _res.get('episodes')
                    episodes = _res.get('actual_episodes', episodes_planned)

                    def _sanitize(seq):
                        if not isinstance(seq, (list, tuple)):
                            return []
                        cleaned = []
                        for value in seq:
                            if isinstance(value, (int, float)) and value == value:
                                cleaned.append(float(value))
                        return cleaned

                    episode_winrates = _sanitize(_res.get('episode_winrates'))
                    if episode_winrates:
                        episode_winrate_last = episode_winrates[-1]
                        episode_winrate_avg = sum(episode_winrates) / len(episode_winrates)
                        if not isinstance(_res.get('best_winrate'), (int, float)):
                            best_winrate = max(episode_winrates)
                        else:
                            best_winrate = _res.get('best_winrate')
                    else:
                        val = _res.get('best_winrate')
                        if isinstance(val, (int, float)):
                            best_winrate = float(val)

                    validation_rewards = _sanitize(_res.get('validation_rewards'))
                    if validation_rewards:
                        validation_last = validation_rewards[-1]
                        validation_avg = sum(validation_rewards) / len(validation_rewards)

                    if seed_value is None:
                        train_meta_seed = (_res.get('train_metadata') or {}).get('seed')
                        cfg_seed = (_res.get('cfg_snapshot') or {}).get('seed')
                        seed_value = train_meta_seed if train_meta_seed is not None else cfg_seed
                except Exception:
                    pass
            runs.append({
                'run_id': run_id,
                'parent_run_id': manifest.get('parent_run_id'),
                'root_id': manifest.get('root_id'),
                'seed': seed_value,
                'episodes_end': manifest.get('episodes_end'),
                'created_at': manifest.get('created_at'),
                'model_path': model_path.as_posix() if model_path else None,
                'replay_path': replay_path.as_posix() if replay_path else None,
                'result_path': result_path.as_posix() if result_path else None,
                'winrate': winrate,
                'pl_ratio': pl_ratio,
                'episodes': episodes,
                'episodes_planned': episodes_planned,
                'best_winrate': best_winrate,
                'episode_winrate_avg': episode_winrate_avg,
                'episode_winrate_last': episode_winrate_last,
                'validation_avg': validation_avg,
                'validation_last': validation_last,
                'symbol_dir': symbol_dir.name if symbol_dir else requested_value,
            'model_type': model_type,
            'agent_type': model_type,
            })
        # Простая сортировка по created_at, затем по run_id
        try:
            runs.sort(key=lambda r: (r.get('created_at') or '', r['run_id']))
        except Exception:
            runs.sort(key=lambda r: r['run_id'])
        return jsonify({'success': True, 'runs': runs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@models_admin_bp.get('/api/runs/oos_results')
def api_runs_oos_results():
    try:
        symbol = (request.args.get('symbol') or '').strip().upper()
        run_id = (request.args.get('run_id') or '').strip()
        if not symbol or not run_id:
            return jsonify({'success': False, 'error': 'symbol and run_id required'}), 400
        oos_file = Path('result') / symbol / 'runs' / run_id / 'oos_results.json'
        if not oos_file.exists():
            return jsonify({'success': False, 'error': 'oos_results.json not found'}), 404
        try:
            data = json.loads(oos_file.read_text(encoding='utf-8'))
            return jsonify({'success': True, 'data': data})
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to parse oos_results.json: {e}'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@models_admin_bp.post('/api/runs/delete')
def api_runs_delete():
    try:
        data = request.get_json(silent=True) or {}
        symbol = (data.get('symbol') or '').strip().upper()
        run_id = (data.get('run_id') or '').strip()
        if not symbol or not run_id:
            return jsonify({'success': False, 'error': 'symbol and run_id required'}), 400
        run_dir = Path('result') / symbol / 'runs' / run_id
        if not run_dir.exists() or not run_dir.is_dir():
            return jsonify({'success': False, 'error': 'run directory not found'}), 404
        deleted = []
        errors = []
        # Удаляем все файлы в директории run
        try:
            for p in run_dir.iterdir():
                if p.is_file():
                    try:
                        os.remove(p)
                        deleted.append(str(p))
                    except Exception as e:
                        errors.append(f'{p.name}: {e}')
        except Exception as e:
            errors.append(f'iterdir: {e}')
        # Пытаемся удалить директорию, если пуста
        removed_dir = False
        try:
            if run_dir.exists() and len(list(run_dir.iterdir())) == 0:
                run_dir.rmdir()
                removed_dir = True
        except Exception:
            removed_dir = False
        return jsonify({'success': True, 'deleted': deleted, 'removed_dir': removed_dir, 'errors': errors})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@models_admin_bp.get('/api/runs/trades')
def api_runs_trades():
    try:
        symbol = (request.args.get('symbol') or '').strip().upper()
        run_id = (request.args.get('run_id') or '').strip()
        trades_filename = (request.args.get('trades_file') or '').strip()

        if not symbol or not run_id or not trades_filename:
            return jsonify({'success': False, 'error': 'symbol, run_id and trades_file required'}), 400
        
        trades_file_path = Path('result') / symbol / 'runs' / run_id / trades_filename.split('/')[-1] # Принимаем относительный путь, но берем только имя файла
        
        if not trades_file_path.exists():
            return jsonify({'success': False, 'error': 'trades file not found'}), 404
        
        try:
            with open(trades_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify({'success': True, 'trades': data.get('trades', [])})
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to parse trades file: {e}'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500



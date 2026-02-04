from flask import Blueprint, render_template, jsonify, request  # type: ignore
from pathlib import Path
from celery.result import AsyncResult  # type: ignore
from tasks import celery  # type: ignore
import json

from utils.trade_lab_dataset import TradeLabDatasetConfig, write_symbol_runs_dataset_to_tmp

trade_opt_bp = Blueprint('trade_opt', __name__)


@trade_opt_bp.route('/trade-lab', methods=['GET'])
def trade_lab_page():
    return render_template('analitika/trade_lab.html')


@trade_opt_bp.get('/api/trade_lab/symbols')
def trade_lab_symbols():
    try:
        model_type = (request.args.get('model_type') or 'dqn').strip().lower()
        base = Path('result')
        if model_type == 'sac':
            base = base / 'sac'
        else:
            base = base / 'dqn'
        symbols = []
        if base.exists():
            for p in base.iterdir():
                if p.is_dir() and (p / 'runs').exists():
                    symbols.append(p.name.upper())
        symbols.sort()
        return jsonify({'success': True, 'symbols': symbols})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@trade_opt_bp.get('/api/trade_lab/runs')
def trade_lab_runs():
    try:
        symbol = (request.args.get('symbol') or '').strip()
        model_type = (request.args.get('model_type') or 'dqn').strip().lower()
        if not symbol:
            return jsonify({'success': False, 'error': 'symbol required'}), 400
        base = Path('result')
        runs_root = (base / 'sac' / symbol.lower() / 'runs') if model_type == 'sac' else (base / 'dqn' / symbol.upper() / 'runs')
        if not runs_root.exists():
            return jsonify({'success': True, 'runs': []})
        items = []
        for rd in runs_root.iterdir():
            if not rd.is_dir():
                continue
            mf = rd / 'manifest.json'
            model_pth = rd / 'model.pth'
            items.append({
                'run_id': rd.name,
                'manifest_exists': mf.exists(),
                'model_path': model_pth.as_posix() if model_pth.exists() else None,
            })
        try:
            items.sort(key=lambda r: (r.get('run_id') or ''))
        except Exception:
            pass
        return jsonify({'success': True, 'runs': items})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@trade_opt_bp.post('/api/trade_lab/dataset/build')
def trade_lab_build_dataset():
    """
    Генерирует JSON датасет по всем обученным run'ам выбранного symbol/model_type.
    Сохраняет в result/trade_lab_tmp/<SYMBOL>/trade_lab_dataset_<type>_<SYMBOL>_<ts>.json
    """
    try:
        data = request.get_json(silent=True) or {}
        symbol = (data.get('symbol') or '').strip()
        model_type = (data.get('model_type') or 'dqn').strip().lower()
        if not symbol:
            return jsonify({'success': False, 'error': 'symbol required'}), 400
        if model_type not in ('dqn', 'sac'):
            model_type = 'dqn'

        cfg = TradeLabDatasetConfig(model_type=model_type, out_dir='result/trade_lab_tmp', max_pkl_mb=64)
        out_path = write_symbol_runs_dataset_to_tmp(symbol, cfg=cfg)
        return jsonify({'success': True, 'dataset_path': out_path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@trade_opt_bp.post('/api/trade_lab/analyze')
def trade_lab_analyze():
    try:
        data = request.get_json(silent=True) or {}
        # TODO: реализовать анализ результатов торговли (парсинг и метрики)
        return jsonify({'success': True, 'message': 'Анализ будет реализован', 'params': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@trade_opt_bp.post('/api/trade_lab/optimize')
def trade_lab_optimize():
    try:
        data = request.get_json(silent=True) or {}
        task = celery.send_task('optimize_trading_params', kwargs={'payload': data})
        return jsonify({'success': True, 'task_id': task.id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@trade_opt_bp.get('/api/trade_lab/jobs/<task_id>/status')
def trade_lab_job_status(task_id: str):
    try:
        res = AsyncResult(task_id, app=celery)
        return jsonify({'success': True, 'state': res.state, 'ready': res.ready()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@trade_opt_bp.get('/api/trade_lab/jobs/<task_id>/result')
def trade_lab_job_result(task_id: str):
    try:
        res = AsyncResult(task_id, app=celery)
        if not res.ready():
            return jsonify({'success': False, 'error': 'not_ready', 'state': res.state}), 202
        return jsonify({'success': True, 'result': res.result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500



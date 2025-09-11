from flask import Blueprint, jsonify, request
from utils.redis_utils import get_redis_client
from tasks.celery_tasks import start_trading_task

trading_bp = Blueprint('trading', __name__)

@trading_bp.post('/api/trading/save_config')
def save_trading_config():
    try:
        data = request.get_json() or {}
        symbols = data.get('symbols') or []
        model_paths = data.get('model_paths') or []
        consensus = data.get('consensus') or None
        import json as _json
        rc = get_redis_client()
        if symbols:
            rc.set('trading:symbols', _json.dumps(symbols, ensure_ascii=False))
        if isinstance(model_paths, list) and len(model_paths) > 0:
            rc.set('trading:model_paths', _json.dumps(model_paths, ensure_ascii=False))
            rc.set('trading:last_model_paths', _json.dumps(model_paths, ensure_ascii=False))
        if consensus is not None:
            rc.set('trading:consensus', _json.dumps(consensus, ensure_ascii=False))
            rc.set('trading:last_consensus', _json.dumps(consensus, ensure_ascii=False))
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.post('/api/trading/start')
def start_trading():
    try:
        data = request.get_json() or {}
        symbols = data.get('symbols', ['BTCUSDT'])
        account_id = str(data.get('account_id') or '').strip()
        model_paths = data.get('model_paths') or []
        model_path = data.get('model_path')
        if (not model_path) and isinstance(model_paths, list) and len(model_paths) > 0:
            model_path = model_paths[0]
        if not model_path:
            model_path = '/workspace/models/btc/ensemble-a/current/dqn_model.pth'

        try:
            import json as _json
            _rc = get_redis_client()
            _rc.set('trading:model_path', model_path)
            if isinstance(model_paths, list):
                _rc.set('trading:model_paths', _json.dumps(model_paths, ensure_ascii=False))
            _rc.set('trading:symbols', _json.dumps(symbols, ensure_ascii=False))
            if account_id:
                _rc.set('trading:account_id', account_id)
            consensus = data.get('consensus')
            if consensus is not None:
                _rc.set('trading:consensus', _json.dumps(consensus, ensure_ascii=False))
                _rc.set('trading:last_consensus', _json.dumps(consensus, ensure_ascii=False))
            if isinstance(model_paths, list) and model_paths:
                _rc.set('trading:last_model_paths', _json.dumps(model_paths, ensure_ascii=False))
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
        except Exception:
            pass

        try:
            _rc_lock = get_redis_client()
            if _rc_lock.get('trading:agent_lock'):
                return jsonify({'success': False, 'error': 'Торговый шаг уже выполняется (agent_lock_active)'}), 429
        except Exception:
            pass

        task = start_trading_task.apply_async(args=[symbols, model_path], countdown=0, expires=300, queue='celery')
        return jsonify({'success': True, 'message': 'Торговля запущена через Celery задачу', 'task_id': task.id}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500



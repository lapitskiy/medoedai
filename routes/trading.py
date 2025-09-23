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
        take_profit_pct = data.get('take_profit_pct')  # Процент тейк-профита
        stop_loss_pct = data.get('stop_loss_pct')      # Процент стоп-лосса
        risk_management_type = data.get('risk_management_type')  # Способ управления рисками
        
        import json as _json
        rc = get_redis_client()
        if symbols:
            rc.set('trading:symbols', _json.dumps(symbols, ensure_ascii=False))
        if isinstance(model_paths, list) and len(model_paths) > 0:
            rc.set('trading:model_paths', _json.dumps(model_paths, ensure_ascii=False))
            rc.set('trading:last_model_paths', _json.dumps(model_paths, ensure_ascii=False))
        if consensus is not None:
            # Сохраняем консенсус для конкретного символа
            symbol = symbols[0] if symbols else 'ALL'
            rc.set(f'trading:consensus:{symbol}', _json.dumps(consensus, ensure_ascii=False))
            # Также сохраняем как общий для обратной совместимости
            rc.set('trading:consensus', _json.dumps(consensus, ensure_ascii=False))
            rc.set('trading:last_consensus', _json.dumps(consensus, ensure_ascii=False))
        if take_profit_pct is not None:
            rc.set('trading:take_profit_pct', str(take_profit_pct))
        if stop_loss_pct is not None:
            rc.set('trading:stop_loss_pct', str(stop_loss_pct))
        if risk_management_type is not None:
            rc.set('trading:risk_management_type', str(risk_management_type))
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
        take_profit_pct = data.get('take_profit_pct')  # Процент тейк-профита
        stop_loss_pct = data.get('stop_loss_pct')      # Процент стоп-лосса
        risk_management_type = data.get('risk_management_type')  # Способ управления рисками
        
        if (not model_path) and isinstance(model_paths, list) and len(model_paths) > 0:
            model_path = model_paths[0]
        if not model_path:
            model_path = '/workspace/models/btc/ensemble-a/current/dqn_model.pth'

        try:
            import json as _json
            _rc = get_redis_client()
            _rc.set('trading:model_path', model_path)
            # НЕ перезаписываем общие модели, чтобы не убить другие агенты
            # if isinstance(model_paths, list):
            #     _rc.set('trading:model_paths', _json.dumps(model_paths, ensure_ascii=False))
            # НЕ перезаписываем общие символы, чтобы не убить другие агенты
            # _rc.set('trading:symbols', _json.dumps(symbols, ensure_ascii=False))
            if account_id:
                _rc.set('trading:account_id', account_id)
            consensus = data.get('consensus')
            if consensus is not None:
                # Сохраняем консенсус для конкретного символа
                symbol = symbols[0] if symbols else 'ALL'
                _rc.set(f'trading:consensus:{symbol}', _json.dumps(consensus, ensure_ascii=False))
            if isinstance(model_paths, list) and model_paths:
                _rc.set('trading:last_model_paths', _json.dumps(model_paths, ensure_ascii=False))
            if take_profit_pct is not None:
                _rc.set('trading:take_profit_pct', str(take_profit_pct))
            if stop_loss_pct is not None:
                _rc.set('trading:stop_loss_pct', str(stop_loss_pct))
            if risk_management_type is not None:
                _rc.set('trading:risk_management_type', str(risk_management_type))
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
                'take_profit_pct': take_profit_pct,
                'stop_loss_pct': stop_loss_pct,
                'risk_management_type': risk_management_type,
            }
            # Сохраняем настройки для конкретного символа
            symbol = symbols[0] if symbols else 'ALL'
            _rc.set(f'trading:symbols:{symbol}', _json.dumps(symbols, ensure_ascii=False))
            _rc.set(f'trading:model_path:{symbol}', model_path)
            if isinstance(model_paths, list):
                _rc.set(f'trading:model_paths:{symbol}', _json.dumps(model_paths, ensure_ascii=False))
            
            # Сохраняем статус для конкретного символа
            _rc.set(f'trading:status:{symbol}', _json.dumps(initial_status, ensure_ascii=False))
            # НЕ перезаписываем общий статус, чтобы не убить другие агенты
            # _rc.set('trading:current_status', _json.dumps(initial_status, ensure_ascii=False))
            from datetime import datetime as _dt
            _rc.set('trading:current_status_ts', _dt.utcnow().isoformat())
        except Exception:
            pass

        try:
            _rc_lock = get_redis_client()
            # Проверяем блокировку для конкретного символа
            symbol = symbols[0] if symbols else 'ALL'
            lock_key = f'trading:agent_lock:{symbol}'
            if _rc_lock.get(lock_key):
                return jsonify({'success': False, 'error': f'Торговый шаг для {symbol} уже выполняется (agent_lock_active)'}), 429
        except Exception:
            pass

        task = start_trading_task.apply_async(args=[symbols, model_path], countdown=0, expires=300, queue='celery')
        return jsonify({'success': True, 'message': 'Торговля запущена через Celery задачу', 'task_id': task.id}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
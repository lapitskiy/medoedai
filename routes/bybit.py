from flask import Blueprint, jsonify, request
from utils.accounts import list_bybit_accounts
from utils.redis_utils import get_redis_client

bybit_bp = Blueprint('bybit', __name__)

@bybit_bp.get('/api/bybit/accounts')
def api_bybit_accounts():
    try:
        return jsonify({'success': True, 'accounts': list_bybit_accounts()}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@bybit_bp.get('/api/bybit/selected')
def api_bybit_selected():
    try:
        rc = get_redis_client()
        sel = rc.get('trading:account_id') if rc else None
        accounts = list_bybit_accounts()
        current = None
        if sel:
            for a in accounts:
                if str(a['id']) == str(sel):
                    current = a
                    break
        return jsonify({'success': True, 'selected': sel, 'account': current}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@bybit_bp.post('/api/bybit/select')
def api_bybit_select():
    try:
        data = request.get_json() or {}
        idx = str(data.get('id') or '').strip()
        if not idx:
            return jsonify({'success': False, 'error': 'id is required'}), 400
        accounts = list_bybit_accounts()
        acc = None
        for a in accounts:
            if str(a.get('id')) == idx:
                acc = a
                break
        if not acc:
            return jsonify({'success': False, 'error': 'account not found'}), 404
        # Строгий режим: не даём выбрать аккаунт без SECRET — иначе дальше неявно сработает фолбэк.
        if not bool(acc.get('has_secret')):
            return jsonify({
                'success': False,
                'error': f'Bybit аккаунт "{acc.get("label")}" (id={idx}) не выбран: нет BYBIT_{idx}_SECRET_KEY'
            }), 400
        rc = get_redis_client()
        if rc:
            rc.set('trading:account_id', idx)
        return jsonify({'success': True, 'selected': idx}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500



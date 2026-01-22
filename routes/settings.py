from __future__ import annotations

import re
from typing import Any

from flask import Blueprint, jsonify, request, render_template

from utils.settings_store import (
    ensure_settings_table,
    list_settings as _list_settings,
    upsert_setting as _upsert_setting,
    delete_setting as _delete_setting,
    get_setting_value as _get_setting_value,
)


settings_bp = Blueprint('settings', __name__)


_RX_BYBIT_KEY = re.compile(r'^BYBIT_(\d+)_(API_KEY|SECRET_KEY|LABEL)$')


def _mask_key(key: str) -> str:
    try:
        s = str(key or '').strip()
        if not s or len(s) < 6:
            return '***'
        return f"{s[:6]}…{s[-4:]}"
    except Exception:
        return '***'


def _safe_value_for_list(v: Any, is_secret: bool) -> Any:
    if is_secret:
        return '❓'
    return v


@settings_bp.get('/settings/api')
def settings_api_page():
    return render_template('settings_api.html')


@settings_bp.get('/settings')
def settings_page():
    return render_template('settings.html')


@settings_bp.get('/settings/encoders')
def settings_encoders_page():
    return render_template('settings_encoders.html')


@settings_bp.get('/api/settings/list')
def api_settings_list():
    """
    Список настроек.
    Query: scope?, group? (если group пустая строка — считаем None)
    """
    try:
        ensure_settings_table()
        scope = (request.args.get('scope') or '').strip() or None
        group_raw = request.args.get('group')
        group = None if (group_raw is None or str(group_raw).strip() == '') else str(group_raw).strip()
        items = _list_settings(scope=scope, group=group)
        for it in items:
            it['value'] = _safe_value_for_list(it.get('value'), bool(it.get('secret')))
        return jsonify({'success': True, 'items': items}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@settings_bp.post('/api/settings/upsert')
def api_settings_upsert():
    try:
        ensure_settings_table()
        data = request.get_json(silent=True) or {}
        scope = str(data.get('scope') or '').strip()
        group = str(data.get('group') or '').strip() or None
        key = str(data.get('key') or '').strip()
        value_type = str(data.get('type') or 'string').strip() or 'string'
        label = str(data.get('label') or '').strip() or None
        description = data.get('description')
        if isinstance(description, str):
            description = description.strip() or None
        secret = bool(data.get('secret') is True or str(data.get('secret') or '').strip().lower() in ('1', 'true', 'yes', 'on'))
        # ВАЖНО: если `value` отсутствует в payload — НЕ затираем текущее значение.
        # Это нужно для редактирования label/description без потери value (особенно secret).
        if 'value' in data:
            value = data.get('value')
            if value is not None:
                value = str(value)
        else:
            value = _get_setting_value(scope, group, key)

        saved = _upsert_setting(
            scope=scope,
            group=group,
            key=key,
            value_type=value_type,
            label=label,
            description=description,
            is_secret=secret,
            value=value,
        )
        return jsonify({'success': True, 'item': saved}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@settings_bp.post('/api/settings/delete')
def api_settings_delete():
    try:
        ensure_settings_table()
        data = request.get_json(silent=True) or {}
        sid = data.get('id')
        if sid is None:
            return jsonify({'success': False, 'error': 'id is required'}), 400
        ok = _delete_setting(int(sid))
        return jsonify({'success': True, 'deleted': bool(ok)}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@settings_bp.get('/api/settings/api/accounts')
def api_settings_list_accounts():
    try:
        exchange = str(request.args.get('exchange') or 'bybit').strip().lower()
        if exchange != 'bybit':
            return jsonify({'success': False, 'error': 'only bybit is supported'}), 400
        ensure_settings_table()
        rows = _list_settings(scope='api', group='bybit')
        acc: dict[str, dict] = {}
        for r in rows:
            k = str(r.get('key') or '').strip()
            m = _RX_BYBIT_KEY.match(k)
            if not m:
                continue
            idx = m.group(1)
            kind = m.group(2)
            acc.setdefault(idx, {'id': idx, 'label': f'Account {idx}', 'api_key_masked': None, 'has_secret': False})
            if kind == 'LABEL':
                if r.get('value'):
                    acc[idx]['label'] = str(r.get('value'))
            elif kind == 'API_KEY':
                # api key считаем секретом, но возвращаем маску
                val = _get_setting_value('api', 'bybit', k) or ''
                acc[idx]['api_key_masked'] = _mask_key(val)
            elif kind == 'SECRET_KEY':
                val = _get_setting_value('api', 'bybit', k) or ''
                acc[idx]['has_secret'] = bool(str(val).strip())
        items = list(acc.values())
        try:
            items.sort(key=lambda a: int(a.get('id') or 0))
        except Exception:
            items.sort(key=lambda a: str(a.get('id')))
        return jsonify({'success': True, 'exchange': 'bybit', 'accounts': items}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@settings_bp.post('/api/settings/api/accounts/save')
def api_settings_save_account():
    """
    Сохраняет аккаунт в Postgres (app_settings) в scope=api, group=bybit.
    Ключи: BYBIT_<id>_{LABEL,API_KEY,SECRET_KEY}
    """
    try:
        ensure_settings_table()
        data = request.get_json(silent=True) or {}
        exchange = str(data.get('exchange') or 'bybit').strip().lower()
        if exchange != 'bybit':
            return jsonify({'success': False, 'error': 'only bybit is supported'}), 400

        account_id = str(data.get('id') or '').strip()
        label = str(data.get('label') or '').strip() or None
        api_key = str(data.get('api_key') or '').strip()
        secret_key = str(data.get('secret_key') or '').strip()

        if account_id and not account_id.isdigit():
            return jsonify({'success': False, 'error': 'id must be numeric'}), 400
        if not api_key or not secret_key:
            return jsonify({'success': False, 'error': 'api_key and secret_key are required'}), 400

        # auto id
        if not account_id:
            # найдём следующий id по существующим ключам BYBIT_<id>_API_KEY
            rows = _list_settings(scope='api', group='bybit')
            ids = set()
            for r in rows:
                k = str(r.get('key') or '')
                m = re.match(r'^BYBIT_(\d+)_API_KEY$', k)
                if m:
                    ids.add(int(m.group(1)))
            account_id = str((max(ids) + 1) if ids else 1)
        if not label:
            label = f'Account {account_id}'

        _upsert_setting(scope='api', group='bybit', key=f'BYBIT_{account_id}_LABEL', value_type='string', is_secret=False, value=label)
        _upsert_setting(scope='api', group='bybit', key=f'BYBIT_{account_id}_API_KEY', value_type='string', is_secret=True, value=api_key)
        _upsert_setting(scope='api', group='bybit', key=f'BYBIT_{account_id}_SECRET_KEY', value_type='string', is_secret=True, value=secret_key)

        return jsonify({
            'success': True,
            'exchange': 'bybit',
            'id': account_id,
            'label': label,
            'api_key_masked': _mask_key(api_key),
            'restart_required': False,
            'message': 'Saved to Postgres (app_settings).'
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@settings_bp.post('/api/settings/api/accounts/delete')
def api_settings_delete_account():
    try:
        ensure_settings_table()
        data = request.get_json(silent=True) or {}
        exchange = str(data.get('exchange') or 'bybit').strip().lower()
        if exchange != 'bybit':
            return jsonify({'success': False, 'error': 'only bybit is supported'}), 400
        account_id = str(data.get('id') or '').strip()
        if not account_id or not account_id.isdigit():
            return jsonify({'success': False, 'error': 'id must be numeric'}), 400

        # удаляем три записи по (scope, group, key)
        rows = _list_settings(scope='api', group='bybit')
        target = {f'BYBIT_{account_id}_LABEL', f'BYBIT_{account_id}_API_KEY', f'BYBIT_{account_id}_SECRET_KEY'}
        deleted = 0
        for r in rows:
            if str(r.get('key') or '') in target:
                if r.get('id') is not None and _delete_setting(int(r['id'])):
                    deleted += 1

        return jsonify({
            'success': True,
            'exchange': 'bybit',
            'id': account_id,
            'deleted_count': deleted,
            'restart_required': False,
            'message': 'Deleted from Postgres (app_settings).'
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500



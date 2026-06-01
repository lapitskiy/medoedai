from __future__ import annotations

import json
import re
from typing import Any

from flask import Blueprint, jsonify, request, render_template

from utils.redis_utils import get_redis_client
from utils.settings_store import (
    ensure_settings_table,
    list_settings as _list_settings,
    upsert_setting as _upsert_setting,
    delete_setting as _delete_setting,
    get_setting_value as _get_setting_value,
)
from utils.bot_users_store import list_platform_users


from utils.bot_promo_store import create_promo_codes, list_promo_codes, redeem_promo_code, delete_promo_code

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


def _reset_trading_account_selection_cache() -> None:
    """
    Сбрасывает выбор аккаунта Bybit в Redis.
    Нужен после save/delete в /settings/api, чтобы не оставались stale account_id.
    """
    try:
        rc = get_redis_client()
        if not rc:
            return
        try:
            rc.delete('trading:account_id')
        except Exception:
            pass
        try:
            for k in rc.scan_iter(match='trading:account_id:*'):
                try:
                    rc.delete(k)
                except Exception:
                    continue
        except Exception:
            pass
    except Exception:
        pass


@settings_bp.get('/settings/api')
def settings_api_page():
    return render_template('settings_api.html')


@settings_bp.get('/settings/bots')
def settings_bots_page():
    return render_template('settings_bots.html')


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


from datetime import datetime
from orm.database import get_db_session
from orm.models import BotSubscription, BotUserIdentity

@settings_bp.get('/api/internal/active-telegram-chats')
def api_internal_active_telegram_chats():
    session_id = request.args.get('session_id')
    
    # load master sessions
    master_sessions_raw = _get_setting_value('trading', 'master', 'MASTER_SESSION_IDS')
    try:
        master_sessions = json.loads(master_sessions_raw) if master_sessions_raw else []
    except Exception:
        master_sessions = []
        
    is_master = bool(session_id and session_id in master_sessions)
    
    admin_chat_ids_raw = (_get_setting_value('api', 'telegram', 'TELEGRAM_CHAT_IDS') or '').strip()
    admin_chat_ids = [item.strip() for item in admin_chat_ids_raw.split(',') if item.strip()]
    
    if not is_master:
        return jsonify({'success': True, 'chat_ids': admin_chat_ids}), 200

    session = get_db_session()
    try:
        now = datetime.utcnow()
        active_identities = (
            session.query(BotUserIdentity.platform_user_id)
            .join(BotSubscription, BotUserIdentity.user_id == BotSubscription.user_id)
            .filter(
                BotUserIdentity.platform == 'telegram',
                BotSubscription.product_code == 'signals',
                BotSubscription.status == 'active',
                BotSubscription.paid_until > now
            )
            .all()
        )
        chat_ids = [str(r[0]) for r in active_identities if r[0]]
        final_chat_ids = list(set(chat_ids + admin_chat_ids))
        
        return jsonify({'success': True, 'chat_ids': final_chat_ids}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        session.close()

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
        _reset_trading_account_selection_cache()

        return jsonify({
            'success': True,
            'exchange': 'bybit',
            'id': account_id,
            'label': label,
            'api_key_masked': _mask_key(api_key),
            'restart_required': False,
            'message': 'Saved to Postgres (app_settings). Redis account selection cache reset.'
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
        _reset_trading_account_selection_cache()

        return jsonify({
            'success': True,
            'exchange': 'bybit',
            'id': account_id,
            'deleted_count': deleted,
            'restart_required': False,
            'message': 'Deleted from Postgres (app_settings). Redis account selection cache reset.'
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@settings_bp.get('/api/settings/bots/telegram')
def api_settings_get_telegram_bot():
    try:
        ensure_settings_table()
        bot_token = (_get_setting_value('api', 'telegram', 'TELEGRAM_BOT_TOKEN') or '').strip()
        chat_ids = (_get_setting_value('api', 'telegram', 'TELEGRAM_CHAT_IDS') or '').strip()
        proxy_url = (_get_setting_value('api', 'telegram', 'TELEGRAM_PROXY_URL') or '').strip()
        vless_url = (_get_setting_value('api', 'telegram', 'TELEGRAM_VLESS_URL') or '').strip()
        return jsonify({
            'success': True,
            'telegram': {
                'bot_token_masked': _mask_key(bot_token) if bot_token else None,
                'has_bot_token': bool(bot_token),
                'chat_ids': chat_ids,
                'proxy_url': proxy_url,
                'vless_url': vless_url,
            },
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@settings_bp.get('/api/bot/accounts/telegram')
def api_bot_accounts_telegram():
    try:
        users = list_platform_users(platform='telegram')
        return jsonify({'success': True, 'platform': 'telegram', 'users': users}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@settings_bp.get('/api/bot/accounts/max')
def api_bot_accounts_max():
    try:
        users = list_platform_users(platform='max')
        return jsonify({'success': True, 'platform': 'max', 'users': users}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@settings_bp.post('/api/settings/bots/telegram/save')
def api_settings_save_telegram_bot():
    try:
        ensure_settings_table()
        data = request.get_json(silent=True) or {}
        bot_token = str(data.get('bot_token') or '').strip()
        chat_ids = str(data.get('chat_ids') or '').strip()
        proxy_url = str(data.get('proxy_url') or '').strip()
        vless_url = str(data.get('vless_url') or '').strip()
        existing_token = (_get_setting_value('api', 'telegram', 'TELEGRAM_BOT_TOKEN') or '').strip()

        if not bot_token and not existing_token:
            return jsonify({'success': False, 'error': 'bot_token is required'}), 400

        if bot_token:
            _upsert_setting(
                scope='api',
                group='telegram',
                key='TELEGRAM_BOT_TOKEN',
                value_type='string',
                label='Telegram bot token',
                description='Bot token from @BotFather.',
                is_secret=True,
                value=bot_token,
            )
        if 'chat_ids' in data:
            _upsert_setting(
                scope='api',
                group='telegram',
                key='TELEGRAM_CHAT_IDS',
                value_type='string',
                label='Telegram chat IDs',
                description='Comma-separated Telegram chat IDs for signal delivery.',
                is_secret=False,
                value=chat_ids,
            )
        if 'proxy_url' in data:
            _upsert_setting(
                scope='api',
                group='telegram',
                key='TELEGRAM_PROXY_URL',
                value_type='string',
                label='Telegram proxy URL',
                description='HTTP/SOCKS proxy URL used only for Telegram Bot API requests.',
                is_secret=False,
                value=proxy_url,
            )
        if 'vless_url' in data:
            generated_proxy_url = None
            if vless_url:
                from utils.vless_proxy_config import write_sing_box_config_from_vless
                generated_proxy_url = write_sing_box_config_from_vless(vless_url)
            _upsert_setting(
                scope='api',
                group='telegram',
                key='TELEGRAM_VLESS_URL',
                value_type='string',
                label='Telegram VLESS URL',
                description='VLESS link for Telegram VPN/proxy infrastructure.',
                is_secret=True,
                value=vless_url,
            )
            if generated_proxy_url:
                _upsert_setting(
                    scope='api',
                    group='telegram',
                    key='TELEGRAM_PROXY_URL',
                    value_type='string',
                    label='Telegram proxy URL',
                    description='HTTP/SOCKS proxy URL used only for Telegram Bot API requests.',
                    is_secret=False,
                    value=generated_proxy_url,
                )

        saved_token = (_get_setting_value('api', 'telegram', 'TELEGRAM_BOT_TOKEN') or '').strip() or bot_token or existing_token
        saved_chat_ids = (_get_setting_value('api', 'telegram', 'TELEGRAM_CHAT_IDS') or '').strip()
        saved_proxy_url = (_get_setting_value('api', 'telegram', 'TELEGRAM_PROXY_URL') or '').strip()
        saved_vless_url = (_get_setting_value('api', 'telegram', 'TELEGRAM_VLESS_URL') or '').strip()
        return jsonify({
            'success': True,
            'telegram': {
                'bot_token_masked': _mask_key(saved_token),
                'has_bot_token': bool(saved_token),
                'chat_ids': saved_chat_ids,
                'proxy_url': saved_proxy_url,
                'vless_url': saved_vless_url,
            },
            'message': 'Saved to Postgres (app_settings).',
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@settings_bp.get('/api/settings/bots/max')
def api_settings_get_max_bot():
    try:
        ensure_settings_table()
        bot_token = (_get_setting_value('api', 'max', 'MAX_BOT_TOKEN') or '').strip()
        api_url = (_get_setting_value('api', 'max', 'MAX_API_URL') or '').strip()
        chat_ids = (_get_setting_value('api', 'max', 'MAX_CHAT_IDS') or '').strip()
        en_raw = (_get_setting_value('api', 'max', 'ENABLE_MAX_BOT_POLLING') or '').strip().lower()
        enable_max_bot_polling = en_raw in ('1', 'true', 'yes', 'on')
        return jsonify({
            'success': True,
            'max': {
                'bot_token_masked': _mask_key(bot_token) if bot_token else None,
                'has_bot_token': bool(bot_token),
                'api_url': api_url,
                'chat_ids': chat_ids,
                'enable_max_bot_polling': enable_max_bot_polling,
            },
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@settings_bp.get('/api/settings/bots/notification-models')
def api_settings_get_notification_models():
    try:
        ensure_settings_table()
        raw = _get_setting_value('notifications', 'xgb', 'ENTRY_MODEL_PATHS')
        selected = json.loads(raw) if raw else []
        if not isinstance(selected, list):
            raise RuntimeError('ENTRY_MODEL_PATHS must be a JSON list')
        selected = [str(item).strip() for item in selected if str(item or '').strip()]
        return jsonify({
            'success': True,
            'configured': raw is not None,
            'selected_model_paths': selected,
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@settings_bp.post('/api/settings/bots/notification-models/save')
def api_settings_save_notification_models():
    try:
        ensure_settings_table()
        data = request.get_json(silent=True) or {}
        raw_paths = data.get('model_paths')
        if not isinstance(raw_paths, list):
            return jsonify({'success': False, 'error': 'model_paths must be a list'}), 400
        selected = []
        seen = set()
        for item in raw_paths:
            path = str(item or '').replace('\\', '/').strip()
            if not path or path in seen:
                continue
            selected.append(path)
            seen.add(path)
        _upsert_setting(
            scope='notifications',
            group='xgb',
            key='ENTRY_MODEL_PATHS',
            value_type='json',
            label='XGB entry notification model paths',
            description='Only XGB entry signals from these model paths can be sent to Telegram/MAX.',
            is_secret=False,
            value=json.dumps(selected, ensure_ascii=False),
        )
        return jsonify({
            'success': True,
            'configured': True,
            'selected_model_paths': selected,
            'message': 'Saved to Postgres (app_settings).',
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@settings_bp.get('/api/bot/promo-codes')
def api_bot_promo_codes_get():
    try:
        codes = list_promo_codes()
        return jsonify({'success': True, 'promo_codes': codes}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@settings_bp.post('/api/bot/promo-codes')
def api_bot_promo_codes_create():
    try:
        data = request.get_json(silent=True) or {}
        count = int(data.get('count') or 1)
        duration_days = int(data.get('duration_days') or 14)
        max_uses = int(data.get('max_uses') or 1)
        note = data.get('note')
        
        if count < 1 or count > 100:
            return jsonify({'success': False, 'error': 'count must be between 1 and 100'}), 400
            
        codes = create_promo_codes(count=count, duration_days=duration_days, max_uses=max_uses, note=note)
        return jsonify({'success': True, 'created': len(codes), 'codes': codes}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@settings_bp.delete('/api/bot/promo-codes/<int:code_id>')
def api_bot_promo_codes_delete(code_id):
    try:
        ok = delete_promo_code(code_id)
        if ok:
            return jsonify({'success': True}), 200
        else:
            return jsonify({'success': False, 'error': 'Promo code not found or could not be deleted'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@settings_bp.post('/api/bot/promo/redeem')
def api_bot_promo_redeem():
    """Для тестов через curl/postman"""
    try:
        data = request.get_json(silent=True) or {}
        platform = str(data.get('platform') or '').strip()
        platform_user_id = str(data.get('platform_user_id') or '').strip()
        code = str(data.get('code') or '').strip()
        
        if not platform or not platform_user_id or not code:
            return jsonify({'success': False, 'error': 'platform, platform_user_id, and code are required'}), 400
            
        result = redeem_promo_code(platform=platform, platform_user_id=platform_user_id, code=code)
        status_code = 200 if result.get('success') else 400
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@settings_bp.post('/api/bot/broadcast')
def api_bot_broadcast():
    try:
        from utils.telegram_bot_poller import _send_message
        from orm.database import get_db_session
        from orm.models import BotSubscription, BotUserIdentity
        from datetime import datetime

        data = request.get_json(silent=True) or {}
        target = str(data.get('target') or '').strip()
        message = str(data.get('message') or '').strip()
        
        if not target or not message:
            return jsonify({'success': False, 'error': 'target and message are required'}), 400
            
        session = get_db_session()
        try:
            now = datetime.utcnow()
            
            # Базовый запрос: все пользователи Telegram
            query = session.query(BotUserIdentity.platform_user_id).filter(
                BotUserIdentity.platform == 'telegram'
            )

            if target == 'active':
                query = query.join(BotSubscription, BotUserIdentity.user_id == BotSubscription.user_id).filter(
                    BotSubscription.product_code == 'signals',
                    BotSubscription.status == 'active',
                    BotSubscription.paid_until > now
                )
            elif target == 'inactive':
                active_query = session.query(BotUserIdentity.platform_user_id).join(BotSubscription, BotUserIdentity.user_id == BotSubscription.user_id).filter(
                    BotUserIdentity.platform == 'telegram',
                    BotSubscription.product_code == 'signals',
                    BotSubscription.status == 'active',
                    BotSubscription.paid_until > now
                ).subquery()
                query = query.filter(~BotUserIdentity.platform_user_id.in_(active_query))
            elif target == 'all':
                pass # Без дополнительных фильтров
            else:
                return jsonify({'success': False, 'error': 'Invalid target'}), 400

            chat_ids = [str(r[0]) for r in query.all() if r[0]]
            
            sent_count = 0
            for chat_id in chat_ids:
                try:
                    _send_message(chat_id, message, with_default_keyboard=False)
                    sent_count += 1
                except Exception as e:
                    print(f"Error sending broadcast to {chat_id}: {e}", flush=True)

            return jsonify({'success': True, 'sent_count': sent_count, 'total_targeted': len(chat_ids)}), 200
        finally:
            session.close()
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@settings_bp.post('/api/settings/bots/max/save')
def api_settings_save_max_bot():
    try:
        ensure_settings_table()
        data = request.get_json(silent=True) or {}
        bot_token = str(data.get('bot_token') or '').strip()
        api_url = str(data.get('api_url') or '').strip()
        chat_ids = str(data.get('chat_ids') or '').strip()
        existing_token = (_get_setting_value('api', 'max', 'MAX_BOT_TOKEN') or '').strip()

        if not bot_token and not existing_token:
            if api_url or chat_ids:
                return jsonify({'success': False, 'error': 'bot_token is required'}), 400
            if 'enable_max_bot_polling' not in data:
                return jsonify({'success': False, 'error': 'bot_token is required'}), 400

        if bot_token:
            _upsert_setting(
                scope='api',
                group='max',
                key='MAX_BOT_TOKEN',
                value_type='string',
                label='MAX bot token',
                description='Bot token for MAX.',
                is_secret=True,
                value=bot_token,
            )
        if 'api_url' in data:
            _upsert_setting(
                scope='api',
                group='max',
                key='MAX_API_URL',
                value_type='string',
                label='MAX API URL',
                description='MAX Bot API base URL.',
                is_secret=False,
                value=api_url,
            )
        if 'chat_ids' in data:
            _upsert_setting(
                scope='api',
                group='max',
                key='MAX_CHAT_IDS',
                value_type='string',
                label='MAX chat IDs',
                description='Comma-separated MAX chat IDs for signal delivery.',
                is_secret=False,
                value=chat_ids,
            )
        if 'enable_max_bot_polling' in data:
            en = data.get('enable_max_bot_polling')
            on = en is True or str(en or '').strip().lower() in ('1', 'true', 'yes', 'on')
            _upsert_setting(
                scope='api',
                group='max',
                key='ENABLE_MAX_BOT_POLLING',
                value_type='bool',
                label='MAX long polling',
                description='Poll GET /updates for user registration. Stored in app_settings only.',
                is_secret=False,
                value='1' if on else '0',
            )

        saved_token = (_get_setting_value('api', 'max', 'MAX_BOT_TOKEN') or '').strip() or bot_token or existing_token
        saved_api_url = (_get_setting_value('api', 'max', 'MAX_API_URL') or '').strip()
        saved_chat_ids = (_get_setting_value('api', 'max', 'MAX_CHAT_IDS') or '').strip()
        en_raw = (_get_setting_value('api', 'max', 'ENABLE_MAX_BOT_POLLING') or '').strip().lower()
        enable_max_bot_polling = en_raw in ('1', 'true', 'yes', 'on')
        return jsonify({
            'success': True,
            'max': {
                'bot_token_masked': _mask_key(saved_token) if saved_token else None,
                'has_bot_token': bool(saved_token),
                'api_url': saved_api_url,
                'chat_ids': saved_chat_ids,
                'enable_max_bot_polling': enable_max_bot_polling,
            },
            'message': 'Saved to Postgres (app_settings).',
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500



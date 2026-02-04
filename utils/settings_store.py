from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime

from orm.database import get_engine, get_db_session
from orm.models import AppSetting


def ensure_settings_table() -> None:
    """Создаёт таблицу app_settings, если её ещё нет (checkfirst=True)."""
    try:
        engine = get_engine()
        AppSetting.__table__.create(bind=engine, checkfirst=True)
    except Exception:
        # Не валим приложение из-за проблем с миграцией/правами
        return None
    # Bootstrap: если ключи Bybit уже заданы через ENV, один раз импортируем их в БД,
    # чтобы после перехода на Postgres всё продолжило работать "как раньше".
    try:
        _bootstrap_bybit_from_env()
    except Exception:
        pass
    # Bootstrap: дефолты для LLM (GigaChat) — чтобы они появлялись в /settings без ручных вызовов.
    try:
        _bootstrap_gigachat_defaults()
    except Exception:
        pass
    # Bootstrap: дефолты для торгового сайзинга — чтобы они появлялись в /settings.
    try:
        _bootstrap_trading_sizing_defaults()
    except Exception:
        pass


def _bootstrap_bybit_from_env() -> None:
    import os
    import re
    session = get_db_session()
    try:
        # Если в БД уже есть хотя бы один BYBIT_* ключ — ничего не делаем.
        existing = session.query(AppSetting).filter(
            AppSetting.scope == 'api',
            AppSetting.group == 'bybit',
            AppSetting.key.like('BYBIT_%')
        ).limit(1).all()
        if existing:
            return

        rx = re.compile(r'^BYBIT_(\d+)_(API_KEY|SECRET_KEY|LABEL)$')
        # Соберём индексированные env
        found: dict[str, dict] = {}
        for k, v in os.environ.items():
            m = rx.match(k)
            if not m:
                continue
            idx = m.group(1)
            kind = m.group(2)
            found.setdefault(idx, {})
            found[idx][kind] = str(v or '').strip()

        if not found:
            return

        # Вставим только те, у кого есть хотя бы API_KEY
        for idx, data in found.items():
            api_key = (data.get('API_KEY') or '').strip()
            secret_key = (data.get('SECRET_KEY') or '').strip()
            label = (data.get('LABEL') or '').strip() or f'Account {idx}'
            if not api_key:
                continue
            rows = [
                ('BYBIT_%s_LABEL' % idx, 'string', False, label),
                ('BYBIT_%s_API_KEY' % idx, 'string', True, api_key),
                ('BYBIT_%s_SECRET_KEY' % idx, 'string', True, secret_key),
            ]
            for key_name, vt, is_secret, val in rows:
                row = AppSetting(scope='api', group='bybit', key=key_name)
                row.value_type = vt
                row.is_secret = bool(is_secret)
                row.value = val if (val is None or str(val).strip()) else None
                session.add(row)
        session.commit()
    finally:
        try:
            session.close()
        except Exception:
            pass


def _bootstrap_gigachat_defaults() -> None:
    """
    Создаёт дефолтные ключи (scope=llm, group=gigachat), если их ещё нет.
    ВАЖНО: без зависимостей на utils.gigachat_* чтобы не получить циклический импорт.
    """
    session = get_db_session()
    try:
        rows = session.query(AppSetting).filter(
            AppSetting.scope == 'llm',
            AppSetting.group == 'gigachat',
        ).all()
        existing = {str(r.key or '') for r in rows}

        defaults = [
            ('ENABLED', 'bool', False, '0'),
            ('ONLY_ON_BUY', 'bool', False, '0'),
            ('AUTH_TYPE', 'string', False, 'oauth'),
            ('CREDENTIALS', 'string', True, None),
            ('ACCESS_TOKEN', 'string', True, None),
            ('OAUTH_URL', 'string', False, 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth'),
            ('OAUTH_SCOPE', 'string', False, 'GIGACHAT_API_PERS'),
            ('CHAT_URL', 'string', False, 'https://gigachat.devices.sberbank.ru/api/v1/chat/completions'),
            ('MODEL', 'string', False, 'GigaChat'),
            ('TIMEOUT_SEC', 'number', False, '10'),
            ('TEMPERATURE', 'number', False, '0.2'),
            ('MAX_TOKENS', 'number', False, '256'),
            ('VERIFY_SSL', 'bool', False, '0'),
            ('SYSTEM_PROMPT', 'string', False, 'Ты опытный трейдер. Ответь кратко: 1) что видишь, 2) риск, 3) что бы сделал.'),
        ]

        changed = False
        for key, vt, is_secret, val in defaults:
            if key in existing:
                continue
            row = AppSetting(scope='llm', group='gigachat', key=key)
            row.value_type = vt
            row.is_secret = bool(is_secret)
            row.value = val
            session.add(row)
            changed = True

        if changed:
            session.commit()
    finally:
        try:
            session.close()
        except Exception:
            pass


def _bootstrap_trading_sizing_defaults() -> None:
    """
    Создаёт дефолтные ключи (scope=trading, group=sizing), если их ещё нет.
    """
    session = get_db_session()
    try:
        rows = session.query(AppSetting).filter(
            AppSetting.scope == 'trading',
            AppSetting.group == 'sizing',
        ).all()
        existing = {str(r.key or '') for r in rows}

        defaults = [
            ('ACCOUNT_PCT', 'number', False, '100', 'Account %', 'Доля свободного USDT для входа (1..100)'),
            ('ENTRY_SPLITS', 'number', False, '1', 'Entry splits', 'Сколько market-ордеров делать при входе (1=одним, 2=по половине)'),
        ]

        changed = False
        for key, vt, is_secret, val, label, desc in defaults:
            if key in existing:
                continue
            row = AppSetting(scope='trading', group='sizing', key=key)
            row.value_type = vt
            row.is_secret = bool(is_secret)
            row.value = val
            row.label = label
            row.description = desc
            session.add(row)
            changed = True

        if changed:
            session.commit()
    finally:
        try:
            session.close()
        except Exception:
            pass


def _norm(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def list_settings(scope: Optional[str] = None, group: Optional[str] = None) -> List[Dict[str, Any]]:
    ensure_settings_table()
    session = get_db_session()
    try:
        q = session.query(AppSetting)
        if scope:
            q = q.filter(AppSetting.scope == scope)
        if group is not None:
            q = q.filter(AppSetting.group == group)
        rows = q.order_by(AppSetting.scope.asc(), AppSetting.group.asc().nullsfirst(), AppSetting.key.asc()).all()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({
                'id': r.id,
                'scope': r.scope,
                'group': r.group,
                'key': r.key,
                'type': r.value_type,
                'label': r.label,
                'description': r.description,
                'secret': bool(r.is_secret),
                # value отдаём как есть — контролируй на уровне API, чтобы не утекли secret
                'value': r.value,
                'updated_at': r.updated_at.isoformat() if isinstance(r.updated_at, datetime) else None,
            })
        return out
    finally:
        try:
            session.close()
        except Exception:
            pass


def get_setting_value(scope: str, group: Optional[str], key: str) -> Optional[str]:
    ensure_settings_table()
    session = get_db_session()
    try:
        q = session.query(AppSetting).filter(
            AppSetting.scope == scope,
            AppSetting.key == key,
        )
        if group is None:
            q = q.filter(AppSetting.group.is_(None))
        else:
            q = q.filter(AppSetting.group == group)
        row = q.first()
        return row.value if row else None
    finally:
        try:
            session.close()
        except Exception:
            pass


def upsert_setting(
    *,
    scope: str,
    group: Optional[str],
    key: str,
    value_type: str = 'string',
    label: Optional[str] = None,
    description: Optional[str] = None,
    is_secret: bool = False,
    value: Optional[str] = None,
) -> Dict[str, Any]:
    ensure_settings_table()
    session = get_db_session()
    try:
        scope_n = _norm(scope)
        key_n = _norm(key)
        if not scope_n or not key_n:
            raise ValueError('scope and key are required')
        group_n = _norm(group)
        vt = _norm(value_type) or 'string'

        q = session.query(AppSetting).filter(AppSetting.scope == scope_n, AppSetting.key == key_n)
        if group_n is None:
            q = q.filter(AppSetting.group.is_(None))
        else:
            q = q.filter(AppSetting.group == group_n)
        row = q.first()
        if not row:
            row = AppSetting(scope=scope_n, group=group_n, key=key_n)
            session.add(row)

        row.value_type = vt
        row.label = _norm(label)
        row.description = description if (description is None or str(description).strip()) else None
        row.is_secret = bool(is_secret)
        row.value = value if (value is None or str(value).strip()) else None

        session.commit()
        session.refresh(row)
        return {
            'id': row.id,
            'scope': row.scope,
            'group': row.group,
            'key': row.key,
            'type': row.value_type,
            'label': row.label,
            'description': row.description,
            'secret': bool(row.is_secret),
            'updated_at': row.updated_at.isoformat() if row.updated_at else None,
        }
    finally:
        try:
            session.close()
        except Exception:
            pass


def delete_setting(setting_id: int) -> bool:
    ensure_settings_table()
    session = get_db_session()
    try:
        row = session.query(AppSetting).filter(AppSetting.id == int(setting_id)).first()
        if not row:
            return False
        session.delete(row)
        session.commit()
        return True
    finally:
        try:
            session.close()
        except Exception:
            pass



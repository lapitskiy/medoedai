import re
from typing import List, Dict

from utils.settings_store import ensure_settings_table, list_settings, get_setting_value

_RX_BYBIT_API = re.compile(r'^BYBIT_(\d+)_API_KEY$')

def _mask_key(key: str) -> str:
    try:
        if not key or len(key) < 6:
            return '***'
        return f"{key[:6]}…{key[-4:]}"
    except Exception:
        return '***'

def list_bybit_accounts() -> List[Dict]:
    """
    Читаем Bybit аккаунты из Postgres (app_settings), scope=api, group=bybit.
    Ожидаемые ключи: BYBIT_<id>_{LABEL,API_KEY,SECRET_KEY}
    """
    ensure_settings_table()
    accounts_map: Dict[str, Dict] = {}
    try:
        rows = list_settings(scope='api', group='bybit')
    except Exception:
        rows = []

    for r in rows:
        k = str(r.get('key') or '').strip()
        m = re.match(r'^BYBIT_(\d+)_(API_KEY|SECRET_KEY|LABEL)$', k)
        if not m:
            continue
        idx = m.group(1)
        kind = m.group(2)
        accounts_map.setdefault(idx, {'id': idx, 'label': f'Account {idx}', 'api_key_masked': None, 'has_secret': False})
        if kind == 'LABEL':
            v = get_setting_value('api', 'bybit', k) or ''
            if v.strip():
                accounts_map[idx]['label'] = v.strip()
        elif kind == 'API_KEY':
            v = get_setting_value('api', 'bybit', k) or ''
            accounts_map[idx]['api_key_masked'] = _mask_key(v)
        elif kind == 'SECRET_KEY':
            v = get_setting_value('api', 'bybit', k) or ''
            accounts_map[idx]['has_secret'] = bool(v.strip())

    accounts: List[Dict] = list(accounts_map.values())
    try:
        accounts.sort(key=lambda a: int(a['id']))
    except Exception:
        accounts.sort(key=lambda a: a['id'])
    return accounts


def get_bybit_account(account_id: str) -> Dict | None:
    """
    Возвращает описание Bybit аккаунта по id (как в BYBIT_<id>_* ENV).
    Нужен для UI/статуса, чтобы показывать "через какой API торгуем".
    """
    try:
        aid = str(account_id or '').strip()
        if not aid:
            return None
        for a in list_bybit_accounts():
            if str(a.get('id')) == aid:
                return a
    except Exception:
        return None
    return None

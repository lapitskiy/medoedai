import os
import re
from typing import List, Dict

_RX_BYBIT_API = re.compile(r'^BYBIT_(\d+)_API_KEY$')

def _mask_key(key: str) -> str:
    try:
        if not key or len(key) < 6:
            return '***'
        return f"{key[:6]}…{key[-4:]}"
    except Exception:
        return '***'

def list_bybit_accounts() -> List[Dict]:
    accounts: List[Dict] = []
    for k, v in os.environ.items():
        m = _RX_BYBIT_API.match(k)
        if not m:
            continue
        idx = m.group(1)
        label = os.environ.get(f'BYBIT_{idx}_LABEL', f'Account {idx}')
        secret = os.environ.get(f'BYBIT_{idx}_SECRET_KEY')
        accounts.append({
            'id': idx,
            'label': label,
            'api_key_masked': _mask_key(v or ''),
            'has_secret': bool(secret)
        })
    try:
        accounts.sort(key=lambda a: int(a['id']))
    except Exception:
        accounts.sort(key=lambda a: a['id'])
    return accounts



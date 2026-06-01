from __future__ import annotations

import json
import time
import uuid
from typing import Any


SESSIONS_INDEX_KEY = "trading:sessions"


def _json_loads(raw: Any, default: Any) -> Any:
    try:
        if raw in (None, ""):
            return default
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="ignore")
        value = json.loads(raw)
        return value if value is not None else default
    except Exception:
        return default


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def session_prefix(session_id: str) -> str:
    sid = str(session_id or "").strip()
    if not sid:
        raise ValueError("session_id is required")
    return f"trading:session:{sid}"


def session_config_key(session_id: str) -> str:
    return f"{session_prefix(session_id)}:config"


def session_status_key(session_id: str) -> str:
    return f"{session_prefix(session_id)}:status"


def session_lock_key(session_id: str) -> str:
    return f"{session_prefix(session_id)}:lock"


def session_runtime_key(session_id: str, name: str) -> str:
    return f"{session_prefix(session_id)}:{str(name).strip(':')}"


def symbol_sessions_key(symbol: str) -> str:
    return f"trading:symbol_sessions:{str(symbol or '').strip().upper()}"


def account_symbol_session_key(account_id: str, symbol: str) -> str:
    return f"trading:account_symbol_session:{str(account_id or '').strip()}:{str(symbol or '').strip().upper()}"


def make_session_id() -> str:
    return f"s_{uuid.uuid4().hex}"


def create_session(rc, config: dict[str, Any]) -> dict[str, Any]:
    session_id = str(config.get("session_id") or make_session_id())
    symbol = str(config.get("symbol") or "").strip().upper()
    account_id = str(config.get("account_id") or "").strip()
    if not symbol:
        raise ValueError("symbol is required")
    if not account_id:
        raise ValueError("account_id is required")

    doc = dict(config)
    doc["session_id"] = session_id
    doc["symbol"] = symbol
    doc["account_id"] = account_id
    doc["created_at"] = str(doc.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    doc["updated_at"] = str(doc.get("updated_at") or doc["created_at"])

    rc.set(session_config_key(session_id), _json_dumps(doc))
    rc.sadd(SESSIONS_INDEX_KEY, session_id)
    rc.sadd(symbol_sessions_key(symbol), session_id)
    rc.set(account_symbol_session_key(account_id, symbol), session_id)
    return doc


def load_session(rc, session_id: str) -> dict[str, Any] | None:
    doc = _json_loads(rc.get(session_config_key(session_id)), None)
    return doc if isinstance(doc, dict) else None


def save_session(rc, session_id: str, doc: dict[str, Any]) -> dict[str, Any]:
    payload = dict(doc)
    payload["session_id"] = str(session_id)
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rc.set(session_config_key(session_id), _json_dumps(payload))
    return payload


def get_runtime_value(rc, session_id: str, name: str, default: Any = None) -> Any:
    raw = rc.get(session_runtime_key(session_id, name))
    if raw in (None, ""):
        return default
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")
    return raw


def set_runtime_value(rc, session_id: str, name: str, value: Any) -> None:
    rc.set(session_runtime_key(session_id, name), str(value))


def delete_runtime_value(rc, session_id: str, name: str) -> None:
    rc.delete(session_runtime_key(session_id, name))


def get_runtime_json(rc, session_id: str, name: str, default: Any = None) -> Any:
    return _json_loads(rc.get(session_runtime_key(session_id, name)), default)


def set_runtime_json(rc, session_id: str, name: str, value: Any) -> None:
    rc.set(session_runtime_key(session_id, name), _json_dumps(value))


def get_status(rc, session_id: str) -> dict[str, Any] | None:
    status = _json_loads(rc.get(session_status_key(session_id)), None)
    return status if isinstance(status, dict) else None


def set_status(rc, session_id: str, status: dict[str, Any]) -> None:
    rc.set(session_status_key(session_id), _json_dumps(status))


def list_session_ids(rc, symbol: str | None = None) -> list[str]:
    if symbol:
        values = rc.smembers(symbol_sessions_key(symbol))
    else:
        values = rc.smembers(SESSIONS_INDEX_KEY)
    result = []
    for item in values or []:
        sid = str(item or "").strip()
        if sid:
            result.append(sid)
    return sorted(set(result))


def get_active_session_for_account_symbol(rc, account_id: str, symbol: str) -> str | None:
    raw = rc.get(account_symbol_session_key(account_id, symbol))
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")
    value = str(raw or "").strip()
    return value or None


def remove_session(rc, session_id: str) -> None:
    doc = load_session(rc, session_id) or {}
    symbol = str(doc.get("symbol") or "").strip().upper()
    account_id = str(doc.get("account_id") or "").strip()
    prefix = session_prefix(session_id)
    for key in rc.scan_iter(match=f"{prefix}:*"):
        try:
            rc.delete(key)
        except Exception:
            continue
    try:
        rc.delete(prefix)
    except Exception:
        pass
    try:
        rc.srem(SESSIONS_INDEX_KEY, session_id)
    except Exception:
        pass
    if symbol:
        try:
            rc.srem(symbol_sessions_key(symbol), session_id)
        except Exception:
            pass
    if symbol and account_id:
        try:
            current = get_active_session_for_account_symbol(rc, account_id, symbol)
            if current == session_id:
                rc.delete(account_symbol_session_key(account_id, symbol))
        except Exception:
            pass

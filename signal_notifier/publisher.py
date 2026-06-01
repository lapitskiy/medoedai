import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any


def _allowed_actions() -> set[str]:
    raw = os.getenv("SIGNAL_PUBLISH_ACTIONS", "buy,sell")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _norm_model_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").strip()


def _configured_entry_model_paths() -> set[str] | None:
    from utils.settings_store import ensure_settings_table, get_setting_value

    ensure_settings_table()
    raw = get_setting_value("notifications", "xgb", "ENTRY_MODEL_PATHS")
    if raw is None:
        return None
    parsed = json.loads(raw or "[]")
    if not isinstance(parsed, list):
        raise RuntimeError("ENTRY_MODEL_PATHS must be a JSON list")
    return {_norm_model_path(item) for item in parsed if _norm_model_path(item)}


def _event_model_paths(payload: dict[str, Any]) -> set[str]:
    raw_paths = payload.get("model_paths")
    if not isinstance(raw_paths, list):
        raw_paths = [payload.get("model_path")]
    return {_norm_model_path(item) for item in raw_paths if _norm_model_path(item)}


def _allowed_by_entry_model_filter(payload: dict[str, Any]) -> bool:
    configured_paths = _configured_entry_model_paths()
    if configured_paths is None:
        return True
    return bool(configured_paths and (_event_model_paths(payload) & configured_paths))


def publish_signal_event(redis_client: Any, payload: dict[str, Any]) -> str | None:
    if redis_client is None:
        return None

    event_type = str(payload.get("event_type") or "trade_signal").strip().lower()
    if event_type == "trade_signal":
        action = str(payload.get("action") or payload.get("decision") or "").lower()
        if action not in _allowed_actions():
            return None
        if not _allowed_by_entry_model_filter(payload):
            return None
    elif event_type == "trade_entry":
        action = str(payload.get("action") or payload.get("decision") or "").lower()
        if action not in {"buy", "sell"}:
            raise RuntimeError("trade_entry requires action buy/sell")
        if not str(payload.get("session_id") or "").strip():
            raise RuntimeError("trade_entry requires session_id")
        if not str(payload.get("symbol") or "").strip():
            raise RuntimeError("trade_entry requires symbol")
        entry_ts_ms = payload.get("entry_ts_ms")
        if entry_ts_ms in (None, "", "0"):
            raise RuntimeError("trade_entry requires entry_ts_ms")
        if not _allowed_by_entry_model_filter(payload):
            return None
    elif event_type != "xgb_exit_soon":
        return None

    stream_name = os.getenv("SIGNAL_STREAM_NAME", "signals:events")
    signal_id = payload.get("signal_id") or str(uuid.uuid4())
    event = {
        **payload,
        "signal_id": signal_id,
        "event_type": event_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    fields = {
        key: value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        for key, value in event.items()
        if value is not None
    }
    redis_client.xadd(stream_name, fields, maxlen=10000, approximate=True)
    return str(signal_id)


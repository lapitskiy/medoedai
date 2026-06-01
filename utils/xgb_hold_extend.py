from __future__ import annotations

from typing import Any

from utils.redis_utils import get_redis_client
from utils.trading_sessions import get_runtime_value, load_session, save_session, set_runtime_value


def extend_xgb_hold_steps(session_id: str, increment_steps: int = 50) -> dict[str, Any]:
    rc = get_redis_client()
    sid = str(session_id or "").strip()
    if not sid:
        raise RuntimeError("session_id is required")

    session_doc = load_session(rc, sid)
    if not isinstance(session_doc, dict):
        raise RuntimeError("session not found")

    symbol = str(session_doc.get("symbol") or "").strip().upper()
    model_path = str(session_doc.get("model_path") or "").strip()
    if "/models/xgb/" not in model_path.replace("\\", "/"):
        raise RuntimeError("active XGB agent not found for session")

    exit_mode = str(get_runtime_value(rc, sid, "exit_mode", session_doc.get("exit_mode")) or "").strip().lower()
    if exit_mode != "hold_steps":
        raise RuntimeError("session exit_mode is not hold_steps")

    increment = int(float(increment_steps))
    if increment < 5 or increment > 200:
        raise RuntimeError("increment_steps must be between 5 and 200")

    raw_current = get_runtime_value(rc, sid, "max_hold_steps", session_doc.get("max_hold_steps"))
    if raw_current in (None, ""):
        raise RuntimeError("current max_hold_steps not found")

    current_steps = int(float(raw_current))
    new_steps = current_steps + increment
    set_runtime_value(rc, sid, "max_hold_steps", str(new_steps))

    updated_doc = dict(session_doc)
    updated_doc["max_hold_steps"] = new_steps
    save_session(rc, sid, updated_doc)

    return {
        "success": True,
        "session_id": sid,
        "symbol": symbol,
        "previous_max_hold_steps": current_steps,
        "increment_steps": increment,
        "max_hold_steps": new_steps,
    }

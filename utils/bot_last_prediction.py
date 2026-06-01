from __future__ import annotations

import json
import re
from datetime import datetime, timezone, timedelta
from html import escape
from typing import Any

from utils.redis_utils import get_redis_client
from utils.trade_utils import get_model_predictions


LABEL_LAST_PREDICTION = "Последнее предсказание"
LABEL_EXTEND_XGB_HOLD = "Продлить +50"
XGB_HOLD_EXTEND_STEPS = 50
XGB_HOLD_EXTEND_CALLBACK_PREFIX = "xgb_hold50:"


def normalize_prediction_trigger(text: str) -> bool:
    """Текст кнопки / команды (без учёта регистра и краевых пробелов)."""
    return str(text or "").strip().casefold() == LABEL_LAST_PREDICTION.casefold()


def build_xgb_hold_extend_command(session_id: Any) -> str:
    return f"{LABEL_EXTEND_XGB_HOLD} {str(session_id or '').strip()}"


def build_xgb_hold_extend_callback(session_id: Any) -> str:
    return f"{XGB_HOLD_EXTEND_CALLBACK_PREFIX}{str(session_id or '').strip()}"


def parse_xgb_hold_extend_trigger(text: str) -> str | None:
    raw = str(text or "").strip()
    if raw.startswith(XGB_HOLD_EXTEND_CALLBACK_PREFIX):
        session_id = raw[len(XGB_HOLD_EXTEND_CALLBACK_PREFIX):].strip()
        return session_id or None
    prefix = LABEL_EXTEND_XGB_HOLD.casefold()
    if not raw.casefold().startswith(prefix):
        return None
    session_id = raw[len(LABEL_EXTEND_XGB_HOLD):].strip()
    return session_id or None


def format_last_prediction_text() -> str:
    """Последний BUY XGB-сигнал из источников, которые питают /xgb_predictions."""
    prediction = _latest_xgb_buy_from_db() or _latest_xgb_buy_from_redis()
    if not prediction:
        return "BUY-сигналов XGB пока нет."

    ts = _fmt_ts_with_age(prediction.get("timestamp") or prediction.get("created_at"))
    price = prediction.get("current_price")
    conf = prediction.get("confidence")
    lines = [
        f"Символ: {escape(str(prediction.get('symbol') or '?'))}",
        f"Действие: {escape(str(prediction.get('action') or 'buy').upper())}",
        f"Уверенность: {conf if conf is not None else '—'}",
        f"Цена: {price if price is not None else '—'}",
        f"<b>Время: {escape(ts)}</b>",
    ]
    mp = prediction.get("model_path")
    if mp:
        lines.append(f"Модель {_model_version_label(mp)}")
    return "\n".join(lines)


def _latest_xgb_buy_from_db() -> dict[str, Any] | None:
    for p in get_model_predictions(action="buy", limit=200) or []:
        model_path = getattr(p, "model_path", None)
        if not _is_xgb_model_path(model_path):
            continue
        return {
            "symbol": getattr(p, "symbol", None),
            "action": getattr(p, "action", None),
            "q_values": _loads_json(getattr(p, "q_values", None), []),
            "current_price": getattr(p, "current_price", None),
            "confidence": getattr(p, "confidence", None),
            "model_path": model_path,
            "timestamp": getattr(p, "timestamp", None),
            "created_at": getattr(p, "created_at", None),
        }
    return None


def _latest_xgb_buy_from_redis() -> dict[str, Any] | None:
    try:
        rc = get_redis_client()
        keys = sorted(rc.keys("trading:latest_result_*") or [], reverse=True)
        for key in keys:
            raw = rc.get(key)
            if not raw:
                continue
            snapshot = json.loads(raw)
            symbols = snapshot.get("symbols") or []
            for item in snapshot.get("predictions") or []:
                if not isinstance(item, dict):
                    continue
                if str(item.get("action") or "").strip().lower() != "buy":
                    continue
                if not _is_xgb_model_path(item.get("model_path")):
                    continue
                return {
                    "symbol": symbols[0] if symbols else None,
                    "action": item.get("action"),
                    "q_values": item.get("q_values") or [],
                    "current_price": None,
                    "confidence": item.get("confidence"),
                    "model_path": item.get("model_path"),
                    "timestamp": snapshot.get("timestamp"),
                    "created_at": snapshot.get("timestamp"),
                }
    except Exception:
        return None
    return None


def _is_xgb_model_path(value: Any) -> bool:
    try:
        return "/models/xgb/" in str(value or "").replace("\\", "/")
    except Exception:
        return False


def _loads_json(raw: Any, default: Any) -> Any:
    try:
        if raw in (None, ""):
            return default
        if isinstance(raw, str):
            return json.loads(raw)
        return raw
    except Exception:
        return default


def _fmt_ts_with_age(value: Any) -> str:
    if value is None:
        return "—"
    dt_utc = _coerce_utc_datetime(value)
    if dt_utc is not None:
        dt_msk = dt_utc + timedelta(hours=3)
        return f"{dt_msk.strftime('%Y-%m-%d %H:%M MSK')} ({_age_text(dt_utc)} назад)"
    return str(value)


def _model_version_label(model_path: Any) -> str:
    path = str(model_path or "").replace("\\", "/")
    match = re.search(r"/(v\d+)(?:/|$)", path)
    if match:
        return match.group(1)
    return "unknown"


def _coerce_utc_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                return parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
        except Exception:
            return None
    return None


def _age_text(dt_utc: datetime) -> str:
    delta = datetime.utcnow() - dt_utc
    total_minutes = max(0, int(delta.total_seconds() // 60))
    hours, minutes = divmod(total_minutes, 60)
    if hours <= 0:
        return f"{minutes} мин"
    return f"{hours} ч {minutes} мин"


def telegram_reply_keyboard_markup() -> dict[str, Any]:
    """ReplyKeyboard (Telegram Bot API): кнопки."""
    return {
        "keyboard": [
            [{"text": LABEL_LAST_PREDICTION}],
            [{"text": "Настройки"}]
        ],
        "resize_keyboard": True,
    }


def telegram_xgb_hold_extend_markup(session_id: Any) -> dict[str, Any]:
    return {
        "inline_keyboard": [[{
            "text": LABEL_EXTEND_XGB_HOLD,
            "callback_data": build_xgb_hold_extend_callback(session_id),
        }]]
    }


def max_inline_keyboard_attachment() -> dict[str, Any]:
    """MAX POST /messages: inline_keyboard; type message — шлёт в чат текст (см. MAX API)."""
    return {
        "type": "inline_keyboard",
        "payload": {
            "buttons": [
                [
                    {
                        "type": "message",
                        "text": LABEL_LAST_PREDICTION,
                        "payload": LABEL_LAST_PREDICTION,
                    }
                ]
            ]
        },
    }


def max_xgb_hold_extend_attachment(session_id: Any) -> dict[str, Any]:
    command = build_xgb_hold_extend_command(session_id)
    return {
        "type": "inline_keyboard",
        "payload": {
            "buttons": [[{
                "type": "message",
                "text": command,
                "payload": command,
            }]]
        },
    }

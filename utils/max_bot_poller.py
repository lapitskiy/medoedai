from __future__ import annotations

import threading
import time
from typing import Any, Optional

import requests

from utils.bot_last_prediction import (
    format_last_prediction_text,
    max_inline_keyboard_attachment,
    normalize_prediction_trigger,
    parse_xgb_hold_extend_trigger,
)
from utils.bot_users_store import register_platform_user
from utils.settings_store import get_setting_value, upsert_setting
from utils.time_log import msk_tag
from utils.xgb_hold_extend import extend_xgb_hold_steps


_STARTED = False

_DEFAULT_BASE = "https://platform-api.max.ru"


def _polling_enabled() -> bool:
    raw = (get_setting_value("api", "max", "ENABLE_MAX_BOT_POLLING") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def start_max_bot_poller() -> None:
    """Фоновый Long Polling GET /updates для регистрации пользователей MAX (dev/test).

    Не совмещать с Webhook: в API MAX одновременно допускается только один способ уведомлений.
    См. https://dev.max.ru/docs-api/methods/GET/updates
    """
    global _STARTED
    if _STARTED:
        return
    _STARTED = True
    thread = threading.Thread(target=_poll_loop, name="max-bot-poller", daemon=True)
    thread.start()
    print(msk_tag("[max_bot] poller started (long polling /updates)"), flush=True)


def _api_base() -> str:
    raw = (get_setting_value("api", "max", "MAX_API_URL") or "").strip().rstrip("/")
    if not raw:
        return _DEFAULT_BASE
    if raw.endswith("/messages"):
        return raw[: -len("/messages")].rstrip("/") or _DEFAULT_BASE
    return raw


def _get_token() -> str:
    return (get_setting_value("api", "max", "MAX_BOT_TOKEN") or "").strip()


def _load_marker() -> Optional[int]:
    raw = (get_setting_value("api", "max", "MAX_LP_MARKER") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _save_marker(marker: Optional[int]) -> None:
    if marker is None:
        return
    upsert_setting(
        scope="api",
        group="max",
        key="MAX_LP_MARKER",
        value_type="string",
        label="MAX long-poll marker",
        description="Cursor for GET /updates (do not edit manually).",
        is_secret=False,
        value=str(marker),
    )


def _poll_loop() -> None:
    marker: Optional[int] = _load_marker()
    while True:
        if not _polling_enabled():
            time.sleep(4)
            continue

        token = _get_token()
        if not token:
            time.sleep(15)
            continue

        base = _api_base()
        url = f"{base}/updates"
        headers = {"Authorization": token}
        params: dict[str, Any] = {
            "timeout": 30,
            "limit": 100,
            "types": "bot_started,bot_added,message_created",
        }
        if marker is not None:
            params["marker"] = marker

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=45)
            if not resp.ok:
                print(
                    msk_tag(f"[max_bot] updates HTTP {resp.status_code} {_mask(resp.text)[:300]}"),
                    flush=True,
                )
                time.sleep(10)
                continue

            data = resp.json()
            updates = data.get("updates")
            if updates is None:
                print(msk_tag(f"[max_bot] unexpected JSON keys: {list(data.keys())[:12]}"), flush=True)
                time.sleep(5)
                continue

            next_marker = data.get("marker")
            if next_marker is not None:
                try:
                    marker = int(next_marker)
                    _save_marker(marker)
                except (TypeError, ValueError):
                    pass

            for upd in updates:
                _handle_update(upd)

        except Exception as exc:
            print(msk_tag(f"[max_bot] poll error: {_mask(str(exc))}"), flush=True)
            time.sleep(10)


def _handle_update(update: dict[str, Any]) -> None:
    ut = str(update.get("update_type") or "")
    if ut == "bot_added" and update.get("is_channel") is True:
        return

    user, chat_id = _extract_user_and_chat(update)
    if not user or chat_id is None:
        return
    if user.get("is_bot"):
        return

    uid = user.get("user_id")
    if uid is None:
        return

    registered = register_platform_user(
        platform="max",
        platform_user_id=uid,
        username=user.get("username"),
        first_name=user.get("first_name"),
        last_name=user.get("last_name"),
        display_name=_display_name(user),
        language_code=None,
        raw_profile=user,
    )
    _ensure_max_chat_id(chat_id)

    incoming = _incoming_text(update)
    extend_session_id = parse_xgb_hold_extend_trigger(incoming)
    if extend_session_id:
        _send_chat_message(chat_id, _extend_hold_text(extend_session_id))
        return

    if normalize_prediction_trigger(incoming):
        _send_chat_message(
            chat_id,
            format_last_prediction_text(),
            with_keyboard=True,
        )
        return

    if ut == "bot_started":
        _send_chat_message(
            chat_id,
            "Готово. Профиль сохранён — уведомления о сделках можно направлять в этот чат.",
            with_keyboard=True,
        )
        print(msk_tag(f"[max_bot] bot_started user_id={registered['user_id']} chat_id={chat_id}"), flush=True)


def _extend_hold_text(session_id: str) -> str:
    try:
        result = extend_xgb_hold_steps(session_id, 50)
        return (
            f"✅ Hold продлён: {result['symbol']}\n"
            f"{result['previous_max_hold_steps']} + {result['increment_steps']} = {result['max_hold_steps']}"
        )
    except Exception as exc:
        return f"❌ Не удалось продлить hold: {exc}"


def _incoming_text(update: dict[str, Any]) -> str:
    ut = str(update.get("update_type") or "")
    if ut != "message_created":
        return ""
    msg = update.get("message") or {}
    body = msg.get("body")
    if isinstance(body, dict):
        text = body.get("text")
        if isinstance(text, str):
            return text.strip()
        if isinstance(text, dict):
            return str(text.get("value") or text.get("text") or "").strip()
    return str(msg.get("text") or "").strip()


def _extract_user_and_chat(update: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[int]]:
    ut = str(update.get("update_type") or "")
    chat_id = _as_int(update.get("chat_id"))
    user = update.get("user")

    if ut == "message_created":
        msg = update.get("message") or {}
        user = msg.get("sender") or user
        rec = msg.get("recipient") or {}
        if chat_id is None:
            chat_id = _as_int(rec.get("chat_id")) or _as_int(rec.get("id"))

    if user and chat_id is not None:
        return user if isinstance(user, dict) else None, chat_id
    return None, None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _display_name(user: dict[str, Any]) -> Optional[str]:
    full_name = " ".join(
        str(user.get(k) or "").strip()
        for k in ("first_name", "last_name")
        if str(user.get(k) or "").strip()
    ).strip()
    if full_name:
        return full_name
    username = str(user.get("username") or "").strip()
    return f"@{username}" if username else None


def _ensure_max_chat_id(chat_id: int) -> None:
    sid = str(chat_id).strip()
    if not sid:
        return
    current = (get_setting_value("api", "max", "MAX_CHAT_IDS") or "").strip()
    values = [item.strip() for item in current.split(",") if item.strip()]
    if sid not in values:
        values.append(sid)
        upsert_setting(
            scope="api",
            group="max",
            key="MAX_CHAT_IDS",
            value_type="string",
            label="MAX chat IDs",
            description="Comma-separated MAX chat IDs for signal delivery.",
            is_secret=False,
            value=",".join(values),
        )


def _send_chat_message(chat_id: int, text: str, *, with_keyboard: bool = False) -> None:
    token = _get_token()
    if not token:
        return
    base = _api_base()
    url = f"{base}/messages"
    headers = {"Authorization": token, "Content-Type": "application/json"}
    payload: dict[str, Any] = {"text": text, "format": "html"}
    if with_keyboard:
        payload["attachments"] = [max_inline_keyboard_attachment()]
    try:
        resp = requests.post(
            url,
            headers=headers,
            params={"chat_id": chat_id},
            json=payload,
            timeout=15,
        )
        if not resp.ok:
            print(msk_tag(f"[max_bot] send message failed: {resp.status_code} {_mask(resp.text)[:300]}"), flush=True)
    except Exception as exc:
        print(msk_tag(f"[max_bot] send message error: {_mask(str(exc))}"), flush=True)


def _mask(text: str) -> str:
    token = _get_token()
    if token:
        return str(text).replace(token, "***")
    return str(text)

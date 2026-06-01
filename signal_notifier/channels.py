import requests

LABEL_EXTEND_XGB_HOLD = "Продлить +50"
XGB_HOLD_EXTEND_CALLBACK_PREFIX = "xgb_hold50:"


def _as_int(value) -> int | None:
    try:
        return int(float(value))
    except Exception:
        return None


def build_signal_text(event: dict) -> str:
    if str(event.get("event_type") or "").strip().lower() == "xgb_exit_soon":
        symbol = str(event.get("symbol") or "").upper()
        session_id = event.get("session_id") or "unknown"
        remaining = _as_int(event.get("remaining_steps"))
        max_hold = _as_int(event.get("max_hold_steps"))
        candles_in_pos = _as_int(event.get("candles_in_pos"))
        direction = str(event.get("direction") or event.get("position_type") or "").upper()
        price = event.get("price")
        lines = [
            f"XGB скоро выход: {symbol} {direction}".strip(),
            f"Осталось hold_steps: {remaining if remaining is not None else 'n/a'}",
            f"В позиции: {candles_in_pos if candles_in_pos is not None else 'n/a'} / {max_hold if max_hold is not None else 'n/a'}",
            f"Цена: {price if price not in (None, '') else 'n/a'}",
            f"Session: {session_id}",
            "Кнопка ниже продлит удержание на +50 steps.",
        ]
        return "\n".join(lines)

    if str(event.get("event_type") or "").strip().lower() == "trade_entry":
        symbol = str(event.get("symbol") or "").upper()
        action = str(event.get("action") or event.get("decision") or "").upper()
        price = event.get("price")
        session_id = event.get("session_id") or "unknown"
        pos_type = str(event.get("position_type") or "").strip().upper()
        entry_ts_ms = _as_int(event.get("entry_ts_ms"))
        entry_time = "n/a"
        try:
            if entry_ts_ms:
                # UTC+3 (MSK) without tz libs, consistent with the rest of the project.
                import datetime as _dt
                dt = _dt.datetime.utcfromtimestamp(int(entry_ts_ms) / 1000) + _dt.timedelta(hours=3)
                entry_time = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            entry_time = "n/a"
        lines = [
            f"ENTRY: {symbol} {action}".strip(),
            f"Side: {pos_type if pos_type else 'n/a'}",
            f"Entry time (MSK): {entry_time}",
            f"Price: {price if price not in (None, '') else 'n/a'}",
            f"Session: {session_id}",
            f"ID: {event.get('signal_id')}",
        ]
        return "\n".join(lines)

    symbol = str(event.get("symbol") or "").upper()
    action = str(event.get("action") or event.get("decision") or "").upper()
    price = event.get("price")
    regime = event.get("market_regime") or "unknown"
    session_id = event.get("session_id") or "unknown"

    lines = [
        f"Signal: {symbol} {action}",
        f"Price: {price if price not in (None, '') else 'n/a'}",
        f"Regime: {regime}",
        f"Session: {session_id}",
        f"ID: {event.get('signal_id')}",
        "Повторное предсказание на вход.",
    ]
    return "\n".join(lines)


def build_signal_controls(event: dict) -> tuple[dict | None, list[dict] | None]:
    if str(event.get("event_type") or "").strip().lower() != "xgb_exit_soon":
        return None, None
    session_id = str(event.get("session_id") or "").strip()
    if not session_id:
        return None, None
    command = f"{LABEL_EXTEND_XGB_HOLD} {session_id}"
    telegram_markup = {
        "inline_keyboard": [[{
            "text": LABEL_EXTEND_XGB_HOLD,
            "callback_data": f"{XGB_HOLD_EXTEND_CALLBACK_PREFIX}{session_id}",
        }]]
    }
    max_attachment = {
        "type": "inline_keyboard",
        "payload": {
            "buttons": [[{
                "type": "message",
                "text": command,
                "payload": command,
            }]]
        },
    }
    return telegram_markup, [max_attachment]


def send_telegram(bot_token: str, chat_id: str, text: str, reply_markup: dict | None = None) -> None:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    resp = requests.post(url, json=payload, timeout=10)
    if not resp.ok:
        raise RuntimeError(f"Telegram send failed: {resp.status_code} {resp.text[:300]}")


def send_max(api_url: str, bot_token: str, chat_id: str, text: str, attachments: list[dict] | None = None) -> None:
    """Official MAX Bot API: POST /messages?chat_id=… Header Authorization: <token>."""
    base = (api_url or "").strip().rstrip("/")
    if not base:
        base = "https://platform-api.max.ru"
    if base.endswith("/messages"):
        url = base
    else:
        url = f"{base}/messages"
    headers = {"Authorization": bot_token, "Content-Type": "application/json"}
    payload = {"text": text}
    if attachments:
        payload["attachments"] = attachments
    resp = requests.post(
        url,
        headers=headers,
        params={"chat_id": chat_id},
        json=payload,
        timeout=15,
    )
    if not resp.ok:
        raise RuntimeError(f"MAX send failed: {resp.status_code} {resp.text[:300]}")


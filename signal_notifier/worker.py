import json
import time

import redis
import requests

from signal_notifier.channels import build_signal_controls, build_signal_text, send_max, send_telegram
from signal_notifier.config import NotifierConfig, load_config


ENTRY_ACTIONS = {"buy", "sell"}
ENTRY_STREAK_TTL_SECONDS = 86400 * 7
LAST_SIGNAL_TTL_SECONDS = 86400 * 7
SIGNATURE_IGNORED_FIELDS = {"signal_id", "created_at", "time", "timestamp", "closed_ts_ms"}


def _decode_event(fields: dict) -> dict:
    event = {}
    for key, value in fields.items():
        k = key.decode() if isinstance(key, bytes) else key
        v = value.decode() if isinstance(value, bytes) else value
        try:
            event[k] = json.loads(v)
        except Exception:
            event[k] = v
    return event


def _log_delivery(rc: redis.Redis, event: dict, channel: str, target: str, ok: bool, error: str | None = None) -> None:
    rc.xadd(
        "signals:deliveries",
        {
            "signal_id": str(event.get("signal_id") or ""),
            "channel": channel,
            "target": target,
            "ok": "1" if ok else "0",
            "error": error or "",
        },
        maxlen=50000,
        approximate=True,
    )


def _sent_key(channel: str, target: str, signal_id: str) -> str:
    return f"signals:sent:{channel}:{target}:{signal_id}"


def _already_sent(rc: redis.Redis, channel: str, target: str, signal_id: str) -> bool:
    key = f"signals:sent:{channel}:{target}:{signal_id}"
    return bool(rc.exists(key))


def _mark_sent(rc: redis.Redis, channel: str, target: str, signal_id: str) -> None:
    rc.set(_sent_key(channel, target, signal_id), "1", ex=86400 * 30)


def _last_signature_key(channel: str, target: str, session_id: str, symbol: str) -> str:
    return f"signals:last_signature:{channel}:{target}:{session_id}:{symbol}"


def _signal_signature(event: dict) -> str:
    payload = {
        str(k): v for k, v in event.items()
        if str(k) not in SIGNATURE_IGNORED_FIELDS and v not in (None, "")
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def _already_sent_previous(rc: redis.Redis, channel: str, target: str, event: dict, signature: str) -> bool:
    session_id = str(event.get("session_id") or "").strip()
    symbol = str(event.get("symbol") or "").strip().upper()
    if not session_id or not symbol:
        return False
    raw = rc.get(_last_signature_key(channel, target, session_id, symbol))
    prev = raw.decode() if isinstance(raw, bytes) else raw
    return str(prev or "") == signature


def _mark_previous_sent(rc: redis.Redis, channel: str, target: str, event: dict, signature: str) -> None:
    session_id = str(event.get("session_id") or "").strip()
    symbol = str(event.get("symbol") or "").strip().upper()
    if session_id and symbol:
        rc.set(_last_signature_key(channel, target, session_id, symbol), signature, ex=LAST_SIGNAL_TTL_SECONDS)


def _entry_streak_key(event: dict) -> str:
    session_id = str(event.get("session_id") or "").strip()
    symbol = str(event.get("symbol") or "").strip().upper()
    if not session_id or not symbol:
        raise RuntimeError("session_id and symbol are required for entry signal gating")
    return f"signals:entry_streak:{session_id}:{symbol}"


def _should_send_entry_signal(rc: redis.Redis, event: dict) -> bool:
    action = str(event.get("action") or event.get("decision") or "").strip().lower()
    if action not in ENTRY_ACTIONS:
        return True

    event_type = str(event.get("event_type") or "trade_signal").strip().lower()
    entry_executed = event.get("entry_executed")
    entry_executed_truthy = str(entry_executed).strip().lower() in ("1", "true", "yes", "on")

    # Entry should be tied to actual execution (position opened).
    # Send immediately for trade_entry (or explicitly marked executed). Otherwise do not send.
    if event_type == "trade_entry" or entry_executed_truthy:
        return True
    return False


def _get_active_telegram_chat_info(cfg: NotifierConfig, session_id: str) -> list[dict]:
    try:
        resp = requests.get(f"http://medoedai:5050/api/internal/active-telegram-chats?session_id={session_id}", timeout=5)
        if resp.ok:
            data = resp.json()
            if data.get("success"):
                if "chat_info" in data:
                    return data.get("chat_info", [])
                else:
                    return [{"chat_id": cid, "has_keys": False} for cid in data.get("chat_ids", [])]
    except Exception as e:
        print(f"Failed to fetch active chat info: {e}", flush=True)
    return [{"chat_id": cid, "has_keys": False} for cid in cfg.telegram_chat_ids]


def _send_event(rc: redis.Redis, cfg: NotifierConfig, event: dict) -> None:
    signal_id = str(event.get("signal_id") or "")
    if not signal_id:
        raise RuntimeError("signal_id is required")
    if not _should_send_entry_signal(rc, event):
        return
    signature = _signal_signature(event)
    telegram_markup, max_attachments = build_signal_controls(event)

    session_id = str(event.get("session_id") or "")

    if "telegram" in cfg.channels:
        active_chat_info = _get_active_telegram_chat_info(cfg, session_id)
        for info in active_chat_info:
            chat_id = info["chat_id"]
            has_keys = info.get("has_keys", False)
            
            if _already_sent(rc, "telegram", chat_id, signal_id):
                continue
            if _already_sent_previous(rc, "telegram", chat_id, event, signature):
                continue
                
            text = build_signal_text(event, has_keys=has_keys)
            send_telegram(cfg.telegram_bot_token or "", chat_id, text, reply_markup=telegram_markup)
            _mark_sent(rc, "telegram", chat_id, signal_id)
            _mark_previous_sent(rc, "telegram", chat_id, event, signature)
            _log_delivery(rc, event, "telegram", chat_id, True)

    if "max" in cfg.channels:
        text = build_signal_text(event, has_keys=False)
        for chat_id in cfg.max_chat_ids:
            if _already_sent(rc, "max", chat_id, signal_id):
                continue
            if _already_sent_previous(rc, "max", chat_id, event, signature):
                continue
            send_max(cfg.max_api_url or "", cfg.max_bot_token or "", chat_id, text, attachments=max_attachments)
            _mark_sent(rc, "max", chat_id, signal_id)
            _mark_previous_sent(rc, "max", chat_id, event, signature)
            _log_delivery(rc, event, "max", chat_id, True)


def _ensure_group(rc: redis.Redis, cfg: NotifierConfig) -> None:
    try:
        rc.xgroup_create(cfg.stream_name, cfg.consumer_group, id="0", mkstream=True)
    except redis.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


def run() -> None:
    cfg = load_config()
    rc = redis.from_url(cfg.redis_url, decode_responses=False)
    _ensure_group(rc, cfg)
    print(f"signal-notifier started stream={cfg.stream_name} channels={cfg.channels}", flush=True)

    while True:
        messages = rc.xreadgroup(
            cfg.consumer_group,
            cfg.consumer_name,
            {cfg.stream_name: ">"},
            count=10,
            block=5000,
        )
        if not messages:
            time.sleep(1)
            continue
        for _, entries in messages:
            for msg_id, fields in entries:
                event = _decode_event(fields)
                try:
                    _send_event(rc, cfg, event)
                    rc.xack(cfg.stream_name, cfg.consumer_group, msg_id)
                except Exception as exc:
                    _log_delivery(rc, event, "system", str(msg_id), False, str(exc))
                    print(f"delivery failed id={msg_id}: {exc}", flush=True)


if __name__ == "__main__":
    run()


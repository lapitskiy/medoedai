import time
import pandas as pd
import json
import requests
import re
from redis import Redis
import numpy as np
from utils.db_utils import db_get_or_fetch_ohlcv # Импортируем функцию загрузки данных
from utils.db_utils import load_latest_candles_from_csv_to_db
from utils.parser import parser_download_and_combine_with_library
from utils.trade_utils import create_model_prediction
import os
from datetime import datetime, timedelta
import uuid
import math
from html import escape
from utils.config_loader import get_config_value
import logging
import traceback
from utils.redis_utils import get_redis_client
from signal_notifier.publisher import publish_signal_event
from utils.trading_sessions import (
    delete_runtime_value,
    get_runtime_json,
    get_runtime_value,
    get_status as get_session_status,
    list_session_ids,
    load_session,
    session_lock_key,
    session_runtime_key,
    set_runtime_json,
    set_runtime_value,
    set_status as set_session_status,
)
from tasks import celery
from trading_agent.risk_trailing import setup_trailing_stop_bybit

logger = logging.getLogger(__name__)


def _xgb_side_risk_defaults(session_doc: dict | None, side: str) -> tuple[float | None, float | None]:
    try:
        side_name = str(side or "").strip().lower()
        if side_name not in ("long", "short") or not isinstance(session_doc, dict):
            return None, None
        model_paths = [str(item) for item in (session_doc.get("model_paths") or []) if item]
        model_roles = session_doc.get("model_roles") if isinstance(session_doc.get("model_roles"), dict) else {}
        side_path = None
        for path_item in model_paths:
            path_key = str(path_item).replace("\\", "/")
            path_abs = path_key if path_key.startswith("/") else ("/workspace/" + path_key.lstrip("/"))
            role = str(model_roles.get(str(path_item)) or model_roles.get(path_abs) or "").strip().lower()
            if role == side_name:
                side_path = str(path_item)
                break
        if side_path is None and len(model_paths) == 1:
            direction = str(session_doc.get("direction") or "").strip().lower()
            if direction == side_name:
                side_path = model_paths[0]
        if not side_path:
            return None, None
        from tasks.xgb_live import _load_xgb_runtime_meta
        cfg, _, _ = _load_xgb_runtime_meta(side_path)
        raw_tp = getattr(cfg, "entry_tp_pct", None)
        raw_sl = getattr(cfg, "entry_sl_pct", None)
        tp_pct = abs(float(raw_tp)) * 100.0 if raw_tp not in (None, "") else None
        sl_pct = abs(float(raw_sl)) * 100.0 if raw_sl not in (None, "") else None
        return tp_pct, sl_pct
    except Exception:
        return None, None


def _xgb_buy_best_prediction(pred_json: dict | None) -> dict | None:
    preds = (pred_json or {}).get("predictions") or []
    buy_preds = [
        p for p in preds
        if isinstance(p, dict)
        and str(p.get("action") or "").strip().lower() == "buy"
        and "/models/xgb/" in str(p.get("model_path") or "").replace("\\", "/")
    ]
    return max(buy_preds, key=lambda p: float(p.get("confidence") or 0.0)) if buy_preds else None


def _xgb_buy_prediction_signature(
    *,
    symbol: str,
    pred_json: dict | None,
    current_price: float | None,
    market_regime: str | None,
) -> str:
    best = _xgb_buy_best_prediction(pred_json)
    return json.dumps(
        {
            "symbol": str(symbol or "").strip().upper(),
            "action": "buy",
            "confidence": best.get("confidence") if isinstance(best, dict) else (pred_json or {}).get("confidence"),
            "price": current_price,
            "market_regime": str(market_regime or "unknown"),
            "model_path": str(best.get("model_path") or "").replace("\\", "/") if isinstance(best, dict) else "",
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def _is_repeated_last_xgb_buy_prediction(rc, key: str, signature: str) -> bool:
    if rc is None:
        return False
    raw = rc.get(key)
    prev = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
    return str(prev or "") == signature


def _mark_last_xgb_buy_prediction(rc, key: str, signature: str) -> None:
    if rc is not None:
        rc.setex(key, 86400 * 7, signature)


def _fmt_entry_time_msk(entry_time) -> str:
    if hasattr(entry_time, "replace") and hasattr(entry_time, "strftime"):
        dt = entry_time.replace(tzinfo=None) + timedelta(hours=3)
    else:
        dt = datetime.utcnow() + timedelta(hours=3)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _notify_max_position_entry(
    *,
    rc,
    session_id: str,
    symbol: str,
    side: str,
    entry_price: float | None,
    entry_time,
    stop_loss_pct: float | None,
    trade_number: str | None = None,
) -> None:
    """
    Шлёт MAX-уведомление о фактическом входе в позицию.
    Источник настроек только app_settings: api/max/MAX_*.
    """
    try:
        from utils.settings_store import get_setting_value

        if not entry_price or float(entry_price) <= 0:
            logger.warning("[max_entry_notify] skip: entry_price is empty session=%s symbol=%s", session_id, symbol)
            return

        dedupe_id = str(trade_number or "")
        if not dedupe_id:
            try:
                dedupe_id = str(int(float(entry_time.timestamp() * 1000))) if hasattr(entry_time, "timestamp") else ""
            except Exception:
                dedupe_id = ""
        dedupe_key = f"max:entry_notify:{session_id}:{symbol}:{dedupe_id or 'unknown'}"
        if rc is not None:
            try:
                if rc.get(dedupe_key):
                    return
                rc.setex(dedupe_key, 86400 * 7, "1")
            except Exception:
                pass

        token = (get_setting_value("api", "max", "MAX_BOT_TOKEN") or "").strip()
        api_url = (get_setting_value("api", "max", "MAX_API_URL") or "").strip().rstrip("/")
        chat_ids_raw = (get_setting_value("api", "max", "MAX_CHAT_IDS") or "").strip()
        if not token or not api_url or not chat_ids_raw:
            logger.warning("[max_entry_notify] skip: MAX_BOT_TOKEN/MAX_API_URL/MAX_CHAT_IDS required in app_settings")
            return

        if api_url.endswith("/messages"):
            url = api_url
        else:
            url = f"{api_url}/messages"
        chat_ids = [item.strip() for item in chat_ids_raw.split(",") if item.strip()]
        if not chat_ids:
            logger.warning("[max_entry_notify] skip: MAX_CHAT_IDS is empty")
            return

        side_n = str(side or "").strip().lower()
        side_ru = "LONG" if side_n in ("buy", "long") else "SHORT" if side_n in ("sell", "short") else side_n.upper()
        entry_price_f = float(entry_price)
        stop_line = "Стоп: не задан"
        if stop_loss_pct is not None:
            sl_pct_f = float(stop_loss_pct)
            if side_n in ("sell", "short"):
                stop_price = entry_price_f * (1.0 + sl_pct_f / 100.0)
            else:
                stop_price = entry_price_f * (1.0 - sl_pct_f / 100.0)
            stop_line = f"Стоп: {sl_pct_f:g}% (~{stop_price:.6g})"

        time_s = _fmt_entry_time_msk(entry_time)

        text = "\n".join([
            "Вход в позицию",
            f"Крипта: {symbol}",
            f"Сторона: {side_ru}",
            f"Цена входа: {entry_price_f:.6g}",
            f"Время: {time_s}",
            stop_line,
            "Сигнал на выход придёт отдельным сообщением.",
        ])

        for chat_id in chat_ids:
            try:
                resp = requests.post(
                    url,
                    headers={"Authorization": token, "Content-Type": "application/json"},
                    params={"chat_id": chat_id},
                    json={"text": text},
                    timeout=15,
                )
                if not resp.ok:
                    logger.warning("[max_entry_notify] send failed chat_id=%s http=%s body=%s", chat_id, resp.status_code, resp.text[:300])
            except Exception as exc:
                logger.warning("[max_entry_notify] send error chat_id=%s: %s", chat_id, exc)
    except Exception as exc:
        logger.warning("[max_entry_notify] failed: %s", exc)


def _notify_telegram_position_entry(
    *,
    rc,
    session_id: str,
    symbol: str,
    side: str,
    entry_price: float | None,
    entry_time,
    stop_loss_pct: float | None,
    trade_number: str | None = None,
) -> None:
    """Шлёт Telegram-уведомление о фактическом входе в позицию."""
    try:
        from utils.bot_users_store import list_platform_users
        from utils.settings_store import get_setting_value

        if not entry_price or float(entry_price) <= 0:
            logger.warning("[telegram_entry_notify] skip: entry_price is empty session=%s symbol=%s", session_id, symbol)
            return

        token = (get_setting_value("api", "telegram", "TELEGRAM_BOT_TOKEN") or "").strip()
        if not token:
            logger.warning("[telegram_entry_notify] skip: TELEGRAM_BOT_TOKEN required in app_settings")
            return

        users = list_platform_users(platform="telegram")
        chat_ids = [str(u.get("platform_user_id") or "").strip() for u in users if str(u.get("status") or "") == "active"]
        chat_ids = [cid for cid in dict.fromkeys(chat_ids) if cid]
        if not chat_ids:
            logger.warning("[telegram_entry_notify] skip: no registered telegram users")
            return

        side_n = str(side or "").strip().lower()
        side_ru = "LONG" if side_n in ("buy", "long") else "SHORT" if side_n in ("sell", "short") else side_n.upper()
        entry_price_f = float(entry_price)
        stop_line = "Стоп: не задан"
        if stop_loss_pct is not None:
            sl_pct_f = float(stop_loss_pct)
            stop_price = entry_price_f * (1.0 + sl_pct_f / 100.0) if side_n in ("sell", "short") else entry_price_f * (1.0 - sl_pct_f / 100.0)
            stop_line = f"Стоп: {sl_pct_f:g}% (~{stop_price:.6g})"

        time_s = _fmt_entry_time_msk(entry_time)
        text = "\n".join([
            "Вход в позицию",
            f"Крипта: {symbol}",
            f"Сторона: {side_ru}",
            f"Цена входа: {entry_price_f:.6g}",
            f"Время: {time_s}",
            stop_line,
            f"Trade#: {trade_number or '—'}",
        ])

        dedupe_id = str(trade_number or "")
        if not dedupe_id:
            try:
                dedupe_id = str(int(float(entry_time.timestamp() * 1000))) if hasattr(entry_time, "timestamp") else ""
            except Exception:
                dedupe_id = ""

        proxy_url = (get_setting_value("api", "telegram", "TELEGRAM_PROXY_URL") or "").strip()
        proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

        for chat_id in chat_ids:
            key = f"telegram:entry_notify:{session_id}:{symbol}:{dedupe_id or 'unknown'}:{chat_id}"
            if rc is not None and rc.get(key):
                continue
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text},
                timeout=10,
                proxies=proxies,
            )
            if not resp.ok:
                logger.warning("[telegram_entry_notify] send failed chat_id=%s http=%s body=%s", chat_id, resp.status_code, resp.text[:300])
                continue
            if rc is not None:
                rc.setex(key, 86400 * 7, "1")
    except Exception as exc:
        logger.warning("[telegram_entry_notify] failed: %s", exc)


def _notify_position_exit(
    *,
    rc,
    session_id: str,
    symbol: str,
    reason: str,
    side: str,
    entry_price: float | None,
    exit_price: float | None,
    qty: float | None,
    pnl: float | None,
    exit_ts_ms: int | None,
    order_id: str | None,
) -> None:
    """Шлёт уведомление о фактическом выходе в MAX и Telegram."""
    text = _build_position_exit_text(
        symbol=symbol,
        reason=reason,
        side=side,
        entry_price=entry_price,
        exit_price=exit_price,
        qty=qty,
        pnl=pnl,
        exit_ts_ms=exit_ts_ms,
    )
    dedupe_id = str(order_id or exit_ts_ms or "unknown")
    _send_max_position_exit(rc=rc, session_id=session_id, symbol=symbol, dedupe_id=dedupe_id, text=text)
    _send_telegram_position_exit(rc=rc, session_id=session_id, symbol=symbol, dedupe_id=dedupe_id, text=text)


def _build_position_exit_text(
    *,
    symbol: str,
    reason: str,
    side: str,
    entry_price: float | None,
    exit_price: float | None,
    qty: float | None,
    pnl: float | None,
    exit_ts_ms: int | None,
) -> str:
    reason_s = str(reason or "unknown").strip().lower()
    reason_ru = {
        "take_profit": "take profit",
        "stop_loss": "stop loss",
        "trailing": "trailing stop",
        "timeout": "timeout",
        "signal": "signal exit",
        "manual": "manual",
    }.get(reason_s, reason_s or "unknown")
    lines = [
        "Выход из позиции",
        f"Символ: {escape(str(symbol or '?'))}",
        f"Причина: {escape(reason_ru)}",
        f"Сторона закрытия: {escape(str(side or '—').upper())}",
        f"Цена входа: {entry_price:.6g}" if entry_price else "Цена входа: —",
        f"Цена выхода: {exit_price:.6g}" if exit_price else "Цена выхода: —",
        f"Количество: {qty:.8g}" if qty else "Количество: —",
        f"PnL: {pnl:.6g}" if pnl is not None else "PnL: —",
        f"<b>Время: {escape(_fmt_closed_ts_msk_with_age(exit_ts_ms))}</b>",
    ]
    return "\n".join(lines)


def _send_max_position_exit(*, rc, session_id: str, symbol: str, dedupe_id: str, text: str) -> None:
    try:
        from utils.settings_store import get_setting_value

        token = (get_setting_value("api", "max", "MAX_BOT_TOKEN") or "").strip()
        api_url = (get_setting_value("api", "max", "MAX_API_URL") or "").strip().rstrip("/")
        chat_ids_raw = (get_setting_value("api", "max", "MAX_CHAT_IDS") or "").strip()
        if not token or not api_url or not chat_ids_raw:
            logger.warning("[max_exit_notify] skip: MAX_BOT_TOKEN/MAX_API_URL/MAX_CHAT_IDS required in app_settings")
            return
        url = api_url if api_url.endswith("/messages") else f"{api_url}/messages"
        for chat_id in [item.strip() for item in chat_ids_raw.split(",") if item.strip()]:
            key = f"max:exit_notify:{session_id}:{symbol}:{dedupe_id}:{chat_id}"
            if rc is not None and rc.get(key):
                continue
            resp = requests.post(
                url,
                headers={"Authorization": token, "Content-Type": "application/json"},
                params={"chat_id": chat_id},
                json={"text": text, "format": "html"},
                timeout=15,
            )
            if not resp.ok:
                logger.warning("[max_exit_notify] send failed chat_id=%s http=%s body=%s", chat_id, resp.status_code, resp.text[:300])
                continue
            if rc is not None:
                rc.setex(key, 86400 * 7, "1")
    except Exception as exc:
        logger.warning("[max_exit_notify] failed: %s", exc)


def _send_telegram_position_exit(*, rc, session_id: str, symbol: str, dedupe_id: str, text: str) -> None:
    try:
        from utils.bot_users_store import list_platform_users
        from utils.settings_store import get_setting_value

        token = (get_setting_value("api", "telegram", "TELEGRAM_BOT_TOKEN") or "").strip()
        if not token:
            logger.warning("[telegram_exit_notify] skip: TELEGRAM_BOT_TOKEN required in app_settings")
            return
        users = list_platform_users(platform="telegram")
        chat_ids = [str(u.get("platform_user_id") or "").strip() for u in users if str(u.get("status") or "") == "active"]
        proxy_url = (get_setting_value("api", "telegram", "TELEGRAM_PROXY_URL") or "").strip()
        proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
        for chat_id in [cid for cid in dict.fromkeys(chat_ids) if cid]:
            key = f"telegram:exit_notify:{session_id}:{symbol}:{dedupe_id}:{chat_id}"
            if rc is not None and rc.get(key):
                continue
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
                proxies=proxies,
            )
            if not resp.ok:
                logger.warning("[telegram_exit_notify] send failed chat_id=%s http=%s body=%s", chat_id, resp.status_code, resp.text[:300])
                continue
            if rc is not None:
                rc.setex(key, 86400 * 7, "1")
    except Exception as exc:
        logger.warning("[telegram_exit_notify] failed: %s", exc)


def _notify_max_xgb_buy_prediction(
    *,
    rc,
    session_id: str,
    symbol: str,
    pred_json: dict | None,
    current_price: float | None,
    market_regime: str | None,
    closed_ts_ms: int | None,
) -> None:
    """Шлёт в MAX новый итоговый XGB BUY prediction. Настройки только из app_settings."""
    try:
        from utils.settings_store import get_setting_value

        signature = _xgb_buy_prediction_signature(
            symbol=symbol,
            pred_json=pred_json,
            current_price=current_price,
            market_regime=market_regime,
        )
        last_key = f"max:xgb_buy_prediction:last:{session_id}:{symbol}"
        if _is_repeated_last_xgb_buy_prediction(rc, last_key, signature):
            logger.info("[max_xgb_buy_notify] skipped repeated previous signal session=%s symbol=%s", session_id, symbol)
            return

        dedupe_key = f"max:xgb_buy_prediction:{session_id}:{symbol}:{closed_ts_ms or 'unknown'}"
        if rc is not None:
            try:
                if rc.get(dedupe_key):
                    return
                rc.setex(dedupe_key, 86400 * 7, "1")
            except Exception:
                pass

        token = (get_setting_value("api", "max", "MAX_BOT_TOKEN") or "").strip()
        api_url = (get_setting_value("api", "max", "MAX_API_URL") or "").strip().rstrip("/")
        chat_ids_raw = (get_setting_value("api", "max", "MAX_CHAT_IDS") or "").strip()
        if not token or not api_url or not chat_ids_raw:
            logger.warning("[max_xgb_buy_notify] skip: MAX_BOT_TOKEN/MAX_API_URL/MAX_CHAT_IDS required in app_settings")
            return

        url = api_url if api_url.endswith("/messages") else f"{api_url}/messages"
        chat_ids = [item.strip() for item in chat_ids_raw.split(",") if item.strip()]
        if not chat_ids:
            return

        text = _build_xgb_buy_prediction_text(
            symbol=symbol,
            pred_json=pred_json,
            current_price=current_price,
            market_regime=market_regime,
            closed_ts_ms=closed_ts_ms,
        )

        for chat_id in chat_ids:
            try:
                resp = requests.post(
                    url,
                    headers={"Authorization": token, "Content-Type": "application/json"},
                    params={"chat_id": chat_id},
                    json={"text": text, "format": "html"},
                    timeout=15,
                )
                if not resp.ok:
                    logger.warning("[max_xgb_buy_notify] send failed chat_id=%s http=%s body=%s", chat_id, resp.status_code, resp.text[:300])
                    continue
                _mark_last_xgb_buy_prediction(rc, last_key, signature)
            except Exception as exc:
                logger.warning("[max_xgb_buy_notify] send error chat_id=%s: %s", chat_id, exc)
    except Exception as exc:
        logger.warning("[max_xgb_buy_notify] failed: %s", exc)


def _notify_telegram_xgb_buy_prediction(
    *,
    rc,
    session_id: str,
    symbol: str,
    pred_json: dict | None,
    current_price: float | None,
    market_regime: str | None,
    closed_ts_ms: int | None,
) -> None:
    """Шлёт новый XGB BUY prediction всем зарегистрированным Telegram users."""
    try:
        from utils.bot_users_store import list_platform_users
        from utils.settings_store import get_setting_value

        token = (get_setting_value("api", "telegram", "TELEGRAM_BOT_TOKEN") or "").strip()
        if not token:
            logger.warning("[telegram_xgb_buy_notify] skip: TELEGRAM_BOT_TOKEN required in app_settings")
            return

        users = list_platform_users(platform="telegram")
        chat_ids = [str(u.get("platform_user_id") or "").strip() for u in users if str(u.get("status") or "") == "active"]
        chat_ids = [chat_id for chat_id in dict.fromkeys(chat_ids) if chat_id]
        if not chat_ids:
            logger.warning("[telegram_xgb_buy_notify] skip: no registered telegram users")
            return

        signature = _xgb_buy_prediction_signature(
            symbol=symbol,
            pred_json=pred_json,
            current_price=current_price,
            market_regime=market_regime,
        )
        text = _build_xgb_buy_prediction_text(
            symbol=symbol,
            pred_json=pred_json,
            current_price=current_price,
            market_regime=market_regime,
            closed_ts_ms=closed_ts_ms,
        )
        proxy_url = (get_setting_value("api", "telegram", "TELEGRAM_PROXY_URL") or "").strip()
        proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

        for chat_id in chat_ids:
            dedupe_key = f"telegram:xgb_buy_prediction:{session_id}:{symbol}:{closed_ts_ms or 'unknown'}:{chat_id}"
            last_key = f"telegram:xgb_buy_prediction:last:{session_id}:{symbol}:{chat_id}"
            if rc is not None:
                try:
                    if _is_repeated_last_xgb_buy_prediction(rc, last_key, signature):
                        logger.info("[telegram_xgb_buy_notify] skipped repeated previous signal session=%s symbol=%s chat_id=%s", session_id, symbol, chat_id)
                        continue
                    if rc.get(dedupe_key):
                        continue
                except Exception:
                    pass
            try:
                resp = requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                    timeout=10,
                    proxies=proxies,
                )
                if not resp.ok:
                    logger.warning("[telegram_xgb_buy_notify] send failed chat_id=%s http=%s body=%s", chat_id, resp.status_code, resp.text[:300])
                    continue
                if rc is not None:
                    try:
                        rc.setex(dedupe_key, 86400 * 7, "1")
                        _mark_last_xgb_buy_prediction(rc, last_key, signature)
                    except Exception:
                        pass
            except Exception as exc:
                logger.warning("[telegram_xgb_buy_notify] send error chat_id=%s: %s", chat_id, exc)
    except Exception as exc:
        logger.warning("[telegram_xgb_buy_notify] failed: %s", exc)


def _build_xgb_buy_prediction_text(
    *,
    symbol: str,
    pred_json: dict | None,
    current_price: float | None,
    market_regime: str | None,
    closed_ts_ms: int | None,
) -> str:
    best = _xgb_buy_best_prediction(pred_json)
    conf = best.get("confidence") if isinstance(best, dict) else (pred_json or {}).get("confidence")
    model_path = best.get("model_path") if isinstance(best, dict) else None
    lines = [
        "Новый XGB BUY-сигнал",
        f"Символ: {escape(str(symbol or '?'))}",
        "Действие: BUY",
        f"Уверенность: {conf if conf is not None else '—'}",
        f"Цена: {current_price if current_price is not None else '—'}",
        f"<b>Время: {escape(_fmt_closed_ts_msk_with_age(closed_ts_ms))}</b>",
        f"Режим рынка: {escape(str(market_regime or 'unknown'))}",
    ]
    if model_path:
        lines.append(f"Модель {_model_version_label(model_path)}")
    lines.append("Повторное предсказание на вход.")
    return "\n".join(lines)


def _fmt_closed_ts_msk_with_age(closed_ts_ms: int | None) -> str:
    if not closed_ts_ms:
        return "—"
    dt_utc = datetime.utcfromtimestamp(int(closed_ts_ms) / 1000)
    dt_msk = dt_utc + timedelta(hours=3)
    return dt_msk.strftime("%Y-%m-%d %H:%M")


def _model_version_label(model_path: object) -> str:
    match = re.search(r"/(v\d+)(?:/|$)", str(model_path or "").replace("\\", "/"))
    return match.group(1) if match else "unknown"


def _norm_notify_model_path(model_path: object) -> str:
    return str(model_path or "").replace("\\", "/").strip()


def _entry_notifications_allowed_for_model_paths(model_paths: list | None) -> bool:
    from utils.settings_store import get_setting_value

    raw = get_setting_value("notifications", "xgb", "ENTRY_MODEL_PATHS")
    if raw is None:
        return True
    selected = json.loads(raw or "[]")
    if not isinstance(selected, list):
        raise RuntimeError("ENTRY_MODEL_PATHS must be a JSON list")
    selected_set = {_norm_notify_model_path(item) for item in selected if _norm_notify_model_path(item)}
    event_set = {_norm_notify_model_path(item) for item in (model_paths or []) if _norm_notify_model_path(item)}
    return bool(selected_set and (event_set & selected_set))


def _safe_json_loads(raw: str) -> dict:
    try:
        if not raw:
            return {}
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="ignore")
        return json.loads(raw) if isinstance(raw, str) else {}
    except Exception:
        return {}

def _attach_exec_error_to_group(symbol: str, ensemble_group_id: str, exec_error: str, exec_error_type: str | None = None, extra: dict | None = None) -> bool:
    """Дописывает exec_error в market_conditions у предсказаний текущего тика ансамбля.

    Почему так: UI дедуплицирует по ensemble_group_id, поэтому отдельную запись "error" он может не показать.
    """
    try:
        from orm.models import ModelPrediction
        from orm.database import get_db_session
        updated = False
        with get_db_session() as session:
            # Берём небольшой хвост последних записей по символу и ищем нужный gid
            rows = (
                session.query(ModelPrediction)
                .filter(ModelPrediction.symbol == symbol)
                .order_by(ModelPrediction.id.desc())
                .limit(100)
                .all()
            )
            for r in rows:
                mc = _safe_json_loads(getattr(r, "market_conditions", None))
                if not isinstance(mc, dict):
                    continue
                if str(mc.get("ensemble_group_id") or "") != str(ensemble_group_id):
                    continue
                mc["exec_error"] = str(exec_error)
                if exec_error_type:
                    mc["exec_error_type"] = str(exec_error_type)
                if isinstance(extra, dict) and extra:
                    # Не перетираем уже существующие ключи, но добавляем новые
                    for k, v in extra.items():
                        if k not in mc:
                            mc[k] = v
                r.market_conditions = json.dumps(mc, ensure_ascii=False)
                updated = True
            if updated:
                session.commit()
        return bool(updated)
    except Exception:
        return False

def _get_price_tick(exchange, symbol: str) -> float:
    """Определяет шаг цены (tickSize) для символа: сначала info.priceFilter.tickSize, затем precision.price.

    Возвращает float тик; фолбэк 0.0001.
    """
    try:
        mkt = exchange.market(symbol)
        info = (mkt.get('info') or {})
        pf = (info.get('priceFilter') or {})
        tick = pf.get('tickSize')
        if tick is not None:
            return float(tick)
        prec = (mkt.get('precision') or {}).get('price')
        if prec is not None and int(prec) >= 0:
            return 10 ** (-int(prec))
    except Exception:
        pass
    return 0.0001

def _round_to_tick(price: float, tick: float) -> float:
    """Округляет цену к ближайшему шагу тика."""
    try:
        if tick and tick > 0:
            return float(round(float(price) / tick) * tick)
    except Exception:
        pass
    return float(f"{float(price):.4f}")

def _apply_safety_buffer_qty(amount: float, buffer_pct: float | None = None) -> float:
    """Возвращает уменьшенное количество с запасом под комиссию/округления.

    buffer_pct можно задать через ENV ORDER_FEE_BUFFER_PCT (в процентах, например 0.5 для 0.5%).
    По умолчанию 0.5%.
    """
    try:
        if buffer_pct is None:
            env_raw = os.getenv('ORDER_FEE_BUFFER_PCT')
            if env_raw is not None and str(env_raw).strip() != '':
                buffer_pct = float(env_raw)
            else:
                buffer_pct = 0.5
        frac = max(0.0, min(5.0, float(buffer_pct))) / 100.0
        safe = float(amount) * (1.0 - frac)
        return max(0.0, float(f"{safe:.12f}"))
    except Exception:
        return float(amount)

XGB_SIGNAL_EXIT_WINDOW_DEFAULT = 20
XGB_SIGNAL_EXIT_THRESHOLD_DEFAULT = 0.5
XGB_ENTRY_ATTEMPTS_HISTORY_KEY = "trading:xgb_entry_attempts_history"
XGB_ENTRY_ATTEMPTS_HISTORY_LIMIT = 300

def _to_float_or_none(value) -> float | None:
    try:
        if value in (None, ''):
            return None
        return float(value)
    except Exception:
        return None

def _derive_xgb_signal_score(q_values, task_name=None, confidence=None) -> float | None:
    try:
        task = str(task_name).strip().lower() if task_name not in (None, '') else None
        if isinstance(q_values, list) and len(q_values) >= 3:
            if task in ('entry_short', 'exit_long'):
                return _to_float_or_none(q_values[2])
            if task in ('entry_long', 'exit_short', 'directional', None):
                return _to_float_or_none(q_values[1])
    except Exception:
        pass
    return _to_float_or_none(confidence)

def _extract_xgb_signal_snapshot(
    pred_json: dict | None,
    *,
    model_paths: list[str] | None = None,
    threshold_override: float | None = None,
) -> dict | None:
    try:
        threshold_by_model: dict[str, float] = {}
        fallback_threshold = _to_float_or_none(threshold_override)
        if isinstance(model_paths, list):
            try:
                from tasks.xgb_live import _load_xgb_runtime_meta
                for model_path in model_paths:
                    model_path_s = str(model_path or '').strip()
                    if not model_path_s or model_path_s in threshold_by_model:
                        continue
                    try:
                        cfg, _, _ = _load_xgb_runtime_meta(model_path_s)
                        thr = _to_float_or_none(getattr(cfg, 'p_enter_threshold', None))
                        if thr is not None:
                            threshold_by_model[model_path_s] = float(thr)
                    except Exception:
                        continue
            except Exception:
                pass
        preds = (pred_json or {}).get('predictions') or []
        if not isinstance(preds, list) or not preds:
            preds = []
        signals = []
        thresholds = []
        for pred in preds:
            if not isinstance(pred, dict):
                continue
            signal = _derive_xgb_signal_score(
                pred.get('q_values') or [],
                pred.get('task'),
                pred.get('confidence'),
            )
            threshold = _to_float_or_none(pred.get('xgb_runtime_threshold'))
            if threshold is None:
                model_path_s = str(pred.get('model_path') or '').strip()
                if model_path_s:
                    threshold = threshold_by_model.get(model_path_s)
            if threshold is None:
                threshold = fallback_threshold
            if signal is None or threshold is None:
                continue
            signals.append(float(signal))
            thresholds.append(float(threshold))
        if not signals or not thresholds:
            signal = _derive_xgb_signal_score(
                (pred_json or {}).get('q_values') or [],
                (pred_json or {}).get('task'),
                (pred_json or {}).get('confidence'),
            )
            threshold = fallback_threshold
            if threshold is None and len(threshold_by_model) == 1:
                threshold = next(iter(threshold_by_model.values()))
            if signal is not None and threshold is not None:
                signals.append(float(signal))
                thresholds.append(float(threshold))
        if not signals or not thresholds:
            return None
        return {
            'signal': float(sum(signals) / len(signals)),
            'threshold': float(sum(thresholds) / len(thresholds)),
        }
    except Exception:
        return None

def _append_xgb_signal_history(rc, session_id: str, snapshot: dict, window: int = XGB_SIGNAL_EXIT_WINDOW_DEFAULT) -> None:
    try:
        key = session_runtime_key(session_id, "xgb_signal_exit_history")
        raw = rc.get(key) if rc is not None else None
        history = json.loads(raw) if raw else []
        if not isinstance(history, list):
            history = []
        snapshot_ts = int(snapshot.get('ts_ms') or 0) if isinstance(snapshot, dict) else 0
        if history and snapshot_ts:
            try:
                last_ts = int((history[-1] or {}).get('ts_ms') or 0)
            except Exception:
                last_ts = 0
            if last_ts == snapshot_ts:
                history[-1] = snapshot
            else:
                history.append(snapshot)
        else:
            history.append(snapshot)
        keep = max(1, int(window))
        rc.set(key, json.dumps(history[-keep:], ensure_ascii=False))
    except Exception:
        logger.exception("[xgb_signal_exit] failed to append history session=%s", session_id)


def _extract_xgb_long_short_signals(pred_json: dict | None) -> tuple[float | None, float | None]:
    try:
        preds = (pred_json or {}).get('predictions') or []
        long_values: list[float] = []
        short_values: list[float] = []
        for pred in preds:
            if not isinstance(pred, dict):
                continue
            signal = _derive_xgb_signal_score(
                pred.get('q_values') or [],
                pred.get('task'),
                pred.get('confidence'),
            )
            if signal is None:
                continue
            task = str(pred.get('task') or '').strip().lower()
            action = str(pred.get('action') or '').strip().lower()
            side = None
            if task in ('entry_long', 'exit_short'):
                side = 'long'
            elif task in ('entry_short', 'exit_long'):
                side = 'short'
            elif action == 'buy':
                side = 'long'
            elif action == 'sell':
                side = 'short'
            if side == 'long':
                long_values.append(float(signal))
            elif side == 'short':
                short_values.append(float(signal))
        long_signal = (float(sum(long_values) / len(long_values)) if long_values else None)
        short_signal = (float(sum(short_values) / len(short_values)) if short_values else None)
        return long_signal, short_signal
    except Exception:
        return None, None


def _append_xgb_entry_attempt_history(rc, row: dict, limit: int = XGB_ENTRY_ATTEMPTS_HISTORY_LIMIT) -> None:
    try:
        if rc is None or not isinstance(row, dict):
            return
        keep = max(1, int(limit))
        rc.lpush(XGB_ENTRY_ATTEMPTS_HISTORY_KEY, json.dumps(row, ensure_ascii=False, default=str))
        rc.ltrim(XGB_ENTRY_ATTEMPTS_HISTORY_KEY, 0, keep - 1)
    except Exception:
        logger.exception("[xgb_entry_attempt] failed to append row")

def _rebuild_xgb_signal_history_from_predictions(rc, session_id: str, window: int = XGB_SIGNAL_EXIT_WINDOW_DEFAULT) -> list[dict]:
    try:
        from orm.database import get_db_session
        from orm.models import ModelPrediction
        from tasks.xgb_live import _load_xgb_runtime_meta

        entry_ts = _to_float_or_none(get_runtime_value(rc, session_id, "pos_entry_ts")) if rc is not None else None
        start_step = _to_float_or_none(get_runtime_value(rc, session_id, "xgb_signal_exit_start_step")) if rc is not None else None
        min_ts = int(entry_ts + start_step * 300000) if entry_ts is not None and start_step is not None else None
        threshold_override = _to_float_or_none(get_runtime_value(rc, session_id, "xgb_entry_threshold")) if rc is not None else None

        rebuilt = []
        session = get_db_session()
        try:
            rows = (
                session.query(ModelPrediction)
                .filter(ModelPrediction.market_conditions.like(f'%"{session_id}"%'))
                .order_by(ModelPrediction.created_at.desc())
                .limit(max(100, int(window) * 5))
                .all()
            )
        finally:
            session.close()

        for row in reversed(rows):
            meta = _safe_json_loads(getattr(row, "market_conditions", None))
            if str(meta.get("session_id") or "") != str(session_id):
                continue
            ts_ms = int(row.created_at.timestamp() * 1000) if getattr(row, "created_at", None) else 0
            if min_ts is not None and ts_ms < min_ts:
                continue
            model_path = str(getattr(row, "model_path", "") or "")
            if "/models/xgb/" not in model_path.replace("\\", "/"):
                continue
            q_values = json.loads(row.q_values or "[]") if isinstance(row.q_values, str) else (row.q_values or [])
            cfg, task_name, _ = _load_xgb_runtime_meta(model_path)
            threshold = threshold_override
            if threshold is None:
                threshold = _to_float_or_none(getattr(cfg, "p_enter_threshold", None))
            signal = _derive_xgb_signal_score(q_values, task_name, getattr(row, "confidence", None))
            if signal is None or threshold is None:
                continue
            rebuilt.append({"signal": float(signal), "threshold": float(threshold), "ts_ms": ts_ms})

        rebuilt = rebuilt[-max(1, int(window)):]
        if rebuilt and rc is not None:
            rc.set(session_runtime_key(session_id, "xgb_signal_exit_history"), json.dumps(rebuilt, ensure_ascii=False))
        return rebuilt
    except Exception:
        logger.exception("[xgb_signal_exit] failed to rebuild history session=%s", session_id)
        return []

def _load_xgb_signal_history(rc, session_id: str, window: int = XGB_SIGNAL_EXIT_WINDOW_DEFAULT) -> list[dict]:
    try:
        key = session_runtime_key(session_id, "xgb_signal_exit_history")
        raw = rc.get(key) if rc is not None else None
        history = json.loads(raw) if raw else []
        if not isinstance(history, list):
            return []
        cleaned = []
        for item in history[-max(1, int(window)):]:
            if not isinstance(item, dict):
                continue
            signal = _to_float_or_none(item.get('signal'))
            threshold = _to_float_or_none(item.get('threshold'))
            if signal is None or threshold is None:
                continue
            cleaned.append({
                'signal': float(signal),
                'threshold': float(threshold),
                'ts_ms': int(item.get('ts_ms') or 0) if item.get('ts_ms') not in (None, '') else 0,
            })
        if len(cleaned) >= max(1, int(window)):
            return cleaned
        rebuilt = _rebuild_xgb_signal_history_from_predictions(rc, session_id, window)
        return rebuilt if len(rebuilt) > len(cleaned) else cleaned
    except Exception:
        logger.exception("[xgb_signal_exit] failed to load history session=%s", session_id)
        return _rebuild_xgb_signal_history_from_predictions(rc, session_id, window)

def _reset_xgb_signal_history(rc, session_id: str) -> None:
    try:
        if rc is not None:
            rc.delete(session_runtime_key(session_id, "xgb_signal_exit_history"))
    except Exception:
        pass
@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0}, queue='trade')
def start_execution_strategy(
    self,
    session_id: str,
    symbol: str | None = None,
    execution_mode: str = 'market',
    side: str = 'buy',
    qty: float | None = None,
    limit_config: dict | None = None,
    leverage: int | None = None,
):
    """Запуск DDD-стратегии исполнения (market | limit_post_only).

    - Ограничение max_lifetime_sec ≤ 300 соблюдается на уровне сервиса
    - При активном intent для symbol — новая заявка отклоняется (guard)
    - Лог каждой попытки сохранится в Redis (state store)
    """
    try:
        from trading_agent.application.executor_service import ExecutionService
        from trading_agent.infrastructure.state_store_redis import RedisStateStore
        session_doc = load_session(get_redis_client(), session_id)
        if not isinstance(session_doc, dict):
            return {"success": False, "error": f"session not found: {session_id}"}
        symbol = str(symbol or session_doc.get('symbol') or '').strip().upper()
        account_id = str(session_doc.get('account_id') or '').strip()
        if not symbol:
            return {"success": False, "error": "symbol is required"}
        if not account_id:
            return {"success": False, "error": f"account_id not set for session {session_id}"}
        # Реальный Bybit gateway (fallback на stub при ошибке инициализации ключей)
        from trading_agent.infrastructure.exchange_gateway_bybit import BybitExchangeGateway
        _gateway = BybitExchangeGateway(symbol_for_markets=symbol, account_id=account_id)
        from trading_agent.strategies.market_strategy import MarketStrategy
        from trading_agent.strategies.limit_post_only_strategy import LimitPostOnlyStrategy
        from trading_agent.domain.models import LimitConfig, Side

        # 1) Количество, если не задано — рассчитать через TradingAgent
        if qty is None or qty <= 0:
            try:
                from trading_agent.trading_agent import TradingAgent
                _dir = str(get_runtime_value(get_redis_client(), session_id, 'direction', session_doc.get('direction') or 'long')).strip().lower() or 'long'
                session_model_path = str(session_doc.get('model_path') or '').strip() or None
                agent = TradingAgent(
                    model_path=session_model_path,
                    direction=_dir,
                    symbol=symbol,
                    session_id=session_id,
                    account_id=account_id,
                )
                agent.symbol = symbol
                agent.base_symbol = symbol
                qty = float(agent._calculate_trade_amount())
            except Exception:
                qty = 0.001

        # Применяем небольшой буфер под комиссию/округления для лимитного исполнения
        try:
            if str(execution_mode).strip().lower() == 'limit_post_only' and qty and qty > 0:
                qty = _apply_safety_buffer_qty(qty)
        except Exception:
            pass

        # 2) Конфиг лимитной стратегии
        cfg = None
        if isinstance(limit_config, dict):
            try:
                rq = int(limit_config.get('requote_interval_sec', 15))
                ml = int(limit_config.get('max_lifetime_sec', 300))
                mx = int(limit_config.get('offset_max_ticks', 16))
                cfg = LimitConfig(requote_interval_sec=max(5, min(60, rq)), max_lifetime_sec=min(300, max(10, ml)), offset_max_ticks=max(2, min(128, mx)))
            except Exception:
                cfg = LimitConfig()

        # 2.5) Очистка зависшего active_intent (если есть) ДО запуска сервиса
        try:
            _rc_chk = Redis(host='redis', port=6379, db=0, decode_responses=True)
            aid = _rc_chk.get(f'exec:active_intent:{session_id}')
            if aid:
                raw = _rc_chk.get(f'exec:intent:{aid}')
                stale = False
                if not raw:
                    stale = True
                else:
                    try:
                        data = json.loads(raw)
                        upd = float(data.get('updated_at') or 0)
                        if upd <= 0 or (time.time() - upd) > 3600:
                            stale = True
                    except Exception:
                        stale = True
                if stale:
                    try:
                        _rc_chk.delete(f'exec:intent:{aid}')
                    except Exception:
                        pass
                    try:
                        _rc_chk.delete(f'exec:active_intent:{session_id}')
                    except Exception:
                        pass
        except Exception:
            pass

        # 2.6) Агрессивная очистка активного интента при DEBUG_BUY (чтобы не упираться в guard)
        try:
            def _truthy(v):
                try:
                    return str(v).strip().lower() in ('1','true','yes','on')
                except Exception:
                    return False
            dbg_raw = os.getenv(f"DEBUG_BUY_{str(symbol).upper()}") or os.getenv("DEBUG_BUY")
            if dbg_raw is not None and _truthy(dbg_raw):
                aid2 = _rc_chk.get(f'exec:active_intent:{session_id}')
                if aid2:
                    raw2 = _rc_chk.get(f'exec:intent:{aid2}')
                    try:
                        if raw2:
                            data2 = json.loads(raw2)
                            exch_id2 = data2.get('exchange_order_id')
                            if exch_id2:
                                try:
                                    _gateway.cancel_order(symbol, exch_id2)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    try:
                        _rc_chk.delete(f'exec:intent:{aid2}')
                    except Exception:
                        pass
                    try:
                        _rc_chk.delete(f'exec:active_intent:{session_id}')
                    except Exception:
                        pass
                    try:
                        logger.warning(f"[exec] DEBUG_BUY cleared active intent before start: symbol={symbol}, intent_id={aid2}")
                    except Exception:
                        pass
        except Exception:
            pass

        # 3) Создаём сервис и стратегий
        store = RedisStateStore(scope=session_id)
        gateway = _gateway
        # Устанавливаем плечо (1..5) перед стартом стратегии, только для лимитного исполнения
        try:
            lev = int(leverage) if leverage is not None else 1
            if lev < 1 or lev > 5:
                lev = 1
        except Exception:
            lev = 1
        try:
            if execution_mode == 'limit_post_only' and hasattr(gateway, 'ex') and hasattr(gateway.ex, 'set_leverage'):
                try:
                    gateway.ex.set_leverage(lev, symbol)
                except Exception:
                    try:
                        gateway.ex.set_leverage(str(lev), symbol, {'buyLeverage': str(lev), 'sellLeverage': str(lev)})
                    except Exception:
                        pass
        except Exception:
            pass
        strategies = {
            'market': MarketStrategy(store, gateway),
            'limit_post_only': LimitPostOnlyStrategy(store, gateway),
        }
        service = ExecutionService(store, strategies)

        # 3.5) Если позиция уже активна и выход по риск-ордерам — не стартуем новый intent и чистим non-reduceOnly
        try:
            # Читаем тип риск-менеджмента
            risk_type_exec = None
            try:
                rc_rt = Redis(host='redis', port=6379, db=0, decode_responses=True)
                _rtv = get_runtime_value(rc_rt, session_id, 'risk_management_type', session_doc.get('risk_management_type'))
                if _rtv is not None and str(_rtv).strip() != '':
                    risk_type_exec = str(_rtv).strip()
            except Exception:
                risk_type_exec = None

            from trading_agent.trading_agent import TradingAgent as _TA
            ag_pos = _TA(symbol=symbol, session_id=session_id, account_id=account_id)
            ag_pos.symbol = symbol
            ag_pos.base_symbol = symbol
            try:
                ag_pos._restore_position_from_exchange()
            except Exception:
                pass
            pos_now = getattr(ag_pos, 'current_position', None)
            if pos_now and (risk_type_exec and str(risk_type_exec).lower() in ('exchange_orders','both')) and str(execution_mode).strip().lower() == 'limit_post_only':
                try:
                    oo2 = gateway.ex.fetch_open_orders(symbol, params={'category': 'linear'}) or []
                    for _od in oo2:
                        try:
                            _inf = _od.get('info') or {}
                            _ro = bool(_inf.get('reduceOnly') or _inf.get('reduce_only') or False)
                            if not _ro:
                                gateway.ex.cancel_order(_od.get('id'), symbol)
                        except Exception:
                            continue
                except Exception:
                    pass
                return {"success": True, "skipped": True, "reason": "position_active_risk_exit"}
        except Exception:
            pass

        # 4) Запуск
        side_enum = Side.BUY if str(side).lower() == 'buy' else Side.SELL

        # helper: жёсткая очистка активного интента (и отмена ордера), если гард блокирует старт
        def _clear_active_intent_for(sym: str) -> None:
            try:
                rc_loc = Redis(host='redis', port=6379, db=0, decode_responses=True)
                aid_loc = rc_loc.get(f'exec:active_intent:{session_id}')
                raw_loc = rc_loc.get(f'exec:intent:{aid_loc}') if aid_loc else None
                if aid_loc and raw_loc:
                    try:
                        data_loc = json.loads(raw_loc)
                        exch_id_loc = data_loc.get('exchange_order_id')
                        if exch_id_loc:
                            try:
                                gateway.cancel_order(sym, exch_id_loc)
                            except Exception:
                                pass
                    except Exception:
                        pass
                if aid_loc:
                    try:
                        rc_loc.delete(f'exec:intent:{aid_loc}')
                    except Exception:
                        pass
                try:
                    rc_loc.delete(f'exec:active_intent:{session_id}')
                except Exception:
                    pass
                try:
                    logger.warning(f"[exec] guard-retry cleared active intent: symbol={sym}, intent_id={aid_loc}")
                except Exception:
                    pass
            except Exception:
                pass

        try:
            intent = service.start(execution_mode=execution_mode, symbol=symbol, side=side_enum, qty=qty, cfg=cfg)
        except RuntimeError as _guard_err:
            if 'Active intent exists for' in str(_guard_err):
                _clear_active_intent_for(symbol)
                time.sleep(0.25)
                intent = service.start(execution_mode=execution_mode, symbol=symbol, side=side_enum, qty=qty, cfg=cfg)
            else:
                raise

        # Если размещение/исполнение провалилось — запишем причину в market_conditions последнего ансамбля,
        # чтобы UI показывал ошибку рядом с сигналом.
        try:
            exec_reason = None
            try:
                logs = getattr(intent, 'logs', None) or []
                if isinstance(logs, list):
                    for row in reversed(logs):
                        try:
                            r = row.get('reason')
                            if r:
                                exec_reason = str(r)
                                break
                        except Exception:
                            continue
            except Exception:
                exec_reason = None
            if not exec_reason:
                try:
                    le = getattr(intent, 'last_error', None)
                    if le:
                        exec_reason = str(le)
                except Exception:
                    exec_reason = None

            if exec_reason:
                exec_type = None
                low = exec_reason.lower()
                if 'minimum amount' in low or 'minimum amount precision' in low or 'min amount' in low:
                    exec_type = 'min_amount'
                elif 'insufficient funds' in low:
                    exec_type = 'insufficient_funds'
                elif 'error sign' in low or 'signature' in low:
                    exec_type = 'bad_signature'

                gid = None
                try:
                    rc_gid = get_redis_client()
                    gid = get_runtime_value(rc_gid, session_id, "last_ensemble_group_id")
                except Exception:
                    gid = None
                if gid:
                    _attach_exec_error_to_group(
                        symbol=str(symbol),
                        ensemble_group_id=str(gid),
                        exec_error=str(exec_reason),
                        exec_error_type=exec_type,
                        extra={
                            "exec_intent_id": getattr(intent, "intent_id", None),
                            "exec_state": str(getattr(intent, "state", "")),
                            "exec_mode": str(execution_mode),
                        },
                    )
        except Exception:
            pass

        # Идемпотентные постановки TP/SL после запуска лимитной стратегии: накрываем момент фактического fill
        try:
            if str(execution_mode) == 'limit_post_only':
                # 2s часто слишком рано: лимитка может выставляться/перекотироваться десятки секунд до первого fill.
                ensure_risk_orders.apply_async(kwargs={'session_id': session_id, 'symbol': symbol}, countdown=20, queue='trade')
                ensure_risk_orders.apply_async(kwargs={'session_id': session_id, 'symbol': symbol}, countdown=15, queue='trade')
                ensure_risk_orders.apply_async(kwargs={'session_id': session_id, 'symbol': symbol}, countdown=60, queue='trade')
                ensure_risk_orders.apply_async(kwargs={'session_id': session_id, 'symbol': symbol}, countdown=120, queue='trade')
                ensure_risk_orders.apply_async(kwargs={'session_id': session_id, 'symbol': symbol}, countdown=180, queue='trade')
        except Exception:
            pass

        return {"success": True, "intent_id": intent.intent_id, "state": intent.state}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}


# --- Ensure TP/SL idempotent task ---
@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0}, queue='trade')
def ensure_risk_orders(self, session_id: str, symbol: str | None = None):
    """
    Идемпотентная постановка TP/SL для открытой позиции по символу.
    - Читает risk_type/tp/sl из Redis (per-symbol → global → дефолт)
    - Восстанавливает позицию с биржи
    - Если reduceOnly ордера уже есть — не дублирует
    - Для long: TP limit SELL выше, SL stop-market SELL ниже
      Для short: TP limit BUY ниже, SL stop-market BUY выше
    - Антидублирование через Redis setex 120s
    """
    try:
        from redis import Redis
        rc = Redis(host='redis', port=6379, db=0, decode_responses=True)
    except Exception:
        rc = None
    session_doc = load_session(rc, session_id) if rc is not None else None
    if not isinstance(session_doc, dict):
        return {"success": False, "error": f"session not found: {session_id}"}
    symbol = str(symbol or session_doc.get('symbol') or '').strip().upper()
    account_id = str(session_doc.get('account_id') or '').strip()
    if not symbol:
        return {"success": False, "error": f"symbol not set for session {session_id}"}
    try:
        logger.info(f"[ensure_risk] ▶ start for session={session_id} symbol={symbol}")
    except Exception:
        pass

    # Антидублирование
    # Важно: для limit_post_only fill может произойти позже, поэтому при "позиции нет" держим TTL коротким,
    # чтобы отложенные повторы ensure_risk_orders смогли отработать.
    ensure_key = session_runtime_key(session_id, "risk:ensured")
    recently_ensured = False
    try:
        if rc is not None and rc.get(ensure_key):
            recently_ensured = True
            try:
                logger.info(f"[ensure_risk] skip {symbol}: recently_ensured")
            except Exception:
                pass
        # ВАЖНО: не выходим сразу — нам нужно выполнить exit-detection (open->none) даже при антидублировании,
        # иначе last_exit_ts_ms/reason в Redis может остаться старым и post-exit cooldown не сработает.
        if rc is not None and not recently_ensured:
            rc.setex(ensure_key, 180, "1")
    except Exception:
        pass

    from trading_agent.trading_agent import TradingAgent
    try:
        _dir = str(get_runtime_value(rc, session_id, 'direction', session_doc.get('direction') or 'long')).strip().lower() or 'long'
    except Exception:
        _dir = 'long'
    session_model_path = str(session_doc.get('model_path') or '').strip() or None
    agent = TradingAgent(
        model_path=session_model_path,
        direction=_dir,
        symbol=symbol,
        session_id=session_id,
        account_id=account_id,
    )
    agent.symbol = symbol
    agent.base_symbol = symbol

    # Параметры
    def _pick_num(v):
        try:
            return float(v) if v is not None and str(v).strip() != '' else None
        except Exception:
            return None
    risk_type = 'exchange_orders'
    tp_pct = 1.0
    sl_pct = 1.0
    # Новые параметры ATR‑режима
    risk_stop_mode = 'fixed_pct'  # 'fixed_pct' | 'atr_tp_sl'
    atr_k = 2.5
    atr_m = 1.8
    atr_min_sl_mult = 1.0
    # Trailing параметры
    trailing_enabled = False
    trailing_mode = None
    trailing_activate_mode = None
    trailing_activate_value = None
    atr_trail_mult = 1.0
    try:
        if rc is not None:
            _rt = get_runtime_value(rc, session_id, 'risk_management_type', session_doc.get('risk_management_type'))
            if _rt:
                risk_type = str(_rt)
            _tp = get_runtime_value(rc, session_id, 'take_profit_pct', session_doc.get('take_profit_pct'))
            _sl = get_runtime_value(rc, session_id, 'stop_loss_pct', session_doc.get('stop_loss_pct'))
            if _pick_num(_tp) is not None:
                tp_pct = _pick_num(_tp)
            if _pick_num(_sl) is not None:
                sl_pct = _pick_num(_sl)
            # ATR stop mode
            try:
                _rsm = get_runtime_value(rc, session_id, 'risk_stop_mode', session_doc.get('risk_stop_mode'))
                if _rsm:
                    risk_stop_mode = str(_rsm)
                _ak = get_runtime_value(rc, session_id, 'atr_k', session_doc.get('atr_k'))
                _am = get_runtime_value(rc, session_id, 'atr_m', session_doc.get('atr_m'))
                _ams = get_runtime_value(rc, session_id, 'atr_min_sl_mult', session_doc.get('atr_min_sl_mult'))
                if _pick_num(_ak) is not None:
                    atr_k = _pick_num(_ak)
                if _pick_num(_am) is not None:
                    atr_m = _pick_num(_am)
                if _pick_num(_ams) is not None:
                    atr_min_sl_mult = _pick_num(_ams)
            except Exception:
                pass
            # Trailing
            try:
                _ten = get_runtime_value(rc, session_id, 'trailing_enabled', session_doc.get('trailing_enabled'))
                if _ten is not None:
                    trailing_enabled = str(_ten).strip().lower() in ('1','true','yes','on')
                _tmode = get_runtime_value(rc, session_id, 'trailing_mode', session_doc.get('trailing_mode'))
                if _tmode is not None and str(_tmode).strip() != '':
                    trailing_mode = str(_tmode).strip()
                _tact_mode = get_runtime_value(rc, session_id, 'trailing_activate_mode', session_doc.get('trailing_activate_mode'))
                if _tact_mode is not None and str(_tact_mode).strip() != '':
                    trailing_activate_mode = str(_tact_mode).strip()
                _tact_val = get_runtime_value(rc, session_id, 'trailing_activate_value', session_doc.get('trailing_activate_value'))
                if _tact_val is not None:
                    try:
                        trailing_activate_value = float(_tact_val)
                    except Exception:
                        trailing_activate_value = None
                _tatm = get_runtime_value(rc, session_id, 'atr_trail_mult', session_doc.get('atr_trail_mult'))
                if _pick_num(_tatm) is not None:
                    atr_trail_mult = float(_pick_num(_tatm))
            except Exception:
                pass
    except Exception:
        pass

    # Лог текущих настроек трейлинга/стопа (чтобы понимать, почему TrailingStop не ставится)
    try:
        logger.info(
            f"[ensure_risk][cfg] symbol={symbol} risk_type={risk_type} risk_stop_mode={str(risk_stop_mode).strip()} "
            f"trailing_enabled={trailing_enabled} trailing_mode={trailing_mode} "
            f"atr_trail_mult={atr_trail_mult} activate_mode={trailing_activate_mode} activate_value={trailing_activate_value}"
        )
    except Exception:
        pass

    if risk_type not in ('exchange_orders', 'both'):
        try:
            logger.info(f"[ensure_risk] skip {symbol}: risk_type={risk_type}")
        except Exception:
            pass
        return {"success": True, "skipped": True, "reason": f"risk_type={risk_type}"}

    # Позиция
    try:
        agent._restore_position_from_exchange()
    except Exception:
        pass
    pos = getattr(agent, 'current_position', None) or {}
    entry_price = pos.get('entry_price')
    amount = pos.get('amount')
    ptype = str(pos.get('type') or '').lower()
    try:
        if rc is not None and ptype in ('long', 'short'):
            _tp_side = get_runtime_value(rc, session_id, f'take_profit_pct_{ptype}', session_doc.get(f'take_profit_pct_{ptype}'))
            _sl_side = get_runtime_value(rc, session_id, f'stop_loss_pct_{ptype}', session_doc.get(f'stop_loss_pct_{ptype}'))
            if _pick_num(_tp_side) is None or _pick_num(_sl_side) is None:
                _tp_default, _sl_default = _xgb_side_risk_defaults(session_doc, ptype)
                if _pick_num(_tp_side) is None:
                    _tp_side = _tp_default
                if _pick_num(_sl_side) is None:
                    _sl_side = _sl_default
            if _pick_num(_tp_side) is not None:
                tp_pct = _pick_num(_tp_side)
            if _pick_num(_sl_side) is not None:
                sl_pct = _pick_num(_sl_side)
    except Exception:
        pass

    # --- Exit detection (TP/SL) for post-exit cooldown/Q-gate boost ---
    # Мы храним "позиция была/нет" в Redis, чтобы поймать событие закрытия позиции,
    # даже если закрытие произошло биржей по reduceOnly TP/SL/Trailing.
    def _last_closed_5m_ts_ms() -> int:
        try:
            now_utc = datetime.utcnow()
            epoch_sec = int(now_utc.timestamp())
            step = 300
            last_closed = (epoch_sec // step) * step - step
            return int(last_closed * 1000)
        except Exception:
            return 0

    def _detect_exit_event(_exchange, _symbol: str) -> dict:
        """Best-effort exit detector from closed reduceOnly order.
        Returns: {reason, side, qty, price, order_id, ts_ms}."""
        try:
            if _exchange is None:
                return {'reason': 'unknown'}
            orders = []
            try:
                if hasattr(_exchange, 'fetch_closed_orders'):
                    orders = _exchange.fetch_closed_orders(_symbol, params={'category': 'linear'}) or []
            except Exception:
                orders = []
            if not orders:
                return {'reason': 'unknown'}

            def _ts_ms(od):
                try:
                    if od.get('timestamp'):
                        return int(od.get('timestamp'))
                except Exception:
                    pass
                try:
                    info = od.get('info') or {}
                    for k in ('updatedTime', 'updatedTimeMs', 'createdTime', 'createdTimeMs', 'time'):
                        v = info.get(k)
                        if v is not None and str(v).strip() != '':
                            return int(float(v))
                except Exception:
                    pass
                return 0

            # Sort most recent first
            orders_sorted = sorted(list(orders), key=_ts_ms, reverse=True)
            for od in orders_sorted[:50]:
                try:
                    info = od.get('info') or {}
                    ro = bool(info.get('reduceOnly') or info.get('reduce_only') or False)
                    if not ro:
                        continue
                    side = str(od.get('side') or info.get('side') or '').lower()
                    typ = str(od.get('type') or info.get('orderType') or '').lower()
                    stop_kind = str(info.get('stopOrderType') or info.get('tpslMode') or '').lower()
                    trig = info.get('triggerPrice') or info.get('stopLoss') or info.get('takeProfit')

                    # qty/price
                    qty = None
                    try:
                        qty = float(od.get('filled') or info.get('cumExecQty') or info.get('qty') or od.get('amount') or 0.0)
                    except Exception:
                        qty = None
                    price = None
                    try:
                        price = float(od.get('average') or info.get('avgPrice') or info.get('avgPriceE8') or 0.0)
                        if price and price > 1e6:
                            price = price / 1e8
                    except Exception:
                        price = None
                    if (price is None or price <= 0) and od.get('price'):
                        try:
                            price = float(od.get('price'))
                        except Exception:
                            pass
                    if (qty is None or qty <= 0) or (price is None or price <= 0):
                        continue

                    reason = 'unknown'
                    if 'trailing' in typ or 'trailing' in stop_kind:
                        reason = 'trailing'
                    elif trig not in (None, '', '0', '0.0') or ('stop' in typ) or ('stop' in stop_kind):
                        reason = 'stop_loss'
                    elif typ == 'limit':
                        reason = 'take_profit'

                    return {
                        'reason': reason,
                        'side': side or None,
                        'qty': float(qty),
                        'price': float(price),
                        'order_id': od.get('id') or info.get('orderId') or info.get('orderID'),
                        'ts_ms': _ts_ms(od) or None,
                    }
                except Exception:
                    continue
            return {'reason': 'unknown'}
        except Exception:
            return {'reason': 'unknown'}

    def _is_confirmed_exit_event(evt: dict) -> bool:
        try:
            return bool(
                evt.get('order_id')
                and float(evt.get('qty') or 0.0) > 0
                and float(evt.get('price') or 0.0) > 0
            )
        except Exception:
            return False

    try:
        if rc is not None:
            pos_key = session_runtime_key(session_id, "pos_open")
            was_open = str(rc.get(pos_key) or '0').strip() == '1'
            if entry_price and amount and amount > 0:
                # Mark open and remember entry for later heuristics/debug
                try:
                    rc.set(pos_key, "1")
                    set_runtime_value(rc, session_id, "pos_entry_price", entry_price)
                    set_runtime_value(rc, session_id, "pos_type", str(ptype or ''))
                    if not was_open:
                        set_runtime_value(rc, session_id, "pos_entry_ts", int(_last_closed_5m_ts_ms() or 0))
                        _reset_xgb_signal_history(rc, session_id)
                        try:
                            _notify_max_position_entry(
                                rc=rc,
                                session_id=session_id,
                                symbol=symbol,
                                side=str(ptype or ''),
                                entry_price=float(entry_price),
                                entry_time=datetime.now(),
                                stop_loss_pct=float(sl_pct) if sl_pct not in (None, '') else None,
                                trade_number=None,
                            )
                        except Exception:
                            pass
                    else:
                        _existing_entry_ts = get_runtime_value(rc, session_id, "pos_entry_ts")
                        if _existing_entry_ts in (None, '', '0'):
                            set_runtime_value(rc, session_id, "pos_entry_ts", int(_last_closed_5m_ts_ms() or 0))
                except Exception:
                    pass
            else:
                # No position now
                try:
                    rc.set(pos_key, "0")
                except Exception:
                    pass
                if was_open:
                    # Exit event (open -> none)
                    now_bucket_ts = _last_closed_5m_ts_ms()
                    prev_exit = None
                    try:
                        prev_exit = get_runtime_value(rc, session_id, "last_exit_ts_ms")
                        prev_exit = int(prev_exit) if prev_exit is not None and str(prev_exit).strip() != '' else None
                    except Exception:
                        prev_exit = None
                    if now_bucket_ts:
                        evt = _detect_exit_event(getattr(agent, 'exchange', None), symbol) or {}
                        if not _is_confirmed_exit_event(evt):
                            try:
                                rc.set(pos_key, "1")
                            except Exception:
                                pass
                            logger.warning(
                                "[exit_detect] skip unconfirmed exit session=%s symbol=%s evt=%s",
                                session_id,
                                symbol,
                                evt,
                            )
                            return {"success": True, "skipped": True, "reason": "exit_unconfirmed"}
                        # Пишем реальное время выхода из закрытого reduceOnly (если доступно),
                        # иначе fallback на "последнюю закрытую 5m свечу".
                        try:
                            evt_ts = evt.get('ts_ms')
                            evt_ts = int(evt_ts) if evt_ts is not None and str(evt_ts).strip() != '' else None
                        except Exception:
                            evt_ts = None
                        exit_ts_to_save = int(evt_ts) if evt_ts else int(now_bucket_ts)
                        if prev_exit is not None and exit_ts_to_save <= int(prev_exit):
                            return {"success": True, "skipped": True, "reason": "exit_not_newer_than_prev"}
                        reason = str(evt.get('reason') or 'unknown')
                        try:
                            if reason == 'unknown':
                                forced_reason = get_runtime_value(rc, session_id, "forced_exit_reason") if rc is not None else None
                                forced_reason = str(forced_reason or '').strip().lower()
                                if forced_reason:
                                    reason = forced_reason
                            if rc is not None:
                                delete_runtime_value(rc, session_id, "forced_exit_reason")
                        except Exception:
                            pass
                        try:
                            ep_hint_raw = get_runtime_value(rc, session_id, "pos_entry_price")
                            ep_hint = float(ep_hint_raw) if ep_hint_raw not in (None, '') else None
                            px_hint = float(evt.get('price') or 0.0)
                            qty_hint = float(evt.get('qty') or 0.0)
                            side_hint = str(evt.get('side') or '').lower()
                            if ep_hint and px_hint and qty_hint and str(reason).strip().lower() in ('stop_loss', 'take_profit'):
                                pnl_hint = (px_hint - ep_hint) * qty_hint if side_hint == 'sell' else (ep_hint - px_hint) * qty_hint
                                if reason == 'stop_loss' and pnl_hint > 0:
                                    reason = 'take_profit'
                                elif reason == 'take_profit' and pnl_hint < 0:
                                    reason = 'stop_loss'
                        except Exception:
                            pass
                        # anti-dup by exchange order id (best-effort)
                        try:
                            last_oid = get_runtime_value(rc, session_id, "last_exit_order_id")
                        except Exception:
                            last_oid = None
                        oid = evt.get('order_id')
                        if oid and last_oid and str(oid) == str(last_oid):
                            return {"success": True, "skipped": True, "reason": "exit_duplicate_order"}
                        try:
                            rc.setex(session_runtime_key(session_id, "last_exit_ts_ms"), 86400, str(int(exit_ts_to_save)))
                            rc.setex(session_runtime_key(session_id, "last_exit_reason"), 86400, str(reason))
                            if oid:
                                rc.setex(session_runtime_key(session_id, "last_exit_order_id"), 86400, str(oid))
                            _reset_xgb_signal_history(rc, session_id)
                            # SL streak escalation: increment on stop_loss, reset on TP/trailing
                            try:
                                if str(reason).strip().lower() == 'stop_loss':
                                    _streak = int(get_runtime_value(rc, session_id, "sl_streak", 0) or 0) + 1
                                    rc.setex(session_runtime_key(session_id, "sl_streak"), 172800, str(_streak))
                                    logger.warning("[sl_streak] symbol=%s streak=%d", symbol, _streak)
                                else:
                                    delete_runtime_value(rc, session_id, "sl_streak")
                            except Exception:
                                pass
                        except Exception:
                            pass
                        # Persist to DB trades so /agent/<symbol> -> "История торговли" shows this exit.
                        try:
                            from utils.trade_utils import create_trade_record
                            ep = None
                            ptype_prev = None
                            try:
                                ep_raw = get_runtime_value(rc, session_id, "pos_entry_price")
                                ep = float(ep_raw) if ep_raw is not None and str(ep_raw).strip() != '' else None
                            except Exception:
                                ep = None
                            try:
                                ptype_prev = str(get_runtime_value(rc, session_id, "pos_type") or '').strip().lower()
                            except Exception:
                                ptype_prev = None
                            entry_ts_ms = None
                            try:
                                raw_ets = get_runtime_value(rc, session_id, "pos_entry_ts")
                                entry_ts_ms = int(raw_ets) if raw_ets and str(raw_ets).strip() not in ('', '0') else None
                            except Exception:
                                entry_ts_ms = None
                            leverage_meta = 1
                            try:
                                raw_lev = get_runtime_value(rc, session_id, "leverage", session_doc.get("leverage", 1))
                                leverage_meta = max(1, int(float(raw_lev or 1)))
                            except Exception:
                                leverage_meta = 1
                            bal_after = None
                            try:
                                bal_after = agent._get_current_balance()
                            except Exception:
                                pass

                            side = str(evt.get('side') or '').lower()
                            action = side if side in ('buy', 'sell') else 'sell'
                            qty = float(evt.get('qty') or 0.0)
                            px = float(evt.get('price') or 0.0)
                            pnl = None
                            try:
                                if ep and qty and px:
                                    if action == 'sell':  # long close
                                        pnl = (px - ep) * qty
                                    elif action == 'buy': # short close
                                        pnl = (ep - px) * qty
                                        
                                    # Вычитаем примерную комиссию (taker 0.055% за вход и выход)
                                    fee_rate = 0.00055
                                    total_fee = (ep * qty * fee_rate) + (px * qty * fee_rate)
                                    pnl -= total_fee
                            except Exception:
                                pnl = None
                            bal_before = (bal_after - pnl) if (bal_after is not None and pnl is not None) else None
                            create_trade_record(
                                symbol_name=symbol,
                                action=action,
                                status='executed',
                                quantity=max(0.0, qty),
                                price=max(0.0, px),
                                model_prediction=f"exit:{reason}",
                                current_balance=bal_after,
                                position_pnl=pnl,
                                exchange_order_id=str(oid) if oid else None,
                                error_message=json.dumps({
                                    'account_id': account_id,
                                    'exit_reason': reason,
                                    'exit_ts_ms': int(now_bucket_ts),
                                    'exit_side': action,
                                    'entry_price': ep,
                                    'entry_ts_ms': entry_ts_ms,
                                    'pos_type_prev': ptype_prev,
                                    'leverage': leverage_meta,
                                    'model_path': (str(getattr(agent, 'model_path', '') or '') or None),
                                    'model_family': (
                                        'xgb'
                                        if '/models/xgb/' in str(getattr(agent, 'model_path', '') or '').replace('\\', '/')
                                        else 'dqn'
                                    ),
                                    'bal_before': bal_before,
                                    'bal_after': bal_after,
                                }, ensure_ascii=False),
                                is_successful=True
                            )
                            try:
                                _notify_position_exit(
                                    rc=rc,
                                    session_id=session_id,
                                    symbol=symbol,
                                    reason=reason,
                                    side=action,
                                    entry_price=ep,
                                    exit_price=px,
                                    qty=qty,
                                    pnl=pnl,
                                    exit_ts_ms=exit_ts_to_save,
                                    order_id=str(oid) if oid else None,
                                )
                            except Exception:
                                logger.exception("[exit_notify] failed session=%s symbol=%s", session_id, symbol)
                            try:
                                from tasks.celery_task_copy_trade import copy_trade_for_clients
                                copy_trade_for_clients.apply_async(
                                    kwargs={
                                        "session_id": str(session_id),
                                        "symbol": str(symbol),
                                        "action": "exit",
                                        "position_type": str(ptype_prev or "long"),
                                        "entry_price": px,
                                    },
                                    queue="celery"
                                )
                            except Exception as e_cp:
                                logger.exception("[copy_trade] manual/risk exit trigger failed: %s", e_cp)
                            try:
                                delete_runtime_value(rc, session_id, "pos_entry_ts")
                                delete_runtime_value(rc, session_id, "pos_entry_price")
                                delete_runtime_value(rc, session_id, "pos_type")
                            except Exception:
                                pass
                        except Exception:
                            pass
                        try:
                            logger.warning("[exit_detect] symbol=%s exit_ts=%s reason=%s", symbol, now_bucket_ts, reason)
                        except Exception:
                            pass
    except Exception:
        pass

    # Если мы здесь из-за антидублирования — exit-detection уже выполнен, TP/SL не трогаем.
    try:
        if recently_ensured and (entry_price and amount and float(amount) > 0):
            return {"success": True, "skipped": True, "reason": "recently_ensured_exit_checked"}
    except Exception:
        pass

    if not (entry_price and amount and amount > 0):
        # Позиции нет.
        # ВАЖНО: при limit_post_only позиция может появиться позже. Если есть pending входящий ордер (non-reduceOnly),
        # то НЕ трогаем ни active_intent, ни открытые ордера — иначе сами же отменим вход до fill.
        # Также, если в Redis есть активный DDD-intent по symbol, значит стратегия исполнения уже в процессе:
        # не чистим intent и ордера, просто ждём.
        try:
            from redis import Redis as _Rai
            _rc_ai = _Rai(host='redis', port=6379, db=0, decode_responses=True)
            _aid_ai = _rc_ai.get(f'exec:active_intent:{session_id}')
        except Exception:
            _aid_ai = None
        if _aid_ai:
            try:
                logger.info(f"[ensure_risk] no_position: active_intent present ({_aid_ai}) -> skip cleanup, wait fill ({symbol})")
            except Exception:
                pass
            try:
                if rc is not None:
                    rc.setex(ensure_key, 10, "1")
            except Exception:
                pass
            return {"success": True, "skipped": True, "reason": "no_position_active_intent"}
        try:
            oo = agent.exchange.fetch_open_orders(symbol, params={'category': 'linear'}) or []
        except Exception:
            oo = []
        pending_entry = False
        try:
            for od in (oo or []):
                info_o = od.get('info') or {}
                ro = bool(info_o.get('reduceOnly') or info_o.get('reduce_only') or False)
                if not ro:
                    pending_entry = True
                    break
        except Exception:
            pending_entry = False
        if pending_entry:
            try:
                logger.info(f"[ensure_risk] no_position: pending entry order detected -> skip cleanup, wait fill ({symbol})")
            except Exception:
                pass
            # Даем шанс отложенным ensure_risk_orders отработать после фактического fill лимитки
            try:
                if rc is not None:
                    rc.setex(ensure_key, 10, "1")
            except Exception:
                pass
            return {"success": True, "skipped": True, "reason": "no_position_pending_entry"}

        # Позиции нет и pending entry нет – можно чистить reduceOnly ордера, чтобы не мешали новому циклу
        cleaned = 0
        for od in (oo or []):
            try:
                params = od.get('info') or {}
                ro = bool(params.get('reduceOnly') or params.get('reduce_only') or False)
                if ro:
                    try:
                        agent.exchange.cancel_order(od.get('id'), symbol)
                        cleaned += 1
                    except Exception:
                        pass
            except Exception:
                continue
        # Дополнительно: гасим активный DDD‑интент и биржевой ордер из него (если остались следы)
        cleared_intent = False
        try:
            from redis import Redis as _R
            _rc2 = _R(host='redis', port=6379, db=0, decode_responses=True)
            aid2 = _rc2.get(f'exec:active_intent:{session_id}')
            if aid2:
                raw2 = _rc2.get(f'exec:intent:{aid2}')
                exch_id2 = None
                if raw2:
                    try:
                        data2 = json.loads(raw2)
                        exch_id2 = data2.get('exchange_order_id')
                    except Exception:
                        exch_id2 = None
                if exch_id2:
                    try:
                        # ccxt: cancel_order(id, symbol)
                        agent.exchange.cancel_order(exch_id2, symbol)
                    except Exception:
                        pass
                try:
                    _rc2.delete(f'exec:intent:{aid2}')
                except Exception:
                    pass
                _rc2.delete(f'exec:active_intent:{session_id}')
                cleared_intent = True
        except Exception:
            pass
        try:
            logger.info(f"[ensure_risk] flat cleanup {symbol}: cancelled_reduceOnly={cleaned}, cleared_active_intent={cleared_intent}")
        except Exception:
            pass
        # Даем шанс отложенным ensure_risk_orders отработать после фактического fill лимитки
        try:
            if rc is not None:
                rc.setex(ensure_key, 10, "1")
        except Exception:
            pass
        return {"success": True, "skipped": True, "reason": "no_position_cleanup"}

    # Снимем лишние non-reduceOnly ордера (например, оставшийся входящий лимит) при наличии позиции
    try:
        oo_cleanup = agent.exchange.fetch_open_orders(symbol, params={'category': 'linear'}) or []
        for _od in oo_cleanup:
            try:
                _info = _od.get('info') or {}
                _ro = bool(_info.get('reduceOnly') or _info.get('reduce_only') or False)
                if not _ro:
                    agent.exchange.cancel_order(_od.get('id'), symbol)
            except Exception:
                continue
    except Exception:
        pass

    # Уже открытые reduceOnly (для антидублирования)
    try:
        open_orders = agent.exchange.fetch_open_orders(symbol, params={'category': 'linear'}) or []
        try:
            sample_orders = []
            for od in (open_orders or [])[:3]:
                params = od.get('info') or {}
                sample_orders.append({
                    'id': od.get('id'),
                    'side': od.get('side'),
                    'type': od.get('type'),
                    'price': od.get('price'),
                    'reduceOnly': bool(params.get('reduceOnly') or params.get('reduce_only') or False),
                    'triggerPrice': params.get('triggerPrice'),
                    'stopLoss': params.get('stopLoss'),
                })
            logger.info(f"[ensure_risk][orders] open_count={len(open_orders)} sample={sample_orders}")
        except Exception:
            pass
    except Exception:
        open_orders = []

    def _has_reduce_only_limit(side_needed: str) -> bool:
        for od in open_orders or []:
            try:
                params = od.get('info') or {}
                ro = bool(params.get('reduceOnly') or params.get('reduce_only') or False)
                side = str(od.get('side') or '').lower()
                typ = str(od.get('type') or '').lower()
                if ro and side == side_needed and typ == 'limit':
                    return True
            except Exception:
                continue
        return False

    def _has_reduce_only_stop(side_needed: str) -> bool:
        for od in open_orders or []:
            try:
                params = od.get('info') or {}
                ro = bool(params.get('reduceOnly') or params.get('reduce_only') or False)
                side = str(od.get('side') or '').lower()
                typ = str(od.get('type') or '').lower()
                has_trigger = ('triggerPrice' in params) or ('stopLoss' in params) or (typ != 'limit')
                if ro and side == side_needed and has_trigger:
                    return True
            except Exception:
                continue
        return False

    # Нормализация цены по tickSize (точнее, чем precision)
    try:
        tick = _get_price_tick(agent.exchange, symbol)
    except Exception:
        tick = 0.0001
    def _rp(x: float) -> float:
        return _round_to_tick(x, tick)

    # Учтём уже заданные TP/SL на уровне позиции (Bybit v5: поля takeProfit/stopLoss)
    tp_already_set = False
    sl_already_set = False
    trailing_already_set = False
    try:
        pos_list = None
        try:
            pos_list = agent.exchange.fetch_positions([symbol])
        except Exception:
            try:
                pos_list = agent.exchange.fetch_positions(symbol)
            except Exception:
                pos_list = []
        for p in pos_list or []:
            try:
                sym_p = p.get('symbol') or (p.get('info') or {}).get('symbol')
                if str(sym_p).upper() != str(symbol).upper():
                    continue
                info_p = p.get('info') or {}
                try:
                    # Подробный лог по позиции от биржи: какие поля есть и что в них
                    keys_preview = list(info_p.keys())[:30]
                    logger.info(f"[ensure_risk][pos] keys={keys_preview} takeProfit={info_p.get('takeProfit')} stopLoss={info_p.get('stopLoss')} tpSize={info_p.get('tpSize')} slSize={info_p.get('slSize')} tpslMode={info_p.get('tpslMode')} trailingStop={info_p.get('trailingStop')}")
                except Exception:
                    pass
                tp_val = str(info_p.get('takeProfit') or '').strip()
                sl_val = str(info_p.get('stopLoss') or '').strip()
                tr_val = str(info_p.get('trailingStop') or '').strip()
                if tp_val not in ('', '0', '0.0', '0.0000'):
                    tp_already_set = True
                if sl_val not in ('', '0', '0.0', '0.0000'):
                    sl_already_set = True
                if tr_val not in ('', '0', '0.0', '0.0000'):
                    trailing_already_set = True
                break
            except Exception:
                continue
    except Exception:
        tp_already_set = False
        sl_already_set = False
        trailing_already_set = False
    try:
        logger.info(f"[ensure_risk] detected_flags: tp_already_set={tp_already_set} sl_already_set={sl_already_set} trailing_already_set={trailing_already_set}")
    except Exception:
        pass

    is_long = (ptype == 'long') or (ptype == '')

    # Доп. детекция TP/SL по открытым reduceOnly ордерам относительно entry_price
    try:
        if entry_price:
            margin = (tick or 0.0001) * 2.0
            tp_by_orders = False
            sl_by_orders = False
            for od in open_orders or []:
                try:
                    params = od.get('info') or {}
                    ro = bool(params.get('reduceOnly') or params.get('reduce_only') or False)
                    if not ro:
                        continue
                    side_o = str(od.get('side') or '').lower()
                    if side_o != 'sell':
                        continue
                    typ_o = str(od.get('type') or '').lower()
                    trig_raw = params.get('triggerPrice') or params.get('stopLoss')
                    trigger_price = None
                    try:
                        trigger_price = float(trig_raw) if trig_raw not in (None, '', '0', '0.0') else None
                    except Exception:
                        trigger_price = None
                    if typ_o == 'limit':
                        tp_by_orders = True
                    else:
                        if trigger_price is not None:
                            if trigger_price >= float(entry_price) + margin:
                                tp_by_orders = True
                            if trigger_price <= float(entry_price) - margin:
                                sl_by_orders = True
                except Exception:
                    continue
            if tp_by_orders:
                tp_already_set = True
            if sl_by_orders:
                sl_already_set = True
            try:
                logger.info(f"[ensure_risk] inferred_from_orders: tp_by_orders={tp_by_orders} sl_by_orders={sl_by_orders} entry={entry_price} tick={tick}")
            except Exception:
                pass
    except Exception:
        pass
    # Хелпер ATR‑цен
    def _atr_tp_sl_prices(_symbol: str, _entry: float, _k: float, _m: float, _min_k: float, _is_long: bool) -> tuple[float, float]:
        try:
            from utils.indicators import get_atr_1h
            atr_abs, _, _ = get_atr_1h(_symbol, length=None)
        except Exception:
            return None, None
        k_eff = max(float(_k), float(_min_k))
        if _is_long:
            tp = float(_entry) + float(_m) * atr_abs
            sl = float(_entry) - k_eff * atr_abs
        else:
            tp = float(_entry) - float(_m) * atr_abs
            sl = float(_entry) + k_eff * atr_abs
        return tp, sl

    # Биржевой trailing stop (Bybit) для уже открытой позиции (включая limit_post_only), если включён режим atr_trailing
    try:
        if trailing_enabled and str(risk_stop_mode).strip() == 'atr_trailing':
            # Проверяем, что ещё нет TrailingStop-ордера
            has_trailing = bool(trailing_already_set)
            for od in open_orders or []:
                try:
                    info_o = od.get('info') or {}
                    if str(info_o.get('orderType') or '').lower() == 'trailingstop':
                        has_trailing = True
                        break
                except Exception:
                    continue
            try:
                logger.info(f"[ensure_risk][trailing] precheck has_trailing={has_trailing}")
            except Exception:
                pass
            if not has_trailing:
                try:
                    trailing_result = setup_trailing_stop_bybit(
                        agent.exchange,
                        symbol,
                        float(amount),
                        float(entry_price),
                        trailing_mode,
                        trailing_activate_mode,
                        trailing_activate_value,
                        float(atr_trail_mult) if atr_trail_mult is not None else 1.0,
                        side="short" if not is_long else "long",
                    )
                    try:
                        resp = trailing_result.get('response') if isinstance(trailing_result, dict) else None
                        rc = resp.get('retCode') if isinstance(resp, dict) else None
                        rm = resp.get('retMsg') if isinstance(resp, dict) else None
                        logger.info(
                            f"[ensure_risk][trailing] placed: dist={trailing_result.get('trailing_dist')} active={trailing_result.get('active_price')} retCode={rc} retMsg={rm}"
                        )
                    except Exception:
                        pass
                    # Сохраняем последнюю попытку установки трейлинга в Redis (для диагностики)
                    try:
                        import time as _t
                        import json as _json
                        rc0 = get_redis_client()
                        rc0.set(
                            f"trading:trailing:last_setup:{symbol}",
                            _json.dumps(
                                {
                                    "ts": _t.time(),
                                    "ok": True,
                                    "symbol": symbol,
                                    "source": "ensure_risk_orders",
                                    "result": trailing_result,
                                },
                                ensure_ascii=False,
                            ),
                        )
                    except Exception:
                        pass
                except Exception as e_trail:
                    try:
                        logger.error(f"[ensure_risk][trailing] setup failed: {e_trail}")
                    except Exception:
                        pass
                    try:
                        import time as _t
                        import json as _json
                        rc0 = get_redis_client()
                        rc0.set(
                            f"trading:trailing:last_setup:{symbol}",
                            _json.dumps(
                                {
                                    "ts": _t.time(),
                                    "ok": False,
                                    "symbol": symbol,
                                    "source": "ensure_risk_orders",
                                    "error": str(e_trail),
                                },
                                ensure_ascii=False,
                            ),
                        )
                    except Exception:
                        pass
        else:
            try:
                logger.info(f"[ensure_risk][trailing] skipped by cfg: enabled={trailing_enabled} risk_stop_mode={risk_stop_mode}")
            except Exception:
                pass
    except Exception:
        pass

    if is_long:
        # Long
        # TP (limit SELL): ставим, только если нет position-level TP и нет reduceOnly limit SELL
        if not tp_already_set and not _has_reduce_only_limit('sell'):
            if risk_stop_mode == 'atr_tp_sl':
                _tp, _sl_dummy = _atr_tp_sl_prices(symbol, float(entry_price), atr_k, atr_m, atr_min_sl_mult, True)
                if _tp:
                    tp_price = _rp(_tp)
                else:
                    tp_price = _rp(float(entry_price) * (1.0 + float(tp_pct)/100.0))
            elif tp_pct and tp_pct > 0:
                tp_price = _rp(float(entry_price) * (1.0 + float(tp_pct)/100.0))
            else:
                tp_price = None
            if tp_price:
                try:
                    agent.exchange.create_limit_sell_order(symbol, float(amount), tp_price, { 'reduceOnly': True, 'timeInForce': 'GTC' })
                    try:
                        logger.info(f"[ensure_risk] TP SELL placed {symbol}: price={tp_price} amount={amount}")
                    except Exception:
                        pass
                except Exception:
                    pass
        # SL (stop-market SELL): ставим, только если нет position-level SL и нет reduceOnly stop SELL
        if not sl_already_set and not _has_reduce_only_stop('sell'):
            if risk_stop_mode == 'atr_tp_sl':
                _tp_dummy, _sl = _atr_tp_sl_prices(symbol, float(entry_price), atr_k, atr_m, atr_min_sl_mult, True)
                if _sl:
                    sl_price = _rp(_sl)
                else:
                    sl_price = _rp(float(entry_price) * (1.0 - float(sl_pct)/100.0))
            elif sl_pct and sl_pct > 0:
                sl_price = _rp(float(entry_price) * (1.0 - float(sl_pct)/100.0))
            else:
                sl_price = None
            if sl_price:
                try:
                    agent.exchange.create_order(
                        symbol,
                        'market',
                        'sell',
                        float(amount),
                        None,
                        { 'reduceOnly': True, 'triggerPrice': sl_price, 'triggerDirection': 'descending', 'triggerBy': 'LastPrice', 'timeInForce': 'GTC' }
                    )
                    try:
                        logger.info(f"[ensure_risk] SL SELL placed {symbol}: triggerPrice={sl_price} amount={amount}")
                    except Exception:
                        pass
                except Exception:
                    pass
    else:
        # Short
        if not tp_already_set and not _has_reduce_only_limit('buy'):
            if risk_stop_mode == 'atr_tp_sl':
                _tp, _sl_dummy = _atr_tp_sl_prices(symbol, float(entry_price), atr_k, atr_m, atr_min_sl_mult, False)
                if _tp:
                    tp_price = _rp(_tp)
                else:
                    tp_price = _rp(float(entry_price) * (1.0 - float(tp_pct)/100.0))
            elif tp_pct and tp_pct > 0:
                tp_price = _rp(float(entry_price) * (1.0 - float(tp_pct)/100.0))
            else:
                tp_price = None
            if tp_price:
                try:
                    agent.exchange.create_limit_buy_order(symbol, float(amount), tp_price, { 'reduceOnly': True, 'timeInForce': 'GTC' })
                    try:
                        logger.info(f"[ensure_risk] TP BUY placed {symbol}: price={tp_price} amount={amount}")
                    except Exception:
                        pass
                except Exception:
                    pass
        if not sl_already_set and not _has_reduce_only_stop('buy'):
            if risk_stop_mode == 'atr_tp_sl':
                _tp_dummy, _sl = _atr_tp_sl_prices(symbol, float(entry_price), atr_k, atr_m, atr_min_sl_mult, False)
                if _sl:
                    sl_price = _rp(_sl)
                else:
                    sl_price = _rp(float(entry_price) * (1.0 + float(sl_pct)/100.0))
            elif sl_pct and sl_pct > 0:
                sl_price = _rp(float(entry_price) * (1.0 + float(sl_pct)/100.0))
            else:
                sl_price = None
            if sl_price:
                try:
                    agent.exchange.create_order(
                        symbol,
                        'market',
                        'buy',
                        float(amount),
                        None,
                        { 'reduceOnly': True, 'triggerPrice': sl_price, 'triggerDirection': 'ascending', 'triggerBy': 'LastPrice', 'timeInForce': 'GTC' }
                    )
                    try:
                        logger.info(f"[ensure_risk] SL BUY placed {symbol}: triggerPrice={sl_price} amount={amount}")
                    except Exception:
                        pass
                except Exception:
                    pass

    try:
        logger.info(f"[ensure_risk] ✓ ensured for {symbol}")
    except Exception:
        pass
    return {"success": True, "ensured": True}
# Определяем очереди и маршрутизацию задач:
# По умолчанию все задачи идут в очередь 'celery',
# а тренировочные задачи направляем в отдельную очередь 'train'.
# Определяем очереди и маршрутизацию задач:
# По умолчанию все задачи идут в очередь 'celery',
# а тренировочные задачи направляем в отдельную очередь 'train'.


@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0}, queue='trade')
def execute_trade(self, session_id: str, dry_run: bool = False):
    """Исполнение торгового шага: предсказание через serving, торговля через TradingAgent."""
    try:
        from trading_agent.trading_agent import TradingAgent
        from utils.db_utils import db_get_or_fetch_ohlcv

        # 1) Читаем параметры из Redis при необходимости
        try:
            rc = Redis(host='redis', port=6379, db=0, decode_responses=True)
        except Exception:
            rc = None

        session_doc = load_session(rc, session_id) if rc is not None else None
        if not isinstance(session_doc, dict):
            logger.error("Сессия не найдена: %s", session_id)
            return {
                "success": False, 
                "skipped": True,
                "reason": "session_not_found",
                "error": f"session not found: {session_id}",
            }

        symbol = str(session_doc.get('symbol') or '').strip().upper()
        account_id = str(session_doc.get('account_id') or '').strip()
        symbols = [symbol]
        model_paths = list(session_doc.get('model_paths') or [])
        model_path = str(session_doc.get('model_path') or (model_paths[0] if model_paths else '')).strip() or None
        if not symbol:
            return {"success": False, "skipped": True, "reason": "symbol_missing", "error": "symbol missing in session"}
        if not account_id:
            return {"success": False, "skipped": True, "reason": "account_missing", "error": "account_id missing in session"}

        def sget(name: str, default=None):
            if rc is None:
                return default
            return get_runtime_value(rc, session_id, name, default)

        def sset(name: str, value) -> None:
            if rc is not None:
                set_runtime_value(rc, session_id, name, value)

        def sdel(name: str) -> None:
            if rc is not None:
                delete_runtime_value(rc, session_id, name)

        def sget_json(name: str, default=None):
            if rc is None:
                return default
            return get_runtime_json(rc, session_id, name, default)

        def sset_json(name: str, value) -> None:
            if rc is not None:
                set_runtime_json(rc, session_id, name, value)

        try:
            if rc is not None and sget('disabled'):
                return {"success": False, "skipped": True, "reason": "session_disabled", "symbol": symbol, "session_id": session_id}
        except Exception:
            pass

        syms = symbols

        # --- Pre-trade exit detection ---
        # Синхронно вызываем ensure_risk_orders, чтобы зафиксировать выход (trailing/SL/TP),
        # который мог произойти на бирже до этого тика. Без этого postexit cooldown не сработает.
        try:
            ensure_risk_orders.run(session_id=session_id, symbol=symbol)
        except Exception as _e_ensure:
            logger.warning("[execute_trade] ensure_risk_orders pre-call failed for %s: %s", symbol, _e_ensure)
        
        if (model_paths is None or not model_paths) and model_path:
            model_paths = [model_path]
        # Санити: оставляем только существующие файлы и убираем дубликаты
        try:
            import os as _os
            if model_paths:
                mp_clean = []
                seen = set()
                for p in model_paths:
                    try:
                        pn_raw = str(p)
                        # Нормализуем путь: делаем его абсолютным относительно /workspace,
                        # чтобы относительные 'models/...'
                        # корректно находились внутри контейнера
                        pn_norm = pn_raw.replace('\\', '/')
                        pn_abs = pn_norm if pn_norm.startswith('/') else ('/workspace/' + pn_norm.lstrip('/'))
                        # Дедупликация по абсолютному пути
                        if pn_abs in seen:
                            continue
                        seen.add(pn_abs)
                        # Пропускаем директории
                        if _os.path.isdir(pn_abs):
                            continue
                        # Добавляем только реально существующие файлы
                        if _os.path.exists(pn_abs):
                            mp_clean.append(pn_abs)
                    except Exception:
                        continue
                model_paths = mp_clean
        except Exception:
            pass
        if not model_paths:
            return {"success": False, "error": f"model_paths not provided for session {session_id}"}

        # 2) Готовим состояние для serving (как в агенте: закрытые 5m свечи -> плотный вектор)
        # Последняя закрытая метка времени (5m)
        def _last_closed_ts_ms():
            try:
                now_utc = datetime.utcnow().timestamp()
                last_closed = (int(now_utc) // 300) * 300 - 300
                return last_closed * 1000
            except Exception:
                return 0

        def _sync_last_exit_from_db() -> dict | None:
            """Reconcile Redis last_exit_* with the latest persisted exit for this session."""
            if rc is None:
                return None
            db_session = None
            try:
                from orm.database import get_db_session
                from orm.models import Symbol, Trade

                db_session = get_db_session()
                rows = (
                    db_session.query(Trade)
                    .join(Symbol, Symbol.id == Trade.symbol_id)
                    .filter(Symbol.name == symbol)
                    .filter(Trade.status == 'executed')
                    .filter(Trade.is_successful.is_(True))
                    .order_by(Trade.created_at.desc())
                    .limit(100)
                    .all()
                )
                latest = None
                expected_model_path = str(model_path or (model_paths[0] if model_paths else '') or '').strip()
                for tr in rows:
                    meta = {}
                    try:
                        meta = json.loads(tr.error_message or '{}')
                    except Exception:
                        meta = {}
                    meta_session = str(meta.get('session_id') or '').strip()
                    meta_account = str(meta.get('account_id') or '').strip()
                    meta_model_path = str(meta.get('model_path') or '').strip()
                    matches_session = bool(meta_session and meta_session == str(session_id))
                    matches_account_model = bool(
                        meta_account
                        and meta_account == str(account_id)
                        and (not expected_model_path or meta_model_path == expected_model_path)
                    )
                    if not (matches_session or matches_account_model):
                        continue
                    model_prediction = str(getattr(tr, 'model_prediction', '') or '')
                    reason = str(meta.get('exit_reason') or '').strip().lower()
                    if not reason and model_prediction.startswith('exit:'):
                        reason = model_prediction.split(':', 1)[1].strip().lower()
                    if not reason:
                        continue
                    exit_ts = None
                    try:
                        raw_ts = meta.get('exit_ts_ms')
                        exit_ts = int(float(raw_ts)) if raw_ts not in (None, '') else None
                    except Exception:
                        exit_ts = None
                    if not exit_ts:
                        dt = getattr(tr, 'executed_at', None) or getattr(tr, 'created_at', None)
                        if dt is not None and hasattr(dt, 'timestamp'):
                            exit_ts = int(dt.timestamp() * 1000)
                    if exit_ts:
                        latest = {'ts_ms': int(exit_ts), 'reason': reason, 'order_id': getattr(tr, 'exchange_order_id', None)}
                        break
                if not latest:
                    return None
                current_ts = None
                try:
                    raw_current = sget('last_exit_ts_ms')
                    current_ts = int(float(raw_current)) if raw_current not in (None, '') else None
                except Exception:
                    current_ts = None
                if current_ts is None or int(latest['ts_ms']) > int(current_ts):
                    rc.setex(session_runtime_key(session_id, 'last_exit_ts_ms'), 86400, str(int(latest['ts_ms'])))
                    rc.setex(session_runtime_key(session_id, 'last_exit_reason'), 86400, str(latest['reason']))
                    if latest.get('order_id'):
                        rc.setex(session_runtime_key(session_id, 'last_exit_order_id'), 86400, str(latest['order_id']))
                    logger.warning(
                        "[postexit_sync] session=%s symbol=%s redis_last_exit updated from DB: %s reason=%s",
                        session_id, symbol, latest['ts_ms'], latest['reason']
                    )
                return latest
            except Exception as exc:
                logger.warning("[postexit_sync] failed session=%s symbol=%s: %s", session_id, symbol, exc)
                return None
            finally:
                try:
                    if db_session is not None:
                        db_session.close()
                except Exception:
                    pass

        # Hard timeout guard: выход по hold_steps должен срабатывать даже если дальше
        # упадёт OHLCV/prediction-пайплайн. Проверяем таймаут до запроса данных.
        try:
            if rc is not None:
                raw_exit_mode_pre = sget('exit_mode')
                exit_mode_pre = str(raw_exit_mode_pre or '').strip().lower()
                if exit_mode_pre == 'hold_steps':
                    raw_mhs_pre = sget('max_hold_steps')
                    raw_ets_pre = sget('pos_entry_ts')
                    max_hold_steps_pre = int(float(raw_mhs_pre)) if raw_mhs_pre not in (None, '') else 0
                    entry_ts_pre = int(float(raw_ets_pre)) if raw_ets_pre not in (None, '', '0') else 0
                    now_bucket_pre = int(_last_closed_ts_ms() or 0)

                    if max_hold_steps_pre > 0 and entry_ts_pre > 0 and now_bucket_pre >= entry_ts_pre:
                        elapsed_steps_pre = int((now_bucket_pre - entry_ts_pre) // 300000)
                        if elapsed_steps_pre >= int(max_hold_steps_pre):
                            try:
                                rc.setex(session_runtime_key(session_id, "forced_exit_reason"), 3600, "timeout")
                            except Exception:
                                pass

                            try:
                                dir_pre = str(sget('direction', 'long') or 'long').strip().lower()
                            except Exception:
                                dir_pre = 'long'
                            try:
                                pos_type_hint_pre = str(sget('pos_type') or '').strip().lower()
                            except Exception:
                                pos_type_hint_pre = ''

                            try:
                                model_for_timeout = (model_paths[0] if model_paths else model_path)
                            except Exception:
                                model_for_timeout = model_path

                            try:
                                agent_timeout = TradingAgent(
                                    model_path=model_for_timeout,
                                    direction=dir_pre,
                                    symbol=symbol,
                                    session_id=session_id,
                                    account_id=account_id,
                                )
                                agent_timeout.symbol = symbol
                                agent_timeout.base_symbol = symbol
                                agent_timeout._restore_position_from_exchange()
                                pos_live = getattr(agent_timeout, 'current_position', None)
                                pos_type_live = str((pos_live or {}).get('type') or '').strip().lower()
                                pos_type_eff = pos_type_live or pos_type_hint_pre or dir_pre

                                if pos_live:
                                    if pos_type_eff == 'short':
                                        trade_result_timeout = agent_timeout._execute_cover_short()
                                    else:
                                        trade_result_timeout = agent_timeout._execute_sell()

                                    if isinstance(trade_result_timeout, dict) and trade_result_timeout.get('success'):
                                        try:
                                            sset("pos_open", "0")
                                            sset("last_exit_ts_ms", int(now_bucket_pre))
                                            sdel("pos_entry_ts")
                                        except Exception:
                                            pass
                                        try:
                                            status_timeout = agent_timeout.get_trading_status()
                                            status_timeout['last_model_prediction'] = 'timeout_exit'
                                            status_timeout['session_id'] = session_id
                                            status_timeout['bybit_account_id'] = account_id
                                            set_session_status(rc, session_id, status_timeout)
                                        except Exception:
                                            pass
                                        return {
                                            "success": True,
                                            "decision": ("buy" if pos_type_eff == 'short' else "sell"),
                                            "trade_result": trade_result_timeout,
                                            "timeout_exit": True,
                                            "symbol": symbol,
                                        }
                            except Exception as _timeout_close_e:
                                logger.warning("[execute_trade] timeout close attempt failed for %s: %s", symbol, _timeout_close_e)
        except Exception as _timeout_guard_e:
            logger.warning("[execute_trade] timeout guard failed for %s: %s", symbol, _timeout_guard_e)

        # Hard weak-signal guard: для hold_steps early exit должен закрывать позицию
        # сразу, даже если risk_management_type=exchange_orders.
        try:
            if rc is not None:
                raw_exit_mode_pre = sget('exit_mode')
                exit_mode_pre = str(raw_exit_mode_pre or '').strip().lower()
                raw_signal_exit_enabled_pre = sget('xgb_signal_exit_enabled', session_doc.get('xgb_signal_exit_enabled'))
                signal_exit_enabled_pre = str(raw_signal_exit_enabled_pre or '').strip().lower() in ('1', 'true', 'yes', 'on')
                if exit_mode_pre == 'hold_steps' and signal_exit_enabled_pre:
                    dir_pre = str(sget('direction', 'long') or 'long').strip().lower()
                    pos_type_hint_pre = str(sget('pos_type') or '').strip().lower()
                    signal_exit_role_pre = 'short' if (pos_type_hint_pre == 'short' or dir_pre == 'short') else 'long'
                    raw_signal_exit_role_enabled_pre = sget(
                        f'xgb_signal_exit_{signal_exit_role_pre}_enabled',
                        session_doc.get(f'xgb_signal_exit_{signal_exit_role_pre}_enabled'),
                    )
                    role_enabled_pre = str(raw_signal_exit_role_enabled_pre or '').strip().lower() in ('1', 'true', 'yes', 'on')
                    role_exit_disabled_pre = raw_signal_exit_role_enabled_pre not in (None, '') and not role_enabled_pre
                    if role_exit_disabled_pre:
                        raw_signal_exit_start_pre = None
                        raw_signal_exit_threshold_pre = None
                    else:
                        raw_signal_exit_start_pre = sget(
                            f'xgb_signal_exit_{signal_exit_role_pre}_start_step',
                            session_doc.get(f'xgb_signal_exit_{signal_exit_role_pre}_start_step'),
                        )
                        raw_signal_exit_threshold_pre = sget(
                            f'xgb_signal_exit_{signal_exit_role_pre}_threshold',
                            session_doc.get(f'xgb_signal_exit_{signal_exit_role_pre}_threshold'),
                        )
                    if (not role_exit_disabled_pre) and raw_signal_exit_start_pre in (None, ''):
                        raw_signal_exit_start_pre = sget('xgb_signal_exit_start_step', session_doc.get('xgb_signal_exit_start_step'))
                    raw_signal_exit_window_pre = sget('xgb_signal_exit_window', session_doc.get('xgb_signal_exit_window'))
                    if (not role_exit_disabled_pre) and raw_signal_exit_threshold_pre in (None, ''):
                        raw_signal_exit_threshold_pre = sget(
                            'xgb_signal_exit_threshold',
                            session_doc.get('xgb_signal_exit_threshold', XGB_SIGNAL_EXIT_THRESHOLD_DEFAULT),
                        )
                    raw_ets_pre = sget('pos_entry_ts')
                    signal_exit_start_pre = int(float(raw_signal_exit_start_pre)) if raw_signal_exit_start_pre not in (None, '') else 0
                    signal_exit_window_pre = max(1, int(float(raw_signal_exit_window_pre))) if raw_signal_exit_window_pre not in (None, '') else XGB_SIGNAL_EXIT_WINDOW_DEFAULT
                    signal_exit_threshold_pre = (
                        float(raw_signal_exit_threshold_pre)
                        if raw_signal_exit_threshold_pre not in (None, '')
                        else XGB_SIGNAL_EXIT_THRESHOLD_DEFAULT
                    )
                    entry_ts_pre = int(float(raw_ets_pre)) if raw_ets_pre not in (None, '', '0') else 0
                    now_bucket_pre = int(_last_closed_ts_ms() or 0)
                    elapsed_steps_pre = int((now_bucket_pre - entry_ts_pre) // 300000) if entry_ts_pre > 0 and now_bucket_pre >= entry_ts_pre else None

                    if isinstance(elapsed_steps_pre, int) and signal_exit_start_pre > 0 and elapsed_steps_pre >= signal_exit_start_pre:
                        signal_samples_pre = _load_xgb_signal_history(rc, session_id, signal_exit_window_pre)
                        if len(signal_samples_pre) >= signal_exit_window_pre:
                            avg_signal_pre = float(sum(float(item['signal']) for item in signal_samples_pre) / len(signal_samples_pre))
                            avg_threshold_pre = float(signal_exit_threshold_pre)
                            if avg_signal_pre < avg_threshold_pre:
                                rc.setex(session_runtime_key(session_id, "forced_exit_reason"), 3600, "weak_signal_avg")

                                model_for_signal_exit = (model_paths[0] if model_paths else model_path)
                                agent_signal_exit = TradingAgent(
                                    model_path=model_for_signal_exit,
                                    direction=dir_pre,
                                    symbol=symbol,
                                    session_id=session_id,
                                    account_id=account_id,
                                )
                                agent_signal_exit.symbol = symbol
                                agent_signal_exit.base_symbol = symbol
                                agent_signal_exit._forced_exit_reason = "weak_signal_avg"
                                agent_signal_exit.last_model_prediction = "exit:weak_signal_avg"
                                agent_signal_exit._restore_position_from_exchange()
                                pos_live = getattr(agent_signal_exit, 'current_position', None)
                                pos_type_live = str((pos_live or {}).get('type') or '').strip().lower()
                                pos_type_eff = pos_type_live or pos_type_hint_pre or dir_pre
                                if pos_live:
                                    trade_result_signal = (
                                        agent_signal_exit._execute_cover_short()
                                        if pos_type_eff == 'short'
                                        else agent_signal_exit._execute_sell()
                                    )
                                    if isinstance(trade_result_signal, dict) and trade_result_signal.get('success'):
                                        try:
                                            sset("pos_open", "0")
                                            sset("last_exit_ts_ms", int(now_bucket_pre))
                                            sdel("pos_entry_ts")
                                        except Exception:
                                            pass
                                        try:
                                            status_signal = agent_signal_exit.get_trading_status()
                                            status_signal['last_model_prediction'] = 'weak_signal_avg_exit'
                                            status_signal['session_id'] = session_id
                                            status_signal['bybit_account_id'] = account_id
                                            set_session_status(rc, session_id, status_signal)
                                        except Exception:
                                            pass
                                        return {
                                            "success": True,
                                            "decision": ("buy" if pos_type_eff == 'short' else "sell"),
                                            "trade_result": trade_result_signal,
                                            "signal_exit": True,
                                            "symbol": symbol,
                                            "avg_signal": avg_signal_pre,
                                            "avg_threshold": avg_threshold_pre,
                                        }
        except Exception as _signal_exit_guard_e:
            logger.warning("[execute_trade] signal-exit guard failed for %s: %s", symbol, _signal_exit_guard_e)

        max_window = 0
        try:
            if rc is not None:
                raw_cfg = rc.get('trading:regime_config')
                if raw_cfg:
                    cfg = json.loads(raw_cfg)
                    w = cfg.get('windows') if isinstance(cfg, dict) else None
                    if isinstance(w, (list, tuple)) and w:
                        max_window = max(int(abs(float(x))) for x in w if x is not None)
        except Exception:
            max_window = 0
        if not max_window:
            max_window = 2880
        limit_candles = max(120, int(max_window) + 50)
        t_ohlcv_start = time.monotonic()
        logging.warning(f"[EXECUTE] OHLCV call: symbol={symbol}, tf=5m, limit={limit_candles}, exchange=bybit")
        df_5m, data_error = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe='5m', limit_candles=limit_candles, exchange_id='bybit', include_error=True)
        t_ohlcv_dt = time.monotonic() - t_ohlcv_start
        try:
            _rows = 0 if (df_5m is None) else len(df_5m)
            _tmin = int(df_5m['timestamp'].min()) if _rows else None
            _tmax = int(df_5m['timestamp'].max()) if _rows else None
            _tmin_h = datetime.fromtimestamp(_tmin/1000).isoformat() if _tmin else '—'
            _tmax_h = datetime.fromtimestamp(_tmax/1000).isoformat() if _tmax else '—'
            logging.warning(f"[EXECUTE] OHLCV done in {t_ohlcv_dt:.2f}s; rows={_rows}; range={_tmin_h}..{_tmax_h}; error={bool(data_error)}")
        except Exception:
            logging.warning(f"[EXECUTE] OHLCV done in {t_ohlcv_dt:.2f}s; (range format error)")
        if data_error:
            error_msg = data_error
            human_error = error_msg
            try:
                if error_msg.startswith('exchange_init_failed') or error_msg.startswith('exchange_fetch_failed'):
                    parts = error_msg.split(':', 2)
                    exchange_part = parts[1].strip() if len(parts) > 1 else 'exchange'
                    detail = parts[2].strip() if len(parts) > 2 else ''
                    human_error = f"Bybit API недоступно ({exchange_part}): {detail}" if exchange_part.lower() == 'bybit' else f"API {exchange_part} недоступно: {detail}"
            except Exception:
                human_error = error_msg

            # Подсказка: какие API key реально видит контейнер (маска, НЕ полный ключ)
            bybit_account_id = None
            bybit_key_hint = None
            bybit_keys_hints = {}
            bybit_keys_present = []
            bybit_api_key_public_selected = None
            bybit_api_key_public_1 = None
            bybit_api_key_public_2 = None
            try:
                bybit_account_id = None
                if rc is not None:
                    bybit_account_id = rc.get(f'trading:account_id:{symbol}') or rc.get('trading:account_id')
            except Exception:
                bybit_account_id = None
            try:
                def _mask_key(k: str) -> str:
                    ks = str(k or '').strip()
                    if not ks:
                        return None
                    if len(ks) <= 8:
                        return f"{ks[0:2]}…{ks[-2:]}"
                    return f"{ks[0:4]}…{ks[-4:]}"

                # Явно фиксируем аккаунты 1/2, чтобы в ошибке было видно, что именно "видит" ENV
                try:
                    # ВНИМАНИЕ: это публичный ключ (не secret), выводим по запросу пользователя
                    bybit_api_key_public_1 = os.getenv("BYBIT_1_API_KEY")
                    bybit_api_key_public_2 = os.getenv("BYBIT_2_API_KEY")
                    bybit_keys_hints["1"] = _mask_key(os.getenv("BYBIT_1_API_KEY"))
                    bybit_keys_hints["2"] = _mask_key(os.getenv("BYBIT_2_API_KEY"))
                    bybit_keys_hints["1_has_secret"] = bool(os.getenv("BYBIT_1_SECRET_KEY"))
                    bybit_keys_hints["2_has_secret"] = bool(os.getenv("BYBIT_2_SECRET_KEY"))
                except Exception:
                    pass

                # Список доступных BYBIT_<ID>_API_KEY в окружении (имена + маска)
                try:
                    candidates = []
                    for k, v in os.environ.items():
                        if not (k.startswith("BYBIT_") and k.endswith("_API_KEY")):
                            continue
                        idx = k[len("BYBIT_"):-len("_API_KEY")]
                        if v:
                            candidates.append((str(idx), str(v)))
                    candidates.sort(key=lambda x: x[0])
                    bybit_keys_present = [c[0] for c in candidates]
                    # Добавим маски по найденным индексам (не только 1/2)
                    for idx, v in candidates:
                        bybit_keys_hints[str(idx)] = _mask_key(v)
                        try:
                            bybit_keys_hints[f"{idx}_has_secret"] = bool(os.getenv(f"BYBIT_{idx}_SECRET_KEY"))
                        except Exception:
                            pass
                except Exception:
                    pass

                # 1) Выбранный аккаунт из Redis
                api_key_val = None
                try:
                    if bybit_account_id:
                        api_key_val = os.getenv(f'BYBIT_{bybit_account_id}_API_KEY') or None
                except Exception:
                    api_key_val = None
                try:
                    if bybit_account_id:
                        bybit_api_key_public_selected = os.getenv(f'BYBIT_{bybit_account_id}_API_KEY') or None
                except Exception:
                    bybit_api_key_public_selected = None

                # 2) Фолбэк: стандартные имена
                if not api_key_val:
                    api_key_val = os.getenv('BYBIT_1_API_KEY') or os.getenv('BYBIT_API_KEY') or None

                # 3) Автоскан: первый BYBIT_<ID>_API_KEY
                if not api_key_val:
                    try:
                        if bybit_keys_present:
                            # Возьмём самый маленький индекс по строковому сравнению (уже отсортировано)
                            first_idx = bybit_keys_present[0]
                            api_key_val = os.getenv(f"BYBIT_{first_idx}_API_KEY") or None
                    except Exception:
                        pass

                bybit_key_hint = _mask_key(api_key_val)
            except Exception:
                bybit_key_hint = None

            # Сохраняем в предсказания, что проблема с обменом/сетевым уровнем
            try:
                from orm.models import ModelPrediction
                from orm.database import get_db_session
                with get_db_session() as session:
                    prediction = ModelPrediction(
                        symbol=symbol,
                        action='error',
                        confidence=0.0,
                        q_values='[]',
                        current_price=0.0,
                        position_status='none',
                        model_path='error',
                        market_conditions=json.dumps({
                            "error": str(human_error),
                            "bybit_account_id": bybit_account_id,
                            "bybit_api_key_hint": bybit_key_hint,
                            "bybit_api_key_hints": bybit_keys_hints,
                            "bybit_api_keys_present": bybit_keys_present,
                            "bybit_api_key_public_selected": bybit_api_key_public_selected,
                            "bybit_api_key_public_1": bybit_api_key_public_1,
                            "bybit_api_key_public_2": bybit_api_key_public_2,
                        }, ensure_ascii=False),
                        created_at=datetime.utcnow()
                    )
                    session.add(prediction)
                    session.commit()
            except Exception as e:
                print(f"Failed to save error prediction: {e}")
            return {"success": False, "error": human_error}

        if df_5m is None or df_5m.empty:
            error_msg = "no candles in DB"
            # Сохраняем ошибку в предсказания
            try:
                from orm.models import ModelPrediction
                from orm.database import get_db_session
                with get_db_session() as session:
                    prediction = ModelPrediction(
                        symbol=symbol,
                        action='error',
                        confidence=0.0,
                        q_values='[]',
                        current_price=0.0,
                        position_status='none',
                        model_path='error',
                        market_conditions=json.dumps({"error": str(error_msg)}, ensure_ascii=False),
                        created_at=datetime.utcnow()
                    )
                    session.add(prediction)
                    session.commit()
            except Exception as e:
                print(f"Failed to save error prediction: {e}")
            return {"success": False, "error": error_msg}
        cutoff = _last_closed_ts_ms()
        df_5m = df_5m[df_5m['timestamp'] <= cutoff]
        if df_5m is None or df_5m.empty:
            error_msg = "no closed candles available"
            # Сохраняем ошибку в предсказания
            try:
                from orm.models import ModelPrediction
                from orm.database import get_db_session
                with get_db_session() as session:
                    prediction = ModelPrediction(
                        symbol=symbol,
                        action='error',
                        confidence=0.0,
                        q_values='[]',
                        current_price=0.0,
                        position_status='none',
                        model_path='error',
                        market_conditions=json.dumps({"error": str(error_msg)}, ensure_ascii=False),
                        created_at=datetime.utcnow()
                    )
                    session.add(prediction)
                    session.commit()
            except Exception as e:
                print(f"Failed to save error prediction: {e}")
            return {"success": False, "error": error_msg}
        # Простая нормализация: последние 100 строк OHLCV
        ohlcv_cols = ['open','high','low','close','volume']
        arr = df_5m[ohlcv_cols].tail(100).values.astype('float32')
        if arr.shape[0] < 20:
            error_msg = "insufficient data for state"
            # Сохраняем ошибку в предсказания
            try:
                from orm.models import ModelPrediction
                from orm.database import get_db_session
                with get_db_session() as session:
                    prediction = ModelPrediction(
                        symbol=symbol,
                        action='error',
                        confidence=0.0,
                        q_values='[]',
                        current_price=0.0,
                        position_status='none',
                        model_path='error',
                        market_conditions=json.dumps({"error": str(error_msg)}, ensure_ascii=False),
                        created_at=datetime.utcnow()
                    )
                    session.add(prediction)
                    session.commit()
            except Exception as e:
                print(f"Failed to save error prediction: {e}")
            return {"success": False, "error": error_msg}
        max_vals = np.maximum(arr.max(axis=0), 1e-9)
        norm = (arr / max_vals).flatten()
        # Ограничим/дополняем до 100*5=500 признаков
        if norm.size < 500:
            norm = np.pad(norm, (0, 500 - norm.size))
        elif norm.size > 500:
            norm = norm[:500]
        state = norm.tolist()

        # Оценка рыночного режима (flat / uptrend / downtrend) по последним закрытым свечам
        def _compute_regime(df: pd.DataFrame):
            try:
                # Загружаем конфигурацию из Redis (если есть)
                cfg = None
                try:
                    if rc is not None:
                        raw = rc.get('trading:regime_config')
                        if raw:
                            cfg = json.loads(raw)
                except Exception:
                    cfg = None

                # Значения по умолчанию
                windows = (cfg.get('windows') if isinstance(cfg, dict) else None) or [576, 1440, 2880]
                weights = (cfg.get('weights') if isinstance(cfg, dict) else None) or [1, 1, 1]
                voting = (cfg.get('voting') if isinstance(cfg, dict) else None) or 'majority'
                tie_break = (cfg.get('tie_break') if isinstance(cfg, dict) else None) or 'last'
                # Conflict‑veto (UI /settings -> /api/trading/regime_config):
                # если 2 коротких окна (576/1440) совпали и противоположны длинному (2880) → можно блокировать вход (HOLD).
                conflict_veto_enabled = False
                conflict_veto_trend_only = True
                # Trend-veto: запрещаем торговать ПРОТИВ режима (downtrend блокирует BUY, uptrend блокирует SELL).
                trend_veto_enabled = False

                # 1) Читаем флаги из Postgres (app_settings), чтобы настраивать как остальные настройки в /settings.
                #    scope='trading', group='regime'
                #    keys: CONFLICT_VETO, CONFLICT_VETO_TREND_ONLY, TREND_VETO
                def _truthy(v) -> bool:
                    try:
                        return str(v).strip().lower() in ('1', 'true', 'yes', 'on')
                    except Exception:
                        return False

                db_conflict_veto = None
                db_conflict_veto_trend_only = None
                db_trend_veto = None
                try:
                    from utils.settings_store import get_setting_value as _get_setting_value
                    db_conflict_veto = _get_setting_value('trading', 'regime', 'CONFLICT_VETO')
                    db_conflict_veto_trend_only = _get_setting_value('trading', 'regime', 'CONFLICT_VETO_TREND_ONLY')
                    db_trend_veto = _get_setting_value('trading', 'regime', 'TREND_VETO')
                except Exception:
                    db_conflict_veto = None
                    db_conflict_veto_trend_only = None
                    db_trend_veto = None

                if db_conflict_veto is not None:
                    conflict_veto_enabled = _truthy(db_conflict_veto)
                if db_conflict_veto_trend_only is not None:
                    conflict_veto_trend_only = _truthy(db_conflict_veto_trend_only)
                if db_trend_veto is not None:
                    trend_veto_enabled = _truthy(db_trend_veto)

                # 2) Фолбэк: если в БД не задано — читаем из Redis cfg (как раньше)
                try:
                    if isinstance(cfg, dict):
                        if db_conflict_veto is None:
                            conflict_veto_enabled = bool(cfg.get('conflict_veto') is True or str(cfg.get('conflict_veto') or '').strip().lower() in ('1', 'true', 'yes', 'on'))
                        # По умолчанию: veto только для uptrend/downtrend (flat игнорируем)
                        if db_conflict_veto_trend_only is None and ('conflict_veto_trend_only' in cfg):
                            conflict_veto_trend_only = bool(cfg.get('conflict_veto_trend_only') is True or str(cfg.get('conflict_veto_trend_only') or '').strip().lower() in ('1', 'true', 'yes', 'on'))
                        if db_trend_veto is None:
                            trend_veto_enabled = bool(cfg.get('trend_veto') is True or str(cfg.get('trend_veto') or '').strip().lower() in ('1', 'true', 'yes', 'on'))
                except Exception:
                    conflict_veto_enabled = False
                    conflict_veto_trend_only = True
                    trend_veto_enabled = False
                drift_thr = float((cfg.get('drift_threshold') if isinstance(cfg, dict) else 0.001) or 0.001)
                vol_flat_thr = float((cfg.get('flat_vol_threshold') if isinstance(cfg, dict) else 0.003) or 0.003)
                # Пока регрессию/ADX не включаем (флаги можно будет учесть позже)

                closes_full = df['close'].astype(float).values
                if closes_full.size < max(windows) + 5:
                    # данных маловато — считаем flat
                    return 'flat', {
                        'windows': windows,
                        'weights': weights,
                        'voting': voting,
                        'tie_break': tie_break,
                        'labels': ['flat'] * len(windows),
                        'votes_map': {'flat': 0.0, 'uptrend': 0.0, 'downtrend': 0.0},
                        'drift_threshold': float(drift_thr),
                        'flat_vol_threshold': float(vol_flat_thr),
                        'metrics': []
                    }

                def classify_window(n: int):
                    # Возвращает (label, drift, vol) для окна n
                    c = closes_full[-n:]
                    if c.size < max(20, n // 3):
                        return 'flat', 0.0, 0.0
                    start = float(c[0]); end = float(c[-1])
                    drift = (end - start) / max(start, 1e-9)
                    rr = np.diff(c) / np.maximum(c[:-1], 1e-9)
                    vol = float(np.std(rr))
                    if abs(drift) < (0.75 * drift_thr) and vol < vol_flat_thr:
                        return 'flat', drift, vol
                    if drift >= drift_thr:
                        return 'uptrend', drift, vol
                    if drift <= -drift_thr:
                        return 'downtrend', drift, vol
                    return 'flat', drift, vol

                votes = {'flat': 0.0, 'uptrend': 0.0, 'downtrend': 0.0}
                labels = []
                metrics = []
                for i, w in enumerate(windows):
                    lab, drift, vol = classify_window(int(w))
                    labels.append(lab)
                    metrics.append({'window': int(w), 'drift': float(drift), 'vol': float(vol)})
                    wt = float(weights[i] if i < len(weights) else 1.0)
                    votes[lab] += wt

                # Conflict-veto candidate: 2 коротких одинаковые и противоположны длинному (по максимальному window)
                long_idx = None
                long_window = None
                long_label = None
                short_label = None
                conflict_veto_candidate = False
                try:
                    if windows and labels and len(windows) == len(labels) and len(windows) >= 3:
                        long_idx = max(range(len(windows)), key=lambda i: int(windows[i]))
                        long_window = int(windows[long_idx])
                        long_label = str(labels[long_idx])
                        shorts = [str(lab) for i, lab in enumerate(labels) if i != long_idx]
                        if len(shorts) >= 2 and shorts[0] == shorts[1]:
                            short_label = shorts[0]
                            # veto только если это явный трендовый конфликт (up vs down)
                            if short_label != long_label:
                                if conflict_veto_trend_only:
                                    conflict_veto_candidate = ({short_label, long_label} <= {'uptrend', 'downtrend'})
                                else:
                                    conflict_veto_candidate = True
                except Exception:
                    conflict_veto_candidate = False

                if voting == 'majority':
                    # Простое большинство по равным весам
                    counts = {'flat': labels.count('flat'), 'uptrend': labels.count('uptrend'), 'downtrend': labels.count('downtrend')}
                    mx = max(counts.values())
                    winners = [k for k, v in counts.items() if v == mx]
                    if len(winners) == 1:
                        winner = winners[0]
                    else:
                        # Ничья → правило tie_break
                        winner = 'flat' if tie_break == 'flat' else labels[-1]
                else:
                    # Взвешенное голосование по weights
                    best = max(votes.items(), key=lambda kv: kv[1])[0]
                    # Проверка ничьей по сумме весов
                    vals = sorted(votes.values(), reverse=True)
                    if len(vals) >= 2 and abs(vals[0] - vals[1]) < 1e-9:
                        winner = 'flat' if tie_break == 'flat' else labels[-1]
                    else:
                        winner = best

                details = {
                    'windows': [int(w) for w in windows],
                    'weights': [float(weights[i] if i < len(weights) else 1.0) for i in range(len(windows))],
                    'voting': voting,
                    'tie_break': tie_break,
                    'labels': labels,
                    'votes_map': votes,
                    # Conflict-veto diagnostics/config
                    'conflict_veto_enabled': bool(conflict_veto_enabled),
                    'conflict_veto_trend_only': bool(conflict_veto_trend_only),
                    'conflict_veto_candidate': bool(conflict_veto_candidate),
                    'long_window': long_window,
                    'long_label': long_label,
                    'short_label': short_label,
                    # Trend-veto config
                    'trend_veto_enabled': bool(trend_veto_enabled),
                    'drift_threshold': float(drift_thr),
                    'flat_vol_threshold': float(vol_flat_thr),
                    'metrics': metrics
                }
                return winner, details
            except Exception:
                return 'flat', {
                    'windows': [576, 1440, 2880],
                    'weights': [1, 1, 1],
                    'voting': 'majority',
                    'tie_break': 'last',
                    'labels': ['flat', 'flat', 'flat'],
                    'votes_map': {'flat': 0.0, 'uptrend': 0.0, 'downtrend': 0.0},
                    'conflict_veto_enabled': False,
                    'conflict_veto_trend_only': True,
                    'conflict_veto_candidate': False,
                    'long_window': 2880,
                    'long_label': 'flat',
                    'short_label': None,
                    'trend_veto_enabled': False,
                    'drift_threshold': 0.002,
                    'flat_vol_threshold': 0.0025,
                    'metrics': []
                }

        market_regime, market_regime_details = _compute_regime(df_5m)
        try:
            logger.warning(
                "[REGIME] symbol=%s regime=%s windows=%s labels=%s votes=%s metrics=%s",
                symbol,
                market_regime,
                market_regime_details.get('windows') if isinstance(market_regime_details, dict) else None,
                market_regime_details.get('labels') if isinstance(market_regime_details, dict) else None,
                market_regime_details.get('votes_map') if isinstance(market_regime_details, dict) else None,
                market_regime_details.get('metrics') if isinstance(market_regime_details, dict) else None,
            )
        except Exception:
            pass

        # 2.5) long-short режим: роли моделей читаем для диагностики и будущего decision layer.
        # Важно: model_paths не фильтруем до prediction, чтобы long и short считались всегда.
        trade_mode = 'single'
        model_roles_map = {}
        # Для UI/диагностики: покажем, какая роль была бы целевой по старому regime rule.
        ls_target_role = None
        ls_models_before = list(model_paths) if isinstance(model_paths, list) else []
        ls_models_after = None
        try:
            if rc is not None:
                tm = sget('trade_mode')
                tm = str(tm).strip() if tm else ''
                if tm == 'long-short':
                    trade_mode = 'long-short'
                raw_roles = sget_json('model_roles', {})
                if raw_roles:
                    parsed = raw_roles if isinstance(raw_roles, dict) else {}
                    if isinstance(parsed, dict):
                        # Нормализуем ключи так же, как model_paths (абсолютный путь внутри /workspace)
                        for k, v in parsed.items():
                            try:
                                kp = str(k).replace('\\', '/')
                                kp_abs = kp if kp.startswith('/') else ('/workspace/' + kp.lstrip('/'))
                                rv = str(v).strip().lower()
                                model_roles_map[kp_abs] = ('short' if rv == 'short' else 'long')
                            except Exception:
                                continue
        except Exception:
            trade_mode = 'single'
            model_roles_map = {}

        try:
            ignore_trend_filter = str(sget('ignore_trend_filter', session_doc.get('ignore_trend_filter'))).strip().lower() in ('1', 'true', 'yes', 'on')
            if trade_mode == 'long-short' and isinstance(model_paths, list) and model_paths:
                target_role = None
                if not ignore_trend_filter:
                    if market_regime == 'uptrend':
                        target_role = 'long'
                    elif market_regime == 'downtrend':
                        target_role = 'short'
                ls_target_role = target_role
                if target_role:
                    mp_before = list(model_paths)
                    mp_filtered = [p for p in model_paths if model_roles_map.get(str(p)) == target_role]
                    ls_models_before = mp_before
                    ls_models_after = list(mp_filtered) if mp_filtered else list(model_paths)
                    try:
                        logger.warning("[long-short] symbol=%s regime=%s target_role=%s models_for_prediction=%s target_role_models=%s",
                                       symbol, market_regime, target_role, len(model_paths), len(mp_filtered))
                    except Exception:
                        pass
                    # ВАЖНО: model_paths не фильтруем до prediction, чтобы long и short считались всегда.
                    # model_paths = ls_models_after
        except Exception:
            pass

        # 3) Вызов serving (+ передаём настройки консенсуса, если заданы)
        # Читаем консенсус из Redis для конкретного символа: {'counts': {flat, trend, total_selected}, 'percents': {flat, trend}}
        consensus_cfg = None
        try:
            if rc is not None:
                _c = sget_json('consensus', {})
                if _c:
                    consensus_cfg = _c
        except Exception as e:
            logger.warning(f"Ошибка при получении консенсуса для символа {symbol}: {e}")
            consensus_cfg = None

        serving_url = get_config_value('SERVING_URL', 'http://serving:8000/predict_ensemble')
        try:
            print(f"[ensemble] models={len(model_paths)} | files={[p.split('/')[-1] for p in model_paths]}")
        except Exception:
            pass
        payload = {
            "state": state,
            "model_paths": model_paths,
            "symbol": symbol,
            "consensus": consensus_cfg or {},
            "market_regime": market_regime,
            "market_regime_details": market_regime_details
        }
        is_xgb_trade = bool(model_paths) and all('/models/xgb/' in str(p).replace('\\', '/') for p in model_paths)
        threshold_override = None
        threshold_overrides_by_model = {}
        xgb_signal_snapshot = None
        try:
            if is_xgb_trade:
                from tasks.xgb_live import predict_xgb_live
                import time as _time

                try:
                    if rc is not None:
                        for model_path_item in model_paths:
                            model_path_key = str(model_path_item or '').replace('\\', '/')
                            model_path_abs = model_path_key if model_path_key.startswith('/') else ('/workspace/' + model_path_key.lstrip('/'))
                            role_name = str(model_roles_map.get(model_path_key) or model_roles_map.get(model_path_abs) or '').strip().lower()
                            if role_name not in ('long', 'short'):
                                continue
                            thr_override = sget(f"xgb_entry_threshold_{role_name}")
                            if thr_override not in (None, ""):
                                threshold_overrides_by_model[str(model_path_item)] = float(thr_override)
                except Exception:
                    threshold_overrides_by_model = {}

                missing_1m_marker = "missing 1m candles for 5m timestamp="
                last_missing_1m_error = None
                for attempt in range(6):
                    try:
                        pred_json = predict_xgb_live(
                            symbol=symbol,
                            model_paths=[str(p) for p in model_paths],
                            df_5m=df_5m,
                            threshold_override=threshold_override,
                            threshold_overrides_by_model=threshold_overrides_by_model or None,
                        )
                        last_missing_1m_error = None
                        break
                    except Exception as e:
                        msg = str(e)
                        if missing_1m_marker in msg:
                            last_missing_1m_error = msg
                            if attempt < 5:
                                try:
                                    logger.warning(
                                        "[xgb_live] missing 1m candles; retry in 10s (%s/6): %s",
                                        attempt + 1,
                                        msg,
                                    )
                                except Exception:
                                    pass
                                _time.sleep(10)
                                continue
                        raise
                if last_missing_1m_error:
                    pred_json = {
                        "success": False,
                        "error": last_missing_1m_error,
                        "error_code": "missing_1m_candles",
                        "error_human": "не загрузились 1m свечи",
                    }
            else:
                resp = requests.post(serving_url, json=payload, timeout=30)
                # Пытаемся извлечь тело при ошибке
                if not resp.ok:
                    body = None
                    try:
                        body = resp.text
                    except Exception:
                        body = None
                    return {"success": False, "error": f"serving error: {resp.status_code} {resp.reason}", "body": body}
                pred_json = resp.json()
        except Exception as e:
            return {"success": False, "error": f"serving error: {e}"}

        if not pred_json.get('success'):
            error_msg = pred_json.get('error', 'serving failed')
            error_code = pred_json.get('error_code') if isinstance(pred_json, dict) else None
            error_human = pred_json.get('error_human') if isinstance(pred_json, dict) else None
            fallback_model_path = None
            try:
                if isinstance(model_paths, list) and model_paths:
                    fallback_model_path = str(model_paths[0])
            except Exception:
                fallback_model_path = None
            # Сохраняем ошибку в предсказания
            try:
                from orm.models import ModelPrediction
                from orm.database import get_db_session
                with get_db_session() as session:
                    prediction = ModelPrediction(
                        symbol=symbol,
                        action='error',
                        confidence=0.0,
                        q_values='[]',
                        current_price=0.0,
                        position_status='none',
                        model_path=fallback_model_path or '',
                        market_conditions=json.dumps(
                            {
                                "error": str(error_msg),
                                "error_code": str(error_code) if error_code else None,
                                "error_human": str(error_human) if error_human else None,
                            },
                            ensure_ascii=False,
                        ),
                        created_at=datetime.utcnow()
                    )
                    session.add(prediction)
                    session.commit()
            except Exception as e:
                print(f"Failed to save error prediction: {e}")
            return {"success": False, "error": error_msg}

        # --- IMPORTANT: short-model action/Q-values mapping ---
        # В CryptoTradingEnvShort действие=1 означает ENTER_SHORT (SELL), а действие=2 означает COVER (BUY).
        # Serving по умолчанию мапит [0,1,2] как [hold,buy,sell], поэтому short-модель выглядит как BUY и не открывает шорт.
        # Здесь приводим q_values и action к "каноническому" формату [hold,buy,sell] путём swap(1<->2) для моделей роли short.
        try:
            preds_list0 = pred_json.get('predictions') or []
            if (not is_xgb_trade) and isinstance(preds_list0, list) and preds_list0:
                for _p in preds_list0:
                    try:
                        mp0 = str((_p or {}).get('model_path') or '').replace('\\', '/')
                        role0 = model_roles_map.get(mp0)
                        if role0 != 'short':
                            continue
                        qv0 = (_p or {}).get('q_values') or []
                        if not (isinstance(qv0, list) and len(qv0) >= 3):
                            continue
                        # qv: [hold, enter_short, cover] -> [hold, cover, enter_short]
                        qv1 = [qv0[0], qv0[2], qv0[1]]
                        _p['q_values'] = qv1
                        # recompute action from swapped q-values
                        try:
                            idx = max(range(3), key=lambda i: float(qv1[i]))
                            _p['action'] = ['hold', 'buy', 'sell'][int(idx)]
                        except Exception:
                            pass
                    except Exception:
                        continue
        except Exception:
            pass

        try:
            if is_xgb_trade and rc is not None:
                signal_exit_window_cfg = XGB_SIGNAL_EXIT_WINDOW_DEFAULT
                try:
                    raw_window = sget('xgb_signal_exit_window')
                    if raw_window not in (None, ''):
                        signal_exit_window_cfg = max(1, int(float(raw_window)))
                except Exception:
                    signal_exit_window_cfg = XGB_SIGNAL_EXIT_WINDOW_DEFAULT
                signal_snapshot = _extract_xgb_signal_snapshot(
                    pred_json,
                    model_paths=[str(p) for p in (model_paths or []) if p],
                    threshold_override=threshold_override,
                )
                xgb_signal_snapshot = signal_snapshot
                if isinstance(signal_snapshot, dict):
                    signal_snapshot['ts_ms'] = int(_last_closed_ts_ms() or 0)
                    _append_xgb_signal_history(rc, session_id, signal_snapshot, signal_exit_window_cfg)
        except Exception:
            pass

        # Подготовим пороги Q-gate: per-symbol Redis > ENV/Config > JSON > дефолт
        qgate_cfg = pred_json.get('qgate') or {}

        sym0 = symbol

        def _pick_threshold(config_key: str, default_val: float, redis_key: str | None = None) -> float:
            # 0) per-symbol в Redis
            try:
                    if redis_key and sym0 and rc is not None:
                        field_name = redis_key.replace('trading:', '')
                        v = sget(field_name)
                    if v is not None and str(v).strip() != '':
                        return float(v)
            except Exception:
                pass
            try:
                config_val = get_config_value(config_key)
                print(f"DEBUG: {config_key} = {config_val}")
                if config_val is not None and str(config_val).strip() != '':
                    val = float(config_val)
                    print(f"DEBUG: Using CONFIG {config_key} = {val}")
                    return val
            except Exception as e:
                print(f"DEBUG: CONFIG {config_key} error: {e}")
                pass
            print(f"DEBUG: Using default {config_key} = {default_val}")
            return float(default_val)

        T1 = _pick_threshold('QGATE_MAXQ', 0.500, 'trading:qgate_maxq')
        T2 = _pick_threshold('QGATE_GAPQ', 0.440, 'trading:qgate_gapq')
        print(f"QGate thresholds chosen: T1={T1:.3f}, T2={T2:.3f}")

        # --- Post-exit policy (after TP/SL): cooldown + boosted Q-gate ---
        # Требование:
        # - после выхода по stop_loss/take_profit: 5 свечей не входить
        # - затем ещё 10 свечей: повышенные MAXQ/GAPQ (из DB/Redis), чтобы избегать "re-entry"
        postexit = None
        try:
            if rc is not None and sym0:
                if is_xgb_trade:
                    _sync_last_exit_from_db()
                # time bucket: last closed 5m candle
                try:
                    now_utc = datetime.utcnow()
                    epoch_sec = int(now_utc.timestamp())
                    last_closed = (epoch_sec // 300) * 300 - 300
                    now_bucket_ts = int(last_closed * 1000)
                except Exception:
                    now_bucket_ts = 0

                ex_ts_raw = sget("last_exit_ts_ms")
                ex_reason_raw = sget("last_exit_reason")
                try:
                    ex_ts = int(ex_ts_raw) if ex_ts_raw is not None and str(ex_ts_raw).strip() != '' else None
                except Exception:
                    ex_ts = None
                ex_reason = str(ex_reason_raw or '').strip().lower()
                allowed_reasons = ('stop_loss', 'take_profit', 'trailing')
                if is_xgb_trade:
                    allowed_reasons = (
                        'stop_loss', 'take_profit', 'trailing', 'timeout',
                        'weak_signal_avg', 'manual', 'signal', 'holdstep', 'unknown',
                    )
                postexit_enabled = True
                if is_xgb_trade:
                    raw_enabled = sget("xgb_postexit_guard_enabled", "0")
                    postexit_enabled = str(raw_enabled or '').strip().lower() in ('1', 'true', 'yes', 'on')
                def _bucket_last_closed_5m_ms(ts_ms: int) -> int:
                    try:
                        epoch_sec = int(int(ts_ms) / 1000)
                        last_closed = (epoch_sec // 300) * 300 - 300
                        return int(last_closed * 1000)
                    except Exception:
                        return 0

                ex_bucket_ts = _bucket_last_closed_5m_ms(int(ex_ts)) if ex_ts else 0
                if postexit_enabled and ex_bucket_ts and now_bucket_ts and ex_reason in allowed_reasons and now_bucket_ts >= ex_bucket_ts:
                    candles_since = int((now_bucket_ts - ex_bucket_ts) // 300000)

                    def _pick_int(key: str, default_val: int) -> int:
                        try:
                            v = sget(key)
                            if v is None:
                                v = rc.get(f'{key}:{sym0}') or rc.get(key)
                            if v is not None and str(v).strip() != '':
                                return int(float(v))
                        except Exception:
                            pass
                        return int(default_val)

                    # --- SL streak escalation: чтение ступеней из DB (app_settings) ---
                    # scope=trading, group=sl_escalation, keys: SL_COOLDOWN_1..SL_COOLDOWN_4
                    sl_escalation_steps = [48, 96, 192, 288]  # defaults (candles at 5min)
                    try:
                        from utils.settings_store import get_setting_value as _gsv_esc
                        for _ei in range(4):
                            _ev = _gsv_esc('trading', 'sl_escalation', f'SL_COOLDOWN_{_ei+1}')
                            if _ev is not None and str(_ev).strip() != '':
                                sl_escalation_steps[_ei] = int(float(_ev))
                    except Exception:
                        pass

                    # Определяем streak и выбираем cooldown
                    sl_streak = 0
                    try:
                        _streak_raw = rc.get(f"trading:sl_streak:{sym0}")
                        sl_streak = int(_streak_raw) if _streak_raw is not None and str(_streak_raw).strip() != '' else 0
                    except Exception:
                        sl_streak = 0

                    if is_xgb_trade:
                        cooldown_candles = _pick_int('xgb_postexit_cooldown_candles', 5)
                    elif ex_reason == 'stop_loss' and sl_streak >= 1:
                        idx_esc = min(sl_streak - 1, len(sl_escalation_steps) - 1)
                        cooldown_candles = int(sl_escalation_steps[idx_esc])
                    else:
                        cooldown_candles = _pick_int('trading:postexit_cooldown_candles', 48)
                    boost_candles = (
                        _pick_int('xgb_postexit_boost_candles', 20)
                        if is_xgb_trade
                        else _pick_int('trading:postexit_boost_candles', 10)
                    )
                    xgb_boost_pct = None
                    if is_xgb_trade:
                        try:
                            xgb_boost_pct = float(sget("xgb_postexit_threshold_boost_pct", 10) or 10)
                        except Exception:
                            xgb_boost_pct = 10.0

                    # boosted thresholds (absolute) from DB/Redis; fallback: multipliers
                    post_T1 = None
                    post_T2 = None
                    try:
                        v = rc.get(f'trading:postexit_qgate_maxq:{sym0}') or rc.get('trading:postexit_qgate_maxq')
                        if v is not None and str(v).strip() != '':
                            post_T1 = float(v)
                    except Exception:
                        post_T1 = None
                    try:
                        v = rc.get(f'trading:postexit_qgate_gapq:{sym0}') or rc.get('trading:postexit_qgate_gapq')
                        if v is not None and str(v).strip() != '':
                            post_T2 = float(v)
                    except Exception:
                        post_T2 = None
                    if post_T1 is None:
                        try:
                            m = rc.get(f'trading:postexit_qgate_maxq_mult:{sym0}') or rc.get('trading:postexit_qgate_maxq_mult')
                            post_T1 = float(T1) * float(m) if m is not None and str(m).strip() != '' else None
                        except Exception:
                            post_T1 = None
                    if post_T2 is None:
                        try:
                            m = rc.get(f'trading:postexit_qgate_gapq_mult:{sym0}') or rc.get('trading:postexit_qgate_gapq_mult')
                            post_T2 = float(T2) * float(m) if m is not None and str(m).strip() != '' else None
                        except Exception:
                            post_T2 = None

                    postexit = {
                        'exit_ts_ms': int(ex_ts),
                        'exit_bucket_ts_ms': int(ex_bucket_ts),
                        'exit_reason': ex_reason,
                        'now_bucket_ts_ms': int(now_bucket_ts),
                        'candles_since_exit': int(candles_since),
                        'cooldown_candles': int(cooldown_candles),
                        'boost_candles': int(boost_candles),
                        'boost_T1': post_T1,
                        'boost_T2': post_T2,
                        'xgb_threshold_boost_pct': xgb_boost_pct,
                        'sl_streak': int(sl_streak),
                        'sl_escalation_steps': list(sl_escalation_steps),
                    }
                    # Диагностика: что пришло из Redis (сырьё) + ISO время для UI/логов
                    try:
                        postexit['redis_last_exit_ts_ms_raw'] = ex_ts_raw
                        postexit['redis_last_exit_reason_raw'] = ex_reason_raw
                    except Exception:
                        pass
                    try:
                        postexit['exit_ts_iso_utc'] = datetime.utcfromtimestamp(int(ex_ts) / 1000).isoformat()
                        postexit['now_bucket_ts_iso_utc'] = datetime.utcfromtimestamp(int(now_bucket_ts) / 1000).isoformat()
                    except Exception:
                        pass
                    pred_json['postexit'] = postexit

                    # apply boosted thresholds during boost window (after cooldown)
                    if (not is_xgb_trade) and candles_since >= cooldown_candles and candles_since < (cooldown_candles + boost_candles):
                        try:
                            if isinstance(post_T1, (int, float)) and post_T1 > float(T1):
                                T1 = float(post_T1)
                            if isinstance(post_T2, (int, float)) and post_T2 > float(T2):
                                T2 = float(post_T2)
                        except Exception:
                            pass
                    # Явная фаза и применённые пороги (чтобы UI показывал "прошло 5 свечей / пороги подняты")
                    try:
                        if candles_since < cooldown_candles:
                            postexit['phase'] = 'cooldown'
                        elif candles_since < (cooldown_candles + boost_candles):
                            postexit['phase'] = 'boost'
                        else:
                            postexit['phase'] = 'expired'
                    except Exception:
                        postexit['phase'] = 'unknown'
                    try:
                        postexit['applied_T1'] = float(T1)
                        postexit['applied_T2'] = float(T2)
                    except Exception:
                        pass
        except Exception:
            postexit = None

        try:
            flat_factor = float(get_config_value('QGATE_FLAT', '1.0'))
        except Exception:
            flat_factor = 1.0
        if market_regime == 'flat' and flat_factor and flat_factor != 1.0:
            T1 *= flat_factor
            T2 *= flat_factor

        eff_T1 = T1
        eff_T2 = T2

        if is_xgb_trade:
            pred_json['qgate_T1'] = None
            pred_json['qgate_T2'] = None
            pred_json['xgb_qgate_disabled'] = True
        else:
            pred_json['qgate_T1'] = float(T1)
            pred_json['qgate_T2'] = float(T2)

        # 3.1) Консенсус по ансамблю на стороне оркестратора
        preds_list = pred_json.get('predictions') or []
        try:
            if len(model_paths) > 1 and len(preds_list) <= 1:
                print(f"[ensemble] WARNING: requested {len(model_paths)} models, serving returned {len(preds_list)} predictions")
        except Exception:
            pass
        decision = pred_json.get('decision', 'hold')
        # Заполним сводку по голосам/порогам для последующего сохранения в prediction.market_conditions
        votes = {'buy': 0, 'sell': 0, 'hold': 0}
        total_sel = len(model_paths)
        total_sel_buy = total_sel
        total_sel_sell = total_sel
        try:
            if str(trade_mode).strip() == 'long-short':
                total_sel_buy = sum(1 for p in model_paths if str(model_roles_map.get(str(p).replace('\\', '/')) or '').strip().lower() == 'long')
                total_sel_sell = sum(1 for p in model_paths if str(model_roles_map.get(str(p).replace('\\', '/')) or '').strip().lower() == 'short')
        except Exception:
            pass
            
        req_flat = None
        req_trend = None
        required = None
        required_type = 'flat'
        consensus_from_qgate = False
        try:
            if isinstance(preds_list, list) and len(preds_list) > 0:
                # Пер‑модельный Q-gate: учитываем в голосах только те BUY/SELL, что проходят T1/T2
                for p in preds_list:
                    act = str(p.get('action') or 'hold').lower()
                    qv = p.get('q_values') or []
                    gate_ok = True if is_xgb_trade else False
                    try:
                        if is_xgb_trade:
                            gate_ok = True
                        elif isinstance(qv, list) and len(qv) >= 3:
                            if act == 'buy':
                                q_buy = float(qv[1])
                                other = max(float(qv[0]), float(qv[2]))
                                _pass_max = True if float(eff_T1) == 0.0 else (q_buy >= eff_T1)
                                _pass_gap = True if float(eff_T2) == 0.0 else ((q_buy - other) >= eff_T2)
                                gate_ok = _pass_max and _pass_gap
                            elif act == 'sell':
                                q_sell = float(qv[2])
                                other = max(float(qv[0]), float(qv[1]))
                                _pass_max = True if float(eff_T1) == 0.0 else (q_sell >= eff_T1)
                                _pass_gap = True if float(eff_T2) == 0.0 else ((q_sell - other) >= eff_T2)
                                gate_ok = _pass_max and _pass_gap
                            else:
                                gate_ok = True # HOLD не блокируем Q‑gate
                        else:
                            gate_ok = (act == 'hold')
                    except Exception:
                        gate_ok = (act == 'hold')
                    # Remap action для short-моделей: buy↔sell (action 1=ENTER_SHORT→sell, action 2=COVER→buy)
                    # Только для multiclass моделей! У бинарных уже правильный action.
                    vote_act = act
                    try:
                        mp_key = str(p.get('model_path') or '').replace('\\', '/')
                        task_name = str(p.get('task') or '').strip().lower()
                        is_binary = task_name.startswith('entry') or task_name.startswith('exit')
                        if model_roles_map.get(mp_key) == 'short' and not is_binary:
                            if act == 'buy':
                                vote_act = 'sell'
                            elif act == 'sell':
                                vote_act = 'buy'
                    except Exception:
                        pass
                    if vote_act in ('buy','sell'):
                        if gate_ok:
                            votes[vote_act] += 1
                    elif vote_act == 'hold':
                        votes['hold'] += 1
                # Выбираем порог в моделях в зависимости от режима
                if consensus_cfg:
                    counts = (consensus_cfg.get('counts') or {})
                    perc = (consensus_cfg.get('percents') or {})
                    # counts приоритетнее percents — используем только counts; проценты игнорируем, чтобы не было 1/3
                    if isinstance(counts.get('flat'), (int, float)):
                        req_flat = int(max(1, counts.get('flat')))
                    if isinstance(counts.get('trend'), (int, float)):
                        req_trend = int(max(1, counts.get('trend')))
                
                required_type = 'trend' if market_regime in ('uptrend','downtrend') else 'flat'
                
                def calc_req(tsel):
                    req = 2 if tsel >= 3 else max(1, tsel)
                    cfg_val = req_trend if required_type == 'trend' else req_flat
                    if cfg_val is not None:
                        req = cfg_val
                    return int(min(max(1, req), max(1, tsel)))
                
                required_buy = calc_req(total_sel_buy)
                required_sell = calc_req(total_sel_sell)
                
                # Правило консенсуса: если хватает голосов BUY → buy, если SELL → sell; иначе hold
                if votes['buy'] >= required_buy and votes['buy'] > votes['sell']:
                    decision = 'buy'
                    consensus_from_qgate = True
                elif votes['sell'] >= required_sell and votes['sell'] > votes['buy']:
                    decision = 'sell'
                    consensus_from_qgate = True
                else:
                    decision = 'hold'
        except Exception:
            pass
        # --- Server-side Q-gate ---
        try:
            # Если решение уже получено через консенсус моделей, прошедших Q‑gate, не блокируем финальным агрегатом
            if (not consensus_from_qgate) and (not is_xgb_trade):
                q_values = pred_json.get('q_values')
                if not isinstance(q_values, list):
                    preds_list_tmp = pred_json.get('predictions') or []
                    if preds_list_tmp:
                        q_values = preds_list_tmp[0].get('q_values')
                if isinstance(q_values, list) and len(q_values) >= 2:
                    q_sorted = sorted([float(x) for x in q_values], reverse=True)
                    maxQ = q_sorted[0]
                    gapQ = q_sorted[0] - q_sorted[1]
                    pass_max = True if float(eff_T1) == 0.0 else (maxQ >= eff_T1)
                    pass_gap = True if float(eff_T2) == 0.0 else (gapQ >= eff_T2)
                    passed = pass_max and pass_gap
                    if not passed:
                        decision = 'hold'
                    try:
                        print(f"Q‑gate: {'PASS' if passed else 'BLOCK'} (maxQ={maxQ:.3f}, gapQ={gapQ:.3f}, T1={eff_T1:.3f}, T2={eff_T2:.3f})")
                    except Exception:
                        pass
        except Exception:
            pass

        # --- Opposite Signal Max Gate ---
        try:
            if is_xgb_trade and decision in ('buy', 'sell'):
                long_sig, short_sig = _extract_xgb_long_short_signals(pred_json)
                if decision == 'buy':
                    opp_max = get_runtime_value(rc, session_id, 'xgb_short_signal_max_for_long_entry')
                    if opp_max not in (None, '') and short_sig is not None:
                        if float(short_sig) > float(opp_max):
                            decision = 'hold'
                            try:
                                pred_json['opposite_signal_gate'] = {'blocked': True, 'reason': 'short_signal_too_strong', 'short_sig': float(short_sig), 'max': float(opp_max)}
                            except Exception:
                                pass
                elif decision == 'sell':
                    opp_max = get_runtime_value(rc, session_id, 'xgb_long_signal_max_for_short_entry')
                    if opp_max not in (None, '') and long_sig is not None:
                        if float(long_sig) > float(opp_max):
                            decision = 'hold'
                            try:
                                pred_json['opposite_signal_gate'] = {'blocked': True, 'reason': 'long_signal_too_strong', 'long_sig': float(long_sig), 'max': float(opp_max)}
                            except Exception:
                                pass
        except Exception:
            pass

        # --- Conflict-veto gate (window disagreement) ---
        # Если 2 коротких окна согласны и противоположны длинному — блокируем вход (decision -> HOLD),
        # но только если это включено в trading:regime_config.
        try:
            cv_enabled = False
            cv_candidate = False
            cv_long_window = None
            cv_long_label = None
            cv_short_label = None
            if isinstance(market_regime_details, dict):
                cv_enabled = bool(market_regime_details.get('conflict_veto_enabled'))
                cv_candidate = bool(market_regime_details.get('conflict_veto_candidate'))
                cv_long_window = market_regime_details.get('long_window')
                cv_long_label = market_regime_details.get('long_label')
                cv_short_label = market_regime_details.get('short_label')
            if cv_enabled:
                try:
                    pred_json['conflict_veto_gate'] = {
                        'enabled': True,
                        'candidate': bool(cv_candidate),
                        'long_window': cv_long_window,
                        'long_label': cv_long_label,
                        'short_label': cv_short_label,
                    }
                except Exception:
                    pass
            if cv_enabled and cv_candidate and decision in ('buy', 'sell'):
                try:
                    pred_json['decision_before_conflict_veto'] = str(decision)
                except Exception:
                    pass
                decision = 'hold'
                try:
                    logger.warning("[conflict_veto] symbol=%s blocked %s due to short=%s vs long=%s (window=%s)",
                                   symbol, pred_json.get('decision_before_conflict_veto'), cv_short_label, cv_long_label, cv_long_window)
                except Exception:
                    pass
        except Exception:
            pass

        # --- Trend-veto gate (block trading against regime) ---
        # Если включено: downtrend блокирует BUY, uptrend блокирует SELL (decision -> HOLD).
        # В long-short режиме направление УЖЕ следует тренду, поэтому trend-veto пропускаем
        # (иначе он заблокирует cover/exit, т.к. после ремапа cover='buy' в downtrend).
        _short_dir = (str(trade_mode).strip() == 'long-short' and ls_target_role == 'short')
        try:
            tv_enabled = bool(market_regime_details.get('trend_veto_enabled')) if isinstance(market_regime_details, dict) else False
            if tv_enabled and str(trade_mode).strip() != 'long-short':
                try:
                    pred_json['trend_veto_gate'] = {'enabled': True, 'market_regime': str(market_regime)}
                except Exception:
                    pass
                if market_regime == 'downtrend' and decision == 'buy':
                    try:
                        pred_json['decision_before_trend_veto'] = str(decision)
                    except Exception:
                        pass
                    decision = 'hold'
                elif market_regime == 'uptrend' and decision == 'sell':
                    try:
                        pred_json['decision_before_trend_veto'] = str(decision)
                    except Exception:
                        pass
                    decision = 'hold'
            elif tv_enabled:
                try:
                    pred_json['trend_veto_gate'] = {'enabled': True, 'skipped_long_short': True, 'market_regime': str(market_regime)}
                except Exception:
                    pass
        except Exception:
            pass

        # --- MarketState gate (OOD-safety) ---
        # Если рынок не NORMAL (HIGH_VOL/PANIC/DRAWDOWN) — блокируем ENTRY (override -> HOLD).
        # Для long: entry='buy', для short (после ремапа): entry='sell'.
        # ВАЖНО: делаем это здесь (в оркестраторе), т.к. только тут есть OHLCV df_5m.
        _entry_signal = 'sell' if _short_dir else 'buy'
        try:
            if decision == _entry_signal and df_5m is not None and (not getattr(df_5m, 'empty', False)):
                try:
                    from envs.dqn_model.gym.gutils_optimized import compute_market_state, MarketState
                except Exception:
                    compute_market_state = None
                    MarketState = None

                if compute_market_state is not None and MarketState is not None:
                    df_np = None
                    try:
                        df_np = df_5m[['open', 'high', 'low', 'close', 'volume']].astype('float32').to_numpy()
                    except Exception:
                        df_np = None

                    if df_np is not None and hasattr(df_np, 'shape') and int(df_np.shape[0]) >= 3:
                        st = compute_market_state(int(df_np.shape[0]) - 1, df_np, roi_buf=None, vol_buf=None, trend_regime=0, atr_rel=None)
                        st_name = str(getattr(st, 'name', st))
                        # Запишем для UI/БД в market_conditions
                        try:
                            pred_json['market_state'] = st_name
                        except Exception:
                            pass
                        # Явно пишем статус гейта: blocked True/False (чтобы UI показывал и "разрешено")
                        try:
                            pred_json['market_state_gate'] = {'blocked': (st != MarketState.NORMAL), 'state': st_name}
                        except Exception:
                            pass
                        if st != MarketState.NORMAL:
                            try:
                                pred_json['decision_before_market_state_gate'] = str(decision)
                            except Exception:
                                pass
                            decision = 'hold'
                            try:
                                logger.warning("[market_state_gate] symbol=%s blocked BUY due to state=%s", symbol, st_name)
                            except Exception:
                                pass
        except Exception:
            pass

        # --- Peak-extension gate (block late BUY near local peak) ---
        # Идея: если модель дала BUY, но цена уже "расширилась" (сильный рост) и находится около локального high,
        # то мы чаще попадаем на разворот/откат у сопротивления. Поэтому режем ENTRY -> HOLD.
        # ВАЖНО: применяем только для long-entry (BUY). Для short-entry это не тот паттерн.
        try:
            if decision == _entry_signal and _entry_signal == 'buy' and df_5m is not None and (not getattr(df_5m, 'empty', False)):
                # defaults (достаточно консервативные)
                peak_enabled = True
                peak_lookback = 96          # 96*5m = 8h
                peak_ret_thr = 0.03         # +3% за lookback
                peak_near_high_thr = 0.997  # в пределах ~0.3% от локального high

                def _truthy(v) -> bool:
                    try:
                        return str(v).strip().lower() in ('1', 'true', 'yes', 'on')
                    except Exception:
                        return False

                # 1) overrides from Postgres (app_settings): scope='trading', group='regime'
                try:
                    from utils.settings_store import get_setting_value as _get_setting_value
                    v0 = _get_setting_value('trading', 'regime', 'PEAK_EXTENSION_VETO')
                    if v0 is not None:
                        peak_enabled = _truthy(v0)
                    v1 = _get_setting_value('trading', 'regime', 'PEAK_EXTENSION_LOOKBACK')
                    if v1 is not None and str(v1).strip() != '':
                        peak_lookback = int(float(v1))
                    v2 = _get_setting_value('trading', 'regime', 'PEAK_EXTENSION_RET_THR')
                    if v2 is not None and str(v2).strip() != '':
                        peak_ret_thr = float(v2)
                    v3 = _get_setting_value('trading', 'regime', 'PEAK_EXTENSION_NEAR_HIGH_THR')
                    if v3 is not None and str(v3).strip() != '':
                        peak_near_high_thr = float(v3)
                except Exception:
                    pass

                # 2) fallback to Redis trading:regime_config (if DB not configured)
                try:
                    cfg = None
                    if rc is not None:
                        raw = rc.get('trading:regime_config')
                        if raw:
                            cfg = json.loads(raw)
                    if isinstance(cfg, dict):
                        if 'peak_extension_veto' in cfg:
                            peak_enabled = bool(cfg.get('peak_extension_veto') is True or _truthy(cfg.get('peak_extension_veto')))
                        if 'peak_extension_lookback' in cfg and str(cfg.get('peak_extension_lookback') or '').strip() != '':
                            peak_lookback = int(float(cfg.get('peak_extension_lookback')))
                        if 'peak_extension_ret_thr' in cfg and str(cfg.get('peak_extension_ret_thr') or '').strip() != '':
                            peak_ret_thr = float(cfg.get('peak_extension_ret_thr'))
                        if 'peak_extension_near_high_thr' in cfg and str(cfg.get('peak_extension_near_high_thr') or '').strip() != '':
                            peak_near_high_thr = float(cfg.get('peak_extension_near_high_thr'))
                except Exception:
                    pass

                # Sanity bounds
                try:
                    peak_lookback = int(max(12, min(2000, int(peak_lookback))))
                except Exception:
                    peak_lookback = 96
                try:
                    peak_ret_thr = float(max(0.0, min(1.0, float(peak_ret_thr))))
                except Exception:
                    peak_ret_thr = 0.03
                try:
                    peak_near_high_thr = float(max(0.90, min(1.0, float(peak_near_high_thr))))
                except Exception:
                    peak_near_high_thr = 0.997

                if peak_enabled:
                    df_np = None
                    try:
                        df_np = df_5m[['open', 'high', 'low', 'close', 'volume']].astype('float32').to_numpy()
                    except Exception:
                        df_np = None
                    if df_np is not None and hasattr(df_np, 'shape') and int(df_np.shape[0]) >= peak_lookback + 1:
                        close_now = float(df_np[-1, 3])
                        close_lb = float(df_np[-peak_lookback, 3])
                        roll_high = float(np.max(df_np[-peak_lookback:, 1]))
                        ret = (close_now - close_lb) / max(close_lb, 1e-9)
                        near_high = (close_now >= roll_high * float(peak_near_high_thr))
                        extended = (ret >= float(peak_ret_thr))
                        if extended and near_high:
                            try:
                                pred_json['decision_before_peak_extension_gate'] = str(decision)
                            except Exception:
                                pass
                            try:
                                pred_json['peak_extension_gate'] = {
                                    'blocked': True,
                                    'reason': 'high_peak',
                                    'lookback': int(peak_lookback),
                                    'ret': float(ret),
                                    'ret_thr': float(peak_ret_thr),
                                    'near_high': bool(near_high),
                                    'near_high_thr': float(peak_near_high_thr),
                                    'close': float(close_now),
                                    'roll_high': float(roll_high),
                                    'dist_to_high_pct': float((roll_high - close_now) / max(close_now, 1e-9)),
                                }
                            except Exception:
                                pass
                            decision = 'hold'
                            try:
                                logger.warning("[peak_extension_gate] symbol=%s blocked BUY: ret=%.4f lookback=%s close=%.6f high=%.6f",
                                               symbol, float(ret), int(peak_lookback), float(close_now), float(roll_high))
                            except Exception:
                                pass
                        else:
                            # Запишем "не заблокировано", чтобы UI мог показать что гейт активен (по желанию)
                            try:
                                pred_json['peak_extension_gate'] = {
                                    'blocked': False,
                                    'enabled': True,
                                    'lookback': int(peak_lookback),
                                    'ret': float(ret),
                                    'ret_thr': float(peak_ret_thr),
                                    'near_high': bool(near_high),
                                    'near_high_thr': float(peak_near_high_thr),
                                }
                            except Exception:
                                pass
                else:
                    try:
                        pred_json['peak_extension_gate'] = {'blocked': False, 'enabled': False}
                    except Exception:
                        pass
        except Exception:
            pass

        # --- Dip-extension gate (block late SHORT near local bottom) ---
        # Идея: если решение = entry в шорт (SELL), но цена уже сильно упала и находится около локального low,
        # то вход в шорт часто "в дно" (pullback/mean-reversion). Поэтому режем ENTRY -> HOLD.
        # ВАЖНО: применяем только для short-entry (SELL).
        try:
            if decision == _entry_signal and _entry_signal == 'sell' and df_5m is not None and (not getattr(df_5m, 'empty', False)):
                # defaults (консервативные)
                dip_enabled = True
                dip_lookback = 96           # 96*5m = 8h
                dip_ret_thr = 0.03          # 3% падение за lookback (используем модуль)
                dip_near_low_thr = 1.003    # в пределах ~0.3% от локального low (close <= low*thr)

                def _truthy(v) -> bool:
                    try:
                        return str(v).strip().lower() in ('1', 'true', 'yes', 'on')
                    except Exception:
                        return False

                # 1) overrides from Postgres (app_settings): scope='trading', group='regime'
                try:
                    from utils.settings_store import get_setting_value as _get_setting_value
                    v0 = _get_setting_value('trading', 'regime', 'DIP_EXTENSION_VETO')
                    if v0 is not None:
                        dip_enabled = _truthy(v0)
                    v1 = _get_setting_value('trading', 'regime', 'DIP_EXTENSION_LOOKBACK')
                    if v1 is not None and str(v1).strip() != '':
                        dip_lookback = int(float(v1))
                    v2 = _get_setting_value('trading', 'regime', 'DIP_EXTENSION_RET_THR')
                    if v2 is not None and str(v2).strip() != '':
                        dip_ret_thr = float(v2)
                    v3 = _get_setting_value('trading', 'regime', 'DIP_EXTENSION_NEAR_LOW_THR')
                    if v3 is not None and str(v3).strip() != '':
                        dip_near_low_thr = float(v3)
                except Exception:
                    pass

                # 2) fallback to Redis trading:regime_config (if DB not configured)
                try:
                    cfg = None
                    if rc is not None:
                        raw = rc.get('trading:regime_config')
                        if raw:
                            cfg = json.loads(raw)
                    if isinstance(cfg, dict):
                        if 'dip_extension_veto' in cfg:
                            dip_enabled = bool(cfg.get('dip_extension_veto') is True or _truthy(cfg.get('dip_extension_veto')))
                        if 'dip_extension_lookback' in cfg and str(cfg.get('dip_extension_lookback') or '').strip() != '':
                            dip_lookback = int(float(cfg.get('dip_extension_lookback')))
                        if 'dip_extension_ret_thr' in cfg and str(cfg.get('dip_extension_ret_thr') or '').strip() != '':
                            dip_ret_thr = float(cfg.get('dip_extension_ret_thr'))
                        if 'dip_extension_near_low_thr' in cfg and str(cfg.get('dip_extension_near_low_thr') or '').strip() != '':
                            dip_near_low_thr = float(cfg.get('dip_extension_near_low_thr'))
                except Exception:
                    pass

                # Sanity bounds
                try:
                    dip_lookback = int(max(12, min(2000, int(dip_lookback))))
                except Exception:
                    dip_lookback = 96
                try:
                    dip_ret_thr = float(max(0.0, min(1.0, float(dip_ret_thr))))
                except Exception:
                    dip_ret_thr = 0.03
                try:
                    dip_near_low_thr = float(max(1.0, min(1.50, float(dip_near_low_thr))))
                except Exception:
                    dip_near_low_thr = 1.003

                if dip_enabled:
                    df_np = None
                    try:
                        df_np = df_5m[['open', 'high', 'low', 'close', 'volume']].astype('float32').to_numpy()
                    except Exception:
                        df_np = None
                    if df_np is not None and hasattr(df_np, 'shape') and int(df_np.shape[0]) >= dip_lookback + 1:
                        close_now = float(df_np[-1, 3])
                        close_lb = float(df_np[-dip_lookback, 3])
                        roll_low = float(np.min(df_np[-dip_lookback:, 2]))
                        ret = (close_now - close_lb) / max(close_lb, 1e-9)  # отрицательное при падении
                        near_low = (close_now <= roll_low * float(dip_near_low_thr))
                        extended = (ret <= -float(dip_ret_thr))
                        if extended and near_low:
                            try:
                                pred_json['decision_before_dip_extension_gate'] = str(decision)
                            except Exception:
                                pass
                            try:
                                pred_json['dip_extension_gate'] = {
                                    'blocked': True,
                                    'reason': 'low_valley',
                                    'lookback': int(dip_lookback),
                                    'ret': float(ret),
                                    'ret_thr': float(dip_ret_thr),
                                    'near_low': bool(near_low),
                                    'near_low_thr': float(dip_near_low_thr),
                                    'close': float(close_now),
                                    'roll_low': float(roll_low),
                                    'dist_to_low_pct': float((close_now - roll_low) / max(close_now, 1e-9)),
                                }
                            except Exception:
                                pass
                            decision = 'hold'
                            try:
                                logger.warning("[dip_extension_gate] symbol=%s blocked SELL: ret=%.4f lookback=%s close=%.6f low=%.6f",
                                               symbol, float(ret), int(dip_lookback), float(close_now), float(roll_low))
                            except Exception:
                                pass
                        else:
                            try:
                                pred_json['dip_extension_gate'] = {
                                    'blocked': False,
                                    'enabled': True,
                                    'lookback': int(dip_lookback),
                                    'ret': float(ret),
                                    'ret_thr': float(dip_ret_thr),
                                    'near_low': bool(near_low),
                                    'near_low_thr': float(dip_near_low_thr),
                                }
                            except Exception:
                                pass
                else:
                    try:
                        pred_json['dip_extension_gate'] = {'blocked': False, 'enabled': False}
                    except Exception:
                        pass
        except Exception:
            pass

        # --- Post-exit cooldown gate (after TP/SL) ---
        # Если недавно был выход по TP/SL (см. pred_json['postexit']) — блокируем ENTRY на N свечей.
        # Для long: entry='buy', для short (после ремапа): entry='sell'.
        try:
            pe = pred_json.get('postexit') if isinstance(pred_json, dict) else None
            if decision == _entry_signal and isinstance(pe, dict):
                candles_since = int(pe.get('candles_since_exit')) if pe.get('candles_since_exit') is not None else None
                cooldown = int(pe.get('cooldown_candles')) if pe.get('cooldown_candles') is not None else None
                reason = str(pe.get('exit_reason') or '').strip().lower()
                reason_allowed = reason in ('stop_loss', 'take_profit', 'trailing')
                if is_xgb_trade:
                    reason_allowed = reason_allowed or reason in (
                        'timeout', 'weak_signal_avg', 'manual', 'signal', 'holdstep', 'unknown',
                    )
                if candles_since is not None and cooldown is not None and candles_since < cooldown and reason_allowed:
                    _sl_streak_val = int(pe.get('sl_streak') or 0)
                    _hours_left = round((cooldown - candles_since) * 5 / 60, 1)
                    pred_json['decision_before_postexit_cooldown'] = str(decision)
                    pred_json['postexit_cooldown_gate'] = {
                        'blocked': True,
                        'candles_left': int(cooldown - candles_since),
                        'hours_left': _hours_left,
                        'cooldown_candles': int(cooldown),
                        'reason': reason,
                        'sl_streak': _sl_streak_val,
                    }
                    decision = 'hold'
                    try:
                        logger.warning("[postexit_cooldown] symbol=%s blocked BUY reason=%s candles_since=%s cooldown=%s streak=%s hours_left=%s",
                                       symbol, reason, candles_since, cooldown, _sl_streak_val, _hours_left)
                    except Exception:
                        pass
        except Exception:
            pass

        # --- XGB post-exit boosted threshold gate ---
        # После cooldown вход не запрещаем полностью, но требуем signal выше threshold на заданный процент.
        def _apply_xgb_postexit_cooldown_gate(current_decision: str, stage: str = "runtime") -> str:
            try:
                pe = pred_json.get('postexit') if isinstance(pred_json, dict) else None
                if not (is_xgb_trade and current_decision == _entry_signal and isinstance(pe, dict)):
                    return current_decision
                candles_since = int(pe.get('candles_since_exit')) if pe.get('candles_since_exit') is not None else None
                cooldown = int(pe.get('cooldown_candles')) if pe.get('cooldown_candles') is not None else None
                reason = str(pe.get('exit_reason') or '').strip().lower()
                if candles_since is None or cooldown is None or candles_since >= cooldown:
                    return current_decision
                pred_json.setdefault('decision_before_postexit_cooldown', str(current_decision))
                pred_json[f'postexit_cooldown_gate_{stage}'] = {
                    'blocked': True,
                    'candles_left': int(cooldown - candles_since),
                    'cooldown_candles': int(cooldown),
                    'candles_since_exit': int(candles_since),
                    'reason': reason,
                }
                pred_json['decision'] = 'hold'
                for _p in (pred_json.get('predictions') or []):
                    if isinstance(_p, dict):
                        _p['action'] = 'hold'
                logger.warning(
                    "[postexit_cooldown_final] session=%s symbol=%s blocked entry stage=%s reason=%s candles_since=%s cooldown=%s",
                    session_id, symbol, stage, reason, candles_since, cooldown
                )
                return 'hold'
            except Exception:
                return current_decision

        def _apply_xgb_postexit_threshold_gate(current_decision: str) -> str:
            try:
                pe = pred_json.get('postexit') if isinstance(pred_json, dict) else None
                if not (is_xgb_trade and current_decision == _entry_signal and isinstance(pe, dict)):
                    return current_decision
                phase = str(pe.get('phase') or '').strip().lower()
                boost_pct = float(pe.get('xgb_threshold_boost_pct') or 0.0)
                if phase != 'boost' or boost_pct <= 0:
                    return current_decision
                snapshot = xgb_signal_snapshot
                if not isinstance(snapshot, dict):
                    snapshot = _extract_xgb_signal_snapshot(
                        pred_json,
                        model_paths=[str(p) for p in (model_paths or []) if p],
                        threshold_override=threshold_override,
                    )
                signal_value = float(snapshot.get('signal')) if isinstance(snapshot, dict) and snapshot.get('signal') is not None else None
                base_threshold = float(snapshot.get('threshold')) if isinstance(snapshot, dict) and snapshot.get('threshold') is not None else None
                if signal_value is None or base_threshold is None:
                    gate_doc = {'blocked': True, 'reason': 'missing_signal_or_threshold', 'boost_pct': float(boost_pct)}
                else:
                    boosted_threshold = base_threshold * (1.0 + boost_pct / 100.0)
                    pe['xgb_signal'] = float(signal_value)
                    pe['xgb_base_threshold'] = float(base_threshold)
                    pe['xgb_boosted_threshold'] = float(boosted_threshold)
                    gate_doc = {
                        'blocked': bool(signal_value < boosted_threshold),
                        'signal': float(signal_value),
                        'base_threshold': float(base_threshold),
                        'boosted_threshold': float(boosted_threshold),
                        'boost_pct': float(boost_pct),
                    }
                pred_json['xgb_postexit_threshold_gate'] = gate_doc
                if not gate_doc.get('blocked'):
                    return current_decision
                pred_json.setdefault('decision_before_xgb_postexit_threshold', str(current_decision))
                pred_json['decision'] = 'hold'
                for _p in (pred_json.get('predictions') or []):
                    if isinstance(_p, dict):
                        _p['action'] = 'hold'
                return 'hold'
            except Exception:
                return current_decision

        decision = _apply_xgb_postexit_cooldown_gate(decision, "pre_agent")
        decision = _apply_xgb_postexit_threshold_gate(decision)

        # 4) Торговля через TradingAgent (без docker exec)
        # Направление из Redis per-symbol
        try:
            _dir = str(sget('direction', session_doc.get('direction') or 'long')).strip().lower() or 'long'
        except Exception:
            _dir = 'long'
        # long-short: направление выбираем по market_regime (uptrend->long, downtrend->short)
        try:
            ignore_trend_filter = str(sget('ignore_trend_filter', session_doc.get('ignore_trend_filter'))).strip().lower() in ('1', 'true', 'yes', 'on')
            if (str(trade_mode).strip() == 'long-short'):
                if ignore_trend_filter:
                    _dir = 'long' if decision == 'buy' else 'short' if decision == 'sell' else _dir
                else:
                    if market_regime == 'uptrend':
                        _dir = 'long'
                    elif market_regime == 'downtrend':
                        _dir = 'short'
                    else: # flat
                        _dir = 'long' if decision == 'buy' else 'short' if decision == 'sell' else _dir
        except Exception:
            pass
        agent = TradingAgent(
            model_path=(model_paths[0] if model_paths else None),
            direction=_dir,
            symbol=symbol,
            session_id=session_id,
            account_id=account_id,
        )
        agent.symbols = syms
        agent.symbol = symbol
        agent.base_symbol = symbol
        try:
            agent.trade_amount = agent._calculate_trade_amount()
        except Exception:
            agent.trade_amount = getattr(agent, 'trade_amount', 0.0)

        # Отметим, что торговый цикл активен (для UI)
        try:
            agent.is_trading = True
        except Exception:
            pass

        # Проставим последнее предсказание для UI
        try:
            agent.last_model_prediction = decision
        except Exception:
            pass

        current_status_before = agent.get_trading_status()

        # Если уже в позиции и свободной USDT мало — отменим лишние входящие лимитные заявки (кроме SL/TP)
        try:
            if getattr(agent, 'current_position', None):
                free_usdt = 0.0
                try:
                    bal = agent._get_current_balance()
                    free_usdt = float(bal) if isinstance(bal, (int, float)) else 0.0
                except Exception:
                    free_usdt = 0.0
                min_cost = 10.0
                try:
                    limits = agent._get_bybit_limits()
                    if isinstance(limits, dict) and limits.get('min_cost'):
                        min_cost = max(min_cost, float(limits.get('min_cost')))
                except Exception:
                    pass
                if free_usdt < min_cost and hasattr(agent, 'exchange') and agent.exchange:
                    try:
                        open_orders = agent.exchange.fetch_open_orders(symbol)
                        for od in (open_orders or []):
                            try:
                                side = str(od.get('side') or '').lower()
                                params = od.get('info') or {}
                                reduce_only = bool(params.get('reduceOnly') or params.get('reduce_only') or False)
                                if side == 'buy' and not reduce_only:
                                    try:
                                        agent.exchange.cancel_order(od.get('id'), symbol)
                                    except Exception:
                                        pass
                            except Exception:
                                continue
                    except Exception:
                        pass
        except Exception:
            pass

        # 4.1) Сохраняем предсказания в БД (по каждому пути модели) + сводка консенсуса и per‑model Q‑gate
        try:
            # Текущая цена: возьмём close последней закрытой свечи
            try:
                current_price = float(df_5m['close'].iloc[-1]) if (df_5m is not None and not df_5m.empty) else None
            except Exception:
                current_price = None
            position_status = 'open' if getattr(agent, 'current_position', None) else 'none'
            preds_list = pred_json.get('predictions') or []
            is_xgb_trade = any('/models/xgb/' in str(p).replace('\\', '/') for p in (model_paths or []))
            try:
                if is_xgb_trade and rc is not None:
                    signal_exit_window_cfg = XGB_SIGNAL_EXIT_WINDOW_DEFAULT
                    raw_window = sget('xgb_signal_exit_window', session_doc.get('xgb_signal_exit_window'))
                    if raw_window not in (None, ''):
                        signal_exit_window_cfg = max(1, int(float(raw_window)))
                    signal_snapshot = xgb_signal_snapshot
                    if not isinstance(signal_snapshot, dict):
                        signal_snapshot = _extract_xgb_signal_snapshot(
                            pred_json,
                            model_paths=[str(p) for p in (model_paths or []) if p],
                            threshold_override=threshold_override,
                        )
                    if isinstance(signal_snapshot, dict):
                        signal_snapshot = dict(signal_snapshot)
                        signal_snapshot['ts_ms'] = int(_last_closed_ts_ms() or 0)
                        _append_xgb_signal_history(rc, session_id, signal_snapshot, signal_exit_window_cfg)
                        logger.warning(
                            "[xgb_signal_exit] appended session=%s signal=%.6f threshold=%.6f window=%s",
                            session_id,
                            float(signal_snapshot.get('signal')),
                            float(signal_snapshot.get('threshold')),
                            signal_exit_window_cfg,
                        )
            except Exception:
                logger.exception("[xgb_signal_exit] failed to append from prediction save session=%s", session_id)
            # Общий ID группы ансамбля для этого тика, чтобы фронт мог объединять карточки
            ensemble_group_id = str(uuid.uuid4()) if preds_list else None
            entry_shap_snapshot = None
            # Сохраним текущую группу в Redis, чтобы DDD-исполнение могло дописать exec_error в эти же карточки
            try:
                if rc is not None and ensemble_group_id:
                    rc.setex(session_runtime_key(session_id, "last_ensemble_group_id"), 900, str(ensemble_group_id))
            except Exception:
                pass
            # 4.1.a) GigaChat (LLM) — добавляем текст ответа в market_conditions каждой карточки предсказания
            gigachat_mc = None
            try:
                from utils.gigachat_client import GigaChatClient, load_gigachat_config
                from utils.gigachat_features import build_gigachat_prompt, build_semantic_snapshot

                cfg_llm = load_gigachat_config()
                run_llm = bool(cfg_llm.enabled)
                try:
                    if bool(cfg_llm.only_on_buy) and str(decision or "").lower() != _entry_signal:
                        run_llm = False
                except Exception:
                    pass

                if run_llm:
                    snap = build_semantic_snapshot(
                        df_5m,
                        symbol=str(symbol),
                        market_regime=str(market_regime),
                        market_regime_details=market_regime_details if isinstance(market_regime_details, dict) else None,
                        decision=str(decision),
                        votes=votes if isinstance(votes, dict) else None,
                    )
                    prompt = build_gigachat_prompt(snap)
                    res = GigaChatClient().chat(prompt=prompt, cfg=cfg_llm)
                    txt = (res.get("text") if isinstance(res, dict) else None) or ""
                    parsed = (res.get("parsed") if isinstance(res, dict) else None) or {}
                    err = (res.get("error") if isinstance(res, dict) else None)
                    lat = (res.get("latency_ms") if isinstance(res, dict) else None)
                    gigachat_mc = {
                        "gigachat_ok": bool(res.get("ok")) if isinstance(res, dict) else False,
                        "gigachat_latency_ms": int(lat) if isinstance(lat, (int, float)) else None,
                        "gigachat_decision": (parsed.get("decision") if isinstance(parsed, dict) else None),
                        "gigachat_confidence": (parsed.get("confidence") if isinstance(parsed, dict) else None),
                        "gigachat_reason": (parsed.get("reason") if isinstance(parsed, dict) else None),
                        "gigachat_risk": (parsed.get("risk") if isinstance(parsed, dict) else None),
                        "gigachat_text": str(txt)[:4000],  # keep DB/UI reasonable
                        "gigachat_error": str(err)[:400] if err else None,
                    }
                    # Для выдачи наружу (в ответах задач/логах)
                    try:
                        if isinstance(pred_json, dict):
                            pred_json["gigachat"] = {
                                "ok": gigachat_mc.get("gigachat_ok"),
                                "latency_ms": gigachat_mc.get("gigachat_latency_ms"),
                                "decision": gigachat_mc.get("gigachat_decision"),
                                "confidence": gigachat_mc.get("gigachat_confidence"),
                                "reason": gigachat_mc.get("gigachat_reason"),
                                "text": gigachat_mc.get("gigachat_text"),
                                "error": gigachat_mc.get("gigachat_error"),
                            }
                    except Exception:
                        pass
            except Exception:
                gigachat_mc = None
            for p in preds_list:
                try:
                    mp = p.get('model_path')
                    act = p.get('action')
                    qv = p.get('q_values') or []
                    # Per‑model q‑gate метрики
                    q_max = None; q_gap = None; q_pass = None
                    try:
                        if isinstance(qv, list) and len(qv) >= 3:
                            if str(act or 'hold').lower() == 'buy':
                                qb = float(qv[1]); other = max(float(qv[0]), float(qv[2]))
                                q_max = qb; q_gap = qb - other
                                _pass_max = True if float(T1) == 0.0 else (qb >= T1)
                                _pass_gap = True if float(T2) == 0.0 else (q_gap >= T2)
                                q_pass = _pass_max and _pass_gap
                            elif str(act or 'hold').lower() == 'sell':
                                qs = float(qv[2]); other = max(float(qv[0]), float(qv[1]))
                                q_max = qs; q_gap = qs - other
                                _pass_max = True if float(T1) == 0.0 else (qs >= T1)
                                _pass_gap = True if float(T2) == 0.0 else (q_gap >= T2)
                                q_pass = _pass_max and _pass_gap
                            else:
                                q_pass = True
                    except Exception:
                        q_pass = (str(act or 'hold').lower() == 'hold')
                    if is_xgb_trade:
                        q_max = None
                        q_gap = None
                        q_pass = True
                    mc = {
                        'session_id': session_id,
                        'account_id': account_id,
                        'ensemble_total': int(total_sel),
                        'ensemble_votes_buy': int(votes.get('buy', 0)),
                        'ensemble_votes_sell': int(votes.get('sell', 0)),
                        'ensemble_votes_hold': int(votes.get('hold', 0)),
                        'ensemble_required': int(required) if required is not None else None,
                        'ensemble_required_type': required_type,
                        'ensemble_regime': market_regime,
                        'ensemble_decision': decision,
                        'ensemble_group_id': ensemble_group_id,
                        # long-short diagnostics for UI
                        'trade_mode': str(trade_mode) if trade_mode else None,
                        'trade_direction': (str(_dir) if _dir else None),
                        'active_role': (str(ls_target_role) if (str(trade_mode).strip() == 'long-short' and ls_target_role) else None),
                        'models_before_role_filter': (len(ls_models_before) if isinstance(ls_models_before, list) else None),
                        'models_after_role_filter': (len(ls_models_after) if isinstance(ls_models_after, list) else (len(model_paths) if isinstance(model_paths, list) else None)),
                        # Детали режима по окнам, чтобы UI мог показать 60=F,180=U,300=D
                        'regime_windows': list(market_regime_details.get('windows', [])) if isinstance(market_regime_details, dict) else [],
                        'regime_labels': list(market_regime_details.get('labels', [])) if isinstance(market_regime_details, dict) else [],
                        'regime_weights': list(market_regime_details.get('weights', [])) if isinstance(market_regime_details, dict) else [],
                        'regime_voting': (market_regime_details.get('voting') if isinstance(market_regime_details, dict) else None),
                        'regime_tie_break': (market_regime_details.get('tie_break') if isinstance(market_regime_details, dict) else None),
                        'qgate_T1': (None if is_xgb_trade else float(T1)),
                        'qgate_T2': (None if is_xgb_trade else float(T2)),
                        'qgate_maxQ': (None if is_xgb_trade else (float(q_max) if q_max is not None else None)),
                        'qgate_gapQ': (None if is_xgb_trade else (float(q_gap) if q_gap is not None else None)),
                        'qgate_filtered': (False if is_xgb_trade else (False if q_pass is None else (not q_pass))),
                        'xgb_qgate_disabled': bool(is_xgb_trade),
                    }
                    try:
                        shap_doc = p.get('xgb_shap') if isinstance(p, dict) else None
                        if (
                            is_xgb_trade
                            and isinstance(shap_doc, dict)
                            and str(decision or '').lower() in ('buy', 'sell')
                            and str(act or '').lower() == str(decision or '').lower()
                        ):
                            mc['xgb_shap'] = shap_doc
                            mc['entered_trade_candidate'] = True
                            if entry_shap_snapshot is None:
                                entry_shap_snapshot = {
                                    'model_path': str(mp) if mp is not None else '',
                                    'action': str(act or ''),
                                    'confidence': float(max(qv)) if isinstance(qv, list) and qv else None,
                                    'current_price': current_price,
                                    'xgb_shap': shap_doc,
                                    'ensemble_group_id': ensemble_group_id,
                                }
                    except Exception:
                        pass
                    # Добавим результат GigaChat в каждую карточку
                    try:
                        if isinstance(gigachat_mc, dict) and gigachat_mc:
                            for k, v in gigachat_mc.items():
                                mc[k] = v
                    except Exception:
                        pass
                    # Роль конкретной модели (long/short), если задано в режиме long-short
                    try:
                        mp_key = str(mp).replace('\\', '/')
                        mc['model_role'] = model_roles_map.get(mp_key)
                    except Exception:
                        pass
                    try:
                        if is_xgb_trade and isinstance(p, dict):
                            if p.get('task') is not None:
                                mc['xgb_task'] = p.get('task')
                            if p.get('direction') is not None:
                                mc['xgb_direction'] = p.get('direction')
                            if p.get('xgb_runtime_threshold') is not None:
                                mc['xgb_runtime_threshold'] = p.get('xgb_runtime_threshold')
                    except Exception:
                        pass
                    # Прокинем причины runtime-гейтов в market_conditions, чтобы UI (/agent/<symbol>) их показывал.
                    # ВАЖНО: UI читает именно ModelPrediction.market_conditions, а не pred_json напрямую.
                    try:
                        if isinstance(pred_json, dict):
                            for k in (
                                'market_state',
                                'decision_before_market_state_gate',
                                'market_state_gate',
                                'decision_before_peak_extension_gate',
                                'peak_extension_gate',
                                'decision_before_dip_extension_gate',
                                'dip_extension_gate',
                                'postexit',
                                'decision_before_postexit_cooldown',
                                'postexit_cooldown_gate',
                                'decision_before_xgb_postexit_threshold',
                                'xgb_postexit_threshold_gate',
                            ):
                                if k in pred_json:
                                    mc[k] = pred_json.get(k)
                    except Exception:
                        pass
                    create_model_prediction(
                        symbol=symbol,
                        action=str(act or 'hold'),
                        q_values=list(qv) if isinstance(qv, (list, tuple)) else [],
                        current_price=current_price,
                        position_status=position_status,
                        model_path=str(mp) if mp is not None else '' ,
                        market_conditions=mc
                    )
                except Exception:
                    # Не ломаем торговый цикл из-за БД
                    pass
            if is_xgb_trade and not preds_list:
                try:
                    fallback_model_path = None
                    if isinstance(model_paths, list) and model_paths:
                        fallback_model_path = str(model_paths[0])
                    fallback_mc = {
                        'session_id': session_id,
                        'account_id': account_id,
                        'ensemble_total': int(total_sel),
                        'ensemble_votes_buy': int(votes.get('buy', 0)),
                        'ensemble_votes_sell': int(votes.get('sell', 0)),
                        'ensemble_votes_hold': int(votes.get('hold', 0)),
                        'ensemble_required': int(required) if required is not None else None,
                        'ensemble_required_type': required_type,
                        'ensemble_regime': market_regime,
                        'ensemble_decision': decision,
                        'trade_mode': str(trade_mode) if trade_mode else None,
                        'trade_direction': (str(_dir) if _dir else None),
                        'active_role': (
                            str(ls_target_role)
                            if (str(trade_mode).strip() == 'long-short' and ls_target_role)
                            else None
                        ),
                        'regime_windows': list(market_regime_details.get('windows', [])) if isinstance(market_regime_details, dict) else [],
                        'regime_labels': list(market_regime_details.get('labels', [])) if isinstance(market_regime_details, dict) else [],
                        'regime_weights': list(market_regime_details.get('weights', [])) if isinstance(market_regime_details, dict) else [],
                        'regime_voting': (market_regime_details.get('voting') if isinstance(market_regime_details, dict) else None),
                        'regime_tie_break': (market_regime_details.get('tie_break') if isinstance(market_regime_details, dict) else None),
                        'prediction_source': 'xgb_decision_fallback',
                    }
                    if isinstance(gigachat_mc, dict) and gigachat_mc:
                        for k, v in gigachat_mc.items():
                            fallback_mc[k] = v
                    create_model_prediction(
                        symbol=symbol,
                        action=str(decision or 'hold'),
                        q_values=[],
                        current_price=current_price,
                        position_status=position_status,
                        model_path=fallback_model_path or '',
                        market_conditions=fallback_mc,
                    )
                except Exception:
                    pass
        except Exception:
            pass

        # Guard: если активен intent в DDD-исполнении для symbol — блокируем новые BUY/SELL, пока он не завершён
        try:
            has_active_intent = False
            if rc is not None:
                aid = rc.get(f'exec:active_intent:{session_id}')
                if aid:
                    raw_intent = rc.get(f'exec:intent:{aid}')
                    if raw_intent:
                        _ji = json.loads(raw_intent)
                        st = str(_ji.get('state') or '').lower()
                        if st not in ('filled','cancelled','failed','expired'):
                            has_active_intent = True
            if has_active_intent:
                decision = 'hold'
        except Exception:
            pass

        # Debug-buy (ENV only): форсировать BUY при любом предсказании, если включён в окружении
        try:
            def _truthy(v):
                try:
                    return str(v).strip().lower() in ('1','true','yes','on')
                except Exception:
                    return False

            debug_buy = False
            env_sym = os.getenv(f"DEBUG_BUY_{str(symbol).upper()}")
            env_glob = os.getenv("DEBUG_BUY")
            if env_sym is not None:
                debug_buy = _truthy(env_sym)
            elif env_glob is not None:
                debug_buy = _truthy(env_glob)

            if debug_buy and not has_active_intent and not getattr(agent, 'current_position', None):
                decision = 'buy'
                try:
                    logger.warning(f"[debug_buy] Forcing BUY for {symbol} (ENV)")
                except Exception:
                    pass
        except Exception:
            pass

        # NOTE: prediction-level BUY notifications were sending "signals" without an executed entry.
        # Entry notifications are now tied to actual opened positions via signal_notifier trade_entry events.

        trade_result = None
        entry_executed_now = False
        if dry_run:
            trade_result = {"success": True, "action": "dry_run"}
        else:
            # Попробуем прочитать режим исполнения, конфиг лимитной стратегии и стратегию выхода из Redis (per-symbol)
            exec_mode = None
            limit_cfg = None
            exit_mode = 'prediction'
            leverage_val = 1
            risk_type_exec = None
            try:
                if rc is not None:
                    _em = sget('execution_mode', session_doc.get('execution_mode'))
                    if _em:
                        exec_mode = str(_em).strip()
                    _lc = sget_json('limit_config', session_doc.get('limit_config'))
                    if _lc:
                        limit_cfg = _lc if isinstance(_lc, dict) else None
                    _xm = sget('exit_mode', session_doc.get('exit_mode'))
                    if _xm:
                        exit_mode = str(_xm).strip()
                    _lev = sget('leverage', session_doc.get('leverage'))
                    if _lev:
                        try:
                            leverage_val = max(1, min(5, int(str(_lev))))
                        except Exception:
                            leverage_val = 1
                    # читаем тип риск-менеджмента, чтобы управлять выходом
                    _rt = sget('risk_management_type', session_doc.get('risk_management_type'))
                    if _rt is not None and str(_rt).strip() != '':
                        risk_type_exec = str(_rt).strip()
            except Exception:
                exec_mode = None
                limit_cfg = None
                exit_mode = 'prediction'
                leverage_val = 1
                risk_type_exec = None

            # Если включены биржевые risk orders, по умолчанию выходим только по ним
            try:
                if (exit_mode is None or str(exit_mode).strip() == '' or str(exit_mode).strip().lower() == 'prediction') and (risk_type_exec and str(risk_type_exec).lower() in ('exchange_orders','both')):
                    exit_mode = 'risk_orders'
            except Exception:
                pass

            # Всегда обновим позицию заранее, чтобы на следующем тике не создавать входящие лимитки поверх открытой позиции
            try:
                agent._restore_position_from_exchange()
            except Exception:
                pass
            try:
                pos_cur = getattr(agent, 'current_position', None) or {}
                amt_cur = float(pos_cur.get('amount') or 0.0)
                avg_cur = float(pos_cur.get('entry_price') or 0.0)
                if not (amt_cur > 0.0 and avg_cur > 0.0):
                    agent.current_position = None
            except Exception:
                agent.current_position = None

            # Если позиции нет — снимаем все открытые ордера по символу (и reduceOnly, и обычные)
            try:
                if not agent.current_position:
                    try:
                        oo_all = agent.exchange.fetch_open_orders(symbol, params={'category': 'linear'}) or []
                    except Exception:
                        oo_all = []
                    cancelled_all = 0
                    for od in (oo_all or []):
                        try:
                            agent.exchange.cancel_order(od.get('id'), symbol)
                            cancelled_all += 1
                        except Exception:
                            continue
                    try:
                        logger.warning(f"[cleanup] flat state: cancelled_open_orders={cancelled_all}")
                    except Exception:
                        pass
                    # Дополнительно: удаляем активный DDD‑интент и отменяем его ордер, если висит
                    try:
                        if rc is not None:
                            aid_flat = rc.get(f'exec:active_intent:{session_id}')
                            if aid_flat:
                                raw_flat = rc.get(f'exec:intent:{aid_flat}')
                                exch_id_flat = None
                                if raw_flat:
                                    try:
                                        data_flat = json.loads(raw_flat)
                                        exch_id_flat = data_flat.get('exchange_order_id')
                                    except Exception:
                                        exch_id_flat = None
                                if exch_id_flat:
                                    try:
                                        agent.exchange.cancel_order(symbol, exch_id_flat)
                                    except Exception:
                                        pass
                                try:
                                    rc.delete(f'exec:intent:{aid_flat}')
                                except Exception:
                                    pass
                                rc.delete(f'exec:active_intent:{session_id}')
                                try:
                                    logger.warning(f"[cleanup] flat state: cleared active intent {aid_flat}")
                                except Exception:
                                    pass
                    except Exception:
                        pass
            except Exception:
                pass

            # Жёсткая блокировка новых входящих лимиток при активной позиции и выходе через риск-ордера
            try:
                if agent.current_position and ((str(exit_mode).lower() == 'risk_orders') or (risk_type_exec and str(risk_type_exec).lower() in ('exchange_orders','both'))):
                    # Удалим все non-reduceOnly ордера (входящие лимитки), TP/SL оставим
                    try:
                        oo = agent.exchange.fetch_open_orders(symbol, params={'category': 'linear'}) or []
                        for od in oo:
                            try:
                                params = od.get('info') or {}
                                reduce_only = bool(params.get('reduceOnly') or params.get('reduce_only') or False)
                                if not reduce_only:
                                    agent.exchange.cancel_order(od.get('id'), symbol)
                            except Exception:
                                continue
                    except Exception:
                        pass
                    trade_result = {"success": True, "action": "hold_exit_via_risk", "reason": "position_active"}
                    # Принудительно удерживаем hold, чтобы ниже не инициировались новые стратегии
                    decision = 'hold'
            except Exception:
                pass

            # Тип позиции: long/short (если удалось определить)
            try:
                pos_type = (agent.current_position or {}).get('type') if isinstance(getattr(agent, 'current_position', None), dict) else None
            except Exception:
                pos_type = None
            # Направление входа (в long-short задаётся по market_regime выше)
            try:
                dir_now = str(_dir).strip().lower() if _dir else 'long'
            except Exception:
                dir_now = 'long'

            # Hold-steps exit mode for XGB: keep position until timeout, then force close.
            try:
                if getattr(agent, 'current_position', None) and str(exit_mode or '').strip().lower() == 'hold_steps':
                    max_hold_steps_cfg = None
                    entry_ts_ms = None
                    signal_exit_enabled = False
                    signal_exit_start_step = None
                    signal_exit_window = XGB_SIGNAL_EXIT_WINDOW_DEFAULT
                    signal_exit_threshold = XGB_SIGNAL_EXIT_THRESHOLD_DEFAULT
                    now_bucket_ts = _last_closed_ts_ms()
                    try:
                        if rc is not None:
                            raw_mhs = sget('max_hold_steps', session_doc.get('max_hold_steps'))
                            if raw_mhs is not None and str(raw_mhs).strip() != '':
                                max_hold_steps_cfg = int(float(raw_mhs))
                            raw_signal_exit_enabled = sget('xgb_signal_exit_enabled', session_doc.get('xgb_signal_exit_enabled'))
                            signal_exit_enabled = str(raw_signal_exit_enabled or '').strip().lower() in ('1', 'true', 'yes', 'on')
                            signal_exit_role = 'short' if (str(pos_type or '').strip().lower() == 'short' or dir_now == 'short') else 'long'
                            raw_role_enabled = sget(
                                f'xgb_signal_exit_{signal_exit_role}_enabled',
                                session_doc.get(f'xgb_signal_exit_{signal_exit_role}_enabled'),
                            )
                            if raw_role_enabled not in (None, ''):
                                signal_exit_enabled = str(raw_role_enabled or '').strip().lower() in ('1', 'true', 'yes', 'on')
                            raw_signal_exit_start = sget(
                                f'xgb_signal_exit_{signal_exit_role}_start_step',
                                session_doc.get(f'xgb_signal_exit_{signal_exit_role}_start_step'),
                            )
                            if raw_signal_exit_start in (None, ''):
                                raw_signal_exit_start = sget('xgb_signal_exit_start_step', session_doc.get('xgb_signal_exit_start_step'))
                            if raw_signal_exit_start not in (None, ''):
                                signal_exit_start_step = int(float(raw_signal_exit_start))
                            raw_signal_exit_window = sget('xgb_signal_exit_window', session_doc.get('xgb_signal_exit_window'))
                            if raw_signal_exit_window not in (None, ''):
                                signal_exit_window = max(1, int(float(raw_signal_exit_window)))
                            raw_signal_exit_threshold = sget(
                                f'xgb_signal_exit_{signal_exit_role}_threshold',
                                session_doc.get(f'xgb_signal_exit_{signal_exit_role}_threshold'),
                            )
                            if raw_signal_exit_threshold in (None, ''):
                                raw_signal_exit_threshold = sget(
                                    'xgb_signal_exit_threshold',
                                    session_doc.get('xgb_signal_exit_threshold', XGB_SIGNAL_EXIT_THRESHOLD_DEFAULT),
                                )
                            if raw_signal_exit_threshold not in (None, ''):
                                signal_exit_threshold = float(raw_signal_exit_threshold)
                    except Exception:
                        max_hold_steps_cfg = None
                    try:
                        if rc is not None:
                            raw_ets = sget('pos_entry_ts')
                            if raw_ets is not None and str(raw_ets).strip() not in ('', '0'):
                                entry_ts_ms = int(raw_ets)
                    except Exception:
                        entry_ts_ms = None

                    candles_in_pos = None
                    if now_bucket_ts and entry_ts_ms and int(now_bucket_ts) >= int(entry_ts_ms):
                        candles_in_pos = int((int(now_bucket_ts) - int(entry_ts_ms)) // 300000)
                    remaining_hold_steps = None
                    if isinstance(max_hold_steps_cfg, int) and max_hold_steps_cfg > 0 and candles_in_pos is not None:
                        remaining_hold_steps = max(0, int(max_hold_steps_cfg) - int(candles_in_pos))
                    if remaining_hold_steps == 10 and rc is not None and entry_ts_ms is not None:
                        try:
                            notify_key = session_runtime_key(
                                session_id,
                                f"xgb_exit_soon_notified:{int(entry_ts_ms)}:{int(max_hold_steps_cfg)}",
                            )
                            if rc.set(notify_key, "1", nx=True, ex=86400 * 30):
                                publish_signal_event(rc, {
                                    "signal_id": f"{session_id}:{symbol}:{int(entry_ts_ms)}:{int(max_hold_steps_cfg)}:xgb_exit_soon",
                                    "event_type": "xgb_exit_soon",
                                    "session_id": session_id,
                                    "account_id": account_id,
                                    "symbol": symbol,
                                    "action": "exit_soon",
                                    "direction": dir_now,
                                    "position_type": pos_type,
                                    "price": locals().get("current_price"),
                                    "remaining_steps": remaining_hold_steps,
                                    "max_hold_steps": int(max_hold_steps_cfg),
                                    "candles_in_pos": int(candles_in_pos),
                                    "entry_ts_ms": int(entry_ts_ms),
                                    "now_bucket_ts_ms": int(now_bucket_ts) if now_bucket_ts is not None else None,
                                    "extend_steps": 50,
                                    "model_paths": model_paths,
                                })
                        except Exception:
                            logger.exception("[signal_notifier] failed to publish xgb_exit_soon session=%s symbol=%s", session_id, symbol)

                    try:
                        pred_json['hold_steps'] = {
                            'enabled': True,
                            'max_hold_steps': int(max_hold_steps_cfg) if max_hold_steps_cfg is not None else None,
                            'entry_ts_ms': int(entry_ts_ms) if entry_ts_ms is not None else None,
                            'now_bucket_ts_ms': int(now_bucket_ts) if now_bucket_ts is not None else None,
                            'candles_in_pos': int(candles_in_pos) if candles_in_pos is not None else None,
                            'remaining_steps': int(remaining_hold_steps) if remaining_hold_steps is not None else None,
                            'signal_exit': {
                                'enabled': bool(signal_exit_enabled),
                                'start_step': int(signal_exit_start_step) if isinstance(signal_exit_start_step, int) else None,
                                'window': int(signal_exit_window),
                                'threshold': float(signal_exit_threshold),
                            },
                        }
                    except Exception:
                        pass

                    signal_exit_triggered = False
                    if (
                        signal_exit_enabled
                        and isinstance(signal_exit_start_step, int)
                        and signal_exit_start_step > 0
                        and candles_in_pos is not None
                        and candles_in_pos >= int(signal_exit_start_step)
                    ):
                        signal_samples = _load_xgb_signal_history(rc, session_id, signal_exit_window)
                        try:
                            pred_json['hold_steps']['signal_exit']['history_size'] = len(signal_samples)
                        except Exception:
                            pass
                        if len(signal_samples) >= int(signal_exit_window):
                            avg_signal = float(sum(float(item['signal']) for item in signal_samples) / len(signal_samples))
                            avg_threshold = float(signal_exit_threshold)
                            try:
                                pred_json['hold_steps']['signal_exit']['avg_signal'] = avg_signal
                                pred_json['hold_steps']['signal_exit']['avg_threshold'] = avg_threshold
                                pred_json['hold_steps']['signal_exit']['ready'] = True
                            except Exception:
                                pass
                            if avg_signal < avg_threshold:
                                signal_exit_triggered = True
                                if pos_type == 'long':
                                    decision = 'sell'
                                elif pos_type == 'short':
                                    decision = 'buy'
                                try:
                                    if rc is not None:
                                        rc.setex(session_runtime_key(session_id, "forced_exit_reason"), 3600, "weak_signal_avg")
                                    pred_json['hold_steps']['phase'] = 'signal_exit'
                                    pred_json['hold_steps']['signal_exit']['triggered'] = True
                                except Exception:
                                    pass
                        else:
                            try:
                                pred_json['hold_steps']['signal_exit']['ready'] = False
                            except Exception:
                                pass

                    if isinstance(max_hold_steps_cfg, int) and max_hold_steps_cfg > 0 and candles_in_pos is not None:
                        if signal_exit_triggered:
                            pass
                        elif candles_in_pos >= int(max_hold_steps_cfg):
                            if pos_type == 'long':
                                decision = 'sell'
                            elif pos_type == 'short':
                                decision = 'buy'
                            try:
                                if rc is not None:
                                    rc.setex(session_runtime_key(session_id, "forced_exit_reason"), 3600, "timeout")
                                pred_json['hold_steps']['phase'] = 'timeout_exit'
                            except Exception:
                                pass
                        else:
                            decision = 'hold'
                            try:
                                pred_json['hold_steps']['phase'] = 'holding'
                            except Exception:
                                pass
            except Exception:
                pass

            # Сбрасываем gate лимитного входа, если позиция уже открыта
            try:
                _rc_gate_reset = Redis(host='redis', port=6379, db=0, decode_responses=True)
                if getattr(agent, 'current_position', None):
                    _rc_gate_reset.delete(session_runtime_key(session_id, "limit_entry:first_ts"))
                    _rc_gate_reset.delete(session_runtime_key(session_id, "limit_entry:attempts"))
                    _rc_gate_reset.delete(session_runtime_key(session_id, "market_entry:first_ts"))
                    _rc_gate_reset.delete(session_runtime_key(session_id, "market_entry:attempts"))
            except Exception:
                pass

            # --- ENTRY ---
            # long entry: BUY + dir=long + нет позиции
            # short entry: SELL + dir=short + нет позиции
            # Safeguard: перед входом обновим позицию с биржи (иногда current_position устаревает/не восстановилась)
            try:
                if (not getattr(agent, 'current_position', None)) and (decision in ('buy', 'sell')):
                    try:
                        agent._restore_position_from_exchange()
                    except Exception:
                        pass
                    if getattr(agent, 'current_position', None):
                        try:
                            _gid = locals().get('ensemble_group_id')
                            if _gid:
                                _attach_exec_error_to_group(symbol, str(_gid), "already in position: skip entry", exec_error_type="already_in_position")
                        except Exception:
                            pass
                        decision = 'hold'
            except Exception:
                pass

            try:
                if is_xgb_trade and getattr(agent, 'current_position', None):
                    pred_json['xgb_in_position_block'] = True
                    pred_json['xgb_in_position_reason'] = 'already_in_position'
                    preds_adj = pred_json.get('predictions') or []
                    if isinstance(preds_adj, list):
                        for _p in preds_adj:
                            try:
                                _p['action'] = 'hold'
                                _mc0 = _p.get('market_conditions') if isinstance(_p.get('market_conditions'), dict) else {}
                                _mc0['xgb_in_position_block'] = True
                                _mc0['xgb_in_position_reason'] = 'already_in_position'
                                _p['market_conditions'] = _mc0
                            except Exception:
                                continue
                    pred_json['decision'] = 'hold'
                    decision = 'hold'
                elif is_xgb_trade and not getattr(agent, 'current_position', None):
                    pred_json['xgb_in_position_block'] = False
            except Exception:
                pass

            # Final hard guard before any market/limit ENTRY. This prevents later runtime code
            # from re-enabling an XGB entry inside the post-exit boosted-threshold window.
            try:
                decision = _apply_xgb_postexit_cooldown_gate(decision, "final")
                decision = _apply_xgb_postexit_threshold_gate(decision)
            except Exception:
                pass

            # Entry dedupe: only one worker may start a fresh entry for this session/symbol.
            try:
                if (not getattr(agent, 'current_position', None)) and ((decision == 'buy' and dir_now == 'long') or (decision == 'sell' and dir_now == 'short')):
                    if rc is not None:
                        _entry_lock_key = session_runtime_key(session_id, "entry:lock")
                        _entry_lock_token = str(uuid.uuid4())
                        _entry_lock_ok = rc.set(_entry_lock_key, _entry_lock_token, nx=True, ex=90)
                        if not _entry_lock_ok:
                            pred_json['entry_lock_block'] = {
                                'blocked': True,
                                'reason': 'entry already in progress',
                                'ttl_sec': rc.ttl(_entry_lock_key),
                            }
                            trade_result = {"success": False, "action": "entry_lock_blocked", "reason": "entry already in progress"}
                            logger.warning("[entry_lock] session=%s symbol=%s blocked duplicate entry", session_id, symbol)
                            decision = 'hold'
            except Exception:
                pass

            # --- Market entry gate: ограничиваем число рыночных входов за окно ---
            if decision in ('buy', 'sell') and str(exec_mode or '').strip() != 'limit_post_only':
                try:
                    _rc_mgate = Redis(host='redis', port=6379, db=0, decode_responses=True)
                    _mma_raw = sget("market_entry_max_attempts", _rc_mgate.get("trading:market_entry_max_attempts"))
                    _mws_raw = sget("market_entry_window_sec", _rc_mgate.get("trading:market_entry_window_sec"))
                    _m_max = int(float(_mma_raw)) if (_mma_raw is not None and str(_mma_raw).strip() != '') else 2
                    _m_win = int(float(_mws_raw)) if (_mws_raw is not None and str(_mws_raw).strip() != '') else 1800
                    _m_max = max(1, min(100, _m_max))
                    _m_win = max(60, min(24 * 3600, _m_win))
                    _mfk = session_runtime_key(session_id, "market_entry:first_ts")
                    _mak = session_runtime_key(session_id, "market_entry:attempts")
                    _mnow = float(time.time())
                    _mft = 0.0
                    try:
                        _mfr = _rc_mgate.get(_mfk)
                        _mft = float(_mfr) if (_mfr is not None and str(_mfr).strip() != '') else 0.0
                    except Exception:
                        _mft = 0.0
                    if _mft <= 0 or (_mnow - _mft) > float(_m_win):
                        _rc_mgate.set(_mfk, str(_mnow), ex=max(600, _m_win * 2))
                        _rc_mgate.set(_mak, "0", ex=max(600, _m_win * 2))
                    _matt = 0
                    try:
                        _matt = int(float(_rc_mgate.get(_mak) or 0))
                    except Exception:
                        _matt = 0
                    if _matt >= _m_max:
                        _m_block_reason = f"market blocked: attempts={_matt}/{_m_max} window_sec={_m_win}"
                        try:
                            _gid = locals().get('ensemble_group_id')
                            if _gid:
                                _attach_exec_error_to_group(symbol, str(_gid), _m_block_reason, exec_error_type="market_entry_gate")
                        except Exception:
                            pass
                        decision = 'hold'
                        trade_result = {"success": False, "action": "market_entry_blocked", "reason": _m_block_reason}
                        logger.warning("[market_entry_gate] %s blocked: %s", symbol, _m_block_reason)
                    else:
                        _rc_mgate.incr(_mak)
                        _rc_mgate.expire(_mak, max(600, _m_win * 2))
                        _rc_mgate.expire(_mfk, max(600, _m_win * 2))
                except Exception:
                    pass

            if (not agent.current_position) and ((decision == 'buy' and dir_now == 'long') or (decision == 'sell' and dir_now == 'short')):
                entry_side = 'buy' if decision == 'buy' else 'sell'
                try:
                    if isinstance(entry_shap_snapshot, dict) and entry_shap_snapshot:
                        setattr(agent, '_last_xgb_shap_snapshot', entry_shap_snapshot)
                    else:
                        setattr(agent, '_last_xgb_shap_snapshot', None)
                except Exception:
                    pass
                if exec_mode == 'limit_post_only':
                    try:
                        # Gate: не даём лимитному входу крутиться бесконечно каждые 5 минут
                        blocked = False
                        block_reason = None
                        try:
                            _rc_gate = Redis(host='redis', port=6379, db=0, decode_responses=True)
                            # Настройки: per-symbol -> global -> defaults
                            _ma_raw = sget("limit_entry_max_attempts", _rc_gate.get("trading:limit_entry_max_attempts"))
                            _ws_raw = sget("limit_entry_window_sec", _rc_gate.get("trading:limit_entry_window_sec"))
                            max_attempts = int(float(_ma_raw)) if (_ma_raw is not None and str(_ma_raw).strip() != '') else 6
                            window_sec = int(float(_ws_raw)) if (_ws_raw is not None and str(_ws_raw).strip() != '') else 1800  # 30 минут
                            max_attempts = max(1, min(100, int(max_attempts)))
                            window_sec = max(60, min(24 * 3600, int(window_sec)))

                            first_key = session_runtime_key(session_id, "limit_entry:first_ts")
                            att_key = session_runtime_key(session_id, "limit_entry:attempts")
                            now_ts = float(time.time())
                            first_ts = 0.0
                            try:
                                _fr = _rc_gate.get(first_key)
                                first_ts = float(_fr) if (_fr is not None and str(_fr).strip() != '') else 0.0
                            except Exception:
                                first_ts = 0.0

                            # Новое окно
                            if first_ts <= 0 or (now_ts - first_ts) > float(window_sec):
                                try:
                                    _rc_gate.set(first_key, str(now_ts), ex=max(600, window_sec * 2))
                                except Exception:
                                    pass
                                try:
                                    _rc_gate.set(att_key, "0", ex=max(600, window_sec * 2))
                                except Exception:
                                    pass
                                first_ts = now_ts

                            attempts = 0
                            try:
                                attempts = int(float(_rc_gate.get(att_key) or 0))
                            except Exception:
                                attempts = 0

                            if attempts >= max_attempts:
                                blocked = True
                                block_reason = f"limit_post_only blocked: attempts={attempts}/{max_attempts} window_sec={window_sec}"
                            else:
                                # фиксируем попытку до enqueue (чтобы не накрутить вечный цикл)
                                try:
                                    _rc_gate.incr(att_key)
                                    _rc_gate.expire(att_key, max(600, window_sec * 2))
                                    _rc_gate.expire(first_key, max(600, window_sec * 2))
                                except Exception:
                                    pass
                        except Exception as _gate_e:
                            # Если gate сломался — лучше не блокировать, но логируем причину в карточку предсказания
                            try:
                                _gid = locals().get('ensemble_group_id')
                                if _gid:
                                    _attach_exec_error_to_group(symbol, str(_gid), f"limit_entry_gate error: {_gate_e}", exec_error_type="limit_entry_gate_error")
                            except Exception:
                                pass
                            blocked = False
                            block_reason = None

                        if blocked:
                            try:
                                _gid = locals().get('ensemble_group_id')
                                if _gid and block_reason:
                                    _attach_exec_error_to_group(symbol, str(_gid), str(block_reason), exec_error_type="limit_entry_gate")
                            except Exception:
                                pass
                            trade_result = {"success": False, "action": "limit_post_only_blocked", "reason": (block_reason or "blocked")}
                        else:
                            base_qty = float(getattr(agent, 'trade_amount', 0.0)) or 0.0
                            qty_entry = _apply_safety_buffer_qty(base_qty) if base_qty > 0 else None
                            start_execution_strategy.apply_async(kwargs={
                                'session_id': session_id,
                                'symbol': symbol,
                                'execution_mode': 'limit_post_only',
                                'side': entry_side,
                                'qty': qty_entry,
                                'limit_config': (limit_cfg or {}),
                                'leverage': leverage_val
                            }, queue='trade')
                            trade_result = {"success": True, "action": "limit_post_only_enqueued", "side": entry_side, "qty": qty_entry}
                    except Exception:
                        # Без фолбэков: если enqueue не удалось — фиксируем ошибку
                        try:
                            _gid = locals().get('ensemble_group_id')
                            if _gid:
                                _attach_exec_error_to_group(symbol, str(_gid), "limit_post_only enqueue failed", exec_error_type="limit_post_only_enqueue_failed")
                        except Exception:
                            pass
                        trade_result = {"success": False, "action": "limit_post_only_enqueue_failed"}
                else:
                    # market: long -> BUY, short -> OPEN_SHORT
                    # Опционально: split entry (например, 2 входа по 50% вместо 1 на 100%).
                    entry_splits = 1
                    try:
                        from utils.settings_store import get_setting_value as _get_setting_value
                        _spl = _get_setting_value('trading', 'sizing', 'ENTRY_SPLITS')
                        if _spl is not None and str(_spl).strip() != '':
                            entry_splits = int(float(str(_spl)))
                    except Exception:
                        entry_splits = 1
                    entry_splits = max(1, min(5, int(entry_splits or 1)))

                    if entry_splits <= 1:
                        trade_result = agent._execute_buy() if entry_side == 'buy' else agent._execute_open_short()
                        entry_executed_now = bool(isinstance(trade_result, dict) and trade_result.get('success') is True)
                    else:
                        total_qty = 0.0
                        try:
                            total_qty = float(getattr(agent, 'trade_amount', 0.0)) or 0.0
                        except Exception:
                            total_qty = 0.0
                        part_qty = (total_qty / float(entry_splits)) if (total_qty and total_qty > 0) else 0.0
                        results = []
                        ok_all = True
                        for i in range(int(entry_splits)):
                            try:
                                agent.trade_amount = float(part_qty)
                            except Exception:
                                pass
                            try:
                                r = agent._execute_buy() if entry_side == 'buy' else agent._execute_open_short()
                                results.append(r)
                                if not (isinstance(r, dict) and r.get('success') is True):
                                    ok_all = False
                                    break
                            except Exception:
                                ok_all = False
                                break
                        # После нескольких ордеров: восстановим позицию с биржи, чтобы риск-ордера ставились на полный объём
                        try:
                            agent._restore_position_from_exchange()
                        except Exception:
                            pass
                        trade_result = {
                            "success": bool(ok_all),
                            "action": "market_split_entry",
                            "side": entry_side,
                            "splits": int(entry_splits),
                            "qty_total": float(total_qty),
                            "qty_each": float(part_qty),
                            "results": results,
                        }
                        entry_executed_now = bool(ok_all)

                    # После рыночного лонга — выставим TP/SL (как раньше). Для шорта риск-ордера ставятся через ensure_risk_orders ниже.
                    if entry_side == 'buy':
                        try:
                            tp_pct = 1.0; sl_pct = 1.0; rtype = 'exchange_orders'
                            try:
                                if rc is not None:
                                    tp_raw = get_runtime_value(rc, session_id, 'take_profit_pct_long', session_doc.get('take_profit_pct_long'))
                                    sl_raw = get_runtime_value(rc, session_id, 'stop_loss_pct_long', session_doc.get('stop_loss_pct_long'))
                                    if _pick_num(tp_raw) is None or _pick_num(sl_raw) is None:
                                        tp_default, sl_default = _xgb_side_risk_defaults(session_doc, 'long')
                                        if _pick_num(tp_raw) is None:
                                            tp_raw = tp_default
                                        if _pick_num(sl_raw) is None:
                                            sl_raw = sl_default
                                    rt = get_runtime_value(rc, session_id, 'risk_management_type', session_doc.get('risk_management_type'))
                                    if rt is not None and str(rt).strip() != '':
                                        rtype = str(rt)
                                    if tp_raw is not None and str(tp_raw).strip() != '':
                                        tp_pct = float(tp_raw)
                                    if sl_raw is not None and str(sl_raw).strip() != '':
                                        sl_pct = float(sl_raw)
                            except Exception:
                                tp_pct = 1.0; sl_pct = 1.0; rtype = 'exchange_orders'
                            if (rtype in ('exchange_orders','both')) and (tp_pct is not None or sl_pct is not None):
                                entry_price = None; amount = None
                                try:
                                    pos = getattr(agent, 'current_position', None) or {}
                                    entry_price = float(pos.get('entry_price')) if pos and (pos.get('entry_price') is not None) else None
                                    amount = float(pos.get('amount')) if pos and (pos.get('amount') is not None) else None
                                except Exception:
                                    entry_price = None; amount = None
                                if entry_price is None:
                                    try:
                                        entry_price = float(agent._get_current_price())
                                    except Exception:
                                        entry_price = None
                                if amount is None or amount <= 0:
                                    try:
                                        amount = float(getattr(agent, 'trade_amount', 0.0))
                                    except Exception:
                                        amount = 0.0
                                if (entry_price and entry_price > 0) and (amount and amount > 0):
                                    try:
                                        mkt = agent.exchange.market(symbol)
                                        p_prec = int((mkt.get('precision', {}) or {}).get('price', 2))
                                    except Exception:
                                        p_prec = 2
                                    def _roundp(x):
                                        try:
                                            return float(f"{float(x):.{max(0,p_prec)}f}")
                                        except Exception:
                                            return float(x)
                                    if tp_pct is not None and tp_pct > 0:
                                        try:
                                            tp_price = _roundp(entry_price * (1.0 + float(tp_pct)/100.0))
                                            agent.exchange.create_limit_sell_order(
                                                symbol,
                                                amount,
                                                tp_price,
                                                { 'reduceOnly': True, 'timeInForce': 'GTC' }
                                            )
                                        except Exception:
                                            pass
                                    if sl_pct is not None and sl_pct > 0:
                                        try:
                                            sl_price = _roundp(entry_price * (1.0 - float(sl_pct)/100.0))
                                            agent.exchange.create_order(
                                                symbol,
                                                'market',
                                                'sell',
                                                amount,
                                                None,
                                                { 'reduceOnly': True, 'triggerPrice': sl_price, 'triggerDirection': 'descending', 'triggerBy': 'LastPrice', 'timeInForce': 'GTC' }
                                            )
                                        except Exception:
                                            pass
                        except Exception:
                            pass

            # --- EXIT ---
            # long exit: SELL when long position
            # short exit: BUY when short position
            elif agent.current_position and decision == 'sell' and pos_type == 'long':
                if (str(exit_mode).lower() == 'risk_orders') or (risk_type_exec and str(risk_type_exec).lower() in ('exchange_orders','both')):
                    trade_result = {"success": True, "action": "hold_exit_via_risk", "reason": "exit_mode=risk_orders"}
                else:
                    sell_strategy = agent._determine_sell_amount(agent._get_current_price())
                    if exec_mode == 'limit_post_only':
                        try:
                            sell_qty = float(sell_strategy.get('sell_amount', 0) or 0)
                            if sell_strategy.get('sell_all') and sell_qty <= 0:
                                sell_qty = None
                            start_execution_strategy.apply_async(kwargs={
                                'session_id': session_id,
                                'symbol': symbol,
                                'execution_mode': 'limit_post_only',
                                'side': 'sell',
                                'qty': sell_qty,
                                'limit_config': (limit_cfg or {}),
                                'leverage': leverage_val
                            }, queue='trade')
                            trade_result = {"success": True, "action": "limit_post_only_enqueued", "side": "sell"}
                        except Exception:
                            trade_result = agent._execute_sell() if sell_strategy.get('sell_all') else agent._execute_partial_sell(sell_strategy.get('sell_amount', 0))
                    else:
                        trade_result = agent._execute_sell() if sell_strategy.get('sell_all') else agent._execute_partial_sell(sell_strategy.get('sell_amount', 0))

            elif agent.current_position and decision == 'buy' and pos_type == 'short':
                # Закрытие шорта делаем market BUY reduceOnly через TradingAgent (без DDD), чтобы точно не открыть long по ошибке.
                trade_result = agent._execute_cover_short()

            else:
                trade_result = {"success": True, "action": "hold"}

        # Пост-обеспечение биржевых risk-orders (TP/SL/TrailingStop).
        # Нужно для ручного "execute_now" и случаев, когда позиция активна, но трейлинг ещё не выставлен.
        try:
            if risk_type_exec and str(risk_type_exec).lower() in ('exchange_orders', 'both'):
                need_ensure = bool(getattr(agent, 'current_position', None))
                if not need_ensure and isinstance(trade_result, dict):
                    need_ensure = str(trade_result.get('action') or '').startswith('limit_post_only_enqueued')
                if need_ensure:
                    # Для limit_post_only первый ensure лучше делать позже, чтобы попасть ближе к фактическому fill.
                    # Для market/прочих режимов 2s норм.
                    _first = 20 if str(exec_mode or '').strip() == 'limit_post_only' else 2
                    ensure_risk_orders.apply_async(kwargs={'session_id': session_id, 'symbol': symbol}, countdown=_first, queue='trade')
                    if str(exec_mode or '').strip() == 'limit_post_only':
                        ensure_risk_orders.apply_async(kwargs={'session_id': session_id, 'symbol': symbol}, countdown=15, queue='trade')
                        ensure_risk_orders.apply_async(kwargs={'session_id': session_id, 'symbol': symbol}, countdown=60, queue='trade')
        except Exception:
            pass

        # Для hold_steps время входа нужно сохранять не только через ensure_risk,
        # иначе UI может увидеть открытую позицию раньше, чем появится ключ в Redis.
        try:
            if rc is not None and getattr(agent, 'current_position', None):
                _ets_key = session_runtime_key(session_id, 'pos_entry_ts')
                _ets_raw = rc.get(_ets_key)
                _pos = getattr(agent, 'current_position', None) or {}
                try:
                    sset('pos_open', '1')
                    if _pos.get('type'):
                        sset('pos_type', str(_pos.get('type')))
                    if _pos.get('entry_price') not in (None, ''):
                        sset('pos_entry_price', str(float(_pos.get('entry_price'))))
                except Exception:
                    pass
                _should_write_entry_ts = bool(entry_executed_now) or _ets_raw in (None, '', '0')
                try:
                    _last_exit_raw = sget('last_exit_ts_ms')
                    if _last_exit_raw not in (None, '') and _ets_raw not in (None, '', '0'):
                        _should_write_entry_ts = _should_write_entry_ts or int(float(_last_exit_raw)) >= int(float(_ets_raw))
                except Exception:
                    pass
                if _should_write_entry_ts:
                    _entry_ts = None
                    try:
                        _entry_time = _pos.get('entry_time')
                        if _entry_time is not None and hasattr(_entry_time, 'timestamp'):
                            _entry_ts = int(_entry_time.timestamp() * 1000)
                    except Exception:
                        _entry_ts = None
                    if not _entry_ts:
                        _entry_ts = _last_closed_ts_ms()
                    if _entry_ts:
                        rc.set(_ets_key, str(int(_entry_ts)))

                # Publish bot event only when a position is actually open (covers market + delayed fills).
                try:
                    _pos_type_pub = str(_pos.get('type') or '').strip().lower()
                    if _pos_type_pub in ('long', 'short'):
                        _ets_val = rc.get(_ets_key)
                        _ets_val = _ets_val.decode() if isinstance(_ets_val, (bytes, bytearray)) else _ets_val
                        _entry_ts_ms_pub = int(float(_ets_val)) if _ets_val not in (None, '', '0') else None
                        if not _entry_ts_ms_pub:
                            raise RuntimeError("pos_entry_ts is required to publish trade_entry")
                        _pub_key = f"signals:published_entry:{session_id}:{symbol}:{int(_entry_ts_ms_pub)}"
                        if rc.set(_pub_key, "1", nx=True, ex=86400 * 90):
                            _entry_price_pub = None
                            try:
                                if _pos.get('entry_price') not in (None, ''):
                                    _entry_price_pub = float(_pos.get('entry_price'))
                            except Exception:
                                _entry_price_pub = None
                            publish_signal_event(rc, {
                                "event_type": "trade_entry",
                                "entry_executed": True,
                                "signal_id": f"{session_id}:{symbol}:{int(_entry_ts_ms_pub)}:{_pos_type_pub}:entry",
                                "session_id": session_id,
                                "account_id": account_id,
                                "symbol": symbol,
                                "action": ("buy" if _pos_type_pub == "long" else "sell"),
                                "price": _entry_price_pub,
                                "position_type": _pos_type_pub,
                                "entry_ts_ms": int(_entry_ts_ms_pub),
                                "trade_mode": str(trade_mode) if trade_mode else None,
                                "model_paths": model_paths,
                                "trade_number": str((_pos.get('trade_number') if _pos else None) or (trade_result.get('trade_number') if isinstance(trade_result, dict) else '') or ''),
                            })
                            try:
                                _pos_side_notify = str((_pos.get('type') if _pos and _pos.get('type') else None) or entry_side or '').strip().lower()
                                _sl_raw_notify = (
                                    get_runtime_value(
                                        rc,
                                        session_id,
                                        f'stop_loss_pct_{_pos_side_notify}',
                                        session_doc.get(f'stop_loss_pct_{_pos_side_notify}'),
                                    )
                                    if _pos_side_notify in ('long', 'short')
                                    else None
                                )
                                if _sl_raw_notify in (None, '') and _pos_side_notify in ('long', 'short'):
                                    _, _sl_raw_notify = _xgb_side_risk_defaults(session_doc, _pos_side_notify)
                                _sl_notify = float(_sl_raw_notify) if _sl_raw_notify not in (None, '') else None
                            except Exception:
                                _sl_notify = None
                            try:
                                _notify_max_position_entry(
                                    rc=rc,
                                    session_id=session_id,
                                    symbol=symbol,
                                    side=str((_pos.get('type') if _pos and _pos.get('type') else None) or entry_side or '').strip().lower(),
                                    entry_price=float(_pos.get('entry_price')) if _pos and _pos.get('entry_price') not in (None, '') else None,
                                    entry_time=(_pos.get('entry_time') if _pos else None) or datetime.now(),
                                    stop_loss_pct=_sl_notify,
                                    trade_number=str((_pos.get('trade_number') if _pos else None) or (trade_result.get('trade_number') if isinstance(trade_result, dict) else '') or ''),
                                )
                            except Exception:
                                pass
                            try:
                                _notify_telegram_position_entry(
                                    rc=rc,
                                    session_id=session_id,
                                    symbol=symbol,
                                    side=str((_pos.get('type') if _pos and _pos.get('type') else None) or entry_side or '').strip().lower(),
                                    entry_price=float(_pos.get('entry_price')) if _pos and _pos.get('entry_price') not in (None, '') else None,
                                    entry_time=(_pos.get('entry_time') if _pos else None) or datetime.now(),
                                    stop_loss_pct=_sl_notify,
                                    trade_number=str((_pos.get('trade_number') if _pos else None) or (trade_result.get('trade_number') if isinstance(trade_result, dict) else '') or ''),
                                )
                            except Exception:
                                pass
                            try:
                                from tasks.celery_task_copy_trade import copy_trade_for_clients
                                copy_trade_for_clients.apply_async(
                                    kwargs={
                                        "session_id": str(session_id),
                                        "symbol": str(symbol),
                                        "action": "entry",
                                        "position_type": str((_pos.get('type') if _pos and _pos.get('type') else None) or entry_side or "long").strip().lower(),
                                        "entry_price": float(_pos.get('entry_price')) if _pos and _pos.get('entry_price') not in (None, '') else None,
                                    },
                                    queue="celery"
                                )
                            except Exception as e_cp:
                                logger.exception("[copy_trade] entry trigger failed: %s", e_cp)
                except Exception:
                    logger.exception("[signal_notifier] failed to publish trade_entry session=%s symbol=%s", session_id, symbol)
        except Exception:
            pass

        status_after = agent.get_trading_status()

        # 5) Сохранение результата в Redis (как раньше)
        try:
            if rc is not None:
                try:
                    was_flat_before = not bool((current_status_before or {}).get('position'))
                except Exception:
                    was_flat_before = False
                if is_xgb_trade and was_flat_before:
                    long_signal_v, short_signal_v = _extract_xgb_long_short_signals(pred_json)
                    entry_state = 'ожидание'
                    try:
                        if bool(entry_executed_now):
                            dir_now_v = str(locals().get('dir_now') or '').strip().lower()
                            decision_v = str(decision or '').strip().lower()
                            if dir_now_v == 'short' and decision_v == 'sell':
                                entry_state = 'вход short'
                            elif dir_now_v == 'long' and decision_v == 'buy':
                                entry_state = 'вход long'
                            else:
                                pos_type_v = str((getattr(agent, 'current_position', None) or {}).get('type') or '').strip().lower()
                                if pos_type_v == 'short':
                                    entry_state = 'вход short'
                                elif pos_type_v == 'long':
                                    entry_state = 'вход long'
                    except Exception:
                        entry_state = 'ожидание'
                    bybit_account_id_v = None
                    try:
                        bybit_account_id_v = str(account_id or '').strip() or None
                    except Exception:
                        bybit_account_id_v = None
                    _append_xgb_entry_attempt_history(rc, {
                        'ts_ms': int(time.time() * 1000),
                        'timestamp': datetime.utcnow().isoformat(),
                        'session_id': str(session_id),
                        'symbol': str(symbol),
                        'long_signal': long_signal_v,
                        'short_signal': short_signal_v,
                        'state': entry_state,
                        'bybit_account_id': bybit_account_id_v,
                    })
                # Упакуем предсказания по моделям для UI
                try:
                    preds_list = pred_json.get('predictions') or []
                    preds_brief = []
                    for p in preds_list:
                        try:
                            preds_brief.append({
                                'model_path': p.get('model_path'),
                                'action': p.get('action'),
                                'confidence': p.get('confidence'),
                                'q_values': p.get('q_values') or []
                            })
                        except Exception:
                            continue
                except Exception:
                    preds_brief = []

                result_data = {
                    'timestamp': datetime.now().isoformat(),
                    'session_id': session_id,
                    'symbols': syms,
                    'model_paths': model_paths,
                    'decision': decision,
                    'serving_url': serving_url,
                    'predictions_count': len(pred_json.get('predictions', []) or []),
                    'predictions': preds_brief,
                    'consensus': consensus_cfg or {},
                    'market_regime': market_regime,
                    'market_regime_details': market_regime_details,
                    'consensus_applied': {
                        'regime': market_regime,
                        'votes': preds_list and {
                            'buy': sum(1 for _p in preds_list if str((_p.get('action') or 'hold')).lower()=='buy'),
                            'sell': sum(1 for _p in preds_list if str((_p.get('action') or 'hold')).lower()=='sell'),
                            'hold': sum(1 for _p in preds_list if str((_p.get('action') or 'hold')).lower()=='hold')
                        } or {'buy':0,'sell':0,'hold':0}
                    },
                    'trade_result': trade_result,
                    'exit_mode': exit_mode,
                }
                rc.setex(session_runtime_key(session_id, "latest_result"), 3600, json.dumps(result_data, default=str))
                status_after['session_id'] = session_id
                status_after['symbol'] = symbol
                status_after['symbol_display'] = symbol
                status_after['bybit_account_id'] = account_id
                set_session_status(rc, session_id, status_after)
        except Exception:
            pass

        return {
            "success": True,
            "decision": decision,
            "status_before": current_status_before,
            "status_after": status_after,
            "trade_result": trade_result,
            "dry_run": bool(dry_run),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0}, queue='trade')
def start_trading_task(self, session_id: str):
    """Оркестратор запускает один trade-cycle для конкретной session."""
    from datetime import datetime

    def _is_trade_worker_alive() -> bool:
        try:
            insp = celery.control.inspect()
            if not insp:
                return False
            active_queues = insp.active_queues() or {}
            for queues in active_queues.values():
                for q in queues or []:
                    if q.get('name') == 'trade':
                        return True
            return False
        except Exception:
            return True

    rc = get_redis_client()
    session_doc = load_session(rc, session_id) if rc is not None else None
    if not isinstance(session_doc, dict):
        return {"success": False, "skipped": True, "reason": "session_not_found"}

    symbol = str(session_doc.get('symbol') or '').strip().upper()
    if not symbol:
        return {"success": False, "skipped": True, "reason": "symbol_missing"}

    trading_enabled = str(get_config_value('ENABLE_TRADING_BEAT', '0')).lower() in ('1', 'true', 'yes', 'on')
    if not trading_enabled:
        return {"success": False, "skipped": True, "reason": "ENABLE_TRADING_BEAT=0"}

    if not _is_trade_worker_alive():
        payload = {
            'success': False,
            'session_id': session_id,
            'symbol': symbol,
            'symbol_display': symbol,
            'is_trading': False,
            'trading_status': 'Ошибка',
            'trading_status_emoji': '🔴',
            'trading_status_full': '🔴 Ошибка: воркер celery-trade не запущен',
            'reason': 'celery_trade_unavailable',
            'last_error': 'Воркер celery-trade не запущен (очередь trade недоступна).',
            'timestamp': datetime.utcnow().isoformat(),
        }
        if rc is not None:
            set_session_status(rc, session_id, payload)
        return {"success": False, "skipped": True, "reason": "celery_trade_unavailable"}

    lock_key = session_lock_key(session_id)
    got_lock = False
    try:
        if rc is not None:
            got_lock = bool(rc.set(lock_key, self.request.id, nx=True, ex=600))
        if not got_lock:
            return {"success": False, "skipped": True, "reason": "agent_lock_active"}

        prev = get_session_status(rc, session_id) if rc is not None else {}
        provisional = {
            'success': True,
            'session_id': session_id,
            'symbol': symbol,
            'symbol_display': symbol,
            'bybit_account_id': str(session_doc.get('account_id') or '').strip(),
            'is_trading': True,
            'trading_status': 'Активна',
            'trading_status_emoji': '🟢',
            'trading_status_full': '🟢 Активна',
            'amount': None,
            'amount_display': 'Не указано',
            'amount_usdt': 0.0,
            'position': None,
            'trades_count': 0,
            'balance': {},
            'current_price': 0.0,
            'last_model_prediction': None,
        }
        if isinstance(prev, dict):
            for key in (
                'position',
                'amount',
                'amount_display',
                'amount_usdt',
                'trades_count',
                'balance',
                'current_price',
                'last_model_prediction',
            ):
                if provisional.get(key) in (None, 0.0, 0, {}, 'Не указано') and prev.get(key) not in (None, '', {}, 0.0, 0):
                    provisional[key] = prev.get(key)
        if rc is not None:
            set_session_status(rc, session_id, provisional)

        res = execute_trade.apply_async(kwargs={'session_id': session_id}, queue='trade')
        return {"success": True, "enqueued": True, "task_ids": [res.id]}
    finally:
        try:
            if rc is not None and got_lock:
                rc.delete(lock_key)
        except Exception:
            pass


@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0}, queue='trade')
def run_active_trading_sessions(self):
    """Периодически раскладывает trade-cycle по всем активным session."""
    rc = get_redis_client()
    session_ids = list_session_ids(rc) if rc is not None else []
    enqueued = []
    for session_id in session_ids:
        sid = str(session_id or '').strip()
        if not sid:
            continue
        session_doc = load_session(rc, sid) if rc is not None else None
        if not isinstance(session_doc, dict):
            continue
        disabled = str(get_runtime_value(rc, sid, 'disabled', '') or '').strip().lower()
        if disabled in ('1', 'true', 'yes', 'on'):
            continue
        result = start_trading_task.apply_async(args=[sid], countdown=0, expires=300, queue='trade')
        enqueued.append({'session_id': sid, 'task_id': result.id})
    return {'success': True, 'total_sessions': len(session_ids), 'enqueued': enqueued}


# --- Периодический апдейтер статуса в Redis ---
@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0})
def refresh_trading_status(self):
    """Обновляет trading:current_status в Redis, если он отсутствует или устарел.

    Лёгкий хелпер для UI: не лезет в биржу, не вызывает модель.
    Помечает is_trading исходя из наличия активного lock ключа.
    """
    try:
        from redis import Redis as _Redis
        import json as _json
        from datetime import datetime as _dt, timedelta as _td
        import logging

        logger = logging.getLogger(__name__)

        rc = _Redis(host='redis', port=6379, db=0, decode_responses=True)

        # Текущие параметры
        try:
            symbols_raw = rc.get('trading:symbols')
            
            symbols = _json.loads(symbols_raw) if symbols_raw else []
            # Нормализация символов (защита от "TON USDT", "TON/USDT", лишних пробелов)
            try:
                from utils.cctx_utils import normalize_to_db as _normalize_to_db
                if isinstance(symbols, list):
                    symbols = [_normalize_to_db(str(s)) for s in symbols if s]
            except Exception:
                try:
                    if isinstance(symbols, list):
                        symbols = [str(s).replace('/', '').replace(' ', '').upper().strip() for s in symbols if s]
                except Exception:
                    pass
            logger.warning(f"DEBUG: symbols in refresh_trading_status = {symbols}") # Добавлен лог
            
            if not symbols:
                logger.error("Не удалось распознать символы для торговли")
                return {
                    "success": False, 
                    "skipped": True, 
                    "reason": "symbols_parsing_error", 
                    "error": "Не удалось распознать символы для торговли"
                }
        except Exception as e:
            logger.error(f"Ошибка при получении символов для торговли: {e}")
            return {
                "success": False, 
                "skipped": True, 
                "reason": "symbols_retrieval_error", 
                "error": f"Не удалось получить символы для торговли: {e}"
            }
        
        sym = symbols[0]

        # Текущий статус
        cached = rc.get('trading:current_status')
        cached_ts = rc.get('trading:current_status_ts')

        # Проверяем свежесть (6 минут)
        is_fresh = False
        try:
            if cached_ts:
                ts = _dt.fromisoformat(cached_ts)
                is_fresh = _dt.utcnow() <= (ts + _td(minutes=6))
        except Exception:
            is_fresh = False

        if cached and is_fresh:
            return {"success": True, "updated": False, "reason": "fresh"}

        # Активность оцениваем по наличию lock ключа с TTL > 0
        is_active = False
        try:
            lock_key = f'trading:agent_lock:{sym}'
            ttl = rc.ttl(lock_key)
            if ttl is not None and int(ttl) > 0:
                is_active = True
        except Exception:
            is_active = False

        # Базовый статус
        status = {
            'success': True,
            'is_trading': bool(is_active),
            'trading_status': 'Активна' if is_active else 'Остановлена',
            'trading_status_emoji': '🟢' if is_active else '🔴',
            'trading_status_full': ('🟢 Активна' if is_active else '🔴 Остановлена'),
            'symbol': sym,
            'symbol_display': sym,
            'amount': None,
            'amount_display': 'Не указано',
            'amount_usdt': 0.0,
            'position': None,
            'trades_count': 0,
            'balance': {}, 
            'current_price': 0.0,
            'last_model_prediction': None,
        }

        # Не перетираем имеющиеся поля, если cached есть
        try:
            if cached:
                prev = _json.loads(cached)
                if isinstance(prev, dict):
                    prev.update({k: v for k, v in status.items() if k not in prev or prev.get(k) is None})
        except Exception as e:
            logger.error(f"Ошибка при обновлении статуса из кэша: {e}")

    except Exception as e:
        logger.error(f"Общая ошибка в refresh_trading_status: {e}")
        return {"success": False, "error": str(e), "reason": "general_exception"}

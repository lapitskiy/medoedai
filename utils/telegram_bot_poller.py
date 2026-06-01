from __future__ import annotations

import threading
import time
from typing import Any, Optional

import requests

from utils.bot_last_prediction import (
    format_last_prediction_text,
    normalize_prediction_trigger,
    parse_xgb_hold_extend_trigger,
    telegram_reply_keyboard_markup,
)
from utils.bot_users_store import register_platform_user
from utils.settings_store import get_setting_value, upsert_setting
from utils.time_log import msk_tag
from utils.xgb_hold_extend import extend_xgb_hold_steps
from utils.bot_promo_store import redeem_promo_code, user_has_bot_access


import os
import redis
import ccxt
from orm.database import get_db_session
from orm.models import BotUserIdentity

_STARTED = False

def _get_redis() -> redis.Redis:
    return redis.from_url(os.environ.get("REDIS_URL", "redis://redis:6379/0"), decode_responses=True)

def _delete_message(chat_id: str, message_id: int) -> None:
    token = _get_token()
    if not token or not chat_id or not message_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/deleteMessage",
            json={"chat_id": chat_id, "message_id": message_id},
            timeout=10,
            proxies=_telegram_proxies(),
        )
    except Exception as exc:
        print(msk_tag(f"[telegram_bot] deleteMessage failed: {_mask_secret(str(exc))}"), flush=True)

def start_telegram_bot_poller() -> None:
    """Запускает фоновый polling Telegram updates в Flask-процессе."""
    global _STARTED
    if _STARTED:
        return
    _STARTED = True
    thread = threading.Thread(target=_poll_loop, name="telegram-bot-poller", daemon=True)
    thread.start()
    print(msk_tag("[telegram_bot] poller started"), flush=True)


def _poll_loop() -> None:
    offset: Optional[int] = None
    while True:
        token = _get_token()
        if not token:
            time.sleep(10)
            continue

        try:
            params: dict[str, Any] = {"timeout": 25, "allowed_updates": ["message", "callback_query"]}
            if offset is not None:
                params["offset"] = offset
            resp = requests.get(
                f"https://api.telegram.org/bot{token}/getUpdates",
                params=params,
                timeout=35,
                proxies=_telegram_proxies(),
            )
            data = resp.json()
            if not resp.ok or not data.get("ok"):
                print(msk_tag(f"[telegram_bot] getUpdates failed: {resp.status_code} {_mask_secret(resp.text)[:300]}"), flush=True)
                time.sleep(10)
                continue

            for update in data.get("result") or []:
                offset = int(update.get("update_id", 0)) + 1
                _handle_update(update)
        except Exception as exc:
            print(msk_tag(f"[telegram_bot] poll error: {_mask_secret(str(exc))}"), flush=True)
            time.sleep(10)


def _handle_update(update: dict[str, Any]) -> None:
    # 1. Обработка pre_checkout_query (запрос перед оплатой)
    if "pre_checkout_query" in update:
        pre_checkout_query = update["pre_checkout_query"]
        query_id = pre_checkout_query.get("id")
        token = _get_token()
        if token and query_id:
            try:
                requests.post(
                    f"https://api.telegram.org/bot{token}/answerPreCheckoutQuery",
                    json={"pre_checkout_query_id": query_id, "ok": True},
                    timeout=10,
                    proxies=_telegram_proxies(),
                )
            except Exception as exc:
                print(msk_tag(f"[telegram_bot] answerPreCheckoutQuery failed: {_mask_secret(str(exc))}"), flush=True)
        return

    callback = update.get("callback_query")
    if isinstance(callback, dict):
        _handle_callback_query(callback)
        return

    msg = update.get("message") or {}
    
    # 2. Обработка успешной оплаты (successful_payment)
    if "successful_payment" in msg:
        payment_info = msg["successful_payment"]
        chat = msg.get("chat") or {}
        chat_id_s = str(chat.get("id") or "").strip()
        user_id = str(msg.get("from", {}).get("id") or "")
        
        # Обновляем подписку в БД
        session = get_db_session()
        try:
            from datetime import datetime, timedelta
            from orm.models import BotSubscription
            identity = session.query(BotUserIdentity).filter(
                BotUserIdentity.platform == "telegram",
                BotUserIdentity.platform_user_id == user_id
            ).first()
            
            if identity:
                if identity.user.status != 'active':
                    identity.user.status = 'active'
                
                sub = session.query(BotSubscription).filter(
                    BotSubscription.user_id == identity.user_id,
                    BotSubscription.product_code == 'signals'
                ).first()
                
                now = datetime.utcnow()
                if not sub:
                    sub = BotSubscription(
                        user_id=identity.user_id,
                        product_code='signals',
                        status='active',
                        paid_until=now + timedelta(days=30)
                    )
                    session.add(sub)
                else:
                    sub.status = 'active'
                    if sub.paid_until and sub.paid_until > now:
                        sub.paid_until = sub.paid_until + timedelta(days=30)
                    else:
                        sub.paid_until = now + timedelta(days=30)
                session.commit()
                _send_message(chat_id_s, "✅ Оплата прошла успешно! Подписка на сигналы активирована (или продлена) на 30 дней.")
            else:
                _send_message(chat_id_s, "❌ Ошибка: Пользователь не найден в БД при обработке оплаты.")
        finally:
            session.close()
        return

    msg_id = msg.get("message_id")
    user = msg.get("from") or {}
    chat = msg.get("chat") or {}
    if not user.get("id"):
        return

    registered = register_platform_user(
        platform="telegram",
        platform_user_id=user.get("id"),
        username=user.get("username"),
        first_name=user.get("first_name"),
        last_name=user.get("last_name"),
        display_name=_display_name(user),
        language_code=user.get("language_code"),
        raw_profile=user,
    )
    _ensure_private_chat_id(chat)

    text = str(msg.get("text") or "").strip()
    chat_id_s = str(chat.get("id") or "").strip()

    r = _get_redis()
    state_key = f"tg:state:{chat_id_s}"
    state = r.get(state_key)

    if text.lower() in ["/settings", "настройки"]:
        status_text = "Подписка не активна ❌"
        leverage_text = "Плечо: x1"
        bot_status_text = "Бот не активен"
        
        session = get_db_session()
        try:
            from datetime import datetime
            from orm.models import BotSubscription
            identity = session.query(BotUserIdentity).filter(
                BotUserIdentity.platform == "telegram",
                BotUserIdentity.platform_user_id == str(user.get("id"))
            ).first()
            
            if identity:
                leverage = identity.bybit_leverage or 1
                leverage_text = f"Плечо: x{leverage}"
            
            if identity and identity.user.status == 'active':
                sub = session.query(BotSubscription).filter(
                    BotSubscription.user_id == identity.user_id,
                    BotSubscription.product_code == 'signals'
                ).first()
                if sub and sub.status == 'active' and sub.paid_until:
                    now = datetime.utcnow()
                    if sub.paid_until > now:
                        days = (sub.paid_until - now).days
                        if days > 0:
                            status_text = f"Статус: Активна (осталось {days} дн.) ✅"
                        else:
                            hours = (sub.paid_until - now).seconds // 3600
                            status_text = f"Статус: Активна (осталось {hours} ч.) ✅"
        finally:
            session.close()

        # Получаем данные о текущей торговле от мастер-сессии
        try:
            import json
            from utils.trading_sessions import get_status as get_session_status
            from utils.trading_sessions import session_runtime_key
            from utils.settings_store import get_setting_value
            master_sessions_raw = get_setting_value('trading', 'master', 'MASTER_SESSION_IDS')
            master_sessions = json.loads(master_sessions_raw) if master_sessions_raw else []
            if master_sessions:
                master_id = master_sessions[0]
                master_status = get_session_status(r, master_id) or {}
                
                # Timestamp берем из latest_result, так как в статусе его может не быть
                latest_raw = r.get(session_runtime_key(master_id, 'latest_result'))
                last_ts = None
                if latest_raw:
                    try:
                        latest_json = json.loads(latest_raw)
                        last_ts = latest_json.get('timestamp')
                    except Exception:
                        pass
                
                ts_str = "нет данных"
                if last_ts:
                    try:
                        from datetime import datetime, timedelta
                        dt = datetime.fromisoformat(last_ts[:26])
                        dt_msk = dt + timedelta(hours=3)
                        ts_str = dt_msk.strftime("%d.%m.%Y %H:%M") + " (МСК)"
                    except Exception:
                        pass
                        
                position = master_status.get('position')
                last_pred = master_status.get('last_model_prediction')
                
                if position:
                    bot_status_text = f"В позиции"
                elif str(last_pred).lower() == 'hold' or not last_pred:
                    bot_status_text = f"Нет сигнала на вход"
                else:
                    bot_status_text = f"Ожидание"
                
                bot_status_text += f"\n⏳ Последнее предсказание: {ts_str}"
            else:
                bot_status_text = "Мастер-модель не назначена"
        except Exception as e:
            bot_status_text = "Нет данных"

        _send_message(
            chat_id_s,
            f"⚙️ <b>Настройки</b>\n\n{status_text}\n⚖️ {leverage_text}\n🤖 {bot_status_text}\n\nВыберите, что вы хотите настроить:",
            with_default_keyboard=False,
            reply_markup={
                "inline_keyboard": [
                    [{"text": "🔑 Указать API ключи", "callback_data": "settings:api_keys"}],
                    [{"text": "⚖️ Указать плечо", "callback_data": "settings:leverage"}],
                    [{"text": "💳 Оплата", "callback_data": "settings:payment"}]
                ]
            }
        )
        return

    if state == "wait_promo":
        if msg_id:
            _delete_message(chat_id_s, msg_id)
        r.delete(state_key)
        result = redeem_promo_code(platform="telegram", platform_user_id=user.get("id"), code=text)
        if result.get("success"):
            _send_message(chat_id_s, f"✅ {result.get('message')}")
        else:
            _send_message(chat_id_s, f"❌ Ошибка: {result.get('message')}\nПопробуйте еще раз: /settings")
        return

    if state == "wait_leverage":
        if msg_id:
            _delete_message(chat_id_s, msg_id)
        r.delete(state_key)
        try:
            leverage = int(text)
            if leverage < 1 or leverage > 100:
                raise ValueError("Out of bounds")
                
            session = get_db_session()
            try:
                identity = session.query(BotUserIdentity).filter(
                    BotUserIdentity.platform == "telegram",
                    BotUserIdentity.platform_user_id == str(user.get("id"))
                ).first()
                
                if identity:
                    identity.bybit_leverage = leverage
                    session.commit()
                    _send_message(chat_id_s, f"✅ Плечо успешно сохранено: x{leverage}")
                else:
                    _send_message(chat_id_s, "❌ Ошибка: Пользователь не найден в БД.")
            finally:
                session.close()
        except ValueError:
            _send_message(chat_id_s, "❌ Ошибка: Плечо должно быть числом от 1 до 100. Попробуйте еще раз: /settings")
        return

    if state == "wait_keys":
        if msg_id:
            _delete_message(chat_id_s, msg_id)
        r.delete(state_key)
        
        parts = text.split()
        if len(parts) >= 2:
            api_key = parts[0]
            api_secret = parts[1]

            try:
                # Test Bybit keys
                exchange = ccxt.bybit({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                })
                exchange.fetch_balance()

                # Check permissions for security
                try:
                    api_info = exchange.privateGetV5UserQueryApi()
                    permissions = api_info.get('result', {}).get('permissions', {})
                    
                    wallet_perms = permissions.get('Wallet', [])
                    exchange_perms = permissions.get('Exchange', [])
                    
                    forbidden_perms = []
                    if 'AccountTransfer' in wallet_perms:
                        forbidden_perms.append('Перевод с аккаунта')
                    if 'SubMemberTransfer' in wallet_perms:
                        forbidden_perms.append('Перевод с субаккаунта')
                    if 'ExchangeHistory' in exchange_perms:
                        forbidden_perms.append('Конвертер: история обмена')
                        
                    if forbidden_perms:
                        perms_str = ", ".join(forbidden_perms)
                        _send_message(chat_id_s, f"❌ Ошибка: Ваши API ключи имеют доступ к кошельку/обмену ({perms_str}).\n\nПожалуйста, выпустите другие API ключи БЕЗ возможности вывода денег и обмена.\n\nПопробуйте еще раз: /settings")
                        return
                except Exception as perm_exc:
                    print(msk_tag(f"[telegram_bot] permission check failed: {perm_exc}"), flush=True)

                # Save to DB
                session = get_db_session()
                try:
                    identity = session.query(BotUserIdentity).filter(
                        BotUserIdentity.platform == "telegram",
                        BotUserIdentity.platform_user_id == str(user.get("id"))
                    ).first()
                    
                    if identity:
                        identity.bybit_api_key = api_key
                        identity.bybit_api_secret = api_secret
                        # Плечо выносится в отдельную настройку, пока сохраняем 1 по умолчанию, если оно не было установлено
                        if identity.bybit_leverage is None:
                            identity.bybit_leverage = 1
                        session.commit()
                        _send_message(
                            chat_id_s,
                            f"✅ Ключи Bybit успешно проверены и сохранены! (Плечо: x{identity.bybit_leverage})\n\n"
                            f"🛡 <b>Бот имеет доступ только к торговле. Возможность вывода средств отсутствует. Ваши деньги в абсолютной безопасности.</b>"
                        )
                    else:
                        _send_message(chat_id_s, "❌ Ошибка: Пользователь не найден в БД.")
                finally:
                    session.close()

            except Exception as exc:
                _send_message(chat_id_s, f"❌ Ошибка проверки ключей Bybit: {str(exc)}\nПопробуйте еще раз: /settings")
        else:
            _send_message(chat_id_s, "❌ Неверный формат. Нужно ввести: API_KEY API_SECRET\nПопробуйте еще раз: /settings")
        return

    extend_session_id = parse_xgb_hold_extend_trigger(text)
    if extend_session_id:
        if not user_has_bot_access("telegram", user.get("id")):
            _send_message(chat_id_s, "❌ У вас нет активного доступа. Введите промо-код в меню /settings", with_default_keyboard=False)
            return
        _send_message(chat_id_s, _extend_hold_text(extend_session_id), with_default_keyboard=False)
        return

    if normalize_prediction_trigger(text):
        if not user_has_bot_access("telegram", user.get("id")):
            _send_message(chat_id_s, "❌ У вас нет активного доступа. Введите промо-код в меню /settings")
            return
        _send_message(chat_id_s, format_last_prediction_text())
        return

    if text.startswith("/start"):
        _send_message(
            chat_id_s,
            "Готово. Пользователь зарегистрирован, доступ можно проверить через меню.",
        )
        print(msk_tag(f"[telegram_bot] /start registered user_id={registered['user_id']}"), flush=True)


def _handle_callback_query(callback: dict[str, Any]) -> None:
    query_id = str(callback.get("id") or "")
    data = str(callback.get("data") or "")
    message = callback.get("message") or {}
    chat = message.get("chat") or {}
    chat_id_s = str(chat.get("id") or "").strip()
    user = callback.get("from") or {}

    r = _get_redis()
    state_key = f"tg:state:{chat_id_s}"

    if data == "settings:api_keys":
        r.set(state_key, "wait_keys", ex=300)
        _answer_callback_query(query_id, "Ожидаю ввода API ключей")

        existing_keys_msg = ""
        session = get_db_session()
        try:
            from orm.models import BotUserIdentity
            identity = session.query(BotUserIdentity).filter(
                BotUserIdentity.platform == "telegram",
                BotUserIdentity.platform_user_id == str(user.get("id"))
            ).first()
            if identity and identity.bybit_api_key:
                api_key = identity.bybit_api_key
                masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
                existing_keys_msg = f"<b>Текущий API ключ:</b> <code>{masked_key}</code>\n<i>Ввод новых ключей заменит старые.</i>\n\n"
        except Exception as e:
            print(msk_tag(f"[telegram_bot] Error fetching existing keys: {e}"), flush=True)
        finally:
            session.close()

        _send_message(
            chat_id_s,
            f"{existing_keys_msg}Введите ваши ключи API через пробел:\n\n"
            "<code>API_KEY API_SECRET</code>\n\n"
            "Например:\n"
            "<code>QW... TY...</code>",
            with_default_keyboard=False
        )
        return

    if data == "settings:leverage":
        r.set(state_key, "wait_leverage", ex=300)
        _answer_callback_query(query_id, "Ожидаю ввода плеча")
        _send_message(
            chat_id_s,
            "Введите желаемое плечо (число от 1 до 100):\n\n"
            "Например:\n"
            "<code>10</code>",
            with_default_keyboard=False
        )
        return

    if data == "settings:payment":
        _answer_callback_query(query_id, "Оплата")
        _send_message(
            chat_id_s,
            "Выберите способ оплаты или активации:",
            with_default_keyboard=False,
            reply_markup={
                "inline_keyboard": [
                    [{"text": "⭐️ Telegram Stars (Оплата)", "callback_data": "settings:pay_stars"}],
                    [{"text": "🎟 Ввести промо-код", "callback_data": "settings:promo"}],
                    [{"text": "💳 Оплатить (Скоро)", "callback_data": "settings:pay"}]
                ]
            }
        )
        return

    if data == "settings:pay_stars":
        _answer_callback_query(query_id, "Генерирую счет в Stars...")
        
        token = _get_token()
        if token:
            payload = {
                "chat_id": chat_id_s,
                "title": "Доступ к сигналам",
                "description": "Подписка на торговые сигналы (30 дней)",
                "payload": f"sub_signals_30d_{user.get('id', '')}",
                "provider_token": "",
                "currency": "XTR",
                "prices": [{"label": "Подписка", "amount": 1000}]  # 1000 Stars
            }
            try:
                requests.post(
                    f"https://api.telegram.org/bot{token}/sendInvoice",
                    json=payload,
                    timeout=10,
                    proxies=_telegram_proxies(),
                )
            except Exception as exc:
                print(msk_tag(f"[telegram_bot] sendInvoice failed: {_mask_secret(str(exc))}"), flush=True)
                _send_message(chat_id_s, "❌ Не удалось создать счет. Попробуйте позже.")
        return

    if data == "settings:pay":
        _answer_callback_query(query_id, "Оплата в разработке")
        _send_message(chat_id_s, "Оплата картой/криптой находится в разработке.")
        return

    if data == "settings:promo":
        r.set(state_key, "wait_promo", ex=300)
        _answer_callback_query(query_id, "Ожидаю ввода промо-кода")
        _send_message(
            chat_id_s,
            "Введите ваш промо-код:",
            with_default_keyboard=False
        )
        return

    session_id = parse_xgb_hold_extend_trigger(data)
    if not session_id:
        return
        
    if not user_has_bot_access("telegram", str(user.get("id")) if user else ""):
        _answer_callback_query(query_id, "❌ Нет активной подписки")
        _send_message(chat_id_s, "❌ У вас нет активного доступа. Введите промо-код в меню /settings", with_default_keyboard=False)
        return
        
    text = _extend_hold_text(session_id)
    _answer_callback_query(query_id, text[:180])
    _send_message(chat_id_s, text, with_default_keyboard=False)


def _extend_hold_text(session_id: str) -> str:
    try:
        result = extend_xgb_hold_steps(session_id, 50)
        return (
            f"✅ Hold продлён: {result['symbol']}\n"
            f"{result['previous_max_hold_steps']} + {result['increment_steps']} = {result['max_hold_steps']}"
        )
    except Exception as exc:
        return f"❌ Не удалось продлить hold: {exc}"


def _ensure_private_chat_id(chat: dict[str, Any]) -> None:
    chat_id = str(chat.get("id") or "").strip()
    if not chat_id or chat.get("type") != "private":
        return

    current = (get_setting_value("api", "telegram", "TELEGRAM_CHAT_IDS") or "").strip()
    values = [item.strip() for item in current.split(",") if item.strip() and item.strip() != "1"]
    if chat_id not in values:
        values.append(chat_id)
        upsert_setting(
            scope="api",
            group="telegram",
            key="TELEGRAM_CHAT_IDS",
            value_type="string",
            label="Telegram chat IDs",
            description="Comma-separated Telegram chat IDs for signal delivery.",
            is_secret=False,
            value=",".join(values),
        )


def _get_token() -> str:
    return (get_setting_value("api", "telegram", "TELEGRAM_BOT_TOKEN") or "").strip()


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


def _answer_callback_query(query_id: str, text: str) -> None:
    token = _get_token()
    if not token or not query_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/answerCallbackQuery",
            json={"callback_query_id": query_id, "text": text, "show_alert": False},
            timeout=10,
            proxies=_telegram_proxies(),
        )
    except Exception as exc:
        print(msk_tag(f"[telegram_bot] answerCallbackQuery failed: {_mask_secret(str(exc))}"), flush=True)


def _send_message(chat_id: str, text: str, *, with_default_keyboard: bool = True, reply_markup: dict[str, Any] | None = None) -> None:
    token = _get_token()
    if not token or not chat_id:
        return
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    elif with_default_keyboard:
        payload["reply_markup"] = telegram_reply_keyboard_markup()
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json=payload,
            timeout=10,
            proxies=_telegram_proxies(),
        )
    except Exception as exc:
        print(msk_tag(f"[telegram_bot] sendMessage failed: {_mask_secret(str(exc))}"), flush=True)


def _mask_secret(text: str) -> str:
    token = _get_token()
    if token:
        return str(text).replace(token, "***")
    return str(text)


def _telegram_proxies() -> Optional[dict[str, str]]:
    proxy_url = (get_setting_value("api", "telegram", "TELEGRAM_PROXY_URL") or "").strip()
    if not proxy_url:
        return None
    return {"http": proxy_url, "https": proxy_url}

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from orm.database import get_db_session, get_engine
from orm.models import BotSubscription, BotUser, BotUserIdentity


def ensure_bot_user_tables() -> None:
    """Создаёт таблицы пользователей ботов, если их ещё нет."""
    engine = get_engine()
    BotUser.__table__.create(bind=engine, checkfirst=True)
    BotUserIdentity.__table__.create(bind=engine, checkfirst=True)
    BotSubscription.__table__.create(bind=engine, checkfirst=True)


def register_platform_user(
    *,
    platform: str,
    platform_user_id: str | int,
    username: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    display_name: Optional[str] = None,
    language_code: Optional[str] = None,
    raw_profile: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Upsert пользователя после /start или любого входящего сообщения.

    platform='telegram', platform_user_id=<telegram from_user.id>.
    Для MAX позже будет platform='max' и его внешний user id.
    """
    ensure_bot_user_tables()
    session = get_db_session()
    now = datetime.utcnow()
    platform_n = str(platform or '').strip().lower()
    external_id = str(platform_user_id or '').strip()
    if not platform_n or not external_id:
        raise ValueError('platform and platform_user_id are required')

    try:
        identity = session.query(BotUserIdentity).filter(
            BotUserIdentity.platform == platform_n,
            BotUserIdentity.platform_user_id == external_id,
        ).first()

        if identity is None:
            user = BotUser(status='active', role='user', created_at=now, updated_at=now, last_seen_at=now)
            session.add(user)
            session.flush()
            identity = BotUserIdentity(
                user_id=user.id,
                platform=platform_n,
                platform_user_id=external_id,
                created_at=now,
            )
            session.add(identity)
        else:
            user = identity.user
            user.last_seen_at = now
            user.updated_at = now

        identity.username = _norm(username)
        identity.first_name = _norm(first_name)
        identity.last_name = _norm(last_name)
        identity.display_name = _norm(display_name) or _build_display_name(first_name, last_name, username)
        identity.language_code = _norm(language_code)
        identity.raw_profile = json.dumps(raw_profile, ensure_ascii=False) if raw_profile else None
        identity.last_seen_at = now
        identity.updated_at = now

        session.commit()
        return {
            'user_id': user.id,
            'identity_id': identity.id,
            'platform': identity.platform,
            'platform_user_id': identity.platform_user_id,
            'status': user.status,
            'role': user.role,
        }
    finally:
        try:
            session.close()
        except Exception:
            pass


def list_platform_users(*, platform: str) -> list[dict[str, Any]]:
    """Возвращает пользователей, зарегистрированных через указанную bot-платформу."""
    ensure_bot_user_tables()
    session = get_db_session()
    platform_n = str(platform or '').strip().lower()
    if not platform_n:
        raise ValueError('platform is required')

    try:
        rows = (
            session.query(BotUserIdentity)
            .join(BotUser, BotUser.id == BotUserIdentity.user_id)
            .filter(BotUserIdentity.platform == platform_n)
            .order_by(BotUserIdentity.created_at.desc(), BotUserIdentity.id.desc())
            .all()
        )
        return [_identity_to_dict(identity) for identity in rows]
    finally:
        try:
            session.close()
        except Exception:
            pass


def _norm(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value_s = str(value).strip()
    return value_s or None


def _build_display_name(
    first_name: Optional[str],
    last_name: Optional[str],
    username: Optional[str],
) -> Optional[str]:
    full_name = ' '.join(part for part in [first_name, last_name] if _norm(part)).strip()
    if full_name:
        return full_name
    username_n = _norm(username)
    return f'@{username_n}' if username_n else None


def _dt_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.isoformat()


def _identity_to_dict(identity: BotUserIdentity) -> dict[str, Any]:
    user = identity.user
    has_keys = bool(getattr(identity, 'bybit_api_key', None) and getattr(identity, 'bybit_api_secret', None))
    
    paid_until = None
    if user.subscriptions:
        for sub in user.subscriptions:
            if sub.product_code == 'signals' and sub.status == 'active':
                paid_until = _dt_iso(sub.paid_until)
                break
                
    active_promos = []
    unread_support_count = 0
    try:
        from sqlalchemy.orm import object_session
        from orm.models import BotPromoRedemption, BotPromoCode, BotSupportMessage
        session = object_session(identity)
        if session:
            redemptions = session.query(BotPromoCode.note).join(
                BotPromoRedemption, BotPromoRedemption.promo_code_id == BotPromoCode.id
            ).filter(BotPromoRedemption.user_id == user.id).all()
            for r in redemptions:
                if r[0]:
                    active_promos.append(r[0])
                    
            unread_support_count = session.query(BotSupportMessage).filter(
                BotSupportMessage.user_id == user.id,
                BotSupportMessage.direction == 'user_to_admin',
                BotSupportMessage.is_read == False
            ).count()
    except Exception:
        pass

    return {
        'user_id': user.id,
        'identity_id': identity.id,
        'platform': identity.platform,
        'platform_user_id': identity.platform_user_id,
        'username': identity.username,
        'first_name': identity.first_name,
        'last_name': identity.last_name,
        'display_name': identity.display_name,
        'language_code': identity.language_code,
        'status': user.status,
        'role': user.role,
        'registered_at': _dt_iso(identity.created_at),
        'last_seen_at': _dt_iso(identity.last_seen_at or user.last_seen_at),
        'has_active_keys': has_keys,
        'bybit_leverage': getattr(identity, 'bybit_leverage', 1) or 1,
        'paid_until': paid_until,
        'active_promos': list(set(active_promos)),
        'unread_support_count': unread_support_count,
    }

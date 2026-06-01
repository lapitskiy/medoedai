from __future__ import annotations

import random
import string
from datetime import datetime, timedelta
from typing import Any, Optional

from orm.database import get_db_session
from orm.models import BotPromoCode, BotPromoRedemption, BotSubscription, BotUserIdentity

def _generate_code(length: int = 8) -> str:
    # Исключаем похожие символы: 0, O, 1, I, L
    chars = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
    return "".join(random.choice(chars) for _ in range(length))

def create_promo_codes(count: int, duration_days: int = 14, max_uses: int = 1, note: Optional[str] = None) -> list[str]:
    session = get_db_session()
    try:
        codes = []
        for _ in range(count):
            raw = _generate_code()
            # Форматируем как MEDOED-XXXX-XXXX
            code_str = f"MEDOED-{raw[:4]}-{raw[4:]}"
            pc = BotPromoCode(
                code=code_str,
                duration_days=duration_days,
                max_uses=max_uses,
                note=note
            )
            session.add(pc)
            codes.append(code_str)
        session.commit()
        return codes
    finally:
        session.close()

def delete_promo_code(code_id: int) -> bool:
    session = get_db_session()
    try:
        promo = session.query(BotPromoCode).filter(BotPromoCode.id == code_id).first()
        if not promo:
            return False
        
        # Удаляем связанные записи об активации
        session.query(BotPromoRedemption).filter(BotPromoRedemption.promo_code_id == promo.id).delete()
        
        session.delete(promo)
        session.commit()
        return True
    except Exception:
        session.rollback()
        return False
    finally:
        session.close()

def list_promo_codes() -> list[dict[str, Any]]:
    session = get_db_session()
    try:
        rows = session.query(BotPromoCode).order_by(BotPromoCode.created_at.desc()).all()
        result = []
        for r in rows:
            result.append({
                "id": r.id,
                "code": r.code,
                "duration_days": r.duration_days,
                "max_uses": r.max_uses,
                "used_count": r.used_count,
                "is_active": r.is_active,
                "note": r.note,
                "valid_until": r.valid_until.isoformat() if r.valid_until else None,
                "created_at": r.created_at.isoformat()
            })
        return result
    finally:
        session.close()

def redeem_promo_code(*, platform: str, platform_user_id: str | int, code: str) -> dict[str, Any]:
    """
    Активирует промокод для пользователя.
    Возвращает словарь с результатом {'success': bool, 'message': str, 'paid_until': datetime}
    """
    session = get_db_session()
    try:
        code_clean = str(code).strip().upper()
        
        # Ищем пользователя
        identity = session.query(BotUserIdentity).filter(
            BotUserIdentity.platform == str(platform).lower(),
            BotUserIdentity.platform_user_id == str(platform_user_id)
        ).first()
        
        if not identity:
            return {"success": False, "message": "Пользователь не найден."}
            
        user = identity.user
        if user.status != 'active':
            return {"success": False, "message": "Аккаунт заблокирован."}

        # Ищем код
        promo = session.query(BotPromoCode).filter(BotPromoCode.code == code_clean).with_for_update().first()
        if not promo:
            return {"success": False, "message": "Промокод не найден."}
            
        if not promo.is_active:
            return {"success": False, "message": "Промокод отключен."}
            
        now = datetime.utcnow()
        if promo.valid_from and now < promo.valid_from:
            return {"success": False, "message": "Время действия промокода еще не наступило."}
            
        if promo.valid_until and now > promo.valid_until:
            return {"success": False, "message": "Промокод истек."}
            
        if promo.used_count >= promo.max_uses:
            return {"success": False, "message": "Лимит активаций исчерпан."}

        # Проверяем, не вводил ли юзер этот код раньше
        existing_redemption = session.query(BotPromoRedemption).filter(
            BotPromoRedemption.promo_code_id == promo.id,
            BotPromoRedemption.user_id == user.id
        ).first()
        
        if existing_redemption:
            return {"success": False, "message": "Вы уже активировали этот промокод."}

        # Ищем существующую подписку
        sub = session.query(BotSubscription).filter(
            BotSubscription.user_id == user.id,
            BotSubscription.product_code == 'signals'
        ).with_for_update().first()

        if not sub:
            sub = BotSubscription(
                user_id=user.id,
                product_code='signals',
                plan_code='monthly',
                status='active',
                paid_until=now,
                provider='promo',
                provider_payment_id=promo.code
            )
            session.add(sub)
        
        # Продлеваем paid_until
        base_date = sub.paid_until if (sub.paid_until and sub.paid_until > now) else now
        new_paid_until = base_date + timedelta(days=promo.duration_days)
        
        sub.status = 'active'
        sub.paid_until = new_paid_until
        sub.provider = 'promo'
        sub.provider_payment_id = promo.code
        sub.updated_at = now

        # Записываем аудит
        promo.used_count += 1
        redemption = BotPromoRedemption(
            promo_code_id=promo.id,
            user_id=user.id,
            paid_until_after=new_paid_until,
            redeemed_at=now
        )
        session.add(redemption)
        
        session.commit()
        
        return {
            "success": True, 
            "message": f"Промокод активирован! Доступ продлен на {promo.duration_days} дней.",
            "paid_until": new_paid_until
        }
    except Exception as e:
        session.rollback()
        return {"success": False, "message": f"Внутренняя ошибка: {str(e)}"}
    finally:
        session.close()

def user_has_bot_access(platform: str, platform_user_id: str | int) -> bool:
    """Проверяет наличие активной подписки у пользователя."""
    session = get_db_session()
    try:
        identity = session.query(BotUserIdentity).filter(
            BotUserIdentity.platform == str(platform).lower(),
            BotUserIdentity.platform_user_id == str(platform_user_id)
        ).first()
        
        if not identity or identity.user.status != 'active':
            return False
            
        sub = session.query(BotSubscription).filter(
            BotSubscription.user_id == identity.user_id,
            BotSubscription.product_code == 'signals'
        ).first()
        
        if not sub or sub.status != 'active' or not sub.paid_until:
            return False
            
        return sub.paid_until > datetime.utcnow()
    finally:
        session.close()

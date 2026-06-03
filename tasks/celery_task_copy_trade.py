import json
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from celery import shared_task
import ccxt
from datetime import datetime

from orm.database import get_db_session
from orm.models import BotUserIdentity, BotSubscription

logger = logging.getLogger(__name__)

def _execute_client_trade(identity, symbol, action, position_type, entry_price):
    try:
        # Check permissions and keys
        if not identity.bybit_api_key or not identity.bybit_api_secret:
            return {"success": False, "error": "No API keys"}

        leverage = identity.bybit_leverage or 1

        from trading_agent.infrastructure.exchange_gateway_bybit import BybitExchangeGateway
        gateway = BybitExchangeGateway(api_key=identity.bybit_api_key, secret=identity.bybit_api_secret)
        exchange = gateway.ex

        # Set leverage (best effort)
        try:
            exchange.set_leverage(leverage, symbol)
            # Also set margin mode to isolated (best effort, ignore errors)
            try:
                exchange.set_margin_mode('isolated', symbol)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Client {identity.platform_user_id} leverage set error: {e}")

        # Execute order
        if action == "exit":
            # For exit, we just place a reduceOnly market order in the opposite direction
            side = "sell" if position_type == "long" else "buy"
            
            # Fetch current position to know how much to close (or just close all)
            # Bybit v5 specific params to close position
            params = {
                'reduceOnly': True,
                'closeOnTrigger': True
            }
            
            # Find the position amount to close fully
            try:
                positions = exchange.fetch_positions([symbol])
                pos = None
                for p in positions:
                    if p.get('symbol') == symbol or p.get('info', {}).get('symbol') == symbol:
                        pos = p
                        break
                if pos and pos.get('contracts', 0) > 0:
                    amount_to_close = pos['contracts']
                    order = exchange.create_market_order(symbol, side, amount_to_close, params=params)
                    return {"success": True, "order": order, "action": "exit", "amount": amount_to_close}
                else:
                    return {"success": False, "error": "No open position found to close"}
            except Exception as e:
                # Fallback to a large reduceOnly order if fetch fails? Better not, could fail.
                return {"success": False, "error": f"Failed to fetch position or close: {e}"}

        elif action == "entry":
            # For entry, calculate amount based on available USDT
            balance = exchange.fetch_balance()
            free_usdt = balance.get('USDT', {}).get('free', 0)
            
            if free_usdt < 1:
                return {"success": False, "error": "Insufficient USDT balance"}

            side = "buy" if position_type == "long" else "sell"
            
            # Get current price
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Use 95% of available margin to avoid insufficient margin errors
            usable_margin = free_usdt * 0.95
            
            # Calculate amount: (margin * leverage) / price
            raw_amount = (usable_margin * leverage) / current_price
            
            amount = gateway._normalize_qty(symbol, raw_amount)
            
            if float(amount) <= 0:
                 return {"success": False, "error": "Calculated amount is too small"}

            params = {
                'leverage': str(leverage),
                'marginMode': 'isolated',
            }

            order = exchange.create_market_order(symbol, side, float(amount), params=params)
            return {"success": True, "order": order, "action": "entry", "amount": amount, "price": current_price}
            
    except ccxt.AuthenticationError:
        # Invalid keys
        from utils.telegram_bot_poller import _send_message
        _send_message(
            identity.platform_user_id,
            "⚠️ <b>Торговля остановлена.</b>\nВаш API ключ недействителен (удален или истек). Пожалуйста, обновите его в настройках (/settings).",
            with_default_keyboard=False
        )
        # We could also mark the keys as invalid in the DB here
        return {"success": False, "error": "AuthenticationError"}
    except ccxt.InsufficientFunds as e:
        return {"success": False, "error": f"Insufficient funds: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@shared_task(bind=True, queue='celery')
def copy_trade_for_clients(self, session_id: str, symbol: str, action: str, position_type: str, entry_price: float = None):
    """
    Executes a copy trade for all active clients.
    action: 'entry' or 'exit'
    position_type: 'long' or 'short'
    """
    from utils.settings_store import get_setting_value
    import json
    
    # Check if this session is the master model
    master_sessions_raw = get_setting_value('trading', 'master', 'MASTER_SESSION_IDS')
    try:
        master_sessions = json.loads(master_sessions_raw) if master_sessions_raw else []
    except Exception:
        master_sessions = []
        
    if session_id not in master_sessions:
        return {"success": True, "skipped": True, "reason": "not_master_session"}

    logger.info(f"[CopyTrade] Triggered for clients: symbol={symbol}, action={action}, type={position_type}")

    session = get_db_session()
    try:
        now = datetime.utcnow()
        # Get users with active subscriptions and bybit keys
        active_users = (
            session.query(BotUserIdentity)
            .join(BotSubscription, BotUserIdentity.user_id == BotSubscription.user_id)
            .filter(
                BotUserIdentity.platform == 'telegram',
                BotUserIdentity.bybit_api_key.isnot(None),
                BotUserIdentity.bybit_api_secret.isnot(None),
                BotSubscription.product_code == 'signals',
                BotSubscription.status == 'active',
                BotSubscription.paid_until > now
            )
            .all()
        )
        
        if not active_users:
            logger.info("[CopyTrade] No active users with keys found.")
            return {"success": True, "clients_count": 0}

        # Execute trades in parallel
        results = []
        with ThreadPoolExecutor(max_workers=min(20, len(active_users))) as executor:
            futures = {
                executor.submit(_execute_client_trade, user, symbol, action, position_type, entry_price): user.platform_user_id
                for user in active_users
            }
            for future in futures:
                user_id = futures[future]
                try:
                    res = future.result()
                    results.append({"user_id": user_id, "result": res})
                    logger.info(f"[CopyTrade] User {user_id}: {res}")
                except Exception as e:
                    results.append({"user_id": user_id, "result": {"success": False, "error": str(e)}})
                    logger.error(f"[CopyTrade] User {user_id} error: {e}")

        success_count = sum(1 for r in results if r["result"].get("success"))
        logger.info(f"[CopyTrade] Completed for {len(active_users)} clients. Success: {success_count}.")
        return {"success": True, "clients_count": len(active_users), "success_count": success_count, "results": results}
    finally:
        session.close()

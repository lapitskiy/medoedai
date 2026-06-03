import sys
import os
sys.path.append('.')
from orm.database import get_db_session
from orm.models import Trade
from trading_agent.trading_agent import TradingAgent
import time

session = get_db_session()
# Get recent executed trades with PnL
trades = session.query(Trade).filter(
    Trade.status == 'executed', 
    Trade.position_pnl.isnot(None)
).order_by(Trade.executed_at.desc()).limit(20).all()

agent = TradingAgent(symbol='BTCUSDT', account_id=5)

try:
    pnl_res = agent.exchange.privateGetV5PositionClosedPnl({'category': 'linear', 'symbol': 'BTCUSDT', 'limit': 50})
    pnl_by_order = {}
    if pnl_res and 'result' in pnl_res and 'list' in pnl_res['result']:
        for p in pnl_res['result']['list']:
            if p.get('orderId') and p.get('closedPnl') is not None:
                pnl_by_order[str(p['orderId'])] = float(p['closedPnl'])
    
    for t in trades:
        if t.exchange_order_id and str(t.exchange_order_id) in pnl_by_order:
            real_pnl = pnl_by_order[str(t.exchange_order_id)]
            if abs(t.position_pnl - real_pnl) > 0.001:
                print(f"Fixing Trade {t.id} (Order {t.exchange_order_id}): {t.position_pnl} -> {real_pnl}")
                t.position_pnl = real_pnl
    session.commit()
    print("Done fixing PnL.")
except Exception as e:
    print(f"Error: {e}")

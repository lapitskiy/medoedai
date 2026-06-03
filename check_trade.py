import sys
import os
sys.path.append('.')
from orm.database import get_db_session
from orm.models import Trade, Symbol

session = get_db_session()
trades = session.query(Trade).filter(Trade.status == 'executed').order_by(Trade.executed_at.desc()).limit(10).all()

for t in trades:
    sym = t.symbol.name if t.symbol else '?'
    print(f"ID: {t.id}, Symbol: {sym}, Action: {t.action}, Qty: {t.quantity}, Price: {t.price}, PnL: {t.position_pnl}, Executed: {t.executed_at}, Error: {t.error_message}")


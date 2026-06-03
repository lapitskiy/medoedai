import sys
import os
sys.path.append('.')
from trading_agent.trading_agent import TradingAgent
from orm.database import get_db_session

agent = TradingAgent(symbol='BTCUSDT', account_id=5)
orders = agent.exchange.fetch_closed_orders('BTCUSDT', limit=5)
for o in orders:
    print(o.get('id'), o.get('info', {}).get('cumExecFee'))

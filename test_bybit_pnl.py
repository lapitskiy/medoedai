import sys
import os
sys.path.append('.')
from trading_agent.trading_agent import TradingAgent
import json

agent = TradingAgent(symbol='BTCUSDT', account_id=5)
try:
    pnl = agent.exchange.privateGetV5PositionClosedPnl({'category': 'linear', 'symbol': 'BTCUSDT', 'limit': 5})
    print(json.dumps(pnl['result']['list'][0], indent=2))
except Exception as e:
    print(e)

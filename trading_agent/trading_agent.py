import os
from utils.config_loader import get_config_value
import time
import logging
import json
import redis
from typing import Dict, Optional, Tuple
import ccxt
import math
import numpy as np
import torch
import re
from datetime import datetime, timedelta
from utils.trade_utils import create_trade_record, update_trade_status, create_model_prediction
from utils.db_utils import db_get_or_fetch_ohlcv
from envs.dqn_model.gym.crypto_trading_env_optimized import CryptoTradingEnvOptimized
from agents.vdqn.dqnn import DQNN
from envs.dqn_model.gym.indicators_optimized import preprocess_dataframes

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingAgent:
    def __init__(
        self,
        model_path: str = "/workspace/models/btc/ensemble-a/current/dqn_model.pth",
        direction: str = None,
        symbol: str | None = None,
        session_id: str | None = None,
        account_id: str | None = None,
    ):
        """
        Инициализация торгового агента
        
        Args:
            model_path: путь к обученной модели
        """
        self.model_path = model_path
        self.session_id = str(session_id).strip() if session_id else None
        self.account_id = str(account_id).strip() if account_id else None
        try:
            self.direction = (direction or os.getenv("AGENT_DIRECTION", "long")).strip().lower()
        except Exception:
            self.direction = "long"
        self.exchange = None
        self.model = None
        self.is_trading = False
        self.current_position = None
        self.trading_history = []
        self.last_model_prediction = None
        # Значения по умолчанию для символов (чтобы статус и цена работали до start_trading)
        self.symbols = []
        _sym = None
        try:
            _sym = (str(symbol).strip().upper() if symbol else None)
        except Exception:
            _sym = None
        self.symbol = _sym or 'BTCUSDT'
        self.base_symbol = self.symbol
        
        # Для отслеживания Q-Values
        self._last_q_values = None
        # Анти-флип: cooldown между противоположными сделками
        self._last_trade_side = None  # 'buy' | 'sell'
        self._last_trade_ts_ms = None  # timestamp последней закрытой свечи (ms)
        
        # Загружаем модель
        self._load_model()
        
        # Инициализируем биржу
        self._init_exchange()
        
        # Попытка восстановить выбранные символы из Redis и открытую позицию с биржи
        try:
            # Важно: если symbol был передан явно (например, запуск per-symbol),
            # не перетираем его значением из Redis (trading:symbols).
            if not _sym:
                self._load_last_symbols_from_redis()
        except Exception:
            pass
        try:
            self._restore_position_from_exchange()
        except Exception:
            pass

    def _trade_meta(self) -> dict:
        meta = {}
        try:
            if getattr(self, 'session_id', None):
                meta['session_id'] = str(self.session_id)
        except Exception:
            pass
        try:
            if getattr(self, 'account_id', None):
                meta['account_id'] = str(self.account_id)
        except Exception:
            pass
        try:
            meta['leverage'] = int(self._get_target_leverage())
        except Exception:
            pass
        try:
            exit_reason = str(getattr(self, '_forced_exit_reason', '') or '').strip()
            if exit_reason:
                meta['exit_reason'] = exit_reason
        except Exception:
            pass
        try:
            if getattr(self, 'model_path', None):
                meta['model_path'] = str(self.model_path)
                meta['model_family'] = (
                    'xgb'
                    if '/models/xgb/' in str(self.model_path).replace('\\', '/')
                    else 'dqn'
                )
        except Exception:
            pass
        try:
            shap_snapshot = getattr(self, '_last_xgb_shap_snapshot', None)
            if isinstance(shap_snapshot, dict) and shap_snapshot:
                meta['xgb_shap'] = shap_snapshot.get('xgb_shap')
                meta['xgb_shap_model_path'] = shap_snapshot.get('model_path')
                meta['xgb_shap_confidence'] = shap_snapshot.get('confidence')
                meta['xgb_shap_current_price'] = shap_snapshot.get('current_price')
                meta['xgb_shap_ensemble_group_id'] = shap_snapshot.get('ensemble_group_id')
        except Exception:
            pass
        return meta
    
    def _load_model(self):
        """Загрузка обученной модели (поддержка torch.save(obj) и torch.save(state_dict))"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Модель не найдена: {self.model_path}")
                return

            checkpoint = torch.load(self.model_path, map_location='cpu')

            # Если сохранён state_dict (dict), нужно создать модель и загрузить веса
            if isinstance(checkpoint, dict) and not hasattr(checkpoint, 'eval'):
                # Попытка восстановить размерности из окружения (по умолчанию)
                try:
                    temp_env = CryptoTradingEnvOptimized(symbol='BTCUSDT', timeframe='5m')
                    obs_dim = getattr(temp_env, 'observation_space_shape', None)
                    if obs_dim is None and hasattr(temp_env, 'observation_space'):
                        obs_dim = temp_env.observation_space.shape[0]
                    act_dim = 3
                except Exception:
                    # Фолбэк значения, если окружение недоступно
                    obs_dim = 100
                    act_dim = 3

                # Импортируем архитектуру сети
                try:
                    model = DQNN(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=(512, 256, 128))
                except Exception as arch_err:
                    logger.error(f"Не удалось создать архитектуру сети: {arch_err}")
                    return

                # Если в checkpoint есть вложенный ключ state_dict
                state_dict = checkpoint.get('state_dict', checkpoint)
                model.load_state_dict(state_dict, strict=False)
                self.model = model
                self.model.eval()
                logger.info(f"Загружен state_dict модели из {self.model_path}")
            else:
                # Сохранён целый объект модели
                self.model = checkpoint
                self.model.eval()
                logger.info(f"Модель загружена из {self.model_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
    
    def _init_exchange(self):
        """Инициализация подключения к бирже Bybit для деривативов"""
        try:
            api_key = None
            secret_key = None
            selected_account_id = str(getattr(self, 'account_id', '') or '').strip() or None
            if not selected_account_id:
                raise RuntimeError("account_id is required for TradingAgent")
            try:
                from utils.settings_store import ensure_settings_table, get_setting_value
                ensure_settings_table()
                api_key = get_setting_value('api', 'bybit', f'BYBIT_{selected_account_id}_API_KEY') or None
                secret_key = get_setting_value('api', 'bybit', f'BYBIT_{selected_account_id}_SECRET_KEY') or None
            except Exception:
                api_key = None
                secret_key = None
            if not api_key or not secret_key:
                raise RuntimeError(
                    f"Bybit API keys missing for selected account id={selected_account_id} "
                    f"(need BYBIT_{selected_account_id}_API_KEY and BYBIT_{selected_account_id}_SECRET_KEY)"
                )
            
            # Настраиваем для работы с деривативами (фьючерсы)
            self.exchange = ccxt.bybit({
                'apiKey': api_key,
                'secret': secret_key,
                'sandbox': False,  # True для тестового режима
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {
                    'defaultType': 'swap',  # Тип по умолчанию - свопы (фьючерсы)
                    'defaultMarginMode': 'isolated',  # Изолированная маржа
                    'defaultLeverage': 1,  # Плечо по умолчанию (без плеча)
                    'recv_window': 20000,
                    'recvWindow': 20000,
                    'adjustForTimeDifference': True,
                    'timeDifference': True,
                }
            })
            
            try:
                from utils.bybit_rate_limiter import load_markets_cached
                load_markets_cached(self.exchange)
            except Exception:
                self.exchange.load_markets()
            try:
                if hasattr(self.exchange, 'load_time_difference'):
                    self.exchange.load_time_difference()
            except Exception as te:
                logger.warning(f"Не удалось синхронизировать время с Bybit: {te}")
            
            #logger.info("Подключение к Bybit Derivatives (фьючерсы) установлено")
        except Exception as e:
            logger.error(f"Ошибка подключения к бирже: {e}")

    def _load_last_symbols_from_redis(self) -> None:
        """Загружает последний выбор символов из Redis (trading:symbols) и обновляет self.symbol/base_symbol.

        Ничего не делает при ошибках доступа к Redis.
        """
        try:
            r = redis.Redis(host='redis', port=6379, db=0, decode_responses=True, socket_connect_timeout=2)
            raw = r.get('trading:symbols')
            if not raw:
                return
            symbols = None
            try:
                symbols = json.loads(raw)
            except Exception:
                # Если по каким-то причинам строка не JSON, пробуем как одиночный символ
                symbols = [raw] if isinstance(raw, str) and raw else None
            if isinstance(symbols, list) and symbols:
                # Нормализуем символы (защита от "TON USDT", "TON/USDT", лишних пробелов)
                try:
                    from utils.cctx_utils import normalize_to_db as _normalize_to_db
                    symbols = [_normalize_to_db(str(s)) for s in symbols if s]
                except Exception:
                    try:
                        symbols = [str(s).replace('/', '').replace(' ', '').upper().strip() for s in symbols if s]
                    except Exception:
                        pass
                # Берём первый символ как основной
                self.symbols = symbols
                self.symbol = symbols[0]
                self.base_symbol = self.symbol
        except Exception as e:
            logger.debug(f"_load_last_symbols_from_redis: {e}")

    def _restore_position_from_exchange(self) -> None:
        """Пытается восстановить открытую позицию с биржи (Bybit derivatives) после перезапуска.

        Если обнаружена открытая LONG/SHORT позиция по self.symbol, заполняет self.current_position
        (type, amount, entry_price, entry_time=None, trade_number=None).
        """
        try:
            if not self.exchange:
                return
            symbol = getattr(self, 'symbol', 'BTCUSDT') or 'BTCUSDT'
            # Загружаем позиции по символу (если API поддерживает фильтр массивом)
            positions = None
            try:
                if hasattr(self.exchange, 'fetch_positions'):
                    positions = self.exchange.fetch_positions([symbol])
                    if positions:
                        try:
                            logger.info(f"[fetch_positions unified [symbol]] found={len(positions)} sample={str(positions[0])[:500]}")
                        except Exception:
                            pass
            except Exception as e:
                logger.debug(f"fetch_positions([symbol]) не удался: {e}")
                positions = None
            # Пробуем без фильтра
            if positions is None and hasattr(self.exchange, 'fetch_positions'):
                try:
                    positions = self.exchange.fetch_positions()
                    if positions:
                        try:
                            logger.info(f"[fetch_positions unified no-filter] found={len(positions)} sample={str(positions[0])[:500]}")
                        except Exception:
                            pass
                except Exception as e:
                    logger.debug(f"fetch_positions() не удался: {e}")
                    positions = []
            # Пробуем с параметрами Bybit v5 (linear/inverse и разные settleCoin)
            if (not positions) and hasattr(self.exchange, 'fetch_positions'):
                categories = ['linear', 'inverse']
                settle_coins = [None, 'USDT', 'USDC', 'BTC', 'ETH']
                for cat in categories:
                    for sc in settle_coins:
                        params = {'category': cat}
                        if sc:
                            params['settleCoin'] = sc
                        try:
                            positions = self.exchange.fetch_positions([symbol], params)
                            if positions:
                                try:
                                    logger.info(f"[fetch_positions unified params] params={params} found={len(positions)} sample={str(positions[0])[:500]}")
                                except Exception:
                                    pass
                                break
                        except Exception as e:
                            logger.debug(f"fetch_positions([symbol], {params}) не удался: {e}")
                            positions = None
                        if (not positions):
                            try:
                                positions = self.exchange.fetch_positions(params)
                                if positions:
                                    try:
                                        logger.info(f"[fetch_positions unified params no-symbol] params={params} found={len(positions)} sample={str(positions[0])[:500]}")
                                    except Exception:
                                        pass
                                    break
                            except Exception as e:
                                logger.debug(f"fetch_positions({params}) не удался: {e}")
                                positions = None
                    if positions:
                        break
            if not positions:
                logger.info("[positions] not found in unified fetch, trying raw v5 position/list fallback")
                positions = []
            # Ищем позицию по нужному символу (нормализуем представления типа BTC/USDT:USDT)
            pos = None
            def _norm_sym(s: str) -> str:
                """XRP/USDT:USDT → XRPUSDT, XRPUSDT → XRPUSDT"""
                try:
                    raw = str(s).upper()
                    # ccxt unified: 'XRP/USDT:USDT' → отсекаем settle-суффикс после ':'
                    base_part = raw.split(':')[0]
                    return ''.join(ch for ch in base_part if ch.isalnum())
                except Exception:
                    return ''.join(ch for ch in str(s).upper() if ch.isalnum())
            target_norm = _norm_sym(symbol)
            for p in positions:
                try:
                    psym = p.get('symbol') or p.get('info', {}).get('symbol') or ''
                    if psym and _norm_sym(psym) == target_norm:
                        pos = p
                        break
                except Exception:
                    continue
            if not pos:
                # Fallback: прямой вызов Bybit v5 position/list через методы ccxt
                try:
                    categories = ['linear', 'inverse']
                    settle_coins = [None, 'USDT', 'USDC', 'BTC', 'ETH']
                    for cat in categories:
                        for sc in settle_coins:
                            params = {'category': cat, 'symbol': symbol}
                            if sc:
                                params['settleCoin'] = sc
                            resp = None
                            try:
                                if hasattr(self.exchange, 'v5PrivateGetPositionList'):
                                    resp = self.exchange.v5PrivateGetPositionList(params)
                                elif hasattr(self.exchange, 'privateGetV5PositionList'):
                                    resp = self.exchange.privateGetV5PositionList(params)
                            except Exception as e:
                                logger.debug(f"bybit v5 position list {params} failed: {e}")
                                resp = None
                            if not resp:
                                continue
                            data = resp.get('result') or resp
                            items = data.get('list') if isinstance(data, dict) else None
                            try:
                                logger.info(f"[bybit v5 position/list] params={params} items={len(items) if isinstance(items, list) else 'N/A'} sample={str(items[0])[:500] if (isinstance(items, list) and items) else 'N/A'}")
                            except Exception:
                                pass
                            if not (items and isinstance(items, list)):
                                continue
                            found = False
                            for it in items:
                                try:
                                    isym = it.get('symbol') or ''
                                    if _norm_sym(isym) == target_norm:
                                        sz = it.get('size')
                                        side = (it.get('side') or '').lower()
                                        avg = it.get('avgPrice') or it.get('entryPrice')
                                        if sz is None:
                                            continue
                                        fsz = float(sz)
                                        if abs(fsz) <= 0:
                                            continue
                                        position_type = 'long' if side == 'buy' else ('short' if side == 'sell' else ('long' if fsz > 0 else 'short'))
                                        try:
                                            entry_price = float(avg) if avg is not None else self._get_current_price()
                                        except Exception:
                                            entry_price = self._get_current_price()
                                        amount = self._normalize_amount(abs(fsz))
                                        self.current_position = {
                                            'type': position_type,
                                            'amount': amount,
                                            'entry_price': entry_price,
                                            'entry_time': None,
                                            'trade_number': None
                                        }
                                        self.trade_amount = amount
                                        self.is_trading = True
                                        found = True
                                        break
                                except Exception:
                                    continue
                            if found:
                                try:
                                    logger.info(f"[position restored v5] type={self.current_position['type']} amount={self.current_position['amount']} entry_price={self.current_position['entry_price']}")
                                except Exception:
                                    pass
                                return
                except Exception as e:
                    logger.debug(f"bybit v5 position list fallback failed: {e}")
                return
            # Определяем размер позиции (в базовой валюте)
            size = None
            try:
                # ccxt unified
                size = pos.get('contracts')
            except Exception:
                size = None
            if size in (None, 0):
                try:
                    info = pos.get('info', {})
                    # Bybit v5: size бывает строкой
                    s = info.get('size') or info.get('positionQty') or info.get('positionAmt')
                    if s is not None:
                        size = float(s)
                except Exception:
                    size = None
            if not size or abs(float(size)) <= 0:
                return
            size = float(size)
            # Определяем направление
            position_type = None
            try:
                ccxt_side = pos.get('side')
                if ccxt_side in ('long', 'short'):
                    position_type = ccxt_side
            except Exception:
                pass
            if not position_type:
                try:
                    info_side = pos.get('info', {}).get('side')
                    if info_side:
                        position_type = 'long' if str(info_side).lower() == 'buy' else 'short'
                except Exception:
                    pass
            if not position_type:
                position_type = 'long' if size > 0 else 'short'
            # Цена входа
            entry = None
            try:
                entry = pos.get('entryPrice')
            except Exception:
                entry = None
            if not entry:
                try:
                    entry = pos.get('average')
                except Exception:
                    entry = None
            if not entry:
                try:
                    info = pos.get('info', {})
                    entry = info.get('avgPrice') or info.get('entryPrice') or info.get('averagePrice')
                except Exception:
                    entry = None
            try:
                entry_price = float(entry) if entry is not None else self._get_current_price()
            except Exception:
                entry_price = self._get_current_price()
            # Нормализуем количество под шаг (на стороне биржи уже точное значение, но выровняем для логики)
            amount = self._normalize_amount(abs(size))
            # Фиксируем текущую позицию
            self.current_position = {
                'type': position_type,
                'amount': amount,
                'entry_price': entry_price,
                'entry_time': None,
                'trade_number': None
            }
            # Позиция восстановлена — выставим торговую величину для статуса
            self.trade_amount = amount
            # Если позиция существует — считаем торговлю активной логически
            self.is_trading = True
        except Exception as e:
            logger.debug(f"_restore_position_from_exchange: {e}")
    
    def start_trading(self, symbols: list) -> Dict:
        """
        Запуск торговли (одноразовое выполнение)
        
        Args:
            symbols: список торговых пар
            
        Returns:
            Dict с результатом запуска
        """
        if not self.exchange:
            return {"success": False, "error": "Биржа не инициализирована"}
        
        if not self.model:
            return {"success": False, "error": "Модель не загружена"}
        
        try:
            # Для Bybit деривативов используем символы без :USDT
            self.symbols = symbols
            self.symbol = self.symbols[0] if self.symbols else 'BTCUSDT'  # Устанавливаем первый символ как основной
            
            # Базовый символ для внутренних расчетов
            self.base_symbol = self.symbol
            
            # Рассчитываем количество для торговли на основе баланса
            self.trade_amount = self._calculate_trade_amount()
            
            # Устанавливаем статус торговли как активный
            self.is_trading = True
            
            logger.info(f"Торговля запущена для {symbols}, основной символ: {self.symbol}, количество: {self.trade_amount}")
            
            # Выполняем один торговый шаг
            result = self._execute_trading_step()
            
            return {
                "success": True, 
                "message": f"Торговый шаг выполнен для {symbols}",
                "trading_result": result
            }
            
        except Exception as e:
            logger.error(f"Ошибка выполнения торгового шага: {e}")
            return {"success": False, "error": str(e)}
    
    def stop_trading(self) -> Dict:
        """Остановка торговли"""
        try:
            self.is_trading = False
            if hasattr(self, 'trading_thread'):
                self.trading_thread.join(timeout=5)
            
            logger.info("Торговля остановлена")
            return {"success": True, "message": "Торговля остановлена"}
            
        except Exception as e:
            logger.error(f"Ошибка остановки торговли: {e}")
            return {"success": False, "error": str(e)}
    
    def get_trading_status(self) -> Dict:
        """Получение статуса торговли"""
        balance_info = self.get_balance()
        current_price = self._get_current_price()
        
        # Определяем статус торговли (используем простые символы для избежания проблем с кодировкой)
        trading_status = "Активна" if self.is_trading else "Остановлена"
        trading_status_emoji = "🟢" if self.is_trading else "🔴"
        
        # Получаем информацию о символе
        symbol_info = getattr(self, 'symbol', None)
        if symbol_info:
            symbol_display = symbol_info
        else:
            symbol_display = "Не указана"
        
        # Получаем информацию о количестве
        amount_info = getattr(self, 'trade_amount', None)
        if amount_info and amount_info > 0:
            amount_display = f"{amount_info:.6f}"
        else:
            amount_display = "Не указано"
        
        return {
            "is_trading": self.is_trading,
            "trading_status": trading_status,
            "trading_status_emoji": trading_status_emoji,
            "trading_status_full": f"{trading_status_emoji} {trading_status}",
            "symbol": symbol_info,
            "symbol_display": symbol_display,
            "amount": amount_info,
            "amount_display": amount_display,
            "amount_usdt": getattr(self, 'trade_amount', 0) * current_price if current_price > 0 else 0,
            "position": self.current_position,
            "trades_count": len(self.trading_history),
            "balance": balance_info.get('balance', {}) if balance_info.get('success') else {},
            "current_price": current_price,
            "last_model_prediction": getattr(self, 'last_model_prediction', None),
            "risk_management": {
                "risk_percentage": 0.15,  # 15% от баланса
                "min_trade_usdt": 10.0,
                "max_trade_usdt": 100.0
            }
        }
    
    def get_balance(self) -> Dict:
        """Получение баланса"""
        try:
            if not self.exchange:
                return {"success": False, "error": "Биржа не инициализирована"}

            # Определяем базовую валюту текущего символа (например, XRP для XRPUSDT),
            # чтобы sizing мог использовать base_balance, если USDT недоступен.
            base_ccy = None
            try:
                sym0 = str(getattr(self, 'base_symbol', None) or getattr(self, 'symbol', None) or '').upper().strip()
                if sym0:
                    for _q in ('USDT', 'USD', 'USDC', 'BUSD', 'USDP'):
                        if sym0.endswith(_q) and len(sym0) > len(_q):
                            base_ccy = sym0[:-len(_q)]
                            break
            except Exception:
                base_ccy = None

            def _as_float(x):
                try:
                    if x is None:
                        return None
                    return float(x)
                except Exception:
                    return None

            def _extract_currency(balance_obj: dict, code: str) -> float:
                # ccxt: иногда баланс[code] = {'free','used','total'}, иногда balance['free'][code]
                try:
                    cur = balance_obj.get(code)
                    if isinstance(cur, dict):
                        # Bybit UNIFIED/ccxt может хранить доступный баланс в разных ключах.
                        # Предпочтение: free/available/availableToWithdraw, затем total (когда free отсутствует/некорректен).
                        for k in ('free', 'available', 'availableBalance', 'availableToWithdraw', 'total'):
                            v = cur.get(k)
                            if v is None:
                                continue
                            vf = _as_float(v)
                            if vf is None:
                                continue
                            # Если значение > 0 — сразу возвращаем.
                            if vf > 0:
                                return float(vf)
                        # Если все ключи есть, но нули — вернём free (или total) как есть
                        vf = _as_float(cur.get('free'))
                        if vf is not None:
                            return float(vf)
                        vt = _as_float(cur.get('total'))
                        if vt is not None:
                            return float(vt)
                    elif cur is not None:
                        v = _as_float(cur)
                        if v is not None:
                            return v
                except Exception:
                    pass
                try:
                    free_map = balance_obj.get('free') or {}
                    if isinstance(free_map, dict) and free_map.get(code) is not None:
                        v = _as_float(free_map.get(code))
                        if v is not None:
                            return v
                except Exception:
                    pass
                try:
                    total_map = balance_obj.get('total') or {}
                    if isinstance(total_map, dict) and total_map.get(code) is not None:
                        v = _as_float(total_map.get(code))
                        if v is not None:
                            return v
                except Exception:
                    pass
                # Фолбэк: разбор raw info (Bybit v5 walletBalance/availableToWithdraw)
                try:
                    info = balance_obj.get('info') or {}
                    # ожидаемый формат: info.result.list = [{coin, walletBalance, availableToWithdraw, availableBalance, ...}]
                    lst = None
                    if isinstance(info, dict):
                        res = info.get('result') or {}
                        if isinstance(res, dict):
                            lst = res.get('list')
                    if isinstance(lst, list) and lst:
                        for row in lst:
                            try:
                                if not isinstance(row, dict):
                                    continue
                                # Bybit v5: часто row['coin'] = [ {coin, walletBalance, availableToWithdraw, ...}, ... ]
                                coin_field = row.get('coin')
                                if isinstance(coin_field, list) and coin_field:
                                    for crow in coin_field:
                                        try:
                                            if not isinstance(crow, dict):
                                                continue
                                            c2 = crow.get('coin') or crow.get('currencyCoin')
                                            if not c2 or str(c2).upper() != str(code).upper():
                                                continue
                                            for k in ('availableToWithdraw', 'availableBalance', 'walletBalance', 'equity'):
                                                v = _as_float(crow.get(k))
                                                if v is not None:
                                                    return float(v)
                                        except Exception:
                                            continue
                                    continue
                                # Bybit: бывает row сам является coin-объектом
                                c = row.get('coin') or row.get('currencyCoin')
                                if not c or str(c).upper() != str(code).upper():
                                    continue
                                for k in ('availableToWithdraw', 'availableBalance', 'walletBalance', 'equity'):
                                    v = _as_float(row.get(k))
                                    if v is not None:
                                        return float(v)
                            except Exception:
                                continue
                except Exception:
                    pass
                return 0.0

            # Важно: Bybit через ccxt часто по умолчанию отдаёт spot. Нам нужен derivatives UNIFIED.
            variants = [
                {'accountType': 'UNIFIED', 'type': 'swap', 'recv_window': 20000, 'recvWindow': 20000},
                {'accountType': 'UNIFIED', 'recv_window': 20000, 'recvWindow': 20000},
                {'type': 'swap', 'recv_window': 20000, 'recvWindow': 20000},
                {'recv_window': 20000, 'recvWindow': 20000},
            ]
            balance = None
            last_e = None
            for p in variants:
                try:
                    b = self.exchange.fetch_balance(p)
                    if isinstance(b, dict) and b:
                        balance = b
                        break
                except Exception as e:
                    last_e = e
                    continue
            if balance is None:
                raise RuntimeError(f"fetch_balance failed: {last_e}")
            out = {
                "success": True,
                "balance": {
                    "USDT": _extract_currency(balance, 'USDT'),
                    "BTC": _extract_currency(balance, 'BTC')
                }
            }
            try:
                if base_ccy and base_ccy not in ('USDT', 'BTC'):
                    out["balance"][base_ccy] = _extract_currency(balance, base_ccy)  # type: ignore[index]
            except Exception:
                pass
            return out
        except Exception as e:
            logger.error(f"Ошибка получения баланса: {e}")
            return {"success": False, "error": str(e)}

    def _get_last_closed_ts_ms(self, timeframe: str = '5m') -> int:
        """Возвращает timestamp (ms) последней ЗАКРЫТОЙ свечи для заданного таймфрейма.
        Для 5m: floor(UTC now, 5m) - 5m.
        """
        try:
            now_utc = datetime.utcnow()
            epoch_sec = int(now_utc.timestamp())
            if timeframe == '5m':
                step = 300
            elif timeframe == '15m':
                step = 900
            elif timeframe in ('1h', '60m'):
                step = 3600
            else:
                step = 300
            last_closed = (epoch_sec // step) * step - step
            return last_closed * 1000
        except Exception:
            return 0

    def _get_target_leverage(self) -> int:
        try:
            if not getattr(self, 'session_id', None):
                return 1
            r = redis.Redis(host='redis', port=6379, db=0, decode_responses=True, socket_connect_timeout=2)
            from utils.trading_sessions import get_runtime_value, load_session
            
            # Читаем session_doc для фоллбека
            session_doc = load_session(r, self.session_id)
            default_leverage = 1
            if isinstance(session_doc, dict):
                default_leverage = session_doc.get('leverage', 1)
                
            raw = get_runtime_value(r, self.session_id, 'leverage', default_leverage)
            return max(1, min(5, int(float(raw or 1))))
        except Exception:
            return 1

    def _ensure_no_leverage(self, symbol: str, target_leverage: int = 1) -> bool:
        """Выставляет нужное плечо и isolated для символа."""
        ok = True
        try:
            target_leverage = max(1, min(5, int(float(target_leverage or 1))))
        except Exception:
            target_leverage = 1
        try:
            if hasattr(self.exchange, 'set_leverage'):
                try:
                    try:
                        self.exchange.set_leverage(target_leverage, symbol)
                        logger.info(f"Установлено плечо {target_leverage}x для {symbol}")
                    except Exception as inner_e:
                        try:
                            lev_s = str(target_leverage)
                            self.exchange.set_leverage(lev_s, symbol, {'buyLeverage': lev_s, 'sellLeverage': lev_s})
                            logger.info(f"Установлено плечо (buy/sell) {target_leverage}x для {symbol}")
                        except Exception as inner_e2:
                            msg = f"{inner_e} | {inner_e2}"
                            if '110043' in msg or 'leverage not modified' in msg.lower():
                                logger.info(f"Плечо уже {target_leverage}x для {symbol} (Bybit 110043)")
                            else:
                                logger.warning(f"Не удалось установить плечо {target_leverage}x для {symbol}: {msg}")
                                ok = False
                except Exception as e:
                    msg = str(e)
                    if '110043' in msg or 'leverage not modified' in msg.lower():
                        logger.info(f"Плечо уже {target_leverage}x для {symbol} (Bybit 110043)")
                    else:
                        logger.warning(f"Не удалось установить плечо {target_leverage}x для {symbol}: {e}")
                        ok = False
            # Установка режима маржи isolated (не критично для ok)
            if hasattr(self, 'exchange') and hasattr(self.exchange, 'set_margin_mode'):
                try:
                    self.exchange.set_margin_mode('isolated', symbol)
                    logger.info(f"Установлен режим маржи isolated для {symbol}")
                except Exception as e:
                    logger.warning(f"Не удалось установить режим isolated для {symbol}: {e}")
            # Дополнительно проверим текущую позицию и леверидж
            try:
                positions = None
                if hasattr(self.exchange, 'fetch_positions'):
                    positions = self.exchange.fetch_positions([symbol])
                if positions:
                    for p in positions:
                        lev = p.get('leverage') or p.get('info', {}).get('leverage')
                        if lev is not None:
                            try:
                                if float(lev) > target_leverage:
                                    logger.warning(f"У символа {symbol} активное плечо {lev} (>{target_leverage}).")
                                    ok = False
                            except Exception:
                                pass
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Ошибка при попытке выставить параметры без плеча для {symbol}: {e}")
            ok = False
        return ok

    def _extract_order_price(self, order: dict) -> float:
        """Достаёт цену исполнения из ответа биржи с надёжными фолбэками.

        Порядок приоритетов:
        - order['price'] (если есть и >0)
        - order['average'] (средняя цена)
        - order['info'].avgPrice
        - order['cost']/order['filled'] (если доступны)
        - order['info'].cumExecValue / cumExecQty (Bybit v5)
        - текущая цена с _get_current_price()
        """
        try:
            # 1) Прямые поля ccxt
            for key in ("price", "average"):
                v = order.get(key)
                if v is not None:
                    try:
                        f = float(v)
                        if f > 0:
                            return f
                    except Exception:
                        pass

            # 2) Вложенные поля info
            info = order.get("info") or {}
            for key in ("avgPrice", "lastPrice", "orderPrice"):
                v = info.get(key)
                if v is not None:
                    try:
                        f = float(v)
                        if f > 0:
                            return f
                    except Exception:
                        pass

            # 3) Стоимость / объём
            cost = order.get("cost")
            filled = order.get("filled")
            try:
                if cost is not None and filled:
                    c = float(cost)
                    q = float(filled)
                    if q > 0 and c > 0:
                        return c / q
            except Exception:
                pass

            # 4) Bybit v5 cumExecValue/cumExecQty
            cev = info.get("cumExecValue")
            ceq = info.get("cumExecQty")
            try:
                if cev is not None and ceq:
                    c = float(cev)
                    q = float(ceq)
                    if q > 0 and c > 0:
                        return c / q
            except Exception:
                pass

            # 5) Фолбэк - текущая цена
            fallback = self._get_current_price()
            return float(fallback) if fallback else 0.0
        except Exception:
            fallback = self._get_current_price()
            return float(fallback) if fallback else 0.0

    def execute_direct_order(self, action: str, symbol: Optional[str] = None, quantity: Optional[float] = None) -> Dict:
        """
        Выполняет РЕАЛЬНЫЙ рыночный ордер BUY/SELL в обход предсказаний и без записи в БД/мониторинг.
        - Не сохраняет запись в историю внутри агента
        - Не трогает self.current_position
        - Возвращает краткий результат с данными биржи
        """
        try:
            if not self.exchange:
                return {"success": False, "error": "Биржа не инициализирована"}

            order_symbol = symbol or getattr(self, 'symbol', 'BTCUSDT')
            if order_symbol is None:
                order_symbol = 'BTCUSDT'

            # Если количество не задано — рассчитать минимально допустимое по бирже (~$10)
            order_qty = quantity
            if order_qty is None or order_qty <= 0:
                # Рассчитываем минимальный возможный размер через ограничения и текущую цену
                self.symbol = order_symbol
                current_price = self._get_current_price()
                limits = self._get_bybit_limits()
                # Минимальная стоимость сделки и минимальный объем
                min_cost = max(10.0, limits.get('min_cost', 10.0))
                min_amount = limits.get('min_amount', 0.001)
                if current_price and current_price > 0:
                    calc_amount = min_cost / current_price
                    order_qty = max(min_amount, calc_amount)
                else:
                    order_qty = min_amount
                order_qty = self._normalize_amount(order_qty)

            side = action.lower()
            if side not in ('buy', 'sell'):
                return {"success": False, "error": "action должен быть 'buy' или 'sell'"}

            # Гарантируем отсутствие плеча — если не удаётся (нет свободной маржи, активное плечо и т.д.), отменяем покупку
            no_lev_ok = self._ensure_no_leverage(order_symbol)
            if side == 'buy' and not no_lev_ok:
                return {
                    "success": False,
                    "error": "Невозможно купить с 1x: активна позиция с плечом или недостаточно свободной маржи. Покупка отменена."
                }

            # Дополнительная проверка на достаточность средств при покупке
            if side == 'buy':
                try:
                    current_price = current_price if 'current_price' in locals() else self._get_current_price()
                    balance_result = self.get_balance()
                    available_usdt = 0.0
                    if balance_result.get('success'):
                        available_usdt = float(balance_result['balance'].get('USDT', 0.0) or 0.0)
                    # Учитываем комиссии и возможный слиппедж – берём 95% от доступного
                    max_affordable_qty = 0.0
                    if current_price and current_price > 0:
                        max_affordable_qty = max(0.0, (available_usdt * 0.95) / current_price)
                    # Границы по бирже
                    limits = limits if 'limits' in locals() else self._get_bybit_limits()
                    min_amount = float(limits.get('min_amount', 0.001) or 0.001)
                    # Если доступная величина меньше минимально допустимой – откажемся заранее
                    if max_affordable_qty < min_amount:
                        required_usdt = current_price * min_amount
                        return {
                            "success": False,
                            "error": f"Недостаточно USDT для минимального ордера: нужно ≈ ${required_usdt:.2f}, доступно ${available_usdt:.2f}"
                        }
                    # Иначе уменьшим количество до доступного
                    order_qty = min(order_qty, max_affordable_qty)
                    order_qty = self._normalize_amount(order_qty)
                except Exception:
                    pass

            if side == 'buy':
                order = self.exchange.create_market_buy_order(
                    order_symbol,
                    order_qty,
                    {
                        'leverage': '1',
                        'marginMode': 'isolated',
                    }
                )
            else:
                order = self.exchange.create_market_sell_order(
                    order_symbol,
                    order_qty,
                    {
                        'leverage': '1',
                        'marginMode': 'isolated',
                    }
                )

            return {
                "success": True,
                "symbol": order_symbol,
                "action": side,
                "quantity": order_qty,
                "order": order
            }
        except Exception as e:
            logger.error(f"Ошибка прямого ордера ({action}) для {symbol}: {e}")
            return {"success": False, "error": str(e)}
    
    def _calculate_trade_amount(self) -> float:
        """
        Рассчитывает количество для торговли на основе баланса и риск-менеджмента
        
        Returns:
            float: количество в базовой валюте (BTC, ETH, SOL и т.д.) для торговли
        """
        try:
            # Получаем баланс
            balance_result = self.get_balance()
            if not balance_result.get('success'):
                logger.warning("Не удалось получить баланс, используем минимальное количество")
                return 0.001  # Минимальное количество
            
            usdt_balance = balance_result['balance']['USDT']
            btc_balance = balance_result['balance']['BTC']
            
            # Получаем ограничения Bybit
            bybit_limits = self._get_bybit_limits()
            
            # Доля счёта для сделки (по умолчанию 100%)
            account_pct = 100
            
            # Пробуем получить account_pct из session_doc
            try:
                if getattr(self, 'session_id', None):
                    from utils.trading_sessions import load_session
                    r = redis.Redis(host='redis', port=6379, db=0, decode_responses=True, socket_connect_timeout=2)
                    session_doc = load_session(r, self.session_id)
                    if isinstance(session_doc, dict) and session_doc.get('account_pct') is not None:
                        _ap_session = session_doc.get('account_pct')
                        if str(_ap_session).strip() != '':
                            account_pct = max(1, min(100, int(float(str(_ap_session)))))
            except Exception:
                pass
                
            # 1) Пробуем Postgres (app_settings): scope=trading, group=sizing, key=ACCOUNT_PCT
            try:
                from utils.settings_store import get_setting_value as _get_setting_value
                _ap_db = _get_setting_value('trading', 'sizing', 'ACCOUNT_PCT')
                if _ap_db is not None and str(_ap_db).strip() != '':
                    # Если нашли в базе, но не было в сессии (или не удалось)
                    if account_pct == 100: # только если не переопределено сессией
                        account_pct = max(1, min(100, int(float(str(_ap_db)))))
            except Exception:
                pass
            # 2) Фолбэк на Redis (старое поведение)
            try:
                from redis import Redis as _Redis
                _rc = _Redis(host='redis', port=6379, db=0, decode_responses=True)
                _ap = _rc.get('trading:account_pct')
                if _ap is not None and str(_ap).strip() != '':
                    if account_pct == 100: # только если не переопределено сессией
                        account_pct = max(1, min(100, int(str(_ap))))
            except Exception:
                pass
            target_leverage = self._get_target_leverage()
            # Минимальная/максимальная стоимость из лимитов биржи
            min_trade_usdt = max(10.0, float(bybit_limits['min_cost']))
            max_trade_usdt = float(bybit_limits['max_cost']) if bybit_limits['max_cost'] else float('inf')

            # account_pct задает margin-долю, leverage увеличивает notional позиции.
            margin_usdt = float(usdt_balance) * (float(account_pct) / 100.0)
            pre_trade_usdt = margin_usdt * float(target_leverage)

            # Ограничиваем по требованиям биржи
            trade_usdt = max(min_trade_usdt, min(pre_trade_usdt, max_trade_usdt))
            try:
                logger.info(
                    f"[SIZE] symbol={self.symbol} USDT_free={usdt_balance:.4f} account_pct={account_pct}% "
                    f"leverage={target_leverage}x margin_usdt={margin_usdt:.4f} pre_usdt={pre_trade_usdt:.4f} "
                    f"clamp[min={min_trade_usdt:.2f},max={max_trade_usdt if max_trade_usdt!=float('inf') else 'inf'}] -> usdt={trade_usdt:.4f}"
                )
            except Exception:
                pass
            
            # Если USDT недостаточно, используем баланс базовой валюты
            required_margin_usdt = trade_usdt / float(target_leverage)
            if required_margin_usdt > usdt_balance:
                base_currency = self.base_symbol.replace('USDT', '').replace('USD', '')
                base_balance = balance_result['balance'].get(base_currency, 0.0)
                
                if base_balance > bybit_limits['min_amount']:
                    trade_amount = base_balance * (float(account_pct) / 100.0)
                    trade_amount = max(bybit_limits['min_amount'], min(trade_amount, bybit_limits['max_amount']))
                    logger.info(f"Используем {base_currency} баланс: {trade_amount} {base_currency} (${trade_amount * self._get_current_price():.2f})")
                    return self._normalize_amount(trade_amount)
                else:
                    logger.warning(f"Недостаточно {base_currency} для торговли")
                    return self._normalize_amount(bybit_limits['min_amount'])  # Минимальное количество для данной валюты
            
            # Конвертируем USDT в базовую валюту
            current_price = self._get_current_price()
            if current_price > 0:
                base_currency = self.symbol.replace('USDT', '').replace('USD', '')
                trade_amount = trade_usdt / current_price
                
                # Проверяем ограничения Bybit для данной валюты
                min_amount_bybit = bybit_limits['min_amount']
                max_amount_bybit = bybit_limits['max_amount']
                
                # Ограничиваем по требованиям биржи
                if trade_amount < min_amount_bybit:
                    #logger.warning(f"Количество {trade_amount:.6f} {base_currency} меньше минимума Bybit {min_amount_bybit}. Увеличиваем до минимума.")
                    trade_amount = min_amount_bybit
                    # Пересчитываем USDT для логирования
                    actual_usdt = trade_amount * current_price
                    logger.info(f"Скорректировано: {trade_amount} {base_currency} (${actual_usdt:.2f})")
                elif trade_amount > max_amount_bybit:
                    logger.warning(f"Количество {trade_amount:.6f} {base_currency} больше максимума Bybit {max_amount_bybit}. Ограничиваем.")
                    trade_amount = max_amount_bybit
                    actual_usdt = trade_amount * current_price
                    logger.info(f"Скорректировано: {trade_amount} {base_currency} (${actual_usdt:.2f})")
                else:
                    logger.info(f"Рассчитано количество: {trade_amount:.6f} {base_currency} (${trade_usdt:.2f})")
                
                return self._normalize_amount(trade_amount)
            else:
                logger.warning("Не удалось получить текущую цену, используем минимальное количество")
                return self._normalize_amount(bybit_limits['min_amount'])
                
        except Exception as e:
            logger.error(f"Ошибка расчета количества торговли: {e}")
            # Фолбэк на минимальное количество для текущей валюты
            try:
                bybit_limits = self._get_bybit_limits()
                return self._normalize_amount(bybit_limits['min_amount'])
            except:
                return 0.001  # Абсолютный фолбэк

    def _normalize_amount(self, amount: float) -> float:
        """Нормализует количество под шаг qtyStep/precision для текущего символа.

        - Берёт шаг из market.info.lotSizeFilter.qtyStep, иначе из precision.amount
        - Округляет вниз до кратного шагу
        - Приводит к границам min/max amount
        """
        try:
            market = self.exchange.market(self.symbol)
            # Получаем шаг количества
            step = None
            try:
                step_raw = (
                    market.get('info', {})
                          .get('lotSizeFilter', {})
                          .get('qtyStep')
                )
                if step_raw is not None:
                    step = float(step_raw)
            except Exception:
                step = None

            if step is None:
                precision = market.get('precision', {}).get('amount', 3)
                step = 10 ** (-precision)

            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0.0) or 0.0
            max_amount = market.get('limits', {}).get('amount', {}).get('max', float('inf')) or float('inf')

            if step <= 0:
                step = 0.001

            normalized = math.floor(max(amount, 0.0) / step) * step

            if normalized < min_amount:
                normalized = min_amount
            if normalized > max_amount:
                normalized = max_amount

            # Форматируем по количеству знаков шага
            # Вычисляем precision из шага (например, 0.001 -> 3)
            precision = max(0, -int(round(math.log10(step)))) if step < 1 else 0
            normalized = float(f"{normalized:.{precision}f}")
            return normalized
        except Exception:
            try:
                return float(f"{amount:.3f}")
            except Exception:
                return amount
    
    def _determine_sell_amount(self, current_price: float) -> dict:
        """
        ИИ уже решил продавать - определяем СКОЛЬКО продавать
        на основе risk management и текущей ситуации
        
        Args:
            current_price: Текущая цена
            
        Returns:
            dict: Стратегия продажи
        """
        try:
            if not self.current_position:
                return {
                    'sell_all': False,
                    'sell_amount': 0,
                    'keep_amount': 0,
                    'reason': 'Нет открытой позиции'
                }
            
            entry_price = self.current_position['entry_price']
            position_amount = self.current_position['amount']
            
            # Рассчитываем P&L
            pnl = (current_price - entry_price) * position_amount
            pnl_percentage = ((current_price - entry_price) / entry_price) * 100
            
            # Стратегия 1: Экстренная защита (большие убытки)
            if pnl_percentage <= -20:  # Убыток больше 20%
                return {
                    'sell_all': True,
                    'sell_amount': position_amount,
                    'keep_amount': 0,
                    'reason': f'🚨 ЗАЩИТА: убыток {pnl_percentage:.2f}% (${pnl:.2f})'
                }
            
            # Стратегия 2: Частичная защита (средние убытки)
            elif pnl_percentage <= -10:  # Убыток больше 10%
                # Продаем 50% для защиты капитала
                sell_amount = position_amount * 0.5
                keep_amount = position_amount * 0.5
                
                return {
                    'sell_all': False,
                    'sell_amount': sell_amount,
                    'keep_amount': keep_amount,
                    'reason': f'🛡️ Защита капитала: убыток {pnl_percentage:.2f}%, продаем 50%'
                }
            
            # Стратегия 3: Частичная фиксация прибыли
            elif pnl_percentage >= 15:  # Прибыль больше 15%
                # Продаем 40% для фиксации прибыли
                sell_amount = position_amount * 0.4
                keep_amount = position_amount * 0.6
                
                return {
                    'sell_all': False,
                    'sell_amount': sell_amount,
                    'keep_amount': keep_amount,
                    'reason': f'💰 Фиксация прибыли: {pnl_percentage:.2f}%, продаем 40%'
                }
            
            # Стратегия 4: Обычная продажа (по сигналу ИИ)
            else:
                # ИИ решил продавать - продаем 70% позиции
                sell_amount = position_amount * 0.7
                keep_amount = position_amount * 0.3
                
                return {
                    'sell_all': False,
                    'sell_amount': sell_amount,
                    'keep_amount': keep_amount,
                    'reason': f'🤖 ИИ сигнал SELL: продаем 70% (P&L: {pnl_percentage:.2f}%)'
                }
                
        except Exception as e:
            logger.error(f"Ошибка определения количества продажи: {e}")
            # Фолбэк: продаем все
            return {
                'sell_all': True,
                'sell_amount': self.current_position['amount'] if self.current_position else 0,
                'keep_amount': 0,
                'reason': f'Ошибка стратегии, продаем все: {str(e)}'
            }
    
    def _get_current_price(self) -> float:
        """Получает текущую цену из базы данных.

        Логика:
        - Пытаемся взять последнюю закрытую свечу по вычисленному last_closed_ts.
        - Если закрытой свечи нет (например, данные в БД ещё не подтянулись),
          берём предпоследнюю свечу из полученных данных.
        - Если доступна только одна свеча — используем её close как фолбэк.
        """
        try:
            # Сначала пробуем получить из БД
            
            symbol_for_db = getattr(self, 'base_symbol', None) or getattr(self, 'symbol', None) or 'BTCUSDT'
            df_5min = db_get_or_fetch_ohlcv(
                symbol_name=symbol_for_db,  # Используем базовый символ без :USDT для БД
                timeframe='5m',
                limit_candles=5,  # Берём несколько свечей, чтобы был фолбэк на предпоследнюю
                exchange_id='bybit'  # Используем Bybit
            )
            
            # Предпочитаем ТОЛЬКО последнюю ЗАКРЫТУЮ свечу
            last_closed_ts = self._get_last_closed_ts_ms('5m')
            if df_5min is not None and not df_5min.empty:
                df_sorted = df_5min.sort_values('timestamp')
                df_closed = df_sorted[df_sorted['timestamp'] <= last_closed_ts]
                if df_closed is not None and not df_closed.empty:
                    # Берём последнюю закрытую
                    current_price = float(df_closed['close'].iloc[-1])
                else:
                    # Фолбэк: берём предпоследнюю свечу из доступных данных
                    if len(df_sorted) >= 2:
                        current_price = float(df_sorted['close'].iloc[-2])
                        logger.debug("Нет обновлённой закрытой свечи, используем предпоследнюю для цены")
                    else:
                        # Если есть только одна свеча — используем её
                        current_price = float(df_sorted['close'].iloc[-1])
                        logger.debug("Недостаточно данных для закрытой свечи, используем последнюю доступную цену")
                # Защита: если по какой-то причине получили 0/NaN — попробуем взять цену с биржи
                try:
                    if current_price and current_price > 0:
                        return current_price
                except Exception:
                    pass
            else:
                logger.warning("Свечи из БД недоступны — пробуем цену с биржи (ccxt.fetch_ticker)")

            # Fallback: цена с биржи, если БД не дала адекватную цену
            try:
                if getattr(self, 'exchange', None):
                    sym = getattr(self, 'symbol', None) or getattr(self, 'base_symbol', None) or symbol_for_db
                    t = self.exchange.fetch_ticker(sym) or {}
                    px = t.get('last') or t.get('close') or t.get('bid') or t.get('ask')
                    if px is not None and float(px) > 0:
                        return float(px)
            except Exception as te:
                logger.warning(f"fetch_ticker fallback failed: {te}")
            return 0.0
                
        except Exception as e:
            logger.error(f"Ошибка получения цены: {e}")
            return 0.0
    
    def _get_current_balance(self) -> float:
        """Получает текущий баланс USDT"""
        try:
            balance_result = self.get_balance()
            if balance_result['balance'].get('USDT', 0.0):
                return balance_result['balance'].get('USDT', 0.0)
            return 0.0
        except Exception as e:
            logger.error(f"Ошибка получения баланса: {e}")
            return 0.0
    
    def _get_bybit_limits(self) -> dict:
        """Получает ограничения Bybit для деривативов текущего символа"""
        try:
            # Получаем информацию о рынке деривативов
            market = self.exchange.market(self.symbol)
            
            # Определяем базовую валюту (первая часть символа)
            base_currency = self.base_symbol.replace('USDT', '').replace('USD', '')
            
            # Получаем ограничения из API Bybit для деривативов
            limits = {
                'min_amount': market.get('limits', {}).get('amount', {}).get('min', 0.001),
                'max_amount': market.get('limits', {}).get('amount', {}).get('max', 1000.0),
                'min_cost': market.get('limits', {}).get('cost', {}).get('min', 10.0),
                'max_cost': market.get('limits', {}).get('cost', {}).get('max', 100000.0),
                'precision_amount': market.get('precision', {}).get('amount', 3),
                'precision_price': market.get('precision', {}).get('price', 2)
            }

            # Коалесценция None/невалидных значений к безопасным дефолтам
            try:
                if limits['min_amount'] is None or float(limits['min_amount']) <= 0:
                    limits['min_amount'] = 0.001
            except Exception:
                limits['min_amount'] = 0.001
            try:
                if limits['max_amount'] is None or float(limits['max_amount']) <= 0:
                    limits['max_amount'] = 1000.0
            except Exception:
                limits['max_amount'] = 1000.0
            try:
                if limits['min_cost'] is None or float(limits['min_cost']) <= 0:
                    limits['min_cost'] = 10.0
            except Exception:
                limits['min_cost'] = 10.0
            try:
                if limits['max_cost'] is None or float(limits['max_cost']) <= 0:
                    limits['max_cost'] = 100000.0
            except Exception:
                limits['max_cost'] = 100000.0
            
            # Если не удалось получить ограничения из API, используем известные значения для популярных пар
            if limits['min_amount'] == 0.001:  # Значение по умолчанию
                known_limits = {
                    'BTC': {'min_amount': 0.001, 'precision_amount': 3},    # 0.001 BTC
                    'ETH': {'min_amount': 0.01, 'precision_amount': 2},     # 0.01 ETH
                    'SOL': {'min_amount': 0.1, 'precision_amount': 1},      # 0.1 SOL
                    'TON': {'min_amount': 1.0, 'precision_amount': 0},      # 1 TON
                    'ADA': {'min_amount': 1.0, 'precision_amount': 0},      # 1 ADA
                    'BNB': {'min_amount': 0.01, 'precision_amount': 2},     # 0.01 BNB
                    'XRP': {'min_amount': 1.0, 'precision_amount': 0},      # 1 XRP
                }
                
                if base_currency in known_limits:
                    limits['min_amount'] = known_limits[base_currency]['min_amount']
                    limits['precision_amount'] = known_limits[base_currency]['precision_amount']
                    #logger.info(f"Используем известные ограничения для {base_currency}: {limits['min_amount']}")
            
            logger.info(f"Ограничения Bybit Derivatives для {self.symbol}: {limits}")
            return limits
            
        except Exception as e:
            logger.warning(f"Не удалось получить ограничения Bybit Derivatives для {self.symbol}, используем значения по умолчанию: {e}")
            # Возвращаем безопасные значения по умолчанию для разных валют
            base_currency = self.base_symbol.replace('USDT', '').replace('USD', '')
            
            default_limits = {
                'BTC': {'min_amount': 0.001, 'precision_amount': 3},
                'ETH': {'min_amount': 0.01, 'precision_amount': 2},
                'SOL': {'min_amount': 0.1, 'precision_amount': 1},
                'TON': {'min_amount': 1.0, 'precision_amount': 0},
                'ADA': {'min_amount': 1.0, 'precision_amount': 0},
                'BNB': {'min_amount': 0.01, 'precision_amount': 2},
                'XRP': {'min_amount': 1.0, 'precision_amount': 0},
            }
            
            limits = default_limits.get(base_currency, {'min_amount': 0.001, 'precision_amount': 3})
            
            return {
                'min_amount': limits['min_amount'],
                'max_amount': 1000.0,
                'min_cost': 10.0,
                'max_cost': 100000.0,
                'precision_amount': limits['precision_amount'],
                'precision_price': 2
            }
    
    def _execute_trading_step(self) -> Dict:
        """
        Выполняет один торговый шаг (вызывается каждые 5 минут через Celery)
        
        Returns:
            Dict с результатом торгового шага
        """
        try:
            # Получаем текущие данные из БД (эффективнее чем с биржи)
            current_price = self._get_current_price()
            
            if current_price <= 0:
                logger.error("Не удалось получить текущую цену")
                return {
                    "error": "Не удалось получить текущую цену",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Обновляем количество для торговли (каждые 10 шагов)
            if not hasattr(self, '_trade_counter'):
                self._trade_counter = 0
            self._trade_counter += 1
            
            if self._trade_counter % 10 == 0:  # Каждые 50 минут (10 * 5 минут)
                new_amount = self._calculate_trade_amount()
                if abs(new_amount - self.trade_amount) > 0.0001:  # Если изменение больше 0.0001 BTC
                    logger.info(f"Обновляем количество торговли: {self.trade_amount:.6f} -> {new_amount:.6f} BTC")
                    self.trade_amount = new_amount
            
            # Получаем предсказание от модели на основе свежих данных
            action = self._get_model_prediction()
            
            # Дополнительная проверка: если действие 'hold' из-за отрицательных Q-Values, не торгуем
            if hasattr(self, 'last_model_prediction') and self.last_model_prediction == 'hold':
                # Проверяем, было ли это решение из-за отрицательных Q-Values
                if hasattr(self, '_last_q_values') and self._last_q_values:
                    max_q_value = max(self._last_q_values)
                    if max_q_value < 0:
                        logger.warning(f"🚫 Торговля отменена: все Q-Values отрицательные ({self._last_q_values})")
                        result = {
                            "timestamp": datetime.now().isoformat(),
                            "symbol": self.symbol,
                            "price": current_price,
                            "action": "hold",
                            "trade_amount": self.trade_amount,
                            "position": self.current_position,
                            "trade_executed": "hold",
                            "reason": "negative_q_values",
                            "q_values": self._last_q_values
                        }
                        return result
            
            # Логируем текущую цену и действие
            logger.info(f"Цена {self.symbol}: ${current_price:.2f}, Действие: {action}")
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "symbol": self.symbol,
                "price": current_price,
                "action": action,
                "trade_amount": self.trade_amount,
                "position": self.current_position
            }
            
            # Проверяем условия для автоматической продажи остатка
            if self.current_position and self.current_position.get('partial_sell_strategy'):
                auto_sell_result = self._check_auto_sell_remaining()
                if auto_sell_result:
                    result["auto_sell_executed"] = auto_sell_result
            
            # Выполняем торговую операцию с учётом self.direction
            if (getattr(self, 'direction', 'long') == 'short'):
                if action == 'sell' and not self.current_position:
                    logger.info(f"🔻 Открываем шорт {self.trade_amount} по цене ${current_price:.2f}")
                    short_open = self._execute_open_short()
                    result["trade_executed"] = "short_open"
                    result["trade_details"] = short_open
                elif action == 'buy' and self.current_position and self.current_position.get('type') == 'short':
                    logger.info(f"🔺 Покрываем шорт {self.current_position['amount']} по цене ${current_price:.2f}")
                    short_cover = self._execute_cover_short()
                    result["trade_executed"] = "short_cover"
                    result["trade_details"] = short_cover
                elif action == 'sell' and self.current_position and self.current_position.get('type') == 'long':
                    sell_strategy = self._determine_sell_amount(current_price)
                    logger.info(f"🔴 ИИ сигнал SELL (закрытие long): {sell_strategy['reason']}")
                    if sell_strategy['sell_all']:
                        sell_result = self._execute_sell()
                        result["trade_executed"] = "sell_all"
                        result["trade_details"] = sell_result
                        result["sell_strategy"] = sell_strategy
                    else:
                        partial_sell_result = self._execute_partial_sell(sell_strategy['sell_amount'])
                        result["trade_executed"] = "sell_partial"
                        result["trade_details"] = partial_sell_result
                        result["sell_strategy"] = sell_strategy
                else:
                    result["trade_executed"] = "hold"
            else:
                if action == 'buy' and not self.current_position:
                    logger.info(f"🟢 Выполняем покупку {self.trade_amount} BTC по цене ${current_price:.2f}")
                    buy_result = self._execute_buy()
                    result["trade_executed"] = "buy"
                    result["trade_details"] = buy_result
                elif action == 'sell' and self.current_position:
                    # ИИ решил продавать - определяем СКОЛЬКО продавать
                    sell_strategy = self._determine_sell_amount(current_price)
                    logger.info(f"🔴 ИИ сигнал SELL: {sell_strategy['reason']}")
                    
                    if sell_strategy['sell_all']:
                        logger.info(f"🔴 Продаем ВСЕ {self.current_position['amount']} {self.base_symbol} по цене ${current_price:.2f}")
                        sell_result = self._execute_sell()
                        result["trade_executed"] = "sell_all"
                        result["trade_details"] = sell_result
                        result["sell_strategy"] = sell_strategy
                    else:
                        logger.info(f"🟡 Частичная продажа: {sell_strategy['sell_amount']} {self.base_symbol} (оставляем {sell_strategy['keep_amount']})")
                        partial_sell_result = self._execute_partial_sell(sell_strategy['sell_amount'])
                        result["trade_executed"] = "sell_partial"
                        result["trade_details"] = partial_sell_result
                        result["sell_strategy"] = sell_strategy
                elif action == 'hold':
                    if self.current_position:
                        # Показываем текущий P&L для открытой позиции
                        entry_price = self.current_position['entry_price']
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                        logger.info(f"📊 Удерживаем позицию: P&L {pnl_pct:.2f}% (${current_price:.2f} vs ${entry_price:.2f})")
                        result["position_pnl"] = pnl_pct
                    else:
                        logger.info(f"⏸️ Ожидаем сигнал для входа в позицию")
                    result["trade_executed"] = "hold"
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка в торговом шаге: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_model_prediction(self) -> str:
        """Получение предсказания от модели"""
        try:
            if not self.model:
                logger.warning("Модель не загружена, возвращаем 'hold'")
                self.last_model_prediction = 'hold'
                return 'hold'
            
            # Получаем исторические данные для формирования состояния
            state = self._prepare_state_for_model()
            if state is None:
                logger.warning("Не удалось подготовить состояние, возвращаем 'hold'")
                self.last_model_prediction = 'hold'
                return 'hold'
            
            # Получаем предсказание от модели
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Добавляем batch dimension
                q_values = self.model(state_tensor)
                action = torch.argmax(q_values, dim=1).item()
            
            # Проверяем Q-Values для принятия решения о торговле
            q_values_list = q_values[0].tolist()
            max_q_value = max(q_values_list)
            
            # Если все Q-Values отрицательные - не торгуем, возвращаем 'hold'
            if max_q_value < 0:
                logger.warning(f"Все Q-Values отрицательные: {q_values_list}. Не торгуем, возвращаем 'hold'")
                action_str = 'hold'
                # Переопределяем action для логирования
                action = 0  # hold
            else:
                # Преобразуем действие в строку только если есть положительные Q-Values
                action_map = {0: 'hold', 1: 'buy', 2: 'sell'}
                action_str = action_map.get(action, 'hold')
            
            # Сохраняем последнее предсказание для записи в БД
            self.last_model_prediction = action_str
            
            # Сохраняем Q-Values для дополнительной проверки в _execute_trading_step
            self._last_q_values = q_values_list
            
            # Рассчитываем уверенность модели
            q_values_list = q_values[0].tolist()
            max_q_value = max(q_values_list)
            min_q_value = min(q_values_list)
            
            # Уверенность = разница между лучшим и худшим действием
            confidence = ((max_q_value - min_q_value) / (abs(max_q_value) + abs(min_q_value) + 1e-8)) * 100
            
            # Логируем предсказание в БД
            try:
                
                # Определяем статус позиции
                position_status = 'open' if self.current_position else 'none'
                
                # Получаем текущую цену
                current_price = self._get_current_price()
                
                # Получаем условия рынка (технические индикаторы)
                market_conditions = self._get_market_conditions()
                
                # Создаем запись о предсказании
                create_model_prediction(
                    symbol=self.base_symbol,
                    action=action_str,
                    q_values=q_values_list,
                    current_price=current_price,
                    position_status=position_status,
                    model_path=self.model_path,
                    market_conditions=market_conditions
                )
                
                #logger.info(f"Предсказание модели записано в БД: {action_str}")
                
            except Exception as e:
                logger.warning(f"Не удалось записать предсказание в БД: {e}")
            
            # Улучшенное логирование с информацией об уверенности
            if max_q_value < 0:
                logger.warning(f"Предсказание модели: {action_str} (action={action}, q_values={q_values_list}) - ВНИМАНИЕ: Все Q-Values отрицательные! Не торгуем.")
            else:
                logger.info(f"Предсказание модели: {action_str} (action={action}, q_values={q_values_list}, уверенность: {confidence:.1f}%)")
            return action_str
            
        except Exception as e:
            logger.error(f"Ошибка получения предсказания: {e}")
            self.last_model_prediction = 'hold'
            return 'hold'
    
    def _prepare_state_for_model(self) -> Optional[np.ndarray]:
        """
        Подготавливает состояние для модели так же, как во время обучения
        Использует базу данных для получения данных, докачивает недостающие
        
        Returns:
            np.ndarray: состояние для модели или None при ошибке
        """
        try:
            # Используем ту же логику, что и в train_dqn_symbol
            
            # Получаем данные из БД, докачиваем недостающие
            df_5min = db_get_or_fetch_ohlcv(
                symbol_name=self.base_symbol,  # Используем базовый символ без :USDT для БД
                timeframe='5m',
                limit_candles=100,  # Нам нужно 100 свечей для индикаторов
                exchange_id='bybit'  # Используем Bybit
            )
            
            if df_5min is None or df_5min.empty:
                logger.warning(f"Не удалось получить данные для {self.symbol}")
                return None
            
            # Используем только ЗАКРЫТЫЕ свечи (срез по последней закрытой метке времени)
            last_closed_ts = self._get_last_closed_ts_ms('5m')
            df_5min = df_5min[df_5min['timestamp'] <= last_closed_ts]
            if df_5min is None or df_5min.empty:
                logger.warning("Нет закрытых 5m свечей для подготовки состояния")
                return None
            
            if len(df_5min) < 50:  # Минимум 50 свечей для расчета индикаторов
                logger.warning(f"Недостаточно исторических данных: {len(df_5min)} свечей")
                return None
            
            logger.info(f"Получено {len(df_5min)} свечей из БД для {self.symbol}")
            
            # Преобразуем в numpy array (только OHLCV колонки)
            ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
            df_5min_array = df_5min[ohlcv_columns].values.astype(np.float32)
            
            # Подготавливаем данные как в окружении обучения
            state = self._create_state_from_ohlcv(df_5min_array)
            return state
            
        except Exception as e:
            logger.error(f"Ошибка подготовки состояния: {e}")
            return None
    
    def _get_market_conditions(self) -> dict:
        """
        Получает текущие условия рынка (технические индикаторы)
        
        Returns:
            dict: Условия рынка
        """
        try:
            
            # Получаем последние данные для расчета индикаторов
            df_5min = db_get_or_fetch_ohlcv(
                symbol_name=self.base_symbol,
                timeframe='5m',
                limit_candles=50,
                exchange_id='bybit'
            )
            
            if df_5min is None or df_5min.empty:
                return {}
            
            # Рассчитываем простые индикаторы
            close_prices = df_5min['close'].values
            
            # RSI (упрощенный)
            if len(close_prices) >= 14:
                delta = np.diff(close_prices)
                gain = np.where(delta > 0, delta, 0)
                loss = np.where(delta < 0, -delta, 0)
                
                avg_gain = np.mean(gain[-14:])
                avg_loss = np.mean(loss[-14:])
                
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100
            else:
                rsi = 50
            
            # EMA (12 и 26)
            if len(close_prices) >= 26:
                ema12 = np.mean(close_prices[-12:])
                ema26 = np.mean(close_prices[-26:])
                ema_cross = ema12 - ema26
            else:
                ema12 = ema26 = ema_cross = 0
            
            # Текущая цена
            current_price = close_prices[-1] if len(close_prices) > 0 else 0
            
            # Изменение цены
            price_change = ((current_price - close_prices[-2]) / close_prices[-2] * 100) if len(close_prices) >= 2 else 0
            
            # Funding данные (если доступны в df)
            def _last_safe(col):
                try:
                    if col in df_5min.columns and len(df_5min[col].dropna()) > 0:
                        return float(df_5min[col].dropna().iloc[-1])
                except Exception:
                    pass
                return None
            fr = _last_safe('funding_rate')
            fr_bp = _last_safe('funding_rate_bp')
            fr_ema = _last_safe('funding_rate_ema')
            fr_change = _last_safe('funding_rate_change')
            fr_sign = _last_safe('funding_sign')
            # Приведение EMA/change к бипсам для единообразного вывода
            fr_ema_bp = (fr_ema * 10000.0) if (fr_ema is not None) else None
            fr_change_bp = (fr_change * 10000.0) if (fr_change is not None) else None
            
            return {
                'rsi': round(rsi, 2),
                'ema12': round(ema12, 6),
                'ema26': round(ema26, 6),
                'ema_cross': round(ema_cross, 6),
                'current_price': round(current_price, 6),
                'price_change_percent': round(price_change, 2),
                'candles_count': len(close_prices),
                'funding_rate_bp': round(fr_bp, 3) if fr_bp is not None else None,
                'funding_rate_ema_bp': round(fr_ema_bp, 3) if fr_ema_bp is not None else None,
                'funding_rate_change_bp': round(fr_change_bp, 3) if fr_change_bp is not None else None,
                'funding_sign': int(fr_sign) if fr_sign is not None else None,
                'funding_included': True
            }
            
        except Exception as e:
            logger.warning(f"Не удалось получить условия рынка: {e}")
            return {}
    
    def _create_state_from_ohlcv(self, df_5min: np.ndarray) -> np.ndarray:
        """
        Создает состояние из OHLCV данных, имитируя логику окружения обучения
        """
        try:
            # Конфигурация индикаторов (как в обучении)
            indicators_config = {
                'rsi': {'length': 14},
                'rsi_7': {'length': 7},
                'ema': {'lengths': [20, 50, 100, 200]},
                'ema_cross': {'pairs': [(20, 50), (100, 200)], 'include_cross_signal': True},
                'sma': {'length': 14},
                'atr': {'length': 14},
                'obv': {},
                'returns': {'periods': [1, 3, 12, 60]},
                'zscore': {'ema_length': 50, 'window': 20},
            }
            
            # Рассчитываем индикаторы
            
            # Создаем заглушки для 15m и 1h (используем 5m данные)
            df_15min = df_5min[::3]  # Каждая 3-я свеча из 5m
            df_1h = df_5min[::12]    # Каждая 12-я свеча из 5m
            
            # Подготавливаем данные
            df_5min_clean, df_15min_clean, df_1h_clean, indicators_array, individual_indicators = \
                preprocess_dataframes(df_5min, df_15min, df_1h, indicators_config)
            
            # Создаем состояние как в окружении
            lookback_window = 100  # Как в обучении
            current_step = len(df_5min_clean)
            
            if current_step < lookback_window:
                # Недостаточно данных, возвращаем нулевое состояние
                return np.zeros(100, dtype=np.float32)  # Примерный размер состояния
            
            # Берем последние lookback_window свечей
            start_idx = current_step - lookback_window
            end_idx = current_step
            
            # Получаем OHLCV данные для окна
            window_ohlcv = df_5min_clean[start_idx:end_idx]
            
            # Получаем индикаторы для окна
            window_indicators = indicators_array[start_idx:end_idx]
            
            # Нормализуем OHLCV данные
            normalized_ohlcv = self._normalize_ohlcv(window_ohlcv)
            
            # Добавляем funding-фичи при наличии в исходном DataFrame (через повторный fetch как pandas)
            funding_features_flat = np.array([], dtype=np.float32)
            try:
                df_src = db_get_or_fetch_ohlcv(self.base_symbol, '5m', limit_candles=max(100, lookback_window), exchange_id='bybit')
                if df_src is not None and not df_src.empty:
                    # Срез на то же окно
                    df_src = df_src.sort_values('timestamp')
                    if len(df_src) >= lookback_window:
                        df_src = df_src.iloc[-lookback_window:]
                        cols = [
                            ('funding_rate_bp', 50.0),   # нормируем на 50 bp в [-1,1]
                            ('funding_rate_ema', 50.0/10000.0),  # EMA в долях -> bp/50
                            ('funding_rate_change', 50.0/10000.0),
                            ('funding_sign', 1.0)
                        ]
                        feats = []
                        for col, scale in cols:
                            if col in df_src.columns:
                                v = df_src[col].astype(float).values
                                if col in ('funding_rate_ema', 'funding_rate_change'):
                                    v = v * 10000.0  # в bp
                                v = np.clip(v / 50.0, -1.0, 1.0) if col != 'funding_sign' else np.clip(v, -1.0, 1.0)
                            else:
                                v = np.zeros(lookback_window, dtype=np.float32)
                            feats.append(v.astype(np.float32))
                        if feats:
                            funding_features_flat = np.column_stack(feats).astype(np.float32).flatten()
            except Exception:
                pass
            
            # Объединяем OHLCV, индикаторы и funding-фичи
            state_features = np.concatenate([
                normalized_ohlcv.flatten(),
                window_indicators.flatten(),
                funding_features_flat
            ], axis=0)
            
            # Добавляем информацию о позиции (фактическое состояние)
            try:
                if self.current_position:
                    position_is_open = 1.0
                    position_amount = float(self.current_position.get('amount', 0.0) or 0.0)
                    limits = self._get_bybit_limits()
                    max_amount = float(limits.get('max_amount', 1000.0) or 1000.0)
                    amount_norm = max(0.0, min(1.0, position_amount / max_amount)) if max_amount > 0 else 0.0
                    position_info = np.array([position_is_open, amount_norm], dtype=np.float32)
                else:
                    position_info = np.array([0.0, 0.0], dtype=np.float32)
            except Exception:
                position_info = np.array([0.0, 0.0], dtype=np.float32)
            
            # Объединяем все в финальное состояние
            final_state = np.concatenate([state_features, position_info])
            
            # Убеждаемся, что размер правильный
            if len(final_state) > 100:
                final_state = final_state[:100]
            elif len(final_state) < 100:
                # Дополняем нулями
                padding = np.zeros(100 - len(final_state), dtype=np.float32)
                final_state = np.concatenate([final_state, padding])
            
            return final_state.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Ошибка создания состояния: {e}")
            return np.zeros(100, dtype=np.float32)
    
    def _normalize_ohlcv(self, ohlcv_data: np.ndarray) -> np.ndarray:
        """
        Нормализует OHLCV данные
        """
        try:
            # Простая нормализация: делим на максимальное значение
            max_values = np.max(ohlcv_data, axis=0)
            max_values = np.where(max_values == 0, 1, max_values)  # Избегаем деления на 0
            
            normalized = ohlcv_data / max_values
            return normalized
            
        except Exception as e:
            logger.error(f"Ошибка нормализации OHLCV: {e}")
            return ohlcv_data
    
    def _execute_buy(self) -> Dict:
        """Выполнение покупки"""
        try:
            # Анти-флип: если недавно был SELL — усиливаем пороги Q-gate для BUY
            in_cd = False
            cd_mult = 1.0
            try:
                candles = int(get_config_value('FLIP_COOLDOWN_CANDLES', '1'))
            except Exception:
                candles = 1
            if candles > 0 and self._last_trade_side == 'sell':
                try:
                    now_bucket_ts = self._get_last_closed_ts_ms('5m')
                    if isinstance(self._last_trade_ts_ms, (int, float)) and isinstance(now_bucket_ts, (int, float)):
                        required_delta = candles * 300000
                        in_cd = (now_bucket_ts - self._last_trade_ts_ms) < required_delta
                except Exception:
                    in_cd = False
            # Q-gate thresholds: per-symbol Redis > global config/env > default
            t1 = float('nan')
            t2 = float('nan')
            try:
                sym = (self.symbol or self.base_symbol or '').strip() or None
            except Exception:
                sym = None
            try:
                r2 = redis.Redis(host='redis', port=6379, db=0, decode_responses=True, socket_connect_timeout=2)
                if sym:
                    v1 = r2.get(f'trading:qgate_maxq:{sym}') or r2.get('trading:qgate_maxq')
                    v2 = r2.get(f'trading:qgate_gapq:{sym}') or r2.get('trading:qgate_gapq')
                    if v1 is not None and str(v1).strip() != '':
                        t1 = float(v1)
                    if v2 is not None and str(v2).strip() != '':
                        t2 = float(v2)
            except Exception:
                pass
            if t1 != t1:
                try:
                    t1 = float(get_config_value('QGATE_MAXQ', 'nan'))
                except Exception:
                    t1 = float('nan')
            if t2 != t2:
                try:
                    t2 = float(get_config_value('QGATE_GAPQ', 'nan'))
                except Exception:
                    t2 = float('nan')
            if t1 != t1:
                t1 = 0.500
            if t2 != t2:
                t2 = 0.440

            if in_cd:
                # Множитель порогов: FLIP_COOLDOWN_Q_MULT (по умолчанию 1.2)
                try:
                    cd_mult = float(get_config_value('FLIP_COOLDOWN_Q_MULT', '1.2'))
                except Exception:
                    cd_mult = 1.2
                if not (isinstance(cd_mult, (int, float)) and cd_mult > 0):
                    cd_mult = 1.2
                try:
                    old_t1, old_t2 = t1, t2
                    if t1 == t1 and t1 > 0:
                        t1 = t1 * cd_mult
                    if t2 == t2 and t2 > 0:
                        t2 = t2 * cd_mult
                    logger.info(f"Cooldown Q-gate boost for BUY: mult={cd_mult} (T1 {old_t1}→{t1}, T2 {old_t2}→{t2})")
                except Exception as _e:
                    logger.warning(f"Cooldown boost error: {_e}")

            # Применяем Q-gate ВСЕГДА, если есть Q-values (не зависим от last_model_prediction)
            if self._last_q_values:
                try:
                    max_q = max(self._last_q_values)
                    sorted_q = sorted(self._last_q_values, reverse=True)
                    second_q = sorted_q[1] if len(sorted_q) > 1 else None
                    gap_q = (max_q - second_q) if (max_q is not None and second_q is not None) else float('nan')

                    # Если пороги не заданы или нечисловые — не пропускаем слабые сигналы
                    if any(map(lambda v: v is None or (isinstance(v, float) and (v != v)), [max_q, gap_q, t1, t2])):
                        logger.info(f"QGate: skip BUY due to invalid thresholds/values (maxQ={max_q}, gapQ={gap_q}, T1={t1}, T2={t2})")
                        self._save_qgate_filtered_prediction('buy', max_q if max_q is not None else float('nan'), gap_q, t1, t2)
                        return {"success": False, "error": "QGate filtered (invalid thresholds/values)", "qgate": {"max_q": max_q, "gap_q": gap_q, "T1": t1, "T2": t2}}

                    # Фильтруем, если ЛИБО max_q ниже порога, ЛИБО gap_q ниже порога
                    if (max_q < t1) or (gap_q < t2):
                        logger.info(f"QGate: skip BUY (maxQ={max_q}, gapQ={gap_q}, T1={t1}, T2={t2})")
                        self._save_qgate_filtered_prediction('buy', max_q, gap_q, t1, t2)
                        return {"success": False, "error": "QGate filtered", "qgate": {"max_q": max_q, "gap_q": gap_q, "T1": t1, "T2": t2}}
                except Exception as e:
                    logger.warning(f"QGate BUY check error: {e}")
            # Получаем текущий баланс перед покупкой
            balance = self._get_current_balance()
            
            # Создаем запись о сделке в БД (используем базовый символ)
            trade_record = create_trade_record(
                symbol_name=self.base_symbol,
                action='buy',
                status='pending',
                quantity=self.trade_amount,
                price=0,  # Будет обновлено после исполнения
                model_prediction=self.last_model_prediction,
                current_balance=balance,
                error_message=json.dumps(self._trade_meta(), ensure_ascii=False),
                is_successful=False
            )
            
            # Нормализуем количество под шаг qtyStep
            amount = self._normalize_amount(self.trade_amount)

            target_leverage = self._get_target_leverage()
            if not self._ensure_no_leverage(self.symbol, target_leverage):
                return {
                    "success": False,
                    "error": f"Невозможно купить с {target_leverage}x: активна позиция с большим плечом или недостаточно свободной маржи."
                }
            order = self.exchange.create_market_buy_order(
                self.symbol,
                amount,
                {
                    'leverage': str(target_leverage),
                    'marginMode': 'isolated',
                    'recv_window': 20000,
                    'recvWindow': 20000,
                }
            )
            
            executed_price = self._extract_order_price(order)
            if not executed_price or executed_price <= 0:
                executed_price = self._get_current_price()
            # Обновляем запись о сделке
            update_trade_status(
                trade_record.trade_number,
                status='executed',
                price=executed_price,
                exchange_order_id=order.get('id'),
                error_message=json.dumps(self._trade_meta(), ensure_ascii=False),
                is_successful=True
            )
            
            self.current_position = {
                'type': 'long',
                'amount': amount,
                'entry_price': executed_price,
                'entry_time': datetime.now(),
                'trade_number': trade_record.trade_number
            }
            
            self.trading_history.append({
                'action': 'buy',
                'price': executed_price,
                'amount': amount,
                'time': datetime.now(),
                'trade_number': trade_record.trade_number
            })
            
            logger.info(f"Покупка выполнена: {order}, Trade #: {trade_record.trade_number}")
            
            # Успешно: фиксируем анти-флип метки
            try:
                self._last_trade_side = 'buy'
                self._last_trade_ts_ms = self._get_last_closed_ts_ms('5m')
            except Exception:
                pass

            self._trigger_copy_trade(action="entry", position_type="long", entry_price=executed_price)

            return {
                "success": True,
                "order": order,
                "position": self.current_position,
                "trade_number": trade_record.trade_number
            }
            
        except Exception as e:
            logger.error(f"Ошибка покупки: {e}")
            
            # Обновляем запись о сделке с ошибкой
            if 'trade_record' in locals():
                update_trade_status(
                    trade_record.trade_number,
                    status='failed',
                    error_message=str(e),
                    is_successful=False
                )
            # Фиксируем причину в карточке предсказания, чтобы отобразить пользователю
            try:
                error_text = str(e)
                err_lower = error_text.lower()
                insufficient = isinstance(e, ccxt.InsufficientFunds) or '110007' in error_text or 'not enough' in err_lower
                if insufficient:
                    reason = "Не выполнено: недостаточно средств для новой заявки"
                    extra = {
                        'qgate_error': 'insufficient_funds',
                        'qgate_error_details': error_text,
                        'qgate_side': 'buy'
                    }
                    try:
                        if 'balance' in locals() and isinstance(balance, (int, float)):
                            extra['qgate_available_balance'] = float(balance)
                    except Exception:
                        pass
                    try:
                        current_price = self._get_current_price()
                        required_cost = float(self.trade_amount) * float(current_price)
                        extra['qgate_required_cost'] = required_cost
                    except Exception:
                        pass
                    self._save_qgate_filtered_prediction(
                        'buy',
                        float('nan'),
                        float('nan'),
                        float('nan'),
                        float('nan'),
                        reason=reason,
                        extra_market_conditions=extra
                    )
            except Exception:
                pass
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def _save_qgate_filtered_prediction(
        self,
        action: str,
        max_q: float,
        gap_q: float,
        t1: float,
        t2: float,
        reason: Optional[str] = None,
        extra_market_conditions: Optional[dict] = None
    ):
        """Сохраняет предсказание с информацией о Q-gate фильтрации"""
        try:
            
            # Создаем предсказание с пометкой о Q-gate фильтрации
            market_conditions = self._get_market_conditions()
            market_conditions['qgate_filtered'] = True
            if reason:
                market_conditions['qgate_reason'] = reason
            else:
                market_conditions['qgate_reason'] = f"maxQ={max_q:.3f}<{t1:.3f} или gapQ={gap_q:.3f}<{t2:.3f}"
            # Подробности для UI
            market_conditions['qgate_side'] = action
            market_conditions['qgate_T1'] = float(t1) if t1 is not None else None
            market_conditions['qgate_T2'] = float(t2) if t2 is not None else None
            market_conditions['qgate_maxQ'] = float(max_q) if max_q is not None else None
            market_conditions['qgate_gapQ'] = float(gap_q) if gap_q is not None else None
            if isinstance(extra_market_conditions, dict):
                try:
                    for key, value in extra_market_conditions.items():
                        market_conditions[key] = value
                except Exception:
                    pass
            
            create_model_prediction(
                symbol=self.symbol,
                action=action,
                q_values=self._last_q_values or [0, 0, 0],
                current_price=self._get_current_price(),
                position_status='none',
                model_path=getattr(self, 'model_path', 'unknown'),
                market_conditions=market_conditions
            )
            logger.info(f"Сохранено Q-gate отфильтрованное предсказание: {action} - {market_conditions['qgate_reason']}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения Q-gate предсказания: {e}")

    def _save_cooldown_filtered_prediction(self, action: str, reason: str, cooldown_candles: int):
        """Сохраняет предсказание с информацией о фильтрации по cooldown."""
        try:
            market_conditions = self._get_market_conditions()
            market_conditions['cooldown_filtered'] = True
            market_conditions['cooldown_reason'] = reason
            market_conditions['cooldown_candles'] = int(cooldown_candles)

            create_model_prediction(
                symbol=self.symbol,
                action=action,
                q_values=self._last_q_values or [0, 0, 0],
                current_price=self._get_current_price(),
                position_status='none',
                model_path=getattr(self, 'model_path', 'unknown'),
                market_conditions=market_conditions
            )
            logger.info(f"Сохранено cooldown-отфильтрованное предсказание: {action} - {reason}")
        except Exception as e:
            logger.error(f"Ошибка сохранения cooldown предсказания: {e}")

    def _execute_sell(self) -> Dict:
        """Выполнение продажи"""
        try:
            # Q-gate по порогам из окружения (опционально)
            try:
                t1 = float(get_config_value('QGATE_SELL_MAXQ', 'nan'))
                t2 = float(get_config_value('QGATE_SELL_GAPQ', 'nan'))
            except Exception:
                t1 = float('nan'); t2 = float('nan')

            # Применяем Q-gate ВСЕГДА, если есть Q-values (не зависим от last_model_prediction)
            if self._last_q_values:
                try:
                    max_q = max(self._last_q_values)
                    sorted_q = sorted(self._last_q_values, reverse=True)
                    second_q = sorted_q[1] if len(sorted_q) > 1 else None
                    gap_q = (max_q - second_q) if (max_q is not None and second_q is not None) else float('nan')

                    if any(map(lambda v: v is None or (isinstance(v, float) and (v != v)), [max_q, gap_q, t1, t2])):
                        logger.info(f"QGate: skip SELL due to invalid thresholds/values (maxQ={max_q}, gapQ={gap_q}, T1={t1}, T2={t2})")
                        self._save_qgate_filtered_prediction('sell', max_q if max_q is not None else float('nan'), gap_q, t1, t2)
                        return {"success": False, "error": "QGate filtered (invalid thresholds/values)", "qgate": {"max_q": max_q, "gap_q": gap_q, "T1": t1, "T2": t2}}

                    if (max_q < t1) or (gap_q < t2):
                        logger.info(f"QGate: skip SELL (maxQ={max_q}, gapQ={gap_q}, T1={t1}, T2={t2})")
                        self._save_qgate_filtered_prediction('sell', max_q, gap_q, t1, t2)
                        return {"success": False, "error": "QGate filtered", "qgate": {"max_q": max_q, "gap_q": gap_q, "T1": t1, "T2": t2}}
                except Exception as e:
                    logger.warning(f"QGate SELL check error: {e}")
            # Получаем текущий баланс перед продажей
            balance = self._get_current_balance()
            
            # Создаем запись о сделке в БД (используем базовый символ)
            trade_record = create_trade_record(
                symbol_name=self.base_symbol,
                action='sell',
                status='pending',
                quantity=self.current_position['amount'],
                price=0,  # Будет обновлено после исполнения
                model_prediction=self.last_model_prediction,
                current_balance=balance,
                error_message=json.dumps(self._trade_meta(), ensure_ascii=False),
                is_successful=False
            )
            
            # Количество к продаже (нормализуем к шагу)
            amount = self._normalize_amount(self.current_position['amount'])

            # Выполняем продажу фьючерса (закрытие long позиции)
            order = self.exchange.create_market_sell_order(
                self.symbol,
                amount,
                {
                    'reduceOnly': True,
                    'leverage': '1',
                    'marginMode': 'isolated',
                    'recv_window': 20000,
                    'recvWindow': 20000,
                }
            )
            
            # Расчет P&L
            exit_price = self._extract_order_price(order)
            if not exit_price or exit_price <= 0:
                exit_price = self._get_current_price()
            entry_price = self.current_position['entry_price']
            pnl = (exit_price - entry_price) * amount
            
            # Обновляем запись о сделке
            update_trade_status(
                trade_record.trade_number,
                status='executed',
                price=exit_price,
                exchange_order_id=order.get('id'),
                position_pnl=pnl,
                error_message=json.dumps(self._trade_meta(), ensure_ascii=False),
                is_successful=True
            )
            
            self.trading_history.append({
                'action': 'sell',
                'price': exit_price,
                'amount': amount,
                'time': datetime.now(),
                'pnl': pnl,
                'trade_number': trade_record.trade_number
            })
            
            logger.info(f"Продажа выполнена: {order}, P&L: {pnl}, Trade #: {trade_record.trade_number}")
            
            old_position = self.current_position
            self.current_position = None

            # Успешно: фиксируем анти-флип метки
            try:
                self._last_trade_side = 'sell'
                self._last_trade_ts_ms = self._get_last_closed_ts_ms('5m')
            except Exception:
                pass

            self._trigger_copy_trade(action="exit", position_type="long", entry_price=exit_price)

            return {
                "success": True,
                "order": order,
                "pnl": pnl,
                "closed_position": old_position,
                "trade_number": trade_record.trade_number
            }
            
        except Exception as e:
            logger.error(f"Ошибка продажи: {e}")
            
            # Обновляем запись о сделке с ошибкой
            if 'trade_record' in locals():
                update_trade_status(
                    trade_record.trade_number,
                    status='failed',
                    error_message=str(e),
                    is_successful=False
                )
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def _execute_open_short(self) -> Dict:
        """Открытие шорт-позиции (market SELL, reduceOnly=False)."""
        try:
            # QGate аналогично BUY — используем SELL пороги, если заданы
            try:
                t1 = float(get_config_value('QGATE_SELL_MAXQ', 'nan'))
                t2 = float(get_config_value('QGATE_SELL_GAPQ', 'nan'))
            except Exception:
                t1 = float('nan'); t2 = float('nan')

            if self._last_q_values:
                try:
                    max_q = max(self._last_q_values)
                    sorted_q = sorted(self._last_q_values, reverse=True)
                    second_q = sorted_q[1] if len(sorted_q) > 1 else None
                    gap_q = (max_q - second_q) if (max_q is not None and second_q is not None) else float('nan')
                    if (max_q != max_q) or (gap_q != gap_q) or (t1 != t1) or (t2 != t2):
                        self._save_qgate_filtered_prediction('sell', max_q if max_q is not None else float('nan'), gap_q, t1, t2)
                        return {"success": False, "error": "QGate filtered (invalid thresholds/values)"}
                    if (max_q < t1) or (gap_q < t2):
                        self._save_qgate_filtered_prediction('sell', max_q, gap_q, t1, t2)
                        return {"success": False, "error": "QGate filtered"}
                except Exception as e:
                    logger.warning(f"QGate SHORT-OPEN check error: {e}")

            balance = self._get_current_balance()
            trade_record = create_trade_record(
                symbol_name=self.base_symbol,
                action='sell',  # открываем шорт продажей
                status='pending',
                quantity=self.trade_amount,
                price=0,
                model_prediction=self.last_model_prediction,
                current_balance=balance,
                error_message=json.dumps(self._trade_meta(), ensure_ascii=False),
                is_successful=False
            )

            amount = self._normalize_amount(self.trade_amount)

            target_leverage = self._get_target_leverage()
            if not self._ensure_no_leverage(self.symbol, target_leverage):
                return {"success": False, "error": f"Невозможно открыть шорт с {target_leverage}x"}

            order = self.exchange.create_market_sell_order(
                self.symbol,
                amount,
                {
                    'leverage': str(target_leverage),
                    'marginMode': 'isolated',
                    'recv_window': 20000,
                    'recvWindow': 20000,
                }
            )

            executed_price = self._extract_order_price(order) or self._get_current_price()
            update_trade_status(
                trade_record.trade_number,
                status='executed',
                price=executed_price,
                exchange_order_id=order.get('id'),
                error_message=json.dumps(self._trade_meta(), ensure_ascii=False),
                is_successful=True
            )

            self.current_position = {
                'type': 'short',
                'amount': amount,
                'entry_price': executed_price,
                'entry_time': datetime.now(),
                'trade_number': trade_record.trade_number
            }
            self.trading_history.append({
                'action': 'sell',
                'price': executed_price,
                'amount': amount,
                'time': datetime.now(),
                'trade_number': trade_record.trade_number
            })

            try:
                self._last_trade_side = 'sell'
                self._last_trade_ts_ms = self._get_last_closed_ts_ms('5m')
            except Exception:
                pass

            self._trigger_copy_trade(action="entry", position_type="short", entry_price=executed_price)

            return {
                "success": True,
                "order": order,
                "position": self.current_position,
                "trade_number": trade_record.trade_number
            }
        except Exception as e:
            logger.error(f"Ошибка открытия шорта: {e}")
            if 'trade_record' in locals():
                update_trade_status(
                    trade_record.trade_number,
                    status='failed',
                    error_message=str(e),
                    is_successful=False
                )
            return {"success": False, "error": str(e)}

    def _execute_cover_short(self) -> Dict:
        """Закрытие шорт-позиции (market BUY, reduceOnly=True)."""
        try:
            if not self.current_position or self.current_position.get('type') != 'short':
                return {"success": False, "error": "Нет шорт-позиции для закрытия"}

            balance = self._get_current_balance()
            amount = self._normalize_amount(self.current_position['amount'])

            trade_record = create_trade_record(
                symbol_name=self.base_symbol,
                action='buy',  # покрываем шорт покупкой
                status='pending',
                quantity=amount,
                price=0,
                model_prediction=self.last_model_prediction,
                current_balance=balance,
                error_message=json.dumps(self._trade_meta(), ensure_ascii=False),
                is_successful=False
            )

            order = self.exchange.create_market_buy_order(
                self.symbol,
                amount,
                {
                    'reduceOnly': True,
                    'leverage': '1',
                    'marginMode': 'isolated',
                    'recv_window': 20000,
                    'recvWindow': 20000,
                }
            )

            exit_price = self._extract_order_price(order) or self._get_current_price()
            entry_price = self.current_position['entry_price']
            pnl = (entry_price - exit_price) * amount

            update_trade_status(
                trade_record.trade_number,
                status='executed',
                price=exit_price,
                exchange_order_id=order.get('id'),
                position_pnl=pnl,
                error_message=json.dumps(self._trade_meta(), ensure_ascii=False),
                is_successful=True
            )

            self.trading_history.append({
                'action': 'buy',
                'price': exit_price,
                'amount': amount,
                'time': datetime.now(),
                'pnl': pnl,
                'trade_number': trade_record.trade_number
            })

            old_position = self.current_position
            self.current_position = None

            try:
                self._last_trade_side = 'buy'
                self._last_trade_ts_ms = self._get_last_closed_ts_ms('5m')
            except Exception:
                pass

            self._trigger_copy_trade(action="exit", position_type="short", entry_price=exit_price)

            return {
                "success": True,
                "order": order,
                "pnl": pnl,
                "closed_position": old_position,
                "trade_number": trade_record.trade_number
            }
        except Exception as e:
            logger.error(f"Ошибка закрытия шорта: {e}")
            if 'trade_record' in locals():
                update_trade_status(
                    trade_record.trade_number,
                    status='failed',
                    error_message=str(e),
                    is_successful=False
                )
            return {"success": False, "error": str(e)}
    def _check_auto_sell_remaining(self) -> Optional[Dict]:
        """
        Проверяет условия для автоматической продажи оставшейся части позиции
        
        Returns:
            Dict: Результат автоматической продажи или None
        """
        try:
            if not self.current_position or not self.current_position.get('partial_sell_strategy'):
                return None
            
            strategy = self.current_position['partial_sell_strategy']
            current_price = self._get_current_price()
            
            if current_price <= 0:
                return None
            
            # Рассчитываем P&L для оставшейся части
            entry_price = self.current_position['entry_price']
            position_amount = self.current_position['amount']
            pnl_percentage = ((current_price - entry_price) / entry_price) * 100
            
            should_sell = False
            reason = ""
            
            # Проверяем условия продажи
            
            # 1. Убыток превысил порог
            if pnl_percentage <= strategy['sell_threshold']:
                should_sell = True
                reason = f"🚨 Авто-продажа остатка: убыток {pnl_percentage:.2f}% превысил порог {strategy['sell_threshold']}%"
            
            # 2. Прибыль превысила порог
            elif pnl_percentage >= strategy['profit_threshold']:
                should_sell = True
                reason = f"💰 Авто-продажа остатка: прибыль {pnl_percentage:.2f}% превысила порог {strategy['profit_threshold']}%"
            
            # 3. Время истекло
            elif datetime.now() >= strategy['time_threshold']:
                should_sell = True
                reason = f"⏰ Авто-продажа остатка: истекло время ожидания (24 часа)"
            
            if should_sell:
                logger.info(f"🔴 {reason}")
                
                # Продаем весь остаток
                sell_result = self._execute_sell()
                
                # Очищаем стратегию
                if 'partial_sell_strategy' in self.current_position:
                    del self.current_position['partial_sell_strategy']
                
                return {
                    "success": True,
                    "reason": reason,
                    "sell_result": sell_result,
                    "pnl_percentage": pnl_percentage
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка проверки авто-продажи остатка: {e}")
            return None
    
    def _execute_partial_sell(self, sell_amount: float) -> Dict:
        """Выполнение частичной продажи"""
        try:
            if not self.current_position or sell_amount <= 0:
                return {
                    "success": False,
                    "error": "Нет позиции или неверное количество для продажи"
                }
            
            # Проверяем, что продаем не больше чем есть
            if sell_amount > self.current_position['amount']:
                sell_amount = self.current_position['amount']
            
            # Нормализуем количество к шагу qtyStep
            sell_amount = self._normalize_amount(sell_amount)
            
            # Получаем текущий баланс перед продажей
            balance = self._get_current_balance()
            
            # Создаем запись о сделке в БД
            trade_record = create_trade_record(
                symbol_name=self.base_symbol,
                action='sell_partial',
                status='pending',
                quantity=sell_amount,
                price=0,  # Будет обновлено после исполнения
                model_prediction=self.last_model_prediction,
                current_balance=balance,
                error_message=json.dumps(self._trade_meta(), ensure_ascii=False),
                is_successful=False
            )
            
            # Выполняем частичную продажу фьючерса
            order = self.exchange.create_market_sell_order(
                self.symbol,
                sell_amount,
                {
                    'reduceOnly': True,
                    'leverage': '1',
                    'marginMode': 'isolated',
                    'recv_window': 20000,
                    'recvWindow': 20000,
                }
            )
            
            # Расчет P&L для проданной части
            exit_price = self._extract_order_price(order)
            if not exit_price or exit_price <= 0:
                exit_price = self._get_current_price()
            entry_price = self.current_position['entry_price']
            pnl = (exit_price - entry_price) * sell_amount
            
            # Обновляем запись о сделке
            update_trade_status(
                trade_record.trade_number,
                status='executed',
                price=exit_price,
                exchange_order_id=order.get('id'),
                position_pnl=pnl,
                error_message=json.dumps(self._trade_meta(), ensure_ascii=False),
                is_successful=True
            )
            
            self.trading_history.append({
                'action': 'sell_partial',
                'price': exit_price,
                'amount': sell_amount,
                'time': datetime.now(),
                'pnl': pnl,
                'trade_number': trade_record.trade_number
            })
            
            # Обновляем текущую позицию (уменьшаем количество)
            remaining_amount = self.current_position['amount'] - sell_amount
            
            if remaining_amount <= 0.0001:  # Если осталось очень мало - закрываем позицию
                logger.info(f"Частичная продажа: закрываем позицию полностью (осталось: {remaining_amount})")
                self.current_position = None
            else:
                # Пересчитываем среднюю цену входа для оставшейся позиции
                total_cost = self.current_position['entry_price'] * self.current_position['amount']
                sold_cost = exit_price * sell_amount
                remaining_cost = total_cost - sold_cost
                new_entry_price = remaining_cost / remaining_amount
                
                # Добавляем стратегию для оставшейся части
                self.current_position['amount'] = remaining_amount
                self.current_position['entry_price'] = new_entry_price
                self.current_position['partial_sell_strategy'] = {
                    'type': 'remaining_position',
                    'sell_threshold': -5.0,  # Продаем остаток при убытке > 5%
                    'profit_threshold': 8.0,  # Продаем остаток при прибыли > 8%
                    'time_threshold': datetime.now() + timedelta(hours=24),  # Максимум 24 часа
                    'created_at': datetime.now()
                }
                
                logger.info(f"Частичная продажа: обновлена позиция - количество: {remaining_amount}, цена входа: {new_entry_price:.6f}")
                logger.info(f"Стратегия для остатка: убыток > {self.current_position['partial_sell_strategy']['sell_threshold']}% или прибыль > {self.current_position['partial_sell_strategy']['profit_threshold']}%")
            
            logger.info(f"Частичная продажа выполнена: {sell_amount} по цене {exit_price}, P&L: {pnl}, Trade #: {trade_record.trade_number}")
            
            return {
                "success": True,
                "order": order,
                "pnl": pnl,
                "sold_amount": sell_amount,
                "remaining_position": self.current_position,
                "trade_number": trade_record.trade_number
            }
            
        except Exception as e:
            logger.error(f"Ошибка частичной продажи: {e}")
            
            # Обновляем запись о сделке с ошибкой
            if 'trade_record' in locals():
                update_trade_status(
                    trade_record.trade_number,
                    status='failed',
                    error_message=str(e),
                    is_successful=False
                )
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def _trigger_copy_trade(self, action: str, position_type: str, entry_price: float = None):
        try:
            from tasks.celery_task_copy_trade import copy_trade_for_clients
            copy_trade_for_clients.apply_async(
                kwargs={
                    "session_id": str(self.session_id),
                    "symbol": self.symbol,
                    "action": action,
                    "position_type": position_type,
                    "entry_price": entry_price
                },
                queue="celery"
            )
        except Exception as e:
            logger.warning(f"Failed to trigger copy trade: {e}")

    def get_trading_history(self) -> Dict:
        """Получение истории торговли"""
        return {
            "success": True,
            "trades": self.trading_history,
            "total_trades": len(self.trading_history)
        }

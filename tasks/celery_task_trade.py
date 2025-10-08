import time
import pandas as pd
import json
import requests
from redis import Redis
import numpy as np
from utils.db_utils import db_get_or_fetch_ohlcv # Импортируем функцию загрузки данных
from utils.db_utils import load_latest_candles_from_csv_to_db
from utils.parser import parser_download_and_combine_with_library
from utils.trade_utils import create_model_prediction
import os
from datetime import datetime
import uuid
from utils.config_loader import get_config_value
import logging
import traceback
from utils.redis_utils import get_redis_client
from tasks import celery

logger = logging.getLogger(__name__)

# Определяем очереди и маршрутизацию задач:
# По умолчанию все задачи идут в очередь 'celery',
# а тренировочные задачи направляем в отдельную очередь 'train'.
# Определяем очереди и маршрутизацию задач:
# По умолчанию все задачи идут в очередь 'celery',
# а тренировочные задачи направляем в отдельную очередь 'train'.


@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0}, queue='trade')
def execute_trade(self, symbols: list, model_path: str | None = None, model_paths: list | None = None):
    """Исполнение торгового шага: предсказание через serving, торговля через TradingAgent."""
    try:
        from trading_agent.trading_agent import TradingAgent
        from utils.db_utils import db_get_or_fetch_ohlcv

        # 1) Читаем параметры из Redis при необходимости
        try:
            rc = Redis(host='redis', port=6379, db=0, decode_responses=True)
        except Exception:
            rc = None

        if (not symbols) and rc is not None:
            try:
                # Если общий список пуст, берем из статусов активных агентов
                all_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'TONUSDT', 'ADAUSDT', 'BNBUSDT', 'XRPUSDT']
                for sym in all_symbols:
                    status_key = f'trading:status:{sym}'
                    if rc.get(status_key):
                        symbols = [sym]
                        break
            except Exception as e:
                logger.error(f"Ошибка при поиске активных символов: {e}")
                return {
                    "success": False, 
                    "skipped": True, 
                    "reason": "symbol_search_error", 
                    "error": f"Не удалось найти активные символы: {e}"
                }

        if not symbols or not isinstance(symbols, list):
            logger.error("Не переданы символы для торговли или переданы в неверном формате")
            return {
                "success": False, 
                "skipped": True, 
                "reason": "symbols_invalid", 
                "error": "Символы должны быть переданы в виде непустого списка"
            }
        
        syms = symbols

        symbol = syms[0]
        
        # Если не передали model_paths аргументом — читаем из Redis для конкретного символа
        if model_paths is None and rc is not None:
            try:
                # Читаем модели для конкретного символа
                _mps = rc.get(f'trading:model_paths:{symbol}')
                if _mps:
                    parsed = json.loads(_mps)
                    if isinstance(parsed, list) and parsed:
                        model_paths = parsed
                else:
                    # Фолбэк на общие модели
                    _mps = rc.get('trading:model_paths')
                    if _mps:
                        parsed = json.loads(_mps)
                        if isinstance(parsed, list) and parsed:
                            model_paths = parsed
            except Exception as e:
                logger.error(f"Ошибка при получении путей моделей для символа {symbol}: {e}")
                return {
                    "success": False, 
                    "skipped": True, 
                    "reason": "model_paths_error", 
                    "error": f"Не удалось получить пути моделей для символа {symbol}: {e}"
                }
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
            return {"success": False, "error": "model_paths not provided"}

        # 2) Готовим состояние для serving (как в агенте: закрытые 5m свечи -> плотный вектор)
        # Последняя закрытая метка времени (5m)
        def _last_closed_ts_ms():
            try:
                now_utc = datetime.utcnow().timestamp()
                last_closed = (int(now_utc) // 300) * 300 - 300
                return last_closed * 1000
            except Exception:
                return 0

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
        df_5m, data_error = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe='5m', limit_candles=limit_candles, exchange_id='bybit', include_error=True)
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
                        market_conditions='{"error": "' + str(human_error) + '"}',
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
                        market_conditions='{"error": "' + str(error_msg) + '"}',
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
                        market_conditions='{"error": "' + str(error_msg) + '"}',
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
                        market_conditions='{"error": "' + str(error_msg) + '"}',
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

        # 3) Вызов serving (+ передаём настройки консенсуса, если заданы)
        # Читаем консенсус из Redis для конкретного символа: {'counts': {flat, trend, total_selected}, 'percents': {flat, trend}}
        consensus_cfg = None
        try:
            if rc is not None:
                # Сначала пробуем для конкретного символа
                _c = rc.get(f'trading:consensus:{symbol}')
                if _c:
                    consensus_cfg = json.loads(_c)
                # Не используем больше глобальный фолбэк, чтобы BNB не перетирал BTC
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
        try:
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
            # Сохраняем ошибку в предсказания
            try:
                from orm.models import ModelPrediction
                from orm.database import get_db_session
                with get_db_session() as session:
                    prediction = ModelPrediction(
                        symbol=symbols[0],
                        action='error',
                        confidence=0.0,
                        q_values='[]',
                        current_price=0.0,
                        position_status='none',
                        model_path='error',
                        market_conditions='{"error": "' + str(error_msg) + '"}',
                        created_at=datetime.utcnow()
                    )
                    session.add(prediction)
                    session.commit()
            except Exception as e:
                print(f"Failed to save error prediction: {e}")
            return {"success": False, "error": error_msg}

        # Подготовим пороги Q-gate: приоритет ENV > JSON > дефолт
        qgate_cfg = pred_json.get('qgate') or {}

        def _pick_threshold(config_key: str, default_val: float) -> float:
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

        T1 = _pick_threshold('QGATE_MAXQ', 1.11)
        T2 = _pick_threshold('QGATE_GAPQ', 1.11)
        print(f"QGate thresholds chosen: T1={T1:.3f}, T2={T2:.3f}")

        try:
            flat_factor = float(get_config_value('QGATE_FLAT', '1.0'))
        except Exception:
            flat_factor = 1.0
        if market_regime == 'flat' and flat_factor and flat_factor != 1.0:
            T1 *= flat_factor
            T2 *= flat_factor

        eff_T1 = T1
        eff_T2 = T2

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
                    gate_ok = False
                    try:
                        if isinstance(qv, list) and len(qv) >= 3:
                            if act == 'buy':
                                q_buy = float(qv[1])
                                other = max(float(qv[0]), float(qv[2]))
                                gate_ok = (q_buy >= eff_T1) and ((q_buy - other) >= eff_T2)
                            elif act == 'sell':
                                q_sell = float(qv[2])
                                other = max(float(qv[0]), float(qv[1]))
                                gate_ok = (q_sell >= eff_T1) and ((q_sell - other) >= eff_T2)
                            else:
                                gate_ok = True # HOLD не блокируем Q‑gate
                        else:
                            gate_ok = (act == 'hold')
                    except Exception:
                        gate_ok = (act == 'hold')
                    if act in ('buy','sell'):
                        if gate_ok:
                            votes[act] += 1
                    elif act == 'hold':
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
                # Фолбэки: если counts не заданы — фиксируем требование без процентов
                # По умолчанию: 2 для total>=3; для 2 моделей — 2; для 1 — 1
                default_req = 2 if total_sel >= 3 else max(1, total_sel)
                if req_flat is None:
                    req_flat = default_req
                if req_trend is None:
                    req_trend = default_req
                # Ограничим максимумом total_sel
                req_flat = int(min(max(1, req_flat), total_sel))
                req_trend = int(min(max(1, req_trend), total_sel))
                required_type = 'trend' if market_regime in ('uptrend','downtrend') else 'flat'
                required = req_trend if required_type == 'trend' else req_flat
                # Правило консенсуса: если хватает голосов BUY → buy, если SELL → sell; иначе hold
                if votes['buy'] >= required and votes['buy'] > votes['sell']:
                    decision = 'buy'
                    consensus_from_qgate = True
                elif votes['sell'] >= required and votes['sell'] > votes['buy']:
                    decision = 'sell'
                    consensus_from_qgate = True
                else:
                    decision = 'hold'
        except Exception:
            pass
        # --- Server-side Q-gate ---
        try:
            # Если решение уже получено через консенсус моделей, прошедших Q‑gate, не блокируем финальным агрегатом
            if not consensus_from_qgate:
                q_values = pred_json.get('q_values')
                if not isinstance(q_values, list):
                    preds_list_tmp = pred_json.get('predictions') or []
                    if preds_list_tmp:
                        q_values = preds_list_tmp[0].get('q_values')
                if isinstance(q_values, list) and len(q_values) >= 2:
                    q_sorted = sorted([float(x) for x in q_values], reverse=True)
                    maxQ = q_sorted[0]
                    gapQ = q_sorted[0] - q_sorted[1]
                    passed = (maxQ >= eff_T1) and (gapQ >= eff_T2)
                    if not passed:
                        decision = 'hold'
                    try:
                        print(f"Q‑gate: {'PASS' if passed else 'BLOCK'} (maxQ={maxQ:.3f}, gapQ={gapQ:.3f}, T1={eff_T1:.3f}, T2={eff_T2:.3f})")
                    except Exception:
                        pass
        except Exception:
            pass

        # 4) Торговля через TradingAgent (без docker exec)
        agent = TradingAgent(model_path=(model_paths[0] if model_paths else None))
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

        # 4.1) Сохраняем предсказания в БД (по каждому пути модели) + сводка консенсуса и per‑model Q‑gate
        try:
            # Текущая цена: возьмём close последней закрытой свечи
            try:
                current_price = float(df_5m['close'].iloc[-1]) if (df_5m is not None and not df_5m.empty) else None
            except Exception:
                current_price = None
            position_status = 'open' if getattr(agent, 'current_position', None) else 'none'
            preds_list = pred_json.get('predictions') or []
            # Общий ID группы ансамбля для этого тика, чтобы фронт мог объединять карточки
            ensemble_group_id = str(uuid.uuid4()) if preds_list else None
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
                                q_max = qb; q_gap = qb - other; q_pass = (qb >= T1) and (q_gap >= T2)
                            elif str(act or 'hold').lower() == 'sell':
                                qs = float(qv[2]); other = max(float(qv[0]), float(qv[1]))
                                q_max = qs; q_gap = qs - other; q_pass = (qs >= T1) and (q_gap >= T2)
                            else:
                                q_pass = True
                    except Exception:
                        q_pass = (str(act or 'hold').lower() == 'hold')
                    mc = {
                        'ensemble_total': int(total_sel),
                        'ensemble_votes_buy': int(votes.get('buy', 0)),
                        'ensemble_votes_sell': int(votes.get('sell', 0)),
                        'ensemble_votes_hold': int(votes.get('hold', 0)),
                        'ensemble_required': int(required) if required is not None else None,
                        'ensemble_required_type': required_type,
                        'ensemble_regime': market_regime,
                        'ensemble_decision': decision,
                        'ensemble_group_id': ensemble_group_id,
                        # Детали режима по окнам, чтобы UI мог показать 60=F,180=U,300=D
                        'regime_windows': list(market_regime_details.get('windows', [])) if isinstance(market_regime_details, dict) else [],
                        'regime_labels': list(market_regime_details.get('labels', [])) if isinstance(market_regime_details, dict) else [],
                        'regime_weights': list(market_regime_details.get('weights', [])) if isinstance(market_regime_details, dict) else [],
                        'regime_voting': (market_regime_details.get('voting') if isinstance(market_regime_details, dict) else None),
                        'regime_tie_break': (market_regime_details.get('tie_break') if isinstance(market_regime_details, dict) else None),
                        'qgate_T1': float(T1),
                        'qgate_T2': float(T2),
                        'qgate_maxQ': float(q_max) if q_max is not None else None,
                        'qgate_gapQ': float(q_gap) if q_gap is not None else None,
                        'qgate_filtered': (False if q_pass is None else (not q_pass)),
                    }
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
        except Exception:
            pass

        trade_result = None
        if decision == 'buy' and not agent.current_position:
            trade_result = agent._execute_buy()
        elif decision == 'sell' and agent.current_position:
            sell_strategy = agent._determine_sell_amount(agent._get_current_price())
            trade_result = agent._execute_sell() if sell_strategy.get('sell_all') else agent._execute_partial_sell(sell_strategy.get('sell_amount', 0))
        else:
            trade_result = {"success": True, "action": "hold"}

        status_after = agent.get_trading_status()

        # 5) Сохранение результата в Redis (как раньше)
        try:
            if rc is not None:
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
                }
                rc.setex(f'trading:latest_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}', 3600, json.dumps(result_data, default=str))
                # Сохраняем статус для конкретного символа
                symbol = syms[0] if syms else 'ALL'
                rc.set(f'trading:status:{symbol}', json.dumps(status_after, default=str))
                # НЕ перезаписываем общий статус, чтобы не убить другие агенты
                # rc.set('trading:current_status', json.dumps(status_after, default=str))
                rc.set('trading:current_status_ts', datetime.utcnow().isoformat())
        except Exception:
            pass

        return {
            "success": True,
            "decision": decision,
            "status_before": current_status_before,
            "status_after": status_after,
            "trade_result": trade_result,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0}, queue='trade')
def start_trading_task(self, symbols, model_path=None):
    """
    Оркестратор: получает параметры, делает Redis-лок, публикует provisional-статус
    и кладёт торговую задачу в очередь trade (без docker exec).
    """
    import os
    from datetime import datetime

    def _is_trade_worker_alive() -> bool:
        try:
            insp = celery.control.inspect()
            if not insp:
                return False
            active_queues = insp.active_queues() or {}
            for queues in active_queues.values():
                if not queues:
                    continue
                for q in queues:
                    if q.get('name') == 'trade':
                        return True
            return False
        except Exception as e:
            print(f"[start_trading_task] инспекция очередей Celery не удалась: {e}")
            # Если не удалось проверить — не блокируем запуск
            return True

    def _mark_trade_worker_down(sym_list, reason_msg):
        try:
            rc = get_redis_client()
            if not isinstance(sym_list, list) or not sym_list:
                logger.error("Не переданы символы для обработки статуса торгового воркера")
                return {
                    "success": False, 
                    "error": "Не указаны символы для обработки статуса торгового воркера"
                }
            
            status_payload = {
                'success': False,
                'is_trading': False,
                'trading_status': 'Ошибка',
                'trading_status_emoji': '🔴',
                'trading_status_full': '🔴 Ошибка: воркер celery-trade не запущен',
                'reason': reason_msg,
                'last_error': reason_msg,
                'timestamp': datetime.utcnow().isoformat()
            }
            import json as _json
            for sym in sym_list:
                sym_u = str(sym).upper()
                payload = dict(status_payload)
                payload['symbol'] = sym_u
                payload['symbol_display'] = sym_u
                rc.set(f'trading:status:{sym_u}', _json.dumps(payload, ensure_ascii=False))
            
            # Обновим глобальный статус, чтобы /api/trading/status тоже сообщал об ошибке
            try:
                rc.set('trading:current_status', _json.dumps(payload, ensure_ascii=False))
                rc.set('trading:current_status_ts', datetime.utcnow().isoformat())
            except Exception:
                pass
        except Exception as err:
            print(f"[start_trading_task] Не удалось записать статус об ошибке celery-trade: {err}")

    # Проверяем, должна ли работать торговля
    trading_enabled = str(get_config_value('ENABLE_TRADING_BEAT', '0')).lower() in ('1', 'true', 'yes', 'on')
    if not trading_enabled:
        return {"success": False, "skipped": True, "reason": "ENABLE_TRADING_BEAT=0"}

    # Проверяем, что воркер с очередью trade жив; иначе сообщаем в UI и прекращаем
    if not _is_trade_worker_alive():
        reason_msg = 'Воркер celery-trade не запущен (очередь trade недоступна). Перезапустите celery-trade.'
        _mark_trade_worker_down(symbols, reason_msg)
        return {
            "success": False,
            "skipped": True,
            "reason": "celery_trade_unavailable",
            "error": reason_msg
        }

    # Определяем символы для торговли
    final_symbols = symbols
    if not final_symbols:
        from redis import Redis
        import json
        rc = Redis(host='redis', port=6379, db=0, decode_responses=True)
        symbols_raw = rc.get('trading:symbols')
        logger.warning(f"DEBUG: start_trading_task received empty symbols, trying to get from Redis. symbols_raw = {symbols_raw}")

        if not symbols_raw:
            logger.warning("DEBUG: No symbols found in trading:symbols from start_trading_task")
            return {"success": False, "skipped": True, "reason": "no_symbols_in_redis", "error": "Не удалось найти символы для торговли в Redis"}
        
        try:
            parsed_symbols = json.loads(symbols_raw)
            if not isinstance(parsed_symbols, list):
                raise ValueError("Символы из Redis не являются списком")
            final_symbols = parsed_symbols
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"DEBUG: JSON/Value Error in start_trading_task: {e}")
            return {"success": False, "skipped": True, "reason": "symbols_parsing_error", "error": "Ошибка парсинга символов из Redis"}

        if not final_symbols:
            logger.warning("DEBUG: Parsed symbols list is empty from start_trading_task")
            return {"success": False, "skipped": True, "reason": "empty_symbols_list", "error": "Список символов из Redis пуст"}
        
        logger.warning(f"DEBUG: Using symbols from Redis: {final_symbols}")

    # Определяем lock_symbol из final_symbols
    lock_symbol = final_symbols[0] if final_symbols else None
    if not lock_symbol:
        logger.error(f"Не удалось определить lock_symbol для торговли. Итоговые символы: {final_symbols}")
        return {"success": False, "skipped": True, "reason": "symbol_undefined", "error": "Не удалось определить символ для торговли"}

    # Redis-лок: предотвращаем параллельные запуски в пределах ~5 минут (per-symbol)
    _rc_lock = None # Инициализируем здесь, чтобы было доступно в finally
    lock_acquired_by_this_task = False # Флаг для отслеживания получения блокировки
    try:
        from redis import Redis as _Redis
        _rc_lock = _Redis(host='redis', port=6379, db=0, decode_responses=True)
        
        lock_key = f'trading:agent_lock:{lock_symbol}'
        # TTL 600с (10 минут) — чтобы агент не терял lock во время работы
        got_lock = _rc_lock.set(lock_key, self.request.id, nx=True, ex=600)
        if not got_lock:
            return {"success": False, "skipped": True, "reason": "agent_lock_active"}
        lock_acquired_by_this_task = True # Блокировка успешно получена

        symbols = final_symbols # Обновляем переданные символы, чтобы остальной код использовал их
        
        print(f"🚀 Оркестрация торговли: symbols={symbols} | model_path={model_path if model_path else 'default'}")
        self.update_state(state="IN_PROGRESS", meta={"progress": 0})

        # Получаем список путей моделей для конкретного символа
        model_paths = None
        try:
            from redis import Redis as _Redis
            _r2 = _Redis(host='redis', port=6379, db=0, decode_responses=True)
            
            # Определяем символ для чтения моделей
            if not symbols:
                logger.error("Не переданы символы для торговли")
                return {
                    "success": False, 
                    "skipped": True, 
                    "reason": "symbols_not_provided", 
                    "error": "Не указаны символы для торговли"
                }
            
            symbol = symbols[0]
            
            # Читаем модели для конкретного символа
            _mps = _r2.get(f'trading:model_paths:{symbol}')
            if _mps:
                import json as _json
                parsed = _json.loads(_mps)
                if isinstance(parsed, list) and parsed:
                    model_paths = parsed
            
            # Фолбэк: если не нашли для символа — пробуем общие
            if (model_paths is None or not model_paths):
                _mps = _r2.get('trading:model_paths')
                if _mps:
                    import json as _json
                    parsed = _json.loads(_mps)
                    if isinstance(parsed, list) and parsed:
                        model_paths = parsed
            
            # Фолбэк: если не нашли актуальные пути — попробуем последние сохранённые
            if (model_paths is None or not model_paths):
                _last = _r2.get('trading:last_model_paths')
                if _last:
                    import json as _json
                    parsed_last = _json.loads(_last)
                    if isinstance(parsed_last, list) and parsed_last:
                        model_paths = parsed_last
        except Exception:
            model_paths = None
        if (model_paths is None or not model_paths) and model_path:
            model_paths = [model_path]
        # Если нашли список — синхронизируем model_path для совместимости
        if (not model_path) and isinstance(model_paths, list) and len(model_paths) > 0:
            model_path = model_paths[0]

        # Если моделей нет — пробуем использовать дефолтную модель
        if not model_paths:
            # Fallback: используем дефолтную модель если Redis пуст после рестарта
            default_model_path = '/workspace/models/btc/ensemble-a/current/dqn_model.pth'
            try:
                import os
                if os.path.exists(default_model_path):
                    model_paths = [default_model_path]
                    model_path = default_model_path
                    print(f"🔄 Используем дефолтную модель: {default_model_path}")
                else:
                    # Снять лок, чтобы не блокировать следующий запуск
                    if _rc_lock:
                        _rc_lock.delete(lock_key)
                    return {"success": False, "skipped": True, "reason": "no_model_paths"}
            except Exception:
                # Снять лок, чтобы не блокировать следующий запуск
                if _rc_lock:
                    _rc_lock.delete(lock_key)
                return {"success": False, "skipped": True, "reason": "no_model_paths"}

        # Промежуточный статус в Redis для UI
        try:
            from redis import Redis as _Redis
            import json as _json
            # Если нет путей моделей — помечаем как остановлена с причиной
            has_models = bool(model_paths and isinstance(model_paths, list) and len(model_paths) > 0)
            provisional = {
                'success': True,
                'is_trading': bool(has_models),
                'trading_status': ('Активна' if has_models else 'Остановлена'),
                'trading_status_emoji': ('🟢' if has_models else '🔴'),
                'trading_status_full': ('🟢 Активна' if has_models else '🔴 Остановлена (нет моделей)'),
                'symbol': (symbols[0] if symbols else None),
                'symbol_display': (symbols[0] if symbols else 'Не указана'),
                'amount': None,
                'amount_display': 'Не указано',
                'amount_usdt': 0.0,
                'position': None,
                'trades_count': 0,
                'balance': {}, 
                'current_price': 0.0,
                'last_model_prediction': None,
            }
            _rc = _Redis(host='redis', port=6379, db=0, decode_responses=True)
            # Сохраняем статус для конкретного символа
            symbol = symbols[0] if symbols else 'ALL'
            _rc.set(f'trading:status:{symbol}', _json.dumps(provisional, ensure_ascii=False))
            # НЕ перезаписываем общий статус, чтобы не убить другие агенты
            # _rc.set('trading:current_status', _json.dumps(provisional, ensure_ascii=False))
            from datetime import datetime as _dt
            _rc.set('trading:current_status_ts', _dt.utcnow().isoformat())
            # Обновим last_* для фолбэка следующего тика
            try:
                if has_models:
                    _rc.set('trading:last_model_paths', _json.dumps(model_paths, ensure_ascii=False))
                    # Если использовали дефолтную модель - сохраняем её как основную
                    if not _rc.get('trading:model_paths'):
                        _rc.set('trading:model_paths', _json.dumps(model_paths, ensure_ascii=False))
                # last_consensus больше не ведём глобально, чтобы не перетирать per‑symbol
                # Мерджим trading:symbols: добавляем текущие, не дублируем
                try:
                    existing_raw = _rc.get('trading:symbols')
                    existing = _json.loads(existing_raw) if existing_raw else []
                    if not isinstance(existing, list):
                        existing = []
                except Exception:
                    existing = []
                merged = []
                try:
                    for s in (existing + (symbols or [])):
                        if s and (s not in merged):
                            merged.append(s)
                except Exception:
                    merged = (symbols or [])
                _rc.set('trading:symbols', _json.dumps(merged, ensure_ascii=False))
            except Exception:
                pass
        except Exception:
            pass

        # Кладём торговые задачи в очередь trade
        try:
            task_ids = []
            # Если несколько символов — создаём отдельную задачу на каждый, чтобы все получали предсказания
            for _sym in symbols:
                # ВАЖНО: специализированный воркер слушает очередь 'trade' (см. docker-compose). Не менять на 'celery-trade'.
                res = execute_trade.apply_async(kwargs={
                    'symbols': [_sym],
                    'model_path': None, # пусть подберётся по Redis/дефолтам внутри execute_trade
                    'model_paths': None,
                }, queue='trade')
                task_ids.append(res.id)
            return {"success": True, "enqueued": True, "task_ids": task_ids}
        except Exception as e:
            return {"success": False, "error": str(e)}

    finally:
        # Гарантируем снятие блокировки в Redis, если она была успешно получена этой задачей
        if _rc_lock and lock_symbol and lock_acquired_by_this_task:
            lock_key = f'trading:agent_lock:{lock_symbol}'
            _rc_lock.delete(lock_key)
            logger.warning(f"DEBUG: Redis lock for {lock_symbol} released by this task.")


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

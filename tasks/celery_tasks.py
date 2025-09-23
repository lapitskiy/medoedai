from agents.vdqn.v_train_model import train_model
from agents.vdqn.v_train_model_optimized import train_model_optimized
from celery import Celery
from kombu import Queue
import time
import os

import pandas as pd

import json
import requests
from redis import Redis
import numpy as np

from utils.db_utils import db_get_or_fetch_ohlcv  # Импортируем функцию загрузки данных
from utils.db_utils import load_latest_candles_from_csv_to_db
from utils.parser import parser_download_and_combine_with_library
from utils.trade_utils import create_model_prediction
# Загружаем переменные окружения из .env (если есть), чтобы Celery видел ключи
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass
from datetime import datetime
import uuid
from celery.schedules import crontab
from utils.seed import set_global_seed

# API ключи Bybit (поддержка новых имён BYBIT_<N>_*)
def _discover_bybit_api_keys() -> tuple[str | None, str | None]:
    try:
        ak = os.getenv('BYBIT_1_API_KEY')
        sk = os.getenv('BYBIT_1_SECRET_KEY')
        if ak and sk:
            return ak, sk
        # Автоскан: BYBIT_<ID>_API_KEY
        candidates = []
        for k, v in os.environ.items():
            if not k.startswith('BYBIT_') or not k.endswith('_API_KEY'):
                continue
            idx = k[len('BYBIT_'):-len('_API_KEY')]
            sec_name = f'BYBIT_{idx}_SECRET_KEY'
            sec_val = os.getenv(sec_name)
            if v and sec_val:
                candidates.append((k, v, sec_name, sec_val))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1], candidates[0][3]
        return None, None
    except Exception:
        return None, None

BYBIT_API_KEY, BYBIT_SECRET_KEY = _discover_bybit_api_keys()

def are_bybit_keys_configured() -> bool:
    try:
        return bool(BYBIT_API_KEY) and bool(BYBIT_SECRET_KEY)
    except Exception:
        return False

# Настраиваем Celery с Redis как брокером и бекендом
celery = Celery(
    "tasks",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0"
)

# Настройки для автоматической очистки результатов задач
celery.conf.result_expires = 3600 * 24 * 7  # 7 дней TTL для результатов задач
celery.conf.result_backend_transport_options = {
    'master_name': 'mymaster',
    'visibility_timeout': 3600 * 24 * 7,  # 7 дней
}

# Определяем очереди и маршрутизацию задач:
# По умолчанию все задачи идут в очередь 'celery',
# а тренировочные задачи направляем в отдельную очередь 'train'.
celery.conf.task_queues = (
    Queue('celery'),
    Queue('train'),
    Queue('trade'),
)
celery.conf.task_default_queue = 'celery'
celery.conf.task_routes = {
    'tasks.celery_tasks.train_dqn': {'queue': 'train'},
    'tasks.celery_tasks.train_dqn_symbol': {'queue': 'train'},
    'tasks.celery_tasks.train_dqn_multi_crypto': {'queue': 'train'},
    'tasks.celery_tasks.execute_trade': {'queue': 'trade'},
}

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
                # Пробуем взять из общего списка символов
                _sym_raw = rc.get('trading:symbols')
                if _sym_raw:
                    _sym = json.loads(_sym_raw)
                    if isinstance(_sym, list) and _sym:
                        symbols = _sym
                else:
                    # Если общий список пуст, берем из статусов активных агентов
                    all_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'TONUSDT', 'ADAUSDT', 'BNBUSDT', 'XRPUSDT']
                    for sym in all_symbols:
                        status_key = f'trading:status:{sym}'
                        if rc.get(status_key):
                            symbols = [sym]
                            break
            except Exception:
                pass
        if not symbols:
            symbols = ['BTCUSDT']

        # Если не передали model_paths аргументом — читаем из Redis для конкретного символа
        if model_paths is None and rc is not None:
            try:
                symbol = symbols[0] if symbols else 'BTCUSDT'
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
            except Exception:
                model_paths = None
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
        symbol = symbols[0]
        # Последняя закрытая метка времени (5m)
        def _last_closed_ts_ms():
            try:
                now_utc = datetime.utcnow().timestamp()
                last_closed = (int(now_utc) // 300) * 300 - 300
                return last_closed * 1000
            except Exception:
                return 0

        df_5m = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe='5m', limit_candles=120, exchange_id='bybit')
        if df_5m is None or df_5m.empty:
            return {"success": False, "error": f"no candles in DB for {symbol}"}
        cutoff = _last_closed_ts_ms()
        df_5m = df_5m[df_5m['timestamp'] <= cutoff]
        if df_5m is None or df_5m.empty:
            return {"success": False, "error": "no closed candles available"}
        # Простая нормализация: последние 100 строк OHLCV
        ohlcv_cols = ['open','high','low','close','volume']
        arr = df_5m[ohlcv_cols].tail(100).values.astype('float32')
        if arr.shape[0] < 20:
            return {"success": False, "error": "insufficient data for state"}
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
                windows = (cfg.get('windows') if isinstance(cfg, dict) else None) or [60, 180, 300]
                weights = (cfg.get('weights') if isinstance(cfg, dict) else None) or [1, 1, 1]
                voting = (cfg.get('voting') if isinstance(cfg, dict) else None) or 'majority'
                tie_break = (cfg.get('tie_break') if isinstance(cfg, dict) else None) or 'flat'
                drift_thr = float((cfg.get('drift_threshold') if isinstance(cfg, dict) else 0.002) or 0.002)
                vol_flat_thr = float((cfg.get('flat_vol_threshold') if isinstance(cfg, dict) else 0.0025) or 0.0025)
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
                    'windows': [60, 180, 300],
                    'weights': [1, 1, 1],
                    'voting': 'majority',
                    'tie_break': 'flat',
                    'labels': ['flat', 'flat', 'flat'],
                    'votes_map': {'flat': 0.0, 'uptrend': 0.0, 'downtrend': 0.0},
                    'drift_threshold': 0.002,
                    'flat_vol_threshold': 0.0025,
                    'metrics': []
                }

        market_regime, market_regime_details = _compute_regime(df_5m)

        # 3) Вызов serving (+ передаём настройки консенсуса, если заданы)
        # Читаем консенсус из Redis для конкретного символа: {'counts': {flat, trend, total_selected}, 'percents': {flat, trend}}
        consensus_cfg = None
        try:
            if rc is not None:
                # Определяем символ для чтения консенсуса
                symbol = symbols[0] if symbols else 'BTCUSDT'
                # Сначала пробуем для конкретного символа
                _c = rc.get(f'trading:consensus:{symbol}')
                if _c:
                    consensus_cfg = json.loads(_c)
                # Не используем больше глобальный фолбэк, чтобы BNB не перетирал BTC
        except Exception:
            consensus_cfg = None

        serving_url = os.environ.get('SERVING_URL', 'http://serving:8000/predict_ensemble')
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
            return {"success": False, "error": pred_json.get('error', 'serving failed')}

        # Подготовим пороги Q-gate один раз (если serving их не передал — дефолт)
        qgate_cfg = pred_json.get('qgate') or {}
        try:
            T1 = float(qgate_cfg.get('T1', pred_json.get('qgate_T1', 0.35)))
        except Exception:
            T1 = 0.35
        try:
            T2 = float(qgate_cfg.get('T2', pred_json.get('qgate_T2', 0.25)))
        except Exception:
            T2 = 0.25
        # Смягчение/усиление порогов для flat через переменную окружения QGATE_FLAT (например, 1.2)
        try:
            if market_regime == 'flat':
                import os as _os
                factor_flat = float(_os.environ.get('QGATE_FLAT', '1.0') or '1.0')
                if factor_flat != 1.0:
                    T1 *= factor_flat
                    T2 *= factor_flat
        except Exception:
            pass
        # Фактор ужесточения для флэта (env QGATE_FLAT, по умолчанию 1.0)
        try:
            flat_factor = float(os.environ.get('QGATE_FLAT', '1.0'))
        except Exception:
            flat_factor = 1.0
        eff_T1 = T1 * (flat_factor if (market_regime == 'flat') else 1.0)
        eff_T2 = T2 * (flat_factor if (market_regime == 'flat') else 1.0)

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
                                gate_ok = True  # HOLD не блокируем Q‑gate
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
        agent.symbols = symbols
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
                        'qgate_T1': float(eff_T1),
                        'qgate_T2': float(eff_T2),
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
                    'symbols': symbols,
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
                symbol = symbols[0] if symbols else 'ALL'
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

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0})
def search_lstm_task(self, query):
    """Фоновая задача, которая выполняется долго"""
    self.update_state(state="IN_PROGRESS", meta={"progress": 0})

    for i in range(5):  # Имитация долгого вычисления
        time.sleep(2)
        self.update_state(state="IN_PROGRESS", meta={"progress": (i + 1) * 20})

    return {"message": "Task completed!", "query": query}

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0}, queue='train')
def train_dqn(self, seed: int | None = None):
    
    self.update_state(state="IN_PROGRESS", meta={"progress": 0})
    
    # Устанавливаем сид, если задан аргументом
    seed = int(seed) if seed is not None else None
    if seed is not None:
        set_global_seed(seed)
        print(f"🔒 Seed установлен: {seed}")
        # ENV больше не используем для сидов

    print("🚀 Начинаю загрузку данных для мультивалютного обучения...")
    
    # Список всех криптовалют для обучения
    crypto_symbols = [
        'BTCUSDT',  # Биткоин
        'TONUSDT',  # TON
        'ETHUSDT',  # Эфириум
        'SOLUSDT',  # Solana
        'ADAUSDT',  # Cardano
        'BNBUSDT',  # Binance Coin
        'XMRUSDT',  # Monero
        'XRPUSDT'   # Ripple
    ]
    
    all_dfs = {}
    
    for symbol in crypto_symbols:
        try:
            print(f"📥 Загружаю {symbol}...")
            
            # Загружаем данные из базы
            df_5min = db_get_or_fetch_ohlcv(
                symbol_name=symbol, 
                timeframe='5m', 
                limit_candles=100000,
                exchange_id='bybit'
            )
            
            if df_5min is not None and not df_5min.empty:
                print(f"  ✅ {symbol}: {len(df_5min)} свечей загружено")
                
                # Подготавливаем данные для этого символа
                df_5min['datetime'] = pd.to_datetime(df_5min['timestamp'], unit='ms')
                df_5min.set_index('datetime', inplace=True)
                
                # Создаем 15-минутные и 1-часовые данные
                df_15min = df_5min.resample('15min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                }).dropna().reset_index()
                
                df_1h = df_5min.resample('1h').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                }).dropna().reset_index()
                
                # Сохраняем в общий словарь
                all_dfs[symbol] = {
                    'df_5min': df_5min,
                    'df_15min': df_15min,
                    'df_1h': df_1h,
                    'symbol': symbol,
                    'candle_count': len(df_5min)
                }
                
            else:
                print(f"  ⚠️ {symbol}: данные не найдены, пропускаем")
                
        except Exception as e:
            print(f"  ❌ {symbol}: ошибка загрузки - {e}")
            continue
    
    if not all_dfs:
        print("❌ Не удалось загрузить данные ни для одной криптовалюты")
        return {"message": "Ошибка: данные не загружены"}
    
    print(f"\n📈 Успешно загружено {len(all_dfs)} криптовалют")
    
    # Проверяем количество свечей
    for symbol, data in all_dfs.items():
        print(f"  • {symbol}: {data['candle_count']} свечей")
    
    # Используем первую криптовалюту для совместимости с текущим кодом
    # В будущем можно будет переключиться на мультивалютное обучение
    first_symbol = list(all_dfs.keys())[0]
    df = {
        'df_5min': all_dfs[first_symbol]['df_5min'],
        'df_15min': all_dfs[first_symbol]['df_15min'],
        'df_1h': all_dfs[first_symbol]['df_1h']
    }
    
    # Выводим первые значения каждого df в формате JSON
    for key, value in df.items():
        records = value[:2].copy()
        if 'timestamp' in records.columns:
            records['timestamp'] = records['timestamp'].astype(str)
        else:
            for col in records.columns:
                if records[col].dtype.name == 'datetime64[ns]':
                    records[col] = records[col].astype(str)
        print(f"{key}: {json.dumps(records.to_dict(orient='records'), ensure_ascii=False, indent=2)}")
    
    print(f"\n🎯 Запуск обучения на {first_symbol}...")
    
    # Получаем количество эпизодов из переменной окружения
    episodes = int(os.getenv('DEFAULT_EPISODES', 10000))
    print(f"🎯 Количество эпизодов: {episodes}")
    
    result = train_model_optimized(dfs=df, episodes=episodes, seed=seed)
    return {"message": result}

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0}, queue='train')
def train_dqn_symbol(self, symbol: str, episodes: int = None, seed: int | None = None):
    """Обучение DQN для одного символа (BTCUSDT/ETHUSDT/...)

    Загружает данные из БД, готовит 5m/15m/1h, запускает train_model_optimized.
    """
    self.update_state(state="IN_PROGRESS", meta={"progress": 0, "symbol": symbol})

    try:
        # Сид до любых инициализаций
        seed = int(seed) if seed is not None else None
        if seed is not None:
            set_global_seed(seed)
            print(f"🔒 Seed установлен: {seed}")
            # ENV больше не используем для сидов

        print(f"\n🚀 Старт обучения для {symbol} [{datetime.now()}]")
        df_5min = db_get_or_fetch_ohlcv(
            symbol_name=symbol,
            timeframe='5m',
            limit_candles=100000,
            exchange_id='bybit'
        )

        if df_5min is None or df_5min.empty:
            # Пытаемся автоматически скачать и загрузить свечи в БД
            try:
                print(f"📥 Данные не найдены для {symbol}. Пытаюсь скачать и загрузить в БД...")
                csv_file_path = parser_download_and_combine_with_library(
                    symbol=symbol,
                    interval='5m',
                    months_to_fetch=12,
                    desired_candles=100000
                )
                if csv_file_path:
                    loaded_count = load_latest_candles_from_csv_to_db(
                        file_path=csv_file_path,
                        symbol_name=symbol,
                        timeframe='5m'
                    )
                    print(f"✅ Загрузка в БД завершена: {loaded_count} свечей")
                # Повторно пробуем получить из БД
                df_5min = db_get_or_fetch_ohlcv(
                    symbol_name=symbol,
                    timeframe='5m',
                    limit_candles=100000,
                    exchange_id='bybit'
                )
            except Exception as fetch_err:
                print(f"❌ Не удалось автоматически загрузить данные для {symbol}: {fetch_err}")
                df_5min = None
        
        if df_5min is None or df_5min.empty:
            return {"message": f"❌ Данные для {symbol} не найдены"}

        import pandas as pd
        df_5min['datetime'] = pd.to_datetime(df_5min['timestamp'], unit='ms')
        df_5min.set_index('datetime', inplace=True)

        df_15min = df_5min.resample('15min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
        }).dropna().reset_index()

        df_1h = df_5min.resample('1h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
        }).dropna().reset_index()

        dfs = {
            'df_5min': df_5min,
            'df_15min': df_15min,
            'df_1h': df_1h,
            'symbol': symbol,
        }

        print(f"📈 {symbol}: 5m={len(df_5min)}, 15m={len(df_15min)}, 1h={len(df_1h)}")

        # Получаем количество эпизодов из аргумента или переменной окружения
        if episodes is None:
            episodes = int(os.getenv('DEFAULT_EPISODES', 5))
        print(f"🎯 Количество эпизодов: {episodes}")

        # Прокидываем пути для продолжения обучения из ENV/Redis если заданы
        load_model_path = os.environ.get('CONTINUE_MODEL_PATH')
        load_buffer_path = os.environ.get('CONTINUE_BUFFER_PATH')
        # Определим родителя/корень для цепочки run'ов при дообучении
        parent_run_id = None
        root_run_id = None
        try:
            # Попробуем Redis как приоритетный источник
            from redis import Redis
            r = Redis(host='redis', port=6379, db=0, decode_responses=True)
            v_model = r.get('continue:model_path')
            v_buffer = r.get('continue:buffer_path')
            if v_model:
                load_model_path = v_model
            if v_buffer:
                load_buffer_path = v_buffer
            # Чистим ключи, чтобы не повлиять на другие задачи
            if v_model:
                r.delete('continue:model_path')
            if v_buffer:
                r.delete('continue:buffer_path')
        except Exception:
            pass

        # Если дообучаем из структурированного пути runs/.../model.pth — проставим связи
        try:
            if isinstance(load_model_path, str):
                norm_path = load_model_path.replace('\\', '/')
                parts = norm_path.split('/')
                if len(parts) >= 4 and parts[-1] == 'model.pth' and 'runs' in parts:
                    runs_idx = parts.index('runs')
                    if runs_idx + 1 < len(parts):
                        parent_run_id = parts[runs_idx + 1]
                        # Попытаемся прочитать root_id из манифеста родителя
                        try:
                            parent_dir = os.path.dirname(load_model_path)
                            manifest_path = os.path.join(parent_dir, 'manifest.json')
                            if os.path.exists(manifest_path):
                                import json as _json
                                with open(manifest_path, 'r', encoding='utf-8') as mf:
                                    mf_data = _json.load(mf)
                                root_run_id = mf_data.get('root_id') or parent_run_id
                            else:
                                root_run_id = parent_run_id
                        except Exception:
                            root_run_id = parent_run_id
        except Exception:
            parent_run_id = parent_run_id or None
            root_run_id = root_run_id or None

        result = train_model_optimized(
            dfs=dfs,
            episodes=episodes,
            load_model_path=load_model_path,
            load_buffer_path=load_buffer_path,
            seed=seed,
            parent_run_id=parent_run_id,
            root_id=root_run_id
        )
        return {"message": f"✅ Обучение {symbol} завершено: {result}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"message": f"❌ Ошибка обучения {symbol}: {str(e)}"}

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0}, queue='train')
def train_dqn_multi_crypto(self, seed: int | None = None):
    """Задача для мультивалютного обучения DQN"""
    self.update_state(state="IN_PROGRESS", meta={"progress": 0})
    # Сид из аргумента/ENV
    seed = int(seed) if seed is not None else None
    if seed is not None:
        set_global_seed(seed)
        print(f"🔒 Seed установлен: {seed}")
        # ENV больше не используем для сидов

    print("🚀 Начинаю мультивалютное обучение DQN...")
    try:
        # Новый модуль для мульти-обучения
        from agents.multi.v_train_multi import train_multi
        
        # Получаем количество эпизодов из переменной окружения
        episodes = int(os.getenv('DEFAULT_EPISODES', 10001))
        print(f"🎯 Количество эпизодов для мульти-обучения: {episodes}")
        
        result = train_multi(symbols=[
            'BTCUSDT','TONUSDT','ETHUSDT','SOLUSDT','ADAUSDT','BNBUSDT'
        ], episodes=episodes)
        return {"message": f"Мультивалютное обучение завершено: {result}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"message": f"Ошибка мульти-обучения: {str(e)}"}

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0})
def trade_step():
    """
    Выполняет один торговый шаг с использованием API Bybit
    """
    try:
        # Проверяем наличие API ключей (новый формат)
        if not BYBIT_API_KEY or not BYBIT_SECRET_KEY:
            return {"error": "API ключи Bybit не настроены"}
        
        # Получаем текущее состояние рынка (замени на получение реальных данных)
        state = get_current_market_state()  # реализуй функцию получения состояния

        action = trade_once(state)

        # Здесь ты можешь сделать реальный ордер через API биржи
        # Используем API ключи для подключения к Bybit
        import ccxt
        exchange = ccxt.bybit({
            'apiKey': BYBIT_API_KEY,
            'secret': BYBIT_SECRET_KEY,
            'sandbox': False,  # True для тестового режима
            'enableRateLimit': True
        })

        return f"Торговое действие: {action} (API подключен)"
        
    except Exception as e:
        return {"error": f"Ошибка в trade_step: {str(e)}"}

@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0})
def start_trading_task(self, symbols, model_path=None):
    """
    Оркестратор: получает параметры, делает Redis-лок, публикует provisional-статус
    и кладёт торговую задачу в очередь trade (без docker exec).
    """
    import os
    from celery import chain

    # Проверяем, должна ли работать торговля
    trading_enabled = os.environ.get('ENABLE_TRADING_BEAT', '1') in ('1', 'true', 'True')
    if not trading_enabled:
        return {"success": False, "skipped": True, "reason": "ENABLE_TRADING_BEAT=0"}

    # Redis-лок: предотвращаем параллельные запуски в пределах ~5 минут (per-symbol)
    try:
        from redis import Redis as _Redis
        _rc_lock = _Redis(host='redis', port=6379, db=0, decode_responses=True)
        # Определяем символ для ключа
        lock_symbol = None
        try:
            lock_symbol = (symbols[0] if (symbols and len(symbols) > 0) else None)
        except Exception:
            lock_symbol = None
        if not lock_symbol:
            lock_symbol = 'ALL'
        lock_key = f'trading:agent_lock:{lock_symbol}'
        # TTL 600с (10 минут) — чтобы агент не терял lock во время работы
        got_lock = _rc_lock.set(lock_key, self.request.id, nx=True, ex=600)
        if not got_lock:
            return {"success": False, "skipped": True, "reason": "agent_lock_active"}
    except Exception:
        pass

    # Если параметры не передали — пробуем взять их из Redis для конкретного символа
    try:
        if (not symbols) or model_path is None:
            from redis import Redis
            _r = Redis(host='redis', port=6379, db=0, decode_responses=True)
            
            # Определяем символ для чтения настроек
            symbol = symbols[0] if symbols else None
            if not symbol:
                # Пробуем взять из общего списка символов
                try:
                    _sym_raw = _r.get('trading:symbols')
                    if _sym_raw:
                        import json as _json
                        _sym = _json.loads(_sym_raw)
                        if isinstance(_sym, list) and _sym:
                            symbol = _sym[0]
                            symbols = _sym
                except Exception:
                    symbol = 'BTCUSDT'
                    symbols = ['BTCUSDT']
            
            # Читаем настройки для конкретного символа
            if (not symbols):
                try:
                    _sym_raw = _r.get(f'trading:symbols:{symbol}')
                    if _sym_raw:
                        import json as _json
                        _sym = _json.loads(_sym_raw)
                        if isinstance(_sym, list) and _sym:
                            symbols = _sym
                except Exception:
                    pass
            if model_path is None:
                try:
                    # Сначала пробуем для конкретного символа
                    _mp = _r.get(f'trading:model_path:{symbol}')
                    if _mp:
                        model_path = _mp
                    else:
                        # Фолбэк на общий
                        _mp = _r.get('trading:model_path')
                        if _mp:
                            model_path = _mp
                except Exception:
                    pass
    except Exception:
        pass

    # Дефолты
    if not symbols:
        symbols = ['BTCUSDT']

    print(f"🚀 Оркестрация торговли: symbols={symbols} | model_path={model_path if model_path else 'default'}")
    self.update_state(state="IN_PROGRESS", meta={"progress": 0})

    # Получаем список путей моделей для конкретного символа
    model_paths = None
    try:
        from redis import Redis as _Redis
        _r2 = _Redis(host='redis', port=6379, db=0, decode_responses=True)
        
        # Определяем символ для чтения моделей
        symbol = symbols[0] if symbols else 'BTCUSDT'
        
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
                if '_rc_lock' in locals():
                    lock_symbol = (symbols[0] if (symbols and len(symbols) > 0) else 'ALL')
                    lock_key = f'trading:agent_lock:{lock_symbol}'
                    _rc_lock.delete(lock_key)
                return {"success": False, "skipped": True, "reason": "no_model_paths"}
        except Exception:
            # Снять лок, чтобы не блокировать следующий запуск
            if '_rc_lock' in locals():
                lock_symbol = (symbols[0] if (symbols and len(symbols) > 0) else 'ALL')
                lock_key = f'trading:agent_lock:{lock_symbol}'
                _rc_lock.delete(lock_key)
            return {"success": False, "skipped": True, "reason": "no_model_paths"}

    # Промежуточный статус в Redis для UI
    try:
        from redis import Redis as _Redis
        import json as _json
        provisional = {
            'success': True,
            'is_trading': True,
            'trading_status': 'Активна',
            'trading_status_emoji': '🟢',
            'trading_status_full': '🟢 Активна',
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
            if model_paths:
                _rc.set('trading:last_model_paths', _json.dumps(model_paths, ensure_ascii=False))
                # Если использовали дефолтную модель - сохраняем её как основную
                if not _rc.get('trading:model_paths'):
                    _rc.set('trading:model_paths', _json.dumps(model_paths, ensure_ascii=False))
            # last_consensus больше не ведём глобально, чтобы не перетирать per‑symbol
            # Сохраняем символы если их нет
            if not _rc.get('trading:symbols'):
                _rc.set('trading:symbols', _json.dumps(symbols, ensure_ascii=False))
        except Exception:
            pass
    except Exception:
        pass

    # Кладём торговую задачу в очередь trade
    try:
        res = execute_trade.apply_async(kwargs={
            'symbols': symbols,
            'model_path': model_path,
            'model_paths': model_paths,
        }, queue='trade')
        return {"success": True, "enqueued": True, "task_id": res.id}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Включаем периодический запуск торговли
import os
# Настройка расписания Celery Beat по флагу окружения (не перетираем значение)
if os.environ.get('ENABLE_TRADING_BEAT', '0').lower() in ('1', 'true', 'yes', 'on'):
    celery.conf.beat_schedule = {
        'start-trading-every-5-minutes': {
            'task': 'tasks.celery_tasks.start_trading_task',
            'schedule': crontab(minute='*/5'),
            'args': ([], None)  # Символы и путь к модели будут передаваться из веб-интерфейса
        },
    }
    celery.conf.timezone = 'UTC'
    print("✅ Периодическая торговля включена (каждые 5 минут)")
else:
    print("⚠️ Периодическая торговля отключена (ENABLE_TRADING_BEAT=0)")

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

        rc = _Redis(host='redis', port=6379, db=0, decode_responses=True)

        # Текущие параметры
        try:
            symbols_raw = rc.get('trading:symbols')
            symbols = _json.loads(symbols_raw) if symbols_raw else ['BTCUSDT']
            if not isinstance(symbols, list) or not symbols:
                symbols = ['BTCUSDT']
        except Exception:
            symbols = ['BTCUSDT']
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
                    status = prev
        except Exception:
            pass

        # Обновляем статус для конкретного символа
        rc.set(f'trading:status:{sym}', _json.dumps(status, ensure_ascii=False))
        
        # Обновляем общий статус только если это единственный активный агент
        # или если общий статус устарел/отсутствует
        try:
            # Проверяем, есть ли другие активные агенты
            other_active = False
            all_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'TONUSDT', 'ADAUSDT', 'BNBUSDT', 'XRPUSDT']
            for other_sym in all_symbols:
                if other_sym != sym:
                    other_lock_key = f'trading:agent_lock:{other_sym}'
                    other_ttl = rc.ttl(other_lock_key)
                    if other_ttl is not None and int(other_ttl) > 0:
                        other_active = True
                        break
            
            # Обновляем общий статус только если нет других активных агентов
            # или если общий статус устарел
            if not other_active or not cached or not is_fresh:
                rc.set('trading:current_status', _json.dumps(status, ensure_ascii=False))
        except Exception:
            # В случае ошибки обновляем общий статус для безопасности
            rc.set('trading:current_status', _json.dumps(status, ensure_ascii=False))
        
        rc.set('trading:current_status_ts', _dt.utcnow().isoformat())
        return {"success": True, "updated": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


# --- CNN Training Task ---
@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0}, queue='train')
def train_cnn_model(self, symbol: str, model_type: str = "multiframe", 
                   seed: int = None):
    """Обучение CNN модели для анализа паттернов криптовалют"""
    
    self.update_state(state="IN_PROGRESS", meta={"progress": 0, "symbol": symbol})
    
    try:
        print(f"🧠 Начинаю обучение CNN модели для {symbol}")
        
        # Импортируем CNN модули
        try:
            from cnn_training.config import CNNTrainingConfig
            from cnn_training.trainer import CNNTrainer
        except ImportError as ie:
            print(f"❌ Ошибка импорта CNN модулей: {ie}")
            raise Exception(f"CNN модули не найдены: {ie}")
        
        # Создаем конфигурацию (все параметры из config.py)
        config = CNNTrainingConfig(
            symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"],  # Используем все символы из конфига
            timeframes=["5m", "15m", "1h"],
            device="auto"
        )
        
        # Создаем тренер
        trainer = CNNTrainer(config)
        
        self.update_state(state="IN_PROGRESS", meta={"progress": 20, "message": "CNN тренер создан"})
        
        # Обучаем модель prediction типа (рекомендуется)
        print(f"🎯 Обучение CNN модели {model_type} для всех символов: {config.symbols}")
        
        if model_type == "multiframe":
            # Обучаем мультифреймовую модель на всех символах
            result = trainer.train_multiframe_model(config.symbols)
        else:
            # Обучаем отдельную модель для 5m фрейма (основной для DQN)
            result = trainer.train_single_model(symbol, "5m", model_type)

        # Подготовим сериализуемый ответ (без PyTorch-моделей и прочих несериализуемых объектов)
        safe_result = None
        try:
            if isinstance(result, dict):
                best_val_accuracy = result.get('best_val_accuracy')
                safe_result = {
                    "best_val_accuracy": float(best_val_accuracy) if best_val_accuracy is not None else None,
                    "train_steps": int(len(result.get('train_losses', []) or [])),
                    "val_steps": int(len(result.get('val_losses', []) or [])),
                }
        except Exception:
            safe_result = {"best_val_accuracy": None, "train_steps": 0, "val_steps": 0}

        self.update_state(state="SUCCESS", meta={
            "progress": 100,
            "message": f"CNN обучение завершено для {symbol}",
            "result": safe_result
        })

        return {
            "success": True,
            "message": f"CNN обучение завершено для {symbol}",
            "symbol": symbol,
            "model_type": model_type,
            "result": safe_result
        }
        
    except Exception as e:
        error_msg = f"❌ Ошибка обучения CNN для {symbol}: {str(e)}"
        print(error_msg)
        import traceback
        print(traceback.format_exc())
        
        self.update_state(state="FAILURE", meta={
            "progress": 0,
            "error": error_msg
        })
        
        return {
            "success": False,
            "error": error_msg,
            "symbol": symbol
        }


# keep beat schedule extension together with trading beat
if os.environ.get('ENABLE_TRADING_BEAT', '0').lower() in ('1', 'true', 'yes', 'on'):
    try:
        # Расширяем уже созданный beat_schedule
        celery.conf.beat_schedule.update({
            'refresh-trading-status-every-minute': {
                'task': 'tasks.celery_tasks.refresh_trading_status',
                'schedule': crontab(minute='*'),
                'args': (),
            },
        })
    except Exception:
        celery.conf.beat_schedule = {
            'refresh-trading-status-every-minute': {
                'task': 'tasks.celery_tasks.refresh_trading_status',
                'schedule': crontab(minute='*'),
                'args': (),
            },
        }

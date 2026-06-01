from flask import Blueprint, jsonify, request, render_template
from utils.redis_utils import get_redis_client
from tasks.celery_task_trade import start_trading_task
from utils.trade_utils import get_recent_trades, get_trade_statistics, get_trades_by_symbol, get_model_predictions
from utils.indicators import get_atr_1h
from trading_agent.risk_trailing import setup_trailing_stop_bybit
import logging
import docker
import json
from datetime import datetime
from typing import Any, Dict, Optional
import time
import re
from pathlib import Path
from utils.trading_sessions import (
    create_session,
    delete_runtime_value,
    get_active_session_for_account_symbol,
    get_runtime_json,
    get_runtime_value,
    get_status as get_session_status,
    list_session_ids,
    load_session,
    remove_session,
    session_config_key,
    session_lock_key,
    session_runtime_key,
    save_session,
    set_runtime_json,
    set_runtime_value,
    set_status as set_session_status,
)

trading_bp = Blueprint('trading', __name__)
XGB_DEFAULT_TAKE_PROFIT_PCT = 3.0
XGB_SIGNAL_EXIT_THRESHOLD_DEFAULT = 0.5
XGB_ENTRY_ATTEMPTS_HISTORY_KEY = "trading:xgb_entry_attempts_history"
XGB_ENTRY_ATTEMPTS_HISTORY_LIMIT = 300


def _normalize_leverage(value, default: int = 1) -> int:
    try:
        return max(1, int(float(value)))
    except Exception:
        return default


def _xgb_model_leverage_key(model_path: str | None) -> str | None:
    normalized = str(model_path or '').replace('\\', '/').strip()
    if not normalized:
        return None
    return f"trading:xgb_model_leverage:{normalized}"


def _read_xgb_entry_attempts_history(rc, limit: int = XGB_ENTRY_ATTEMPTS_HISTORY_LIMIT) -> list[dict]:
    try:
        if rc is None:
            return []
        keep = max(1, int(limit))
        rows = rc.lrange(XGB_ENTRY_ATTEMPTS_HISTORY_KEY, 0, keep - 1) or []
        out: list[dict] = []
        for row in rows:
            try:
                raw = row.decode("utf-8") if isinstance(row, (bytes, bytearray)) else str(row)
                doc = json.loads(raw)
                if isinstance(doc, dict):
                    out.append(doc)
            except Exception:
                continue
        return out
    except Exception:
        return []


def _get_session_or_400(rc, session_id: str) -> tuple[dict | None, tuple | None]:
    sid = str(session_id or '').strip()
    if not sid:
        return None, (jsonify({'success': False, 'error': 'session_id is required'}), 400)
    doc = load_session(rc, sid)
    if not isinstance(doc, dict):
        return None, (jsonify({'success': False, 'error': f'session not found: {sid}'}), 404)
    return doc, None


def _read_xgb_runtime_meta(model_path: str | None) -> tuple[float | None, str | None, float | None]:
    try:
        if not model_path:
            return None, None, None
        meta_path = Path(str(model_path)).with_name("meta.json")
        manifest_path = Path(str(model_path)).with_name("manifest.json")
        threshold = None
        task_name = None
        stop_loss_pct = None
        if meta_path.exists():
            meta_doc = json.loads(meta_path.read_text(encoding="utf-8"))
            cfg = meta_doc.get("cfg_snapshot") if isinstance(meta_doc, dict) else {}
            if isinstance(cfg, dict):
                raw_thr = cfg.get("p_enter_threshold")
                if raw_thr not in (None, ""):
                    threshold = float(raw_thr)
                raw_task = cfg.get("task")
                if raw_task:
                    task_name = str(raw_task).strip().lower()
                raw_sl = cfg.get("entry_sl_pct")
                if raw_sl not in (None, ""):
                    stop_loss_pct = abs(float(raw_sl)) * 100.0
        if manifest_path.exists():
            manifest_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(manifest_doc, dict) and not task_name:
                raw_task = manifest_doc.get("task")
                if raw_task:
                    task_name = str(raw_task).strip().lower()
        return threshold, task_name, stop_loss_pct
    except Exception:
        return None, None, None


def _read_xgb_threshold_meta(model_path: str | None) -> tuple[float | None, str | None]:
    threshold, task_name, _ = _read_xgb_runtime_meta(model_path)
    return threshold, task_name


def _read_xgb_hold_steps_meta(model_path: str | None) -> int | None:
    try:
        if not model_path:
            return None
        meta_path = Path(str(model_path)).with_name("meta.json")
        if not meta_path.exists():
            return None
        meta_doc = json.loads(meta_path.read_text(encoding="utf-8"))
        cfg = meta_doc.get("cfg_snapshot") if isinstance(meta_doc, dict) else {}
        raw_hold = cfg.get("max_hold_steps") if isinstance(cfg, dict) else None
        return int(float(raw_hold)) if raw_hold not in (None, "") else None
    except Exception:
        return None


def _read_xgb_side_trade_meta(model_path: str | None) -> tuple[float | None, str | None, float | None, float | None]:
    try:
        threshold, task_name, stop_loss_pct = _read_xgb_runtime_meta(model_path)
        take_profit_pct = None
        if model_path:
            meta_path = Path(str(model_path)).with_name("meta.json")
            if meta_path.exists():
                meta_doc = json.loads(meta_path.read_text(encoding="utf-8"))
                cfg = meta_doc.get("cfg_snapshot") if isinstance(meta_doc, dict) else {}
                raw_tp = cfg.get("entry_tp_pct") if isinstance(cfg, dict) else None
                if raw_tp not in (None, ""):
                    take_profit_pct = abs(float(raw_tp)) * 100.0
        return threshold, task_name, stop_loss_pct, take_profit_pct
    except Exception:
        return None, None, None, None


def _normalize_model_path_key(value: Any) -> str:
    path = str(value or "").strip().replace("\\", "/")
    if path and not path.startswith("/"):
        path = "/workspace/" + path.lstrip("/")
    return path


def _xgb_side_model_path(session_doc: Dict[str, Any], side: str) -> str | None:
    side_name = str(side or "").strip().lower()
    if side_name not in ("long", "short"):
        return None
    model_paths = [str(item) for item in (session_doc.get("model_paths") or []) if item]
    model_roles_raw = session_doc.get("model_roles") if isinstance(session_doc.get("model_roles"), dict) else {}
    model_roles = {
        _normalize_model_path_key(path): str(role or "").strip().lower()
        for path, role in model_roles_raw.items()
    }
    for path_item in model_paths:
        if model_roles.get(_normalize_model_path_key(path_item)) == side_name:
            return str(path_item)
    direction = str(session_doc.get("direction") or "").strip().lower()
    model_path = str(session_doc.get("model_path") or "").strip()
    if len(model_paths) == 1 and direction == side_name:
        return model_paths[0]
    if model_path and direction == side_name:
        return model_path
    return None


def _derive_xgb_prediction_signal(q_values: Any, task: Any, confidence: Any, role: Any = None) -> float | None:
    try:
        values = q_values if isinstance(q_values, list) else []
        task_name = str(task or "").strip().lower() or None
        role_name = str(role or "").strip().lower() or None
        if len(values) >= 3:
            if task_name in ("entry_short", "exit_long") or (task_name is None and role_name == "short"):
                return float(values[2])
            if task_name in ("entry_long", "exit_short", "directional", None):
                return float(values[1])
        return float(confidence) if confidence not in (None, "") else None
    except Exception:
        return None


def _latest_xgb_model_prediction(session_id: str, model_path: str | None) -> Dict[str, Any]:
    try:
        if not session_id or not model_path:
            return {}
        from orm.database import get_db_session
        from orm.models import ModelPrediction

        normalized = str(model_path).replace("\\", "/")
        path_candidates = {normalized}
        if normalized.startswith("/workspace/"):
            path_candidates.add(normalized[len("/workspace/"):])
        else:
            path_candidates.add("/workspace/" + normalized.lstrip("/"))

        session = get_db_session()
        try:
            rows = (
                session.query(ModelPrediction)
                .filter(ModelPrediction.model_path.in_(list(path_candidates)))
                .filter(ModelPrediction.market_conditions.like(f'%"{session_id}"%'))
                .order_by(ModelPrediction.created_at.desc())
                .limit(20)
                .all()
            )
        finally:
            session.close()

        for row in rows:
            try:
                mc = json.loads(row.market_conditions) if row.market_conditions else {}
            except Exception:
                mc = {}
            if str(mc.get("session_id") or "") != str(session_id):
                continue
            try:
                q_values = json.loads(row.q_values) if row.q_values else []
            except Exception:
                q_values = []
            task_name = mc.get("xgb_task") or mc.get("task")
            signal = _derive_xgb_prediction_signal(
                q_values,
                task_name,
                getattr(row, "confidence", None),
                mc.get("model_role"),
            )
            threshold = mc.get("xgb_runtime_threshold")
            if threshold in (None, ""):
                threshold = mc.get("xgb_p_enter_threshold")
            return {
                "signal": signal,
                "threshold": float(threshold) if threshold not in (None, "") else None,
                "action": getattr(row, "action", None),
                "created_at": row.created_at.isoformat() if getattr(row, "created_at", None) else None,
            }
    except Exception:
        return {}
    return {}


def _read_xgb_result_meta(model_path: str | None) -> tuple[float | None, float | None, int | None]:
    try:
        if not model_path:
            return None, None, None
        repo_root = Path(__file__).resolve().parents[1]
        meta_path = Path(str(model_path)).with_name("meta.json")
        if not meta_path.exists():
            normalized = str(model_path).replace('\\', '/')
            marker = '/models/'
            if marker in normalized:
                relative_part = normalized.split(marker, 1)[1]
                candidate = repo_root / 'models' / relative_part
                meta_path = candidate.with_name("meta.json")
            elif not normalized.startswith('/'):
                candidate = repo_root / 'models' / 'xgb' / normalized.lstrip('/')
                meta_path = candidate.with_name("meta.json")
        if not meta_path.exists():
            return None, None, None
        meta_doc = json.loads(meta_path.read_text(encoding="utf-8"))
        cfg = meta_doc.get("cfg_snapshot") if isinstance(meta_doc, dict) else {}
        if not isinstance(cfg, dict):
            return None, None, None
        tp_pct = None
        sl_pct = None
        hold_steps = None
        raw_tp = cfg.get("entry_tp_pct")
        raw_sl = cfg.get("entry_sl_pct")
        raw_hold = cfg.get("max_hold_steps")
        if raw_tp not in (None, ""):
            tp_pct = abs(float(raw_tp)) * 100.0
        if raw_sl not in (None, ""):
            sl_pct = abs(float(raw_sl)) * 100.0
        if raw_hold not in (None, ""):
            hold_steps = int(raw_hold)
        return tp_pct, sl_pct, hold_steps
    except Exception:
        return None, None, None


def _read_xgb_model_uuid(model_path: str | None) -> str | None:
    try:
        if not model_path:
            return None
        repo_root = Path(__file__).resolve().parents[1]
        manifest_path = Path(str(model_path)).with_name("manifest.json")
        if not manifest_path.exists():
            normalized = str(model_path).replace('\\', '/')
            marker = '/models/'
            if marker in normalized:
                relative_part = normalized.split(marker, 1)[1]
                manifest_path = (repo_root / 'models' / relative_part).with_name("manifest.json")
        if not manifest_path.exists():
            return None
        manifest_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest_doc, dict):
            return None
        value = manifest_doc.get("run_name") or manifest_doc.get("run_id") or manifest_doc.get("id")
        return str(value).strip() if value else None
    except Exception:
        return None


def _to_float_or_none(value) -> float | None:
    try:
        if value in (None, ''):
            return None
        return float(value)
    except Exception:
        return None

def _read_xgb_signal_exit_status(rc, symbol: str) -> dict:
    status = {
        'enabled': False,
        'start_step': None,
        'window': 20,
        'history_size': 0,
        'avg_signal': None,
        'avg_threshold': None,
        'last_signal': None,
        'last_threshold': None,
        'ready': False,
        'passes_threshold': None,
    }
    try:
        raw_enabled = rc.get(f'trading:xgb_signal_exit_enabled:{symbol}') or rc.get('trading:xgb_signal_exit_enabled')
        status['enabled'] = str(raw_enabled or '').strip().lower() in ('1', 'true', 'yes', 'on')
    except Exception:
        status['enabled'] = False
    try:
        raw_start = rc.get(f'trading:xgb_signal_exit_start_step:{symbol}') or rc.get('trading:xgb_signal_exit_start_step')
        if raw_start not in (None, ''):
            status['start_step'] = int(float(raw_start))
    except Exception:
        status['start_step'] = None
    try:
        raw_window = rc.get(f'trading:xgb_signal_exit_window:{symbol}') or rc.get('trading:xgb_signal_exit_window')
        if raw_window not in (None, ''):
            status['window'] = max(1, int(float(raw_window)))
    except Exception:
        status['window'] = None
    try:
        raw_history = rc.get(f'trading:xgb_signal_exit_history:{symbol}')
        history = json.loads(raw_history) if raw_history else []
        if not isinstance(history, list):
            history = []
        cleaned = []
        for item in history:
            if not isinstance(item, dict):
                continue
            signal = _to_float_or_none(item.get('signal'))
            threshold = _to_float_or_none(item.get('threshold'))
            if signal is None or threshold is None:
                continue
            cleaned.append({
                'signal': float(signal),
                'threshold': float(threshold),
            })
        status['history_size'] = len(cleaned)
        if cleaned:
            status['last_signal'] = cleaned[-1]['signal']
            status['last_threshold'] = cleaned[-1]['threshold']
        window = int(status['window']) if isinstance(status['window'], int) and status['window'] > 0 else None
        if window and len(cleaned) >= window:
            tail = cleaned[-window:]
            avg_signal = float(sum(item['signal'] for item in tail) / len(tail))
            avg_threshold = float(sum(item['threshold'] for item in tail) / len(tail))
            status['avg_signal'] = avg_signal
            status['avg_threshold'] = avg_threshold
            status['ready'] = True
            status['passes_threshold'] = bool(avg_signal >= avg_threshold)
    except Exception:
        pass
    return status


@trading_bp.route('/trading_agent')
def trading_agent_page():
    """Страница торгового агента"""
    return render_template('trading_agent.html')

@trading_bp.get('/trading/results')
def trading_results_page():
    """Страница агрегированных результатов торговли"""
    return render_template('trading_results.html')

@trading_bp.route('/agent/<symbol>')
def agent_symbol_page(symbol: str):
    """Страница агента, отфильтрованная по конкретному символу (BTCUSDT, TONUSDT и т.д.)"""
    try:
        sym = (symbol or '').upper().strip()
        # Простейшая валидация: только латиница и 'USDT' в конце
        import re
        if not re.match(r'^[A-Z]{2,10}USDT$', sym):
            # дефолт на BTCUSDT
            sym = 'BTCUSDT'
        return render_template('agent_symbol.html', symbol=sym)
    except Exception:
        return render_template('agent_symbol.html', symbol='BTCUSDT')


@trading_bp.get('/api/trading/trailing_last_setup')
def trailing_last_setup():
    """Диагностика: последняя попытка установки трейлинга (из Redis)."""
    try:
        sym = (request.args.get('symbol') or '').upper().strip()
        if not sym:
            return jsonify({'success': False, 'error': 'symbol is required'}), 400
        rc = get_redis_client()
        raw = rc.get(f'trading:trailing:last_setup:{sym}') if rc else None
        payload = None
        if raw:
            try:
                payload = json.loads(raw if isinstance(raw, str) else raw.decode('utf-8'))
            except Exception:
                payload = None
        return jsonify({'success': True, 'symbol': sym, 'data': payload}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@trading_bp.get('/api/trading/atr_1h')
def api_atr_1h():
    """Возвращает ATR(1h) для символа и одновременно прогревает Redis-кэш (length берём из app_settings)."""
    try:
        sym = (request.args.get('symbol') or '').upper().strip()
        if not sym:
            return jsonify({'success': False, 'error': 'symbol is required'}), 400
        if not re.match(r'^[A-Z]{2,10}USDT$', sym):
            return jsonify({'success': False, 'error': 'symbol must match [A-Z]{2,10}USDT'}), 400
        atr_abs, atr_norm, close = get_atr_1h(sym, length=None)
        return jsonify({'success': True, 'symbol': sym, 'atr_abs': atr_abs, 'atr_norm': atr_norm, 'close': close}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.post('/api/trading/save_config')
def save_trading_config():
    """Автосохранение выбора моделей и консенсуса без запуска торговли."""
    try:
        data = request.get_json() or {}
        try:
            logging.info(f"[save_config] payload symbols={data.get('symbols')} sel_paths={len(data.get('model_paths') or [])} counts={(data.get('consensus') or {}).get('counts')}")
            logging.info(f"[save_config] FULL payload: {data}")
            # Детальный лог consensus
            consensus = data.get('consensus')
            if consensus:
                logging.info(f"[save_config] consensus counts: {consensus.get('counts')}")
                logging.info(f"[save_config] consensus percents: {consensus.get('percents')}")
        except Exception:
            pass
        symbols = data.get('symbols') or []
        model_paths = data.get('model_paths') or []
        # Режим: 'single' (фикс. направление) | 'long-short' (переключение по режиму рынка)
        trade_mode = str(data.get('trade_mode') or '').strip() or None
        if trade_mode not in ('single', 'long-short'):
            trade_mode = None
        # Роли моделей: {model_path: 'long'|'short'} (используется в режиме long-short)
        model_roles = data.get('model_roles') if isinstance(data.get('model_roles'), dict) else None
        consensus = data.get('consensus') or None
        take_profit_pct = data.get('take_profit_pct')  # Процент тейк-профита
        stop_loss_pct = data.get('stop_loss_pct')      # Процент стоп-лосса
        risk_management_type = data.get('risk_management_type')  # Способ управления рисками
        # Новые параметры ATR‑режима стопов
        atr_k = data.get('atr_k')
        atr_m = data.get('atr_m')
        atr_min_sl_mult = data.get('atr_min_sl_mult')
        risk_stop_mode = data.get('risk_stop_mode')  # 'fixed_pct' | 'atr_tp_sl' | (в будущем 'atr_trailing')
        # Параметры трейлинга
        trailing_enabled = data.get('trailing_enabled')
        trailing_mode = data.get('trailing_mode')
        atr_trail_mult = data.get('atr_trail_mult')
        atr_trail_activate_pct = data.get('atr_trail_activate_pct')
        trailing_activate_mode = data.get('trailing_activate_mode')
        trailing_activate_value = data.get('trailing_activate_value')
        account_pct = data.get('account_pct')  # Доля счёта для сделки, %
        # Q-gate thresholds (per-symbol)
        qgate_maxq = data.get('qgate_maxq')
        qgate_gapq = data.get('qgate_gapq')
        exit_mode = str(data.get('exit_mode') or '').strip() or None  # 'prediction' | 'risk_orders'
        leverage = data.get('leverage')  # 1..5
        import json as _json
        rc = get_redis_client()

        # Нормализация символов (защита от "TON USDT", "TON/USDT", лишних пробелов и т.п.)
        try:
            from utils.cctx_utils import normalize_to_db as _normalize_to_db
            if isinstance(symbols, (list, tuple)):
                symbols = [_normalize_to_db(str(s)) for s in symbols if s]
            elif isinstance(symbols, str) and symbols:
                symbols = [_normalize_to_db(symbols)]
        except Exception:
            try:
                if isinstance(symbols, (list, tuple)):
                    symbols = [str(s).replace('/', '').replace(' ', '').upper().strip() for s in symbols if s]
                elif isinstance(symbols, str) and symbols:
                    symbols = [str(symbols).replace('/', '').replace(' ', '').upper().strip()]
            except Exception:
                pass

        # Изменяем логику: вместо перезаписи, объединяем символы
        if symbols:
            existing_raw = rc.get('trading:symbols')
            existing_symbols = _json.loads(existing_raw) if existing_raw else []
            if not isinstance(existing_symbols, list):
                existing_symbols = []

            # Нормализуем и существующие значения (могли остаться старые "TON USDT")
            try:
                from utils.cctx_utils import normalize_to_db as _normalize_to_db2
                existing_symbols = [_normalize_to_db2(str(s)) for s in existing_symbols if s]
            except Exception:
                try:
                    existing_symbols = [str(s).replace('/', '').replace(' ', '').upper().strip() for s in existing_symbols if s]
                except Exception:
                    pass

            merged_symbols = list(set(existing_symbols + symbols))
            rc.set('trading:symbols', _json.dumps(merged_symbols, ensure_ascii=False))

        # Сохраняем только непустые списки моделей (и глобально, и per‑symbol)
        if isinstance(model_paths, list) and len(model_paths) > 0:
            rc.set('trading:model_paths', _json.dumps(model_paths, ensure_ascii=False))
            rc.set('trading:last_model_paths', _json.dumps(model_paths, ensure_ascii=False))
            # Пер‑символьное хранилище для корректного отображения и оркестрации
            symbol = symbols[0] if symbols else 'ALL'
            rc.set(f'trading:model_paths:{symbol}', _json.dumps(model_paths, ensure_ascii=False))
            try:
                logging.info(f"[save_config] symbol={symbol} model_paths_selected={len(model_paths)} -> saved per-symbol model_paths")
            except Exception:
                pass
            # Сохраняем режим/роли per-symbol (backward compatible: старые ключи не трогаем)
            try:
                if trade_mode:
                    rc.set(f'trading:trade_mode:{symbol}', str(trade_mode))
                # Нормализуем роли и сохраняем только для выбранных model_paths
                if isinstance(model_roles, dict):
                    roles_norm = {}
                    mp_set = set(str(p) for p in (model_paths or []) if p)
                    for k, v in model_roles.items():
                        try:
                            kp = str(k)
                            if kp not in mp_set:
                                continue
                            rv = str(v).strip().lower()
                            roles_norm[kp] = ('short' if rv == 'short' else 'long')
                        except Exception:
                            continue
                    rc.set(f'trading:model_roles:{symbol}', _json.dumps(roles_norm, ensure_ascii=False))
            except Exception:
                pass
        # Не перетираем консенсус пустыми/дефолтными значениями
        if consensus is not None and isinstance(model_paths, list) and len(model_paths) > 0:
            symbol = symbols[0] if symbols else 'ALL'
            # Опционально: синхронизируем total_selected с фактическим списком
            try:
                c = consensus.get('counts') if isinstance(consensus, dict) else None
                if isinstance(c, dict):
                    before = dict(c)
                    c['total_selected'] = len(model_paths)
                    logging.info(f"[save_config] symbol={symbol} counts_in={before} -> counts_saved={c}")
            except Exception:
                pass
            rc.set(f'trading:consensus:{symbol}', _json.dumps(consensus, ensure_ascii=False))
            try:
                logging.info(f"[save_config] symbol={symbol} consensus saved")
            except Exception:
                pass
        if take_profit_pct is not None:
            rc.set('trading:take_profit_pct', str(take_profit_pct))
        if stop_loss_pct is not None:
            rc.set('trading:stop_loss_pct', str(stop_loss_pct))
        if risk_management_type is not None:
            rc.set('trading:risk_management_type', str(risk_management_type))
        # Per‑symbol сохранение для ATR‑/Trailing‑режимов (если указан символ)
        try:
            if symbols:
                sym_ps = symbols[0]
                # Q-gate per-symbol
                try:
                    if qgate_maxq is not None:
                        rc.set(f'trading:qgate_maxq:{sym_ps}', str(float(qgate_maxq)))
                    if qgate_gapq is not None:
                        rc.set(f'trading:qgate_gapq:{sym_ps}', str(float(qgate_gapq)))
                except Exception:
                    pass
                if atr_k is not None:
                    rc.set(f'trading:atr_k:{sym_ps}', str(atr_k))
                if atr_m is not None:
                    rc.set(f'trading:atr_m:{sym_ps}', str(atr_m))
                if atr_min_sl_mult is not None:
                    rc.set(f'trading:atr_min_sl_mult:{sym_ps}', str(atr_min_sl_mult))
                if risk_stop_mode is not None:
                    rc.set(f'trading:risk_stop_mode:{sym_ps}', str(risk_stop_mode))
                # Trailing per-symbol
                if trailing_enabled is not None:
                    try:
                        rc.set(f'trading:trailing_enabled:{sym_ps}', '1' if (str(trailing_enabled).strip().lower() in ('1','true','yes','on')) else '0')
                    except Exception:
                        rc.set(f'trading:trailing_enabled:{sym_ps}', str(trailing_enabled))
                if trailing_mode is not None:
                    rc.set(f'trading:trailing_mode:{sym_ps}', str(trailing_mode))
                if atr_trail_mult is not None:
                    rc.set(f'trading:atr_trail_mult:{sym_ps}', str(atr_trail_mult))
                if atr_trail_activate_pct is not None:
                    rc.set(f'trading:atr_trail_activate_pct:{sym_ps}', str(atr_trail_activate_pct))
                if trailing_activate_mode is not None:
                    rc.set(f'trading:trailing_activate_mode:{sym_ps}', str(trailing_activate_mode))
                if trailing_activate_value is not None:
                    rc.set(f'trading:trailing_activate_value:{sym_ps}', str(trailing_activate_value))
        except Exception:
            pass
        # Глобальные ключи ATR‑режима
        if atr_k is not None:
            rc.set('trading:atr_k', str(atr_k))
        if atr_m is not None:
            rc.set('trading:atr_m', str(atr_m))
        if atr_min_sl_mult is not None:
            rc.set('trading:atr_min_sl_mult', str(atr_min_sl_mult))
        if risk_stop_mode is not None:
            rc.set('trading:risk_stop_mode', str(risk_stop_mode))
        # Глобальные ключи трейлинга
        if trailing_enabled is not None:
            try:
                rc.set('trading:trailing_enabled', '1' if (str(trailing_enabled).strip().lower() in ('1','true','yes','on')) else '0')
            except Exception:
                rc.set('trading:trailing_enabled', str(trailing_enabled))
        if trailing_mode is not None:
            rc.set('trading:trailing_mode', str(trailing_mode))
        if atr_trail_mult is not None:
            rc.set('trading:atr_trail_mult', str(atr_trail_mult))
        if atr_trail_activate_pct is not None:
            rc.set('trading:atr_trail_activate_pct', str(atr_trail_activate_pct))
        if trailing_activate_mode is not None:
            rc.set('trailing_activate_mode', str(trailing_activate_mode))
        if trailing_activate_value is not None:
            rc.set('trailing_activate_value', str(trailing_activate_value))
        if account_pct is not None:
            try:
                ap = int(account_pct)
                if 1 <= ap <= 100:
                    rc.set('trading:account_pct', str(ap))
                    # Дублируем в Postgres app_settings, чтобы настраивать через /settings
                    try:
                        from utils.settings_store import ensure_settings_table, upsert_setting as _upsert_setting
                        ensure_settings_table()
                        _upsert_setting(
                            scope='trading',
                            group='sizing',
                            key='ACCOUNT_PCT',
                            value_type='number',
                            label='Account %',
                            description='Доля свободного USDT для входа (1..100)',
                            is_secret=False,
                            value=str(ap),
                        )
                    except Exception:
                        pass
            except Exception:
                pass
        # Глобальные Q-gate значения (fallback)
        try:
            if qgate_maxq is not None:
                rc.set('trading:qgate_maxq', str(float(qgate_maxq)))
            if qgate_gapq is not None:
                rc.set('trading:qgate_gapq', str(float(qgate_gapq)))
        except Exception:
            pass
        # debug_buy более не сохраняем в Redis: управляется только через ENV
        try:
            if exit_mode and symbols:
                sym = (symbols[0] if isinstance(symbols, list) and symbols else 'ALL')
                rc.set(f'trading:exit_mode:{sym}', exit_mode)
            if leverage is not None and symbols:
                try:
                    lev_int = int(leverage)
                    if 1 <= lev_int <= 5:
                        sym = (symbols[0] if isinstance(symbols, list) and symbols else 'ALL')
                        rc.set(f'trading:leverage:{sym}', str(lev_int))
                except Exception:
                    pass
        except Exception:
            pass
        try:
            logging.info("[save_config] ✓ done")
        except Exception:
            pass
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.post('/api/trading/start')
def start_trading():
    """Запуск новой торговой session без legacy fallback."""
    try:
        data = request.get_json() or {}
        symbols = data.get('symbols') or [data.get('symbol') or 'BTCUSDT']
        try:
            from utils.cctx_utils import normalize_to_db as _normalize_to_db
            if isinstance(symbols, (list, tuple)):
                symbols = [_normalize_to_db(str(s)) for s in symbols if s]
            elif isinstance(symbols, str) and symbols:
                symbols = [_normalize_to_db(symbols)]
        except Exception:
            symbols = [str(symbols[0] if isinstance(symbols, (list, tuple)) and symbols else symbols).replace('/', '').replace(' ', '').upper().strip()]

        if not isinstance(symbols, list) or len(symbols) != 1:
            return jsonify({'success': False, 'error': 'exactly one symbol is required per session'}), 400

        symbol = str(symbols[0] or '').strip().upper()
        account_id = str(data.get('account_id') or '').strip()
        if not account_id:
            return jsonify({'success': False, 'error': 'account_id is required'}), 400

        rc = get_redis_client()
        active_same = get_active_session_for_account_symbol(rc, account_id, symbol)
        if active_same:
            return jsonify({'success': False, 'error': f'active session already exists for {symbol} on account {account_id}', 'session_id': active_same}), 409

        model_paths = data.get('model_paths') if isinstance(data.get('model_paths'), list) else []
        model_path = str(data.get('model_path') or (model_paths[0] if model_paths else '')).strip() or None
        if model_path and not model_paths:
            model_paths = [model_path]
        if not model_paths:
            return jsonify({'success': False, 'error': 'model_paths is required'}), 400

        try:
            from utils.settings_store import ensure_settings_table, get_setting_value
            ensure_settings_table()
            api_key = (get_setting_value('api', 'bybit', f'BYBIT_{account_id}_API_KEY') or '').strip()
            secret_key = (get_setting_value('api', 'bybit', f'BYBIT_{account_id}_SECRET_KEY') or '').strip()
            label = (get_setting_value('api', 'bybit', f'BYBIT_{account_id}_LABEL') or f'Account {account_id}').strip()
            if (not api_key) or (not secret_key):
                return jsonify({
                    'success': False,
                    'error': f'Bybit API не настроено для выбранного аккаунта "{label}" (id={account_id}). '
                             f'Нужны настройки BYBIT_{account_id}_API_KEY и BYBIT_{account_id}_SECRET_KEY в Postgres.'
                }), 400
        except Exception:
            return jsonify({'success': False, 'error': f'Не удалось проверить Bybit API в Postgres для account_id={account_id}'}), 400

        execution_mode = str(data.get('execution_mode') or '').strip() or None
        limit_config = data.get('limit_config') if isinstance(data.get('limit_config'), dict) else None
        immediate = bool(data.get('immediate') or False)
        immediate_side = str(data.get('side') or '').lower() if data.get('side') else None
        exit_mode = str(data.get('exit_mode') or '').strip() or None
        leverage = data.get('leverage')
        account_pct = data.get('account_pct')
        risk_management_type = data.get('risk_management_type')
        take_profit_pct = data.get('take_profit_pct')
        stop_loss_pct = data.get('stop_loss_pct')
        risk_stop_mode = str(data.get('risk_stop_mode') or '').strip() or None
        trailing_enabled = data.get('trailing_enabled')
        trailing_mode = str(data.get('trailing_mode') or '').strip() or None
        trailing_activate_mode = str(data.get('trailing_activate_mode') or '').strip() or None
        trailing_activate_value = data.get('trailing_activate_value')
        atr_k = data.get('atr_k')
        atr_m = data.get('atr_m')
        atr_min_sl_mult = data.get('atr_min_sl_mult')
        atr_trail_mult = data.get('atr_trail_mult')
        max_hold_steps = data.get('max_hold_steps')
        xgb_signal_exit_enabled = data.get('xgb_signal_exit_enabled')
        xgb_signal_exit_start_step = data.get('xgb_signal_exit_start_step')
        xgb_signal_exit_window = data.get('xgb_signal_exit_window')
        xgb_signal_exit_threshold = data.get('xgb_signal_exit_threshold')
        xgb_signal_exit_long_enabled = data.get('xgb_signal_exit_long_enabled')
        xgb_signal_exit_long_start_step = data.get('xgb_signal_exit_long_start_step')
        xgb_signal_exit_long_threshold = data.get('xgb_signal_exit_long_threshold')
        xgb_signal_exit_short_enabled = data.get('xgb_signal_exit_short_enabled')
        xgb_signal_exit_short_start_step = data.get('xgb_signal_exit_short_start_step')
        xgb_signal_exit_short_threshold = data.get('xgb_signal_exit_short_threshold')
        xgb_postexit_guard_enabled = data.get('xgb_postexit_guard_enabled')
        xgb_postexit_cooldown_candles = data.get('xgb_postexit_cooldown_candles')
        xgb_postexit_threshold_boost_pct = data.get('xgb_postexit_threshold_boost_pct')
        xgb_postexit_boost_candles = data.get('xgb_postexit_boost_candles')
        direction = str(data.get('direction') or '').strip().lower() or None
        trade_mode = str(data.get('trade_mode') or '').strip() or None
        if trade_mode not in ('single', 'long-short'):
            trade_mode = None
        model_roles = data.get('model_roles') if isinstance(data.get('model_roles'), dict) else None
        consensus = data.get('consensus') if isinstance(data.get('consensus'), dict) else None
        qgate_maxq = data.get('qgate_maxq')
        qgate_gapq = data.get('qgate_gapq')

        is_xgb_model = bool(model_path and '/models/xgb/' in str(model_path).replace('\\', '/'))
        _, _, xgb_default_stop_loss_pct = _read_xgb_runtime_meta(model_path if is_xgb_model else None)
        if leverage in (None, '') and is_xgb_model:
            model_leverage_key = _xgb_model_leverage_key(model_path)
            leverage = rc.get(model_leverage_key) if model_leverage_key else None
        try:
            leverage = max(1, min(5, int(float(leverage)))) if leverage not in (None, '') else 1
        except Exception:
            leverage = 1
        if risk_management_type in (None, '') and is_xgb_model and xgb_default_stop_loss_pct not in (None, ''):
            risk_management_type = 'exchange_orders'
        if stop_loss_pct in (None, '') and is_xgb_model and xgb_default_stop_loss_pct not in (None, ''):
            stop_loss_pct = float(xgb_default_stop_loss_pct)
        if take_profit_pct in (None, '') and is_xgb_model:
            take_profit_pct = float(XGB_DEFAULT_TAKE_PROFIT_PCT)
        xgb_side_defaults: Dict[str, Any] = {}
        if is_xgb_model:
            side_source = {
                'model_path': model_path,
                'model_paths': [str(p) for p in model_paths],
                'model_roles': model_roles or {},
                'direction': direction,
            }
            for side_name in ('long', 'short'):
                side_path = _xgb_side_model_path(side_source, side_name)
                _, _, side_sl, side_tp = _read_xgb_side_trade_meta(side_path) if side_path else (None, None, None, None)
                if side_sl not in (None, ''):
                    xgb_side_defaults[f'stop_loss_pct_{side_name}'] = float(side_sl)
                if side_tp not in (None, ''):
                    xgb_side_defaults[f'take_profit_pct_{side_name}'] = float(side_tp)

        session_doc = create_session(rc, {
            'symbol': symbol,
            'account_id': account_id,
            'model_path': model_path,
            'model_paths': [str(p) for p in model_paths],
            'direction': direction,
            'trade_mode': trade_mode,
            'model_roles': model_roles or {},
            'consensus': consensus or {},
            'execution_mode': execution_mode,
            'limit_config': limit_config or {},
            'exit_mode': exit_mode,
            'leverage': leverage,
            'account_pct': account_pct,
            'risk_management_type': risk_management_type,
            'take_profit_pct': take_profit_pct,
            'stop_loss_pct': stop_loss_pct,
            **xgb_side_defaults,
            'risk_stop_mode': risk_stop_mode,
            'atr_k': atr_k,
            'atr_m': atr_m,
            'atr_min_sl_mult': atr_min_sl_mult,
            'trailing_enabled': trailing_enabled,
            'trailing_mode': trailing_mode,
            'trailing_activate_mode': trailing_activate_mode,
            'trailing_activate_value': trailing_activate_value,
            'atr_trail_mult': atr_trail_mult,
            'max_hold_steps': max_hold_steps,
            'xgb_signal_exit_enabled': xgb_signal_exit_enabled,
            'xgb_signal_exit_start_step': xgb_signal_exit_start_step,
            'xgb_signal_exit_window': xgb_signal_exit_window,
            'xgb_signal_exit_threshold': xgb_signal_exit_threshold,
            'xgb_signal_exit_long_enabled': xgb_signal_exit_long_enabled,
            'xgb_signal_exit_long_start_step': xgb_signal_exit_long_start_step,
            'xgb_signal_exit_long_threshold': xgb_signal_exit_long_threshold,
            'xgb_signal_exit_short_enabled': xgb_signal_exit_short_enabled,
            'xgb_signal_exit_short_start_step': xgb_signal_exit_short_start_step,
            'xgb_signal_exit_short_threshold': xgb_signal_exit_short_threshold,
            'xgb_postexit_guard_enabled': xgb_postexit_guard_enabled,
            'xgb_postexit_cooldown_candles': xgb_postexit_cooldown_candles,
            'xgb_postexit_threshold_boost_pct': xgb_postexit_threshold_boost_pct,
            'xgb_postexit_boost_candles': xgb_postexit_boost_candles,
        })
        session_id = str(session_doc.get('session_id'))

        runtime_scalars = {
            'direction': direction,
            'trade_mode': trade_mode,
            'execution_mode': execution_mode,
            'exit_mode': exit_mode,
            'leverage': leverage,
            'risk_management_type': risk_management_type,
            'take_profit_pct': take_profit_pct,
            'stop_loss_pct': stop_loss_pct,
            'risk_stop_mode': risk_stop_mode,
            'atr_k': atr_k,
            'atr_m': atr_m,
            'atr_min_sl_mult': atr_min_sl_mult,
            'trailing_enabled': ('1' if str(trailing_enabled).strip().lower() in ('1', 'true', 'yes', 'on') else '0') if trailing_enabled is not None else None,
            'trailing_mode': trailing_mode,
            'trailing_activate_mode': trailing_activate_mode,
            'trailing_activate_value': trailing_activate_value,
            'atr_trail_mult': atr_trail_mult,
            'max_hold_steps': max_hold_steps,
            'xgb_signal_exit_enabled': ('1' if str(xgb_signal_exit_enabled).strip().lower() in ('1', 'true', 'yes', 'on') else '0') if xgb_signal_exit_enabled is not None else None,
            'xgb_signal_exit_start_step': xgb_signal_exit_start_step,
            'xgb_signal_exit_window': xgb_signal_exit_window,
            'xgb_signal_exit_threshold': xgb_signal_exit_threshold,
            'xgb_signal_exit_long_enabled': ('1' if str(xgb_signal_exit_long_enabled).strip().lower() in ('1', 'true', 'yes', 'on') else '0') if xgb_signal_exit_long_enabled is not None else None,
            'xgb_signal_exit_long_start_step': xgb_signal_exit_long_start_step,
            'xgb_signal_exit_long_threshold': xgb_signal_exit_long_threshold,
            'xgb_signal_exit_short_enabled': ('1' if str(xgb_signal_exit_short_enabled).strip().lower() in ('1', 'true', 'yes', 'on') else '0') if xgb_signal_exit_short_enabled is not None else None,
            'xgb_signal_exit_short_start_step': xgb_signal_exit_short_start_step,
            'xgb_signal_exit_short_threshold': xgb_signal_exit_short_threshold,
            'xgb_postexit_guard_enabled': ('1' if str(xgb_postexit_guard_enabled).strip().lower() in ('1', 'true', 'yes', 'on') else '0') if xgb_postexit_guard_enabled is not None else None,
            'xgb_postexit_cooldown_candles': xgb_postexit_cooldown_candles,
            'xgb_postexit_threshold_boost_pct': xgb_postexit_threshold_boost_pct,
            'xgb_postexit_boost_candles': xgb_postexit_boost_candles,
            'xgb_entry_threshold': data.get('xgb_entry_threshold'),
            'qgate_maxq': qgate_maxq,
            'qgate_gapq': qgate_gapq,
        }
        for key, value in runtime_scalars.items():
            if value not in (None, ''):
                set_runtime_value(rc, session_id, key, value)
        set_runtime_json(rc, session_id, 'consensus', consensus or {})
        set_runtime_json(rc, session_id, 'model_roles', model_roles or {})
        set_runtime_json(rc, session_id, 'limit_config', limit_config or {})

        initial_status = {
            'success': True,
            'session_id': session_id,
            'is_trading': True,
            'trading_status': 'Активна',
            'trading_status_emoji': '🟢',
            'trading_status_full': '🟢 Активна',
            'symbol': symbol,
            'symbol_display': symbol,
            'bybit_account_id': account_id,
            'amount': None,
            'amount_display': 'Не указано',
            'amount_usdt': 0.0,
            'position': None,
            'trades_count': 0,
            'balance': {},
            'current_price': 0.0,
            'last_model_prediction': None,
        }
        set_session_status(rc, session_id, initial_status)

        task = start_trading_task.apply_async(args=[session_id], countdown=0, expires=300, queue='trade')

        try:
            from tasks.celery_task_trade import start_execution_strategy as _start_exec
            if execution_mode and immediate and immediate_side in ('buy', 'sell'):
                _start_exec.apply_async(kwargs={
                    'session_id': session_id,
                    'symbol': symbol,
                    'execution_mode': execution_mode,
                    'side': immediate_side,
                    'qty': data.get('qty'),
                    'limit_config': (limit_config or {}),
                    'leverage': leverage,
                }, queue='trade')
        except Exception:
            pass

        return jsonify({
            'success': True,
            'message': 'Trading session started',
            'session_id': session_id,
            'task_id': task.id,
        }), 200
    except Exception as e:
        logging.error(f"Ошибка запуска торговли: {e}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

@trading_bp.post('/api/trading/execute_now')
def execute_now():
    """
    Разовый немедленный торговый шаг для конкретного символа (без ожидания периодики).
    Использует тот же пайплайн, что и периодический цикл, но единично.
    По умолчанию работает в боевом режиме (не dry_run), чтобы реально исполнять сделки.
    """
    try:
        data = request.get_json(silent=True) or {}
        symbols = data.get('symbols') or [data.get('symbol') or 'BTCUSDT']
        # dry_run не передаём в таск для совместимости со старыми воркерами
        # Передаём напрямую в очередь trade ту же задачу, что и периодика
        from tasks.celery_task_trade import execute_trade as _exec
        res = _exec.apply_async(kwargs={
            'symbols': symbols,
            'model_path': None,
            'model_paths': None,
        }, queue='trade')
        return jsonify({'success': True, 'enqueued': True, 'task_id': res.id}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.post('/api/trading/stop')
def stop_trading():
    """Остановка торговли в контейнере trading_agent"""
    try:
        # Подключаемся к Docker
        client = docker.from_env()
        
        try:
            # Получаем контейнер medoedai
            container = client.containers.get('medoedai')
            
            # Проверяем что контейнер запущен
            if container.status != 'running':
                return jsonify({
                    'success': False, 
                    'error': f'Контейнер medoedai не запущен. Статус: {container.status}'
                }), 500
            
            # Получаем ранее выбранный путь к модели (если есть)
            model_path = None
            try:
                rc = get_redis_client()
                mp = rc.get('trading:model_path')
                if mp:
                    model_path = mp.decode('utf-8')
            except Exception:
                pass

            # Останавливаем торговлю через exec
            if model_path:
                cmd = f'python -c "import json; from trading_agent.trading_agent import TradingAgent; agent = TradingAgent(model_path=\\"{model_path}\\"); result = agent.stop_trading(); print(\\"RESULT: \\" + json.dumps(result))"'
            else:
                cmd = 'python -c "import json; from trading_agent.trading_agent import TradingAgent; agent = TradingAgent(); result = agent.stop_trading(); print(\\"RESULT: \\" + json.dumps(result))"'
            
            exec_result = container.exec_run(cmd, tty=True)
            
            # Логируем результат выполнения команды
            logging.info(f"Stop trading - Exit code: {exec_result.exit_code}")
            if exec_result.output:
                output_str = exec_result.output.decode('utf-8')
                logging.info(f"Stop trading - Output: {output_str}")
            
            if exec_result.exit_code == 0:
                output = exec_result.output.decode('utf-8') if exec_result.output else ""
                # Ищем результат в выводе
                if 'RESULT:' in output:
                    result_str = output.split('RESULT:')[1].strip()
                    try:
                        result = json.loads(result_str)
                        return jsonify(result), 200
                    except Exception as parse_error:
                        logging.error(f"Ошибка парсинга результата: {parse_error}")
                        return jsonify({
                            'success': True,
                            'message': 'Торговля остановлена',
                            'output': output
                        }), 200
                else:
                    return jsonify({
                        'success': True,
                        'message': 'Торговля остановлена',
                        'output': output
                    }), 200
            else:
                error_output = exec_result.output.decode('utf-8') if exec_result.output else "No error output"
                logging.error(f"Ошибка выполнения команды остановки торговли: {error_output}")
                return jsonify({
                    'success': False,
                    'error': f'Ошибка выполнения команды: {error_output}'
                }), 500
                
        except docker.errors.NotFound:
            return jsonify({
                'success': False, 
                'error': 'Контейнер medoedai не найден. Запустите docker-compose up medoedai'
            }), 500
        except Exception as e:
            return jsonify({
                'success': False, 
                'error': f'Ошибка Docker: {str(e)}'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Ошибка stop_trading: {str(e)}'
        }), 500

@trading_bp.post('/api/trading/stop_session')
def stop_trading_symbol():
    try:
        data = request.get_json(silent=True) or {}
        session_id = str(data.get('session_id') or '').strip()
        if not session_id:
            return jsonify({'success': False, 'error': 'session_id is required'}), 400
        rc = get_redis_client()
        session_doc, err = _get_session_or_400(rc, session_id)
        if err:
            return err
        symbol = str(session_doc.get('symbol') or '').strip().upper()
        try:
            aid = rc.get(f'exec:active_intent:{session_id}')
            if aid:
                try:
                    rc.delete(f'exec:intent:{aid}')
                except Exception:
                    pass
            rc.delete(f'exec:active_intent:{session_id}')
        except Exception:
            pass
        try:
            set_runtime_value(rc, session_id, 'disabled', '1')
        except Exception:
            pass
        try:
            status = get_session_status(rc, session_id) or {}
            if not isinstance(status, dict):
                status = {}
            status.update({
                'session_id': session_id,
                'symbol': symbol,
                'is_trading': False,
                'trading_status': 'Остановлена',
                'trading_status_emoji': '🔴',
                'trading_status_full': '🔴 Остановлена'
            })
            set_session_status(rc, session_id, status)
        except Exception:
            pass
        try:
            rc.delete(session_lock_key(session_id))
        except Exception:
            pass
        remove_session(rc, session_id)
        return jsonify({'success': True, 'symbol': symbol, 'session_id': session_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.post('/api/trading/force_trailing')
def force_trailing_stop():
    """Принудительно выставляет биржевой trailing stop для открытой позиции."""
    try:
        data = request.get_json(silent=True) or {}
        symbol = str(data.get('symbol') or '').strip().upper()
        if not symbol:
            return jsonify({'success': False, 'error': 'symbol is required'}), 400
        if not re.match(r'^[A-Z]{2,10}USDT$', symbol):
            return jsonify({'success': False, 'error': 'symbol must match [A-Z]{2,10}USDT'}), 400

        rc = get_redis_client()
        if rc is None:
            return jsonify({'success': False, 'error': 'redis not available'}), 500

        acc_id = rc.get(f'trading:account_id:{symbol}')
        if isinstance(acc_id, (bytes, bytearray)):
            acc_id = acc_id.decode('utf-8', errors='ignore')
        if not acc_id:
            return jsonify({'success': False, 'error': f'account_id not set for {symbol}'}), 400

        def _get_rc(key: str):
            v = rc.get(key)
            if isinstance(v, (bytes, bytearray)):
                v = v.decode('utf-8', errors='ignore')
            return v

        trailing_enabled_raw = _get_rc(f'trading:trailing_enabled:{symbol}')
        if trailing_enabled_raw is None:
            return jsonify({'success': False, 'error': f'trailing_enabled not set for {symbol}'}), 400
        trailing_enabled = str(trailing_enabled_raw).strip().lower() in ('1','true','yes','on')
        if not trailing_enabled:
            return jsonify({'success': False, 'error': f'trailing is disabled for {symbol}'}), 400

        risk_stop_mode = _get_rc(f'trading:risk_stop_mode:{symbol}')
        if not risk_stop_mode:
            return jsonify({'success': False, 'error': f'risk_stop_mode not set for {symbol}'}), 400
        if str(risk_stop_mode).strip() != 'atr_trailing':
            return jsonify({'success': False, 'error': f'risk_stop_mode={risk_stop_mode} (need atr_trailing)'}), 400

        trailing_mode = _get_rc(f'trading:trailing_mode:{symbol}')
        atr_trail_mult_raw = _get_rc(f'trading:atr_trail_mult:{symbol}')
        if atr_trail_mult_raw is None or str(atr_trail_mult_raw).strip() == '':
            return jsonify({'success': False, 'error': f'atr_trail_mult not set for {symbol}'}), 400
        try:
            atr_trail_mult = float(atr_trail_mult_raw)
        except Exception:
            return jsonify({'success': False, 'error': f'atr_trail_mult invalid for {symbol}'}), 400

        trailing_activate_mode = _get_rc(f'trading:trailing_activate_mode:{symbol}')
        trailing_activate_value_raw = _get_rc(f'trading:trailing_activate_value:{symbol}')
        if trailing_activate_mode is None or trailing_activate_value_raw is None:
            return jsonify({'success': False, 'error': f'trailing activation not set for {symbol}'}), 400
        try:
            trailing_activate_value = float(trailing_activate_value_raw)
        except Exception:
            return jsonify({'success': False, 'error': f'trailing_activate_value invalid for {symbol}'}), 400

        try:
            from trading_agent.trading_agent import TradingAgent
        except Exception as e:
            return jsonify({'success': False, 'error': f'TradingAgent import error: {e}'}), 500

        agent = TradingAgent(symbol=symbol)
        agent.symbol = symbol
        agent.base_symbol = symbol
        if not getattr(agent, 'exchange', None):
            return jsonify({'success': False, 'error': 'exchange not initialized'}), 500

        try:
            agent._restore_position_from_exchange()
        except Exception as e:
            return jsonify({'success': False, 'error': f'position restore failed: {e}'}), 500

        pos = getattr(agent, 'current_position', None) or {}
        entry_price = pos.get('entry_price')
        amount = pos.get('amount')
        try:
            current_price = float(agent._get_current_price() or 0.0)
        except Exception:
            current_price = 0.0
        trailing_now = pos.get('trailingStop') or pos.get('trailing_stop') or pos.get('trailing_stop_price')
        if trailing_now not in (None, '', 0):
            return jsonify({'success': False, 'error': f'trailing already set for {symbol}'}), 409
        if not entry_price or not amount:
            return jsonify({'success': False, 'error': f'no open position for {symbol}'}), 400
        if current_price <= 0:
            return jsonify({'success': False, 'error': f'current price not available for {symbol}'}), 400

        pos_side = str(pos.get('type') or 'long').strip().lower()
        try:
            trailing_result = setup_trailing_stop_bybit(
                agent.exchange,
                symbol,
                float(amount),
                float(current_price),
                trailing_mode,
                trailing_activate_mode,
                trailing_activate_value,
                float(atr_trail_mult),
                side=pos_side,
            )
        except Exception as e_trail:
            try:
                rc.set(
                    f"trading:trailing:last_setup:{symbol}",
                    json.dumps(
                        {
                            "ts": time.time(),
                            "ok": False,
                            "symbol": symbol,
                            "source": "force_trailing",
                            "error": str(e_trail),
                        },
                        ensure_ascii=False,
                    ),
                )
            except Exception:
                pass
            return jsonify({'success': False, 'error': str(e_trail)}), 500

        try:
            rc.set(
                f"trading:trailing:last_setup:{symbol}",
                json.dumps(
                    {
                        "ts": time.time(),
                        "ok": True,
                        "symbol": symbol,
                        "source": "force_trailing",
                        "result": trailing_result,
                    },
                    ensure_ascii=False,
                ),
            )
        except Exception:
            pass

        return jsonify({'success': True, 'symbol': symbol, 'trailing': trailing_result}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.get('/api/trading/agents')
def list_trading_agents():
    try:
        rc = get_redis_client()
        # Bybit аккаунт теперь выбираем per-symbol (trading:account_id:<SYMBOL>), с fallback на глобальный ключ.
        try:
            from utils.accounts import get_bybit_account
        except Exception:
            get_bybit_account = None
        # Базовый список символов (можно расширить)
        known = ['BTCUSDT','ETHUSDT','SOLUSDT','TONUSDT','ADAUSDT','BNBUSDT','XRPUSDT']
        agents = []
        import os as _os
        for sym in known:
            # Активность по локам (могут отсутствовать после рестарта)
            try:
                ttl = rc.ttl(f'trading:agent_lock:{sym}')
            except Exception:
                ttl = None
            # Статус per-symbol
            try:
                raw = rc.get(f'trading:status:{sym}')
                status = json.loads(raw) if raw else None
            except Exception:
                status = None
            # Фолбэк активности: если статус говорит, что торгуем — считаем активным
            try:
                is_trading_flag = bool(status and (status.get('is_trading') is True or str(status.get('trading_status') or '').strip() in ('Активна','🟢 Активна')))
            except Exception:
                is_trading_flag = False
            is_active = ((ttl is not None and isinstance(ttl, int) and ttl > 0) or is_trading_flag)
            # Консенсус per-symbol
            try:
                cons_raw = rc.get(f'trading:consensus:{sym}')
                consensus = json.loads(cons_raw) if cons_raw else None
            except Exception:
                consensus = None
            try:
                logging.info(f"[agents] {sym}: raw_consensus={consensus}")
            except Exception:
                pass
            # Список моделей
            total_models = 0
            try:
                mps_raw = rc.get(f'trading:model_paths:{sym}')
                if mps_raw:
                    parsed = json.loads(mps_raw)
                    if isinstance(parsed, list):
                        total_models = len(parsed)
            except Exception:
                total_models = 0
            # Вычисляем требуемые пороги так же, как оркестратор
            def _required(total_sel: int, counts: dict | None, regime: str) -> tuple[int,int,int,str,int]:
                req_flat = None
                req_trend = None
                try:
                    c = counts or {}
                    if isinstance(c.get('flat'), (int, float)):
                        req_flat = int(max(1, c.get('flat')))
                    if isinstance(c.get('trend'), (int, float)):
                        req_trend = int(max(1, c.get('trend')))
                except Exception:
                    pass
                default_req = 2 if total_sel >= 3 else max(1, total_sel)
                if req_flat is None:
                    req_flat = default_req
                if req_trend is None:
                    req_trend = default_req
                req_flat = int(min(max(1, req_flat), total_sel if total_sel>0 else 1))
                req_trend = int(min(max(1, req_trend), total_sel if total_sel>0 else 1))
                req_type = 'trend' if regime in ('uptrend','downtrend') else 'flat'
                required = (req_trend if req_type=='trend' else req_flat)
                return total_sel, req_flat, req_trend, req_type, required
            counts = (consensus or {}).get('counts') if isinstance(consensus, dict) else None
            regime = (status or {}).get('market_regime') or 'flat'
            # Насильно приводим total_selected к фактическому выбору моделей
            try:
                if isinstance(counts, dict):
                    before_ts = counts.get('total_selected')
                    counts['total_selected'] = total_models
                    try:
                        logging.info(f"[agents] {sym}: total_models={total_models}, counts_in_ts={before_ts}, counts_out_ts={counts.get('total_selected')}, flat={counts.get('flat')}, trend={counts.get('trend')}, regime={regime}")
                    except Exception:
                        pass
                    # Если исправили в памяти, сохраняем обратно в Redis
                    if before_ts != total_models and total_models > 0:
                        try:
                            consensus['counts'] = counts
                            rc.set(f'trading:consensus:{sym}', json.dumps(consensus, ensure_ascii=False))
                            logging.info(f"[agents] {sym}: FIXED Redis total_selected {before_ts} -> {total_models}")
                        except Exception as e:
                            logging.error(f"[agents] {sym}: Failed to fix Redis: {e}")
            except Exception:
                pass
            # Используем исправленное значение total_selected из counts
            corrected_total = counts.get('total_selected', total_models) if isinstance(counts, dict) else total_models
            total_sel, req_flat, req_trend, req_type, required = _required(corrected_total, counts, regime)
            try:
                logging.info(f"[agents] {sym}: required={required} ({req_type}), req_flat={req_flat}, req_trend={req_trend}")
            except Exception:
                pass
            # Настройки per-symbol/global
            try:
                exec_mode_v = rc.get(f'trading:execution_mode:{sym}')
                exit_mode_v = rc.get(f'trading:exit_mode:{sym}')
                leverage_v = rc.get(f'trading:leverage:{sym}')
                limit_cfg_raw = rc.get(f'trading:limit_config:{sym}')
                trade_mode_v = rc.get(f'trading:trade_mode:{sym}')
                model_roles_raw = rc.get(f'trading:model_roles:{sym}')
                risk_type_v = rc.get(f'trading:risk_management_type:{sym}') or rc.get('trading:risk_management_type')
                tp_pct_v = rc.get(f'trading:take_profit_pct:{sym}') or rc.get('trading:take_profit_pct')
                sl_pct_v = rc.get(f'trading:stop_loss_pct:{sym}') or rc.get('trading:stop_loss_pct')
                account_pct_v = rc.get('trading:account_pct')
                risk_stop_mode_v = rc.get(f'trading:risk_stop_mode:{sym}') or rc.get('trading:risk_stop_mode')
                # Trailing (пер-символ / глобально)
                trailing_enabled_v = rc.get(f'trading:trailing_enabled:{sym}') or rc.get('trading:trailing_enabled')
                trailing_mode_v = rc.get(f'trading:trailing_mode:{sym}') or rc.get('trading:trailing_mode')
                atr_trail_mult_v = rc.get(f'trading:atr_trail_mult:{sym}') or rc.get('trading:atr_trail_mult')
                atr_trail_activate_pct_v = rc.get(f'trading:atr_trail_activate_pct:{sym}') or rc.get('trading:atr_trail_activate_pct')
                trailing_activate_mode_v = rc.get(f'trading:trailing_activate_mode:{sym}') or rc.get('trailing_activate_mode')
                trailing_activate_value_v = rc.get(f'trading:trailing_activate_value:{sym}') or rc.get('trailing_activate_value')
                # ATR (только из Redis cache; без загрузки рынков/свечей, чтобы не тормозить /api/trading/agents)
                atr_1h_abs = None
                try:
                    try:
                        from utils.indicators import get_atr_1h_length
                        _len = int(get_atr_1h_length(default=34))
                    except Exception:
                        _len = 34
                    atr_key = f"atr:1h:{sym}:{_len}:auto"
                    atr_raw = rc.get(atr_key) if rc else None
                    if atr_raw:
                        try:
                            _s = atr_raw if isinstance(atr_raw, str) else atr_raw.decode("utf-8", errors="ignore")
                            _d = json.loads(_s) if _s else None
                            if isinstance(_d, dict) and _d.get("atr_abs") is not None:
                                atr_1h_abs = float(_d.get("atr_abs"))
                        except Exception:
                            atr_1h_abs = None
                except Exception:
                    atr_1h_abs = None
                try:
                    limit_cfg = json.loads(limit_cfg_raw) if limit_cfg_raw else None
                except Exception:
                    limit_cfg = None
                try:
                    if model_roles_raw and isinstance(model_roles_raw, (bytes, bytearray)):
                        model_roles_raw = model_roles_raw.decode('utf-8', errors='ignore')
                    model_roles_v = json.loads(model_roles_raw) if model_roles_raw else None
                    if not isinstance(model_roles_v, dict):
                        model_roles_v = {}
                except Exception:
                    model_roles_v = {}
                try:
                    if trade_mode_v and isinstance(trade_mode_v, (bytes, bytearray)):
                        trade_mode_v = trade_mode_v.decode('utf-8', errors='ignore')
                    trade_mode_v = str(trade_mode_v).strip() if trade_mode_v else None
                except Exception:
                    trade_mode_v = None
                # debug_buy из ENV (пер-символьно > глобально)
                dbg_env_sym = _os.getenv(f'DEBUG_BUY_{sym}')
                dbg_env_glob = _os.getenv('DEBUG_BUY')
                def _truthy(v):
                    try:
                        return str(v).strip().lower() in ('1','true','yes','on')
                    except Exception:
                        return False
                debug_buy = None
                if dbg_env_sym is not None:
                    debug_buy = _truthy(dbg_env_sym)
                elif dbg_env_glob is not None:
                    debug_buy = _truthy(dbg_env_glob)
            except Exception:
                exec_mode_v = exit_mode_v = leverage_v = None
                limit_cfg = None
                trade_mode_v = None
                model_roles_v = {}
                risk_type_v = None
                tp_pct_v = None
                sl_pct_v = None
                account_pct_v = None
                debug_buy = None

            # Q-gate per-symbol thresholds (MAXQ/GAPQ)
            try:
                qmax_v = rc.get(f'trading:qgate_maxq:{sym}') or rc.get('trading:qgate_maxq')
                qgap_v = rc.get(f'trading:qgate_gapq:{sym}') or rc.get('trading:qgate_gapq')
                if isinstance(qmax_v, (bytes, bytearray)):
                    qmax_v = qmax_v.decode('utf-8', errors='ignore')
                if isinstance(qgap_v, (bytes, bytearray)):
                    qgap_v = qgap_v.decode('utf-8', errors='ignore')
                qmax_f = float(str(qmax_v)) if qmax_v not in (None, '') else None
                qgap_f = float(str(qgap_v)) if qgap_v not in (None, '') else None
            except Exception:
                qmax_f = None
                qgap_f = None

            # Направление per-symbol
            try:
                dir_v = rc.get(f'trading:direction:{sym}')
                if isinstance(dir_v, (bytes, bytearray)):
                    dir_v = dir_v.decode('utf-8')
                sel_direction = str(dir_v).strip().lower() if dir_v else None
            except Exception:
                sel_direction = None

            agent_obj = {
                'symbol': sym,
                'active': bool(is_active),
                # Биржа/API (то, что выбираем при старте "Аккаунт Bybit")
                'exchange': 'Bybit',
                'account_id': None,
                'account_id_source': None,  # 'per_symbol' | 'global' | None
                'account_label': None,
                'account_api_key_masked': None,
                'status': status or {},
                'consensus': consensus or {},
                'total_models': total_models,
                'required_flat': req_flat,
                'required_trend': req_trend,
                'required_type': req_type,
                'required': required,
                'lock_ttl': (int(ttl) if ttl is not None else None),
                'settings': {
                    'execution_mode': (str(exec_mode_v).strip() if exec_mode_v else None),
                    'direction': sel_direction,
                    'trade_mode': (str(trade_mode_v).strip() if trade_mode_v else None),
                    'model_roles': (model_roles_v if isinstance(model_roles_v, dict) else {}),
                    'exit_mode': (str(exit_mode_v).strip() if exit_mode_v else None),
                    'leverage': (int(str(leverage_v)) if leverage_v not in (None, '') else None),
                    'limit_config': (limit_cfg or {}),
                    'risk_management_type': (str(risk_type_v).strip() if risk_type_v else None),
                    'take_profit_pct': (float(str(tp_pct_v)) if tp_pct_v not in (None, '') else None),
                    'stop_loss_pct': (float(str(sl_pct_v)) if sl_pct_v not in (None, '') else None),
                    'risk_stop_mode': (str(risk_stop_mode_v).strip() if risk_stop_mode_v else None),
                    'account_pct': (int(str(account_pct_v)) if account_pct_v not in (None, '') else None),
                    # Trailing summary (если задано)
                    'trailing_enabled': (str(trailing_enabled_v).strip().lower() in ('1','true','yes','on') if trailing_enabled_v not in (None, '') else None),
                    'trailing_mode': (str(trailing_mode_v).strip() if trailing_mode_v else None),
                    'atr_trail_mult': (float(str(atr_trail_mult_v)) if atr_trail_mult_v not in (None, '') else None),
                    'atr_trail_activate_pct': (float(str(atr_trail_activate_pct_v)) if atr_trail_activate_pct_v not in (None, '') else None),
                    'trailing_activate_mode': (str(trailing_activate_mode_v).strip() if trailing_activate_mode_v else None),
                    'trailing_activate_value': (float(str(trailing_activate_value_v)) if trailing_activate_value_v not in (None, '') else None),
                    'atr_1h_abs': (float(atr_1h_abs) if atr_1h_abs not in (None, '') else None),
                    'debug_buy_env': debug_buy,
                    'qgate_maxq': qmax_f,
                    'qgate_gapq': qgap_f,
                }
            }
            # Bybit account per-symbol (fallback to global)
            try:
                acc_id = None
                acc_src = None
                if rc:
                    acc_id = rc.get(f'trading:account_id:{sym}')
                    if isinstance(acc_id, (bytes, bytearray)):
                        acc_id = acc_id.decode('utf-8', errors='ignore')
                    acc_id = str(acc_id).strip() if acc_id else None
                    if acc_id:
                        acc_src = 'per_symbol'
                    else:
                        acc_id = rc.get('trading:account_id')
                        if isinstance(acc_id, (bytes, bytearray)):
                            acc_id = acc_id.decode('utf-8', errors='ignore')
                        acc_id = str(acc_id).strip() if acc_id else None
                        if acc_id:
                            acc_src = 'global'
                agent_obj['account_id'] = acc_id
                agent_obj['account_id_source'] = acc_src
                if acc_id and callable(get_bybit_account):
                    acc = get_bybit_account(acc_id)
                    if isinstance(acc, dict):
                        agent_obj['account_label'] = acc.get('label')
                        agent_obj['account_api_key_masked'] = acc.get('api_key_masked')
            except Exception:
                pass
            try:
                logging.info(f"[agents] {sym}: agent_obj={agent_obj}")
            except Exception:
                pass
            agents.append(agent_obj)
        return jsonify({'success': True, 'agents': agents, 'ts': datetime.utcnow().isoformat()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.get('/api/trading/status')
def trading_status():
    """Статус торговли из контейнера trading_agent"""
    try:
        # Входная точка: фиксируем вызов эндпоинта
        try:
            logging.info("[trading_status] ▶ request received")
        except Exception:
            pass
        # Сначала пробуем быстрый статус из Redis (обновляется периодическим таском)
        try:
            _rc = get_redis_client()
            cached = _rc.get('trading:current_status')
            cached_ts = _rc.get('trading:current_status_ts')
            if cached:
                status_obj = json.loads(cached)
                # Проверим свежесть (не старее 6 минут, > интервала beat)
                from datetime import timedelta
                is_fresh = True
                try:
                    if cached_ts:
                        ts = datetime.fromisoformat(cached_ts)
                        is_fresh = datetime.utcnow() <= (ts + timedelta(minutes=6))
                except Exception:
                    is_fresh = True
                if is_fresh:
                    # Возвращаем плоскую структуру для совместимости с фронтендом
                    flat = {'success': True, 'agent_status': status_obj}
                    if isinstance(status_obj, dict):
                        flat.update(status_obj)
                    # Приложим exit_mode per-symbol, если есть
                    try:
                        sym = flat.get('symbol') or flat.get('symbol_display')
                        if sym:
                            em = _rc.get(f'trading:exit_mode:{sym}')
                            if em:
                                flat['exit_mode'] = em.decode('utf-8') if isinstance(em, (bytes, bytearray)) else str(em)
                    except Exception:
                        pass
                    # Добавим pending_order (DDD), если активен intent
                    try:
                        import time as _t
                        import redis as _r
                        _rc2 = _r.Redis(host='redis', port=6379, db=0, decode_responses=True)
                        sym = flat.get('symbol') or flat.get('symbol_display')
                        if sym:
                            aid = _rc2.get(f'exec:active_intent:{sym}')
                            if aid:
                                raw = _rc2.get(f'exec:intent:{aid}')
                                if raw:
                                    import json as _json
                                    data = _json.loads(raw)
                                    flat['pending_order'] = {
                                        'symbol': data.get('symbol'),
                                        'intent_id': data.get('intent_id'),
                                        'state': data.get('state'),
                                        'price': data.get('price'),
                                        'attempts': data.get('attempts'),
                                        'age_sec': int(max(0, (_t.time() - float(data.get('created_at') or 0)))) if data.get('created_at') else None,
                                        'last_error': data.get('last_error')
                                    }
                    except Exception:
                        pass
                    try:
                        logging.info(f"[trading_status] ✓ using cached status | keys={list(flat.keys())}")
                        # Краткий обзор важных полей
                        logging.info("[trading_status] summary: is_trading=%s, position=%s, trades_count=%s",
                                    flat.get('is_trading'), bool(flat.get('position') or flat.get('current_position')), flat.get('trades_count'))
                    except Exception:
                        pass
                    return jsonify(flat), 200
        except Exception:
            pass

        # Нет свежего статуса в Redis — возвращаем понятный OFF статус для UI
        try:
            # Попробуем дополнить OFF статус сведениями о pending intent (DDD)
            pending_block = None
            try:
                import redis as _r
                _rc2 = _r.Redis(host='redis', port=6379, db=0, decode_responses=True)
                # Пройдём по известным символам и найдём active_intent
                known = ['BTCUSDT','ETHUSDT','SOLUSDT','TONUSDT','ADAUSDT','BNBUSDT','XRPUSDT']
                for sym in known:
                    aid = _rc2.get(f'exec:active_intent:{sym}')
                    if aid:
                        raw = _rc2.get(f'exec:intent:{aid}')
                        if raw:
                            import json as _json
                            data = _json.loads(raw)
                            pending_block = {
                                'symbol': data.get('symbol'),
                                'intent_id': data.get('intent_id'),
                                'state': data.get('state'),
                                'price': data.get('price'),
                                'attempts': data.get('attempts'),
                                'age_sec': int(max(0, (time.time() - float(data.get('created_at') or 0)))) if data.get('created_at') else None,
                                'last_error': data.get('last_error')
                            }
                            break
                    # если нашли — прекращаем цикл
                    if pending_block:
                        break
            except Exception:
                pending_block = None

            default_status = {
                'success': True,
                'is_trading': False,
                'trading_status': 'Остановлена',
                'trading_status_emoji': '🔴',
                'trading_status_full': '🔴 Остановлена (агент не запущен)',
                'symbol': None,
                'symbol_display': 'Не указана',
                'amount': None,
                'amount_display': 'Не указано',
                'amount_usdt': 0.0,
                'position': None,
                'trades_count': 0,
                'balance': {},
                'current_price': 0.0,
                'last_model_prediction': None,
                'is_fresh': False,
                'reason': 'status not available in redis',
                'pending_order': pending_block
            }
            flat = {'success': True, 'agent_status': default_status}
            flat.update(default_status)
            try:
                logging.info("[trading_status] ⚠ no redis status, returning OFF state")
            except Exception:
                pass
            return jsonify(flat), 200
        except Exception:
            return jsonify({'success': False, 'error': 'status not available in redis', 'is_fresh': False}), 200
            
    except Exception as e:
        logging.error(f"Ошибка получения статуса: {e}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

# Добавляем остальные торговые эндпоинты
@trading_bp.get('/api/trading/status_all')
def trading_status_all():
    """Итоговый статус: активные agents по session_id."""
    try:
        from utils.settings_store import get_setting_value, ensure_settings_table
        ensure_settings_table()
        master_sessions_raw = get_setting_value('trading', 'master', 'MASTER_SESSION_IDS')
        try:
            master_sessions = set(json.loads(master_sessions_raw)) if master_sessions_raw else set()
        except Exception:
            master_sessions = set()

        rc = get_redis_client()
        active_agents = []
        for session_id in list_session_ids(rc):
            session_doc = load_session(rc, session_id)
            if not isinstance(session_doc, dict):
                continue
            symbol = str(session_doc.get('symbol') or '').strip().upper()
            if not symbol:
                continue
            status_obj = get_session_status(rc, session_id) or {}
            ttl = rc.ttl(session_lock_key(session_id))
            model_path_s = str(session_doc.get('model_path') or '').strip() or None
            is_xgb_agent = bool(model_path_s and '/models/xgb/' in model_path_s.replace('\\', '/'))
            direction_v = get_runtime_value(rc, session_id, 'direction', session_doc.get('direction'))
            trade_mode_v = get_runtime_value(rc, session_id, 'trade_mode', session_doc.get('trade_mode'))
            exit_mode_v = get_runtime_value(rc, session_id, 'exit_mode', session_doc.get('exit_mode'))
            execution_mode_v = get_runtime_value(rc, session_id, 'execution_mode', session_doc.get('execution_mode'))
            risk_type_v = get_runtime_value(rc, session_id, 'risk_management_type', session_doc.get('risk_management_type'))
            max_hold_steps_v = get_runtime_value(rc, session_id, 'max_hold_steps', session_doc.get('max_hold_steps'))
            leverage_v = get_runtime_value(rc, session_id, 'leverage', session_doc.get('leverage', 1))
            xgb_entry_threshold_v = get_runtime_value(rc, session_id, 'xgb_entry_threshold')
            xgb_signal_exit_threshold_raw_v = get_runtime_value(
                rc,
                session_id,
                'xgb_signal_exit_threshold',
                session_doc.get('xgb_signal_exit_threshold', XGB_SIGNAL_EXIT_THRESHOLD_DEFAULT),
            )
            xgb_stop_loss_pct_v = get_runtime_value(rc, session_id, 'stop_loss_pct', session_doc.get('stop_loss_pct'))
            xgb_take_profit_pct_v = get_runtime_value(rc, session_id, 'take_profit_pct', session_doc.get('take_profit_pct'))
            pos_entry_ts_v = get_runtime_value(rc, session_id, 'pos_entry_ts')
            xgb_postexit_guard_enabled_raw_v = get_runtime_value(rc, session_id, 'xgb_postexit_guard_enabled', session_doc.get('xgb_postexit_guard_enabled'))
            xgb_postexit_cooldown_candles_raw_v = get_runtime_value(rc, session_id, 'xgb_postexit_cooldown_candles', session_doc.get('xgb_postexit_cooldown_candles'))
            xgb_postexit_boost_candles_raw_v = get_runtime_value(rc, session_id, 'xgb_postexit_boost_candles', session_doc.get('xgb_postexit_boost_candles'))
            xgb_postexit_threshold_boost_pct_raw_v = get_runtime_value(rc, session_id, 'xgb_postexit_threshold_boost_pct', session_doc.get('xgb_postexit_threshold_boost_pct'))
            model_paths_v = [str(item) for item in (session_doc.get('model_paths') or []) if item]
            model_roles_v = get_runtime_json(rc, session_id, 'model_roles', session_doc.get('model_roles') or {})
            model_roles_v = model_roles_v if isinstance(model_roles_v, dict) else {}
            bybit_account_id_v = str(session_doc.get('account_id') or '').strip() or None
            bybit_account_label_v = None
            bybit_api_key_hint_v = None
            xgb_entry_threshold_default_v = None
            xgb_task_v = None
            xgb_model_uuid_v = None
            xgb_stop_loss_pct_default_v = None
            xgb_stop_loss_override_active_v = False
            xgb_take_profit_override_active_v = False

            if bybit_account_id_v:
                try:
                    from utils.settings_store import ensure_settings_table, get_setting_value
                    ensure_settings_table()
                    bybit_account_label_v = get_setting_value('api', 'bybit', f'BYBIT_{bybit_account_id_v}_LABEL') or f'Account {bybit_account_id_v}'
                    _api_key = str(get_setting_value('api', 'bybit', f'BYBIT_{bybit_account_id_v}_API_KEY') or '').strip()
                    if _api_key:
                        bybit_api_key_hint_v = (f"{_api_key[:4]}...{_api_key[-4:]}" if len(_api_key) > 8 else f"{_api_key[:2]}...{_api_key[-2:]}")
                except Exception:
                    pass

            history = get_runtime_json(rc, session_id, 'xgb_signal_exit_history', [])
            history = history if isinstance(history, list) else []
            window_raw = get_runtime_value(rc, session_id, 'xgb_signal_exit_window', session_doc.get('xgb_signal_exit_window'))
            try:
                window_v = max(1, int(float(window_raw))) if window_raw not in (None, '') else None
            except Exception:
                window_v = None
            avg_signal_v = None
            avg_threshold_v = None
            ready_v = False
            passes_threshold_v = None
            last_signal_v = None
            last_threshold_v = None
            try:
                signal_exit_threshold_v = float(xgb_signal_exit_threshold_raw_v)
            except Exception:
                signal_exit_threshold_v = XGB_SIGNAL_EXIT_THRESHOLD_DEFAULT
            if history:
                try:
                    last_signal_v = float(history[-1].get('signal')) if history[-1].get('signal') not in (None, '') else None
                    last_threshold_v = signal_exit_threshold_v
                except Exception:
                    last_signal_v = None
                    last_threshold_v = signal_exit_threshold_v
            if window_v and len(history) >= window_v:
                tail = history[-window_v:]
                try:
                    avg_signal_v = float(sum(float(item.get('signal') or 0.0) for item in tail) / len(tail))
                    avg_threshold_v = signal_exit_threshold_v
                    ready_v = True
                    passes_threshold_v = bool(avg_signal_v >= avg_threshold_v)
                except Exception:
                    pass

            if is_xgb_agent:
                xgb_entry_threshold_default_v, xgb_task_v, xgb_stop_loss_pct_default_v = _read_xgb_runtime_meta(model_path_s)
                xgb_model_uuid_v = _read_xgb_model_uuid(model_path_s)
                try:
                    if xgb_stop_loss_pct_v not in (None, '') and xgb_stop_loss_pct_default_v is not None:
                        xgb_stop_loss_override_active_v = abs(float(xgb_stop_loss_pct_v) - float(xgb_stop_loss_pct_default_v)) > 1e-9
                except Exception:
                    xgb_stop_loss_override_active_v = False
                try:
                    if xgb_take_profit_pct_v not in (None, ''):
                        xgb_take_profit_override_active_v = abs(float(xgb_take_profit_pct_v) - float(XGB_DEFAULT_TAKE_PROFIT_PCT)) > 1e-9
                except Exception:
                    xgb_take_profit_override_active_v = False

            # Post-exit guard snapshot for monitor (XGB only).
            xgb_postexit_guard_enabled_v = False
            xgb_postexit_cooldown_candles_v = None
            xgb_postexit_boost_candles_v = None
            xgb_postexit_threshold_boost_pct_v = None
            xgb_postexit_phase_v = None
            xgb_postexit_candles_since_exit_v = None
            xgb_postexit_candles_left_v = None
            xgb_postexit_hours_left_v = None
            xgb_last_exit_ts_ms_v = None
            xgb_last_exit_reason_v = None
            try:
                xgb_postexit_guard_enabled_v = str(xgb_postexit_guard_enabled_raw_v or '').strip().lower() in ('1', 'true', 'yes', 'on')
            except Exception:
                xgb_postexit_guard_enabled_v = False
            try:
                xgb_postexit_cooldown_candles_v = int(float(xgb_postexit_cooldown_candles_raw_v)) if xgb_postexit_cooldown_candles_raw_v not in (None, '') else None
            except Exception:
                xgb_postexit_cooldown_candles_v = None
            try:
                xgb_postexit_boost_candles_v = int(float(xgb_postexit_boost_candles_raw_v)) if xgb_postexit_boost_candles_raw_v not in (None, '') else None
            except Exception:
                xgb_postexit_boost_candles_v = None
            try:
                xgb_postexit_threshold_boost_pct_v = float(xgb_postexit_threshold_boost_pct_raw_v) if xgb_postexit_threshold_boost_pct_raw_v not in (None, '') else None
            except Exception:
                xgb_postexit_threshold_boost_pct_v = None
            try:
                if is_xgb_agent and xgb_postexit_guard_enabled_v and rc is not None:
                    ex_ts_raw = get_runtime_value(rc, session_id, 'last_exit_ts_ms')
                    ex_reason_raw = get_runtime_value(rc, session_id, 'last_exit_reason')
                    try:
                        xgb_last_exit_ts_ms_v = int(float(ex_ts_raw)) if ex_ts_raw not in (None, '') else None
                    except Exception:
                        xgb_last_exit_ts_ms_v = None
                    xgb_last_exit_reason_v = str(ex_reason_raw or '').strip().lower() or None
                    if xgb_last_exit_ts_ms_v:
                        now_utc = datetime.utcnow()
                        epoch_sec = int(now_utc.timestamp())
                        last_closed = (epoch_sec // 300) * 300 - 300
                        now_bucket_ts = int(last_closed * 1000)
                        ex_epoch_sec = int(int(xgb_last_exit_ts_ms_v) / 1000)
                        ex_last_closed = (ex_epoch_sec // 300) * 300 - 300
                        ex_bucket_ts = int(ex_last_closed * 1000)
                        if now_bucket_ts >= ex_bucket_ts:
                            xgb_postexit_candles_since_exit_v = int((now_bucket_ts - ex_bucket_ts) // 300000)
                            cd = int(xgb_postexit_cooldown_candles_v) if isinstance(xgb_postexit_cooldown_candles_v, int) else None
                            boost = int(xgb_postexit_boost_candles_v) if isinstance(xgb_postexit_boost_candles_v, int) else None
                            if cd is not None and boost is not None:
                                if xgb_postexit_candles_since_exit_v < cd:
                                    xgb_postexit_phase_v = 'cooldown'
                                    xgb_postexit_candles_left_v = int(cd - xgb_postexit_candles_since_exit_v)
                                    xgb_postexit_hours_left_v = round(xgb_postexit_candles_left_v * 5 / 60, 2)
                                elif xgb_postexit_candles_since_exit_v < (cd + boost):
                                    xgb_postexit_phase_v = 'boost'
                                else:
                                    xgb_postexit_phase_v = 'expired'
            except Exception:
                pass

            try:
                pos_entry_ts_ms = int(float(pos_entry_ts_v)) if pos_entry_ts_v not in (None, '', '0') else None
            except Exception:
                pos_entry_ts_ms = None
            hold_seconds_remaining_v = None
            try:
                if pos_entry_ts_ms and max_hold_steps_v not in (None, ''):
                    now_bucket_ms = (int(time.time()) // 300) * 300 * 1000 - 300000
                    elapsed_steps = int((now_bucket_ms - pos_entry_ts_ms) // 300000)
                    hold_seconds_remaining_v = max(0, int((int(float(max_hold_steps_v)) - elapsed_steps) * 300))
            except Exception:
                hold_seconds_remaining_v = None

            def _bool_runtime(name: str, default=None):
                raw = get_runtime_value(rc, session_id, name, default)
                if raw in (None, ''):
                    return None
                return str(raw).strip().lower() in ('1', 'true', 'yes', 'on')

            def _float_runtime(name: str, default=None):
                raw = get_runtime_value(rc, session_id, name, default)
                try:
                    return float(raw) if raw not in (None, '') else None
                except Exception:
                    return None

            def _int_runtime(name: str, default=None):
                raw = _float_runtime(name, default)
                return int(raw) if raw is not None else None

            xgb_side_models_v = []
            for side_name in ('long', 'short'):
                opposite_name = 'short' if side_name == 'long' else 'long'
                opposite_max_key = (
                    'xgb_short_signal_max_for_long_entry'
                    if side_name == 'long'
                    else 'xgb_long_signal_max_for_short_entry'
                )
                side_path = None
                for path_item in model_paths_v:
                    path_key = _normalize_model_path_key(path_item)
                    role = str(
                        model_roles_v.get(str(path_item))
                        or model_roles_v.get(path_key)
                        or ''
                    ).strip().lower()
                    if role == side_name:
                        side_path = str(path_item)
                        break
                if side_path is None and len(model_paths_v) == 1 and str(direction_v or '').strip().lower() == side_name:
                    side_path = model_paths_v[0]
                side_threshold_default, side_task, side_sl_default, side_tp_default = (
                    _read_xgb_side_trade_meta(side_path) if side_path else (None, None, None, None)
                )
                side_threshold_v = get_runtime_value(rc, session_id, f'xgb_entry_threshold_{side_name}')
                side_sl_v = get_runtime_value(rc, session_id, f'stop_loss_pct_{side_name}')
                side_tp_v = get_runtime_value(rc, session_id, f'take_profit_pct_{side_name}')
                latest_prediction_v = _latest_xgb_model_prediction(session_id, side_path)
                xgb_side_models_v.append({
                    'role': side_name,
                    'model_path': side_path,
                    'model_uuid': _read_xgb_model_uuid(side_path) if side_path else None,
                    'task': side_task,
                    'max_hold_steps': _read_xgb_hold_steps_meta(side_path) if side_path else None,
                    'entry_threshold': (
                        float(str(side_threshold_v))
                        if side_threshold_v not in (None, '')
                        else side_threshold_default
                    ),
                    'entry_threshold_default': side_threshold_default,
                    'entry_threshold_override_active': bool(side_threshold_v not in (None, '')),
                    'stop_loss_pct': (
                        float(str(side_sl_v))
                        if side_sl_v not in (None, '')
                        else side_sl_default
                    ),
                    'stop_loss_pct_default': side_sl_default,
                    'stop_loss_override_active': bool(side_sl_v not in (None, '')),
                    'take_profit_pct': (
                        float(str(side_tp_v))
                        if side_tp_v not in (None, '')
                        else side_tp_default
                    ),
                    'take_profit_pct_default': side_tp_default,
                    'take_profit_override_active': bool(side_tp_v not in (None, '')),
                    'signal_exit_enabled': _bool_runtime(f'xgb_signal_exit_{side_name}_enabled', session_doc.get(f'xgb_signal_exit_{side_name}_enabled')),
                    'signal_exit_start_step': _int_runtime(f'xgb_signal_exit_{side_name}_start_step', session_doc.get(f'xgb_signal_exit_{side_name}_start_step')),
                    'signal_exit_threshold': _float_runtime(f'xgb_signal_exit_{side_name}_threshold', session_doc.get(f'xgb_signal_exit_{side_name}_threshold')),
                    'signal_exit_window': window_v,
                    'opposite_role': opposite_name,
                    'opposite_signal_max': _float_runtime(opposite_max_key, session_doc.get(opposite_max_key)),
                    'last_prediction_signal': latest_prediction_v.get('signal'),
                    'last_prediction_threshold': latest_prediction_v.get('threshold'),
                    'last_prediction_action': latest_prediction_v.get('action'),
                    'last_prediction_at': latest_prediction_v.get('created_at'),
                })

            active_agents.append({
                'session_id': session_id,
                'symbol': symbol,
                'is_active': True,
                'ttl_seconds': int(ttl) if ttl is not None and isinstance(ttl, int) and ttl > 0 else 0,
                'status': status_obj.get('trading_status_full') or status_obj.get('trading_status') or '—',
                'current_price': status_obj.get('current_price'),
                'position': status_obj.get('position'),
                'amount_usdt': status_obj.get('amount_usdt'),
                'position_entry_ts_ms': pos_entry_ts_ms,
                'hold_seconds_remaining': hold_seconds_remaining_v,
                'trades_count': status_obj.get('trades_count'),
                'last_prediction': status_obj.get('last_model_prediction'),
                'amount': status_obj.get('amount'),
                'amount_display': status_obj.get('amount_display'),
                'model_path': model_path_s,
                'model_paths': model_paths_v,
                'model_roles': model_roles_v,
                'xgb_side_models': xgb_side_models_v,
                'is_xgb': is_xgb_agent,
                'execution_mode': (str(execution_mode_v).strip() if execution_mode_v else None),
                'direction': (str(direction_v).strip().lower() if direction_v else None),
                'trade_mode': (str(trade_mode_v).strip() if trade_mode_v else None),
                'ignore_trend_filter': _bool_runtime('ignore_trend_filter', session_doc.get('ignore_trend_filter')),
                'exit_mode': (str(exit_mode_v).strip() if exit_mode_v else None),
                'risk_management_type': (str(risk_type_v).strip() if risk_type_v else None),
                'account_pct': (int(float(session_doc.get('account_pct'))) if session_doc.get('account_pct') not in (None, '') else None),
                'leverage': (int(float(leverage_v)) if leverage_v not in (None, '') else 1),
                'max_hold_steps': (int(float(max_hold_steps_v)) if max_hold_steps_v not in (None, '') else None),
                'xgb_entry_threshold': (
                    float(str(xgb_entry_threshold_v))
                    if xgb_entry_threshold_v not in (None, '')
                    else xgb_entry_threshold_default_v
                ),
                'xgb_entry_threshold_default': xgb_entry_threshold_default_v,
                'xgb_threshold_override_active': bool(xgb_entry_threshold_v not in (None, '')),
                'xgb_stop_loss_pct': (
                    float(str(xgb_stop_loss_pct_v))
                    if xgb_stop_loss_pct_v not in (None, '')
                    else xgb_stop_loss_pct_default_v
                ),
                'xgb_stop_loss_pct_default': xgb_stop_loss_pct_default_v,
                'xgb_stop_loss_override_active': xgb_stop_loss_override_active_v,
                'xgb_take_profit_pct': (
                    float(str(xgb_take_profit_pct_v))
                    if xgb_take_profit_pct_v not in (None, '')
                    else XGB_DEFAULT_TAKE_PROFIT_PCT
                ),
                'xgb_take_profit_pct_default': XGB_DEFAULT_TAKE_PROFIT_PCT,
                'xgb_take_profit_override_active': xgb_take_profit_override_active_v,
                'xgb_task': xgb_task_v,
                'xgb_model_uuid': xgb_model_uuid_v,
                'xgb_signal_exit_enabled': str(get_runtime_value(rc, session_id, 'xgb_signal_exit_enabled', session_doc.get('xgb_signal_exit_enabled')) or '').strip().lower() in ('1', 'true', 'yes', 'on'),
                'xgb_signal_exit_start_step': (int(float(get_runtime_value(rc, session_id, 'xgb_signal_exit_start_step', session_doc.get('xgb_signal_exit_start_step')))) if get_runtime_value(rc, session_id, 'xgb_signal_exit_start_step', session_doc.get('xgb_signal_exit_start_step')) not in (None, '') else None),
                'xgb_signal_exit_window': window_v,
                'xgb_signal_exit_threshold': signal_exit_threshold_v,
                'xgb_signal_exit_history_size': len(history),
                'xgb_signal_exit_avg_signal': avg_signal_v,
                'xgb_signal_exit_avg_threshold': avg_threshold_v,
                'xgb_signal_exit_last_signal': last_signal_v,
                'xgb_signal_exit_last_threshold': last_threshold_v,
                'xgb_signal_exit_ready': ready_v,
                'xgb_signal_exit_passes_threshold': passes_threshold_v,
                'xgb_postexit_guard_enabled': bool(xgb_postexit_guard_enabled_v),
                'xgb_postexit_cooldown_candles': xgb_postexit_cooldown_candles_v,
                'xgb_postexit_boost_candles': xgb_postexit_boost_candles_v,
                'xgb_postexit_threshold_boost_pct': xgb_postexit_threshold_boost_pct_v,
                'xgb_postexit_phase': xgb_postexit_phase_v,
                'xgb_postexit_candles_since_exit': xgb_postexit_candles_since_exit_v,
                'xgb_postexit_candles_left': xgb_postexit_candles_left_v,
                'xgb_postexit_hours_left': xgb_postexit_hours_left_v,
                'xgb_last_exit_ts_ms': xgb_last_exit_ts_ms_v,
                'xgb_last_exit_reason': xgb_last_exit_reason_v,
                'bybit_account_id': bybit_account_id_v,
                'bybit_account_label': bybit_account_label_v,
                'bybit_api_key_hint': bybit_api_key_hint_v,
                'is_client_master': bool(session_id in master_sessions),
            })
        return jsonify({
            'success': True,
            'active_agents': active_agents,
            'total_active': len(active_agents),
            'xgb_entry_attempts_history': _read_xgb_entry_attempts_history(rc),
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.get('/api/trading/latest_results')
def trading_latest_results():
    """Получение последних результатов торговли из Celery"""
    try:
        requested_symbol = (request.args.get('symbol') or '').upper().strip()
        latest_results = []
        try:
            rc = get_redis_client()
            keys = rc.keys('trading:latest_result_*') or []
            for k in keys:
                try:
                    raw = rc.get(k)
                    if not raw:
                        continue
                    if isinstance(raw, (bytes, bytearray)):
                        s = raw.decode('utf-8')
                    else:
                        s = str(raw)
                    result = json.loads(s)
                    latest_results.append(result)
                except Exception:
                    continue
        except Exception:
            pass
        if requested_symbol:
            latest_results = [r for r in latest_results if isinstance(r.get('symbols'), list) and requested_symbol in r.get('symbols')]
        latest_results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return jsonify({
            'success': True,
            'latest_results': latest_results,
            'total_results': len(latest_results)
        }), 200
    except Exception as e:
        logging.error(f"Ошибка получения последних результатов: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@trading_bp.post('/api/trading/xgb_threshold')
def update_xgb_threshold():
    try:
        data = request.get_json(silent=True) or {}
        session_id = str(data.get('session_id') or '').strip()
        rc = get_redis_client()
        session_doc, err = _get_session_or_400(rc, session_id)
        if err:
            return err
        symbol = str(session_doc.get('symbol') or '').strip().upper()
        side = str(data.get('side') or '').strip().lower()
        if side not in ('long', 'short'):
            return jsonify({'success': False, 'error': 'side must be long or short'}), 400
        model_path_s = _xgb_side_model_path(session_doc, side)
        if not model_path_s or '/models/xgb/' not in model_path_s.replace('\\', '/'):
            return jsonify({'success': False, 'error': f'active XGB {side} model not found for session'}), 400

        default_threshold, task_name = _read_xgb_threshold_meta(model_path_s)
        reset_to_default = bool(data.get('reset') is True)
        runtime_key = f'xgb_entry_threshold_{side}'

        if reset_to_default:
            delete_runtime_value(rc, session_id, runtime_key)
            return jsonify({
                'success': True,
                'session_id': session_id,
                'symbol': symbol,
                'side': side,
                'xgb_entry_threshold': default_threshold,
                'xgb_entry_threshold_default': default_threshold,
                'xgb_threshold_override_active': False,
                'xgb_task': task_name,
            }), 200

        raw_threshold = data.get('threshold')
        if raw_threshold in (None, ''):
            return jsonify({'success': False, 'error': 'threshold is required'}), 400

        threshold = float(raw_threshold)
        if not (0.0 < threshold < 1.0):
            return jsonify({'success': False, 'error': 'threshold must be between 0 and 1'}), 400

        set_runtime_value(rc, session_id, runtime_key, str(threshold))
        return jsonify({
            'success': True,
            'session_id': session_id,
            'symbol': symbol,
            'side': side,
            'xgb_entry_threshold': threshold,
            'xgb_entry_threshold_default': default_threshold,
            'xgb_threshold_override_active': True,
            'xgb_task': task_name,
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@trading_bp.post('/api/trading/xgb_signal_exit_threshold')
def update_xgb_signal_exit_threshold():
    try:
        data = request.get_json(silent=True) or {}
        rc = get_redis_client()
        session_id = str(data.get('session_id') or '').strip()
        session_doc, err = _get_session_or_400(rc, session_id)
        if err:
            return err

        symbol = str(session_doc.get('symbol') or '').strip().upper()
        model_path_s = str(session_doc.get('model_path') or '').strip() or None
        if not model_path_s or '/models/xgb/' not in model_path_s.replace('\\', '/'):
            return jsonify({'success': False, 'error': 'active XGB agent not found for session'}), 400

        raw_threshold = data.get('threshold')
        if raw_threshold in (None, ''):
            return jsonify({'success': False, 'error': 'threshold is required'}), 400
        threshold = float(raw_threshold)
        if not (0.0 < threshold < 1.0):
            return jsonify({'success': False, 'error': 'threshold must be between 0 and 1'}), 400

        set_runtime_value(rc, session_id, 'xgb_signal_exit_threshold', str(threshold))
        updated_doc = dict(session_doc)
        updated_doc['xgb_signal_exit_threshold'] = threshold
        save_session(rc, session_id, updated_doc)

        return jsonify({
            'success': True,
            'session_id': session_id,
            'symbol': symbol,
            'xgb_signal_exit_threshold': threshold,
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@trading_bp.post('/api/trading/xgb_opposite_signal_threshold')
def update_xgb_opposite_signal_threshold():
    try:
        data = request.get_json(silent=True) or {}
        rc = get_redis_client()
        session_id = str(data.get('session_id') or '').strip()
        session_doc, err = _get_session_or_400(rc, session_id)
        if err:
            return err

        symbol = str(session_doc.get('symbol') or '').strip().upper()
        side = str(data.get('side') or '').strip().lower()
        if side not in ('long', 'short'):
            return jsonify({'success': False, 'error': 'side must be long or short'}), 400
        model_path_s = _xgb_side_model_path(session_doc, side)
        if not model_path_s or '/models/xgb/' not in model_path_s.replace('\\', '/'):
            return jsonify({'success': False, 'error': f'active XGB {side} model not found for session'}), 400

        raw_threshold = data.get('threshold')
        if raw_threshold in (None, ''):
            return jsonify({'success': False, 'error': 'threshold is required'}), 400
        threshold = float(raw_threshold)
        if not (0.0 <= threshold <= 1.0):
            return jsonify({'success': False, 'error': 'threshold must be between 0 and 1'}), 400

        runtime_key = (
            'xgb_short_signal_max_for_long_entry'
            if side == 'long'
            else 'xgb_long_signal_max_for_short_entry'
        )
        set_runtime_value(rc, session_id, runtime_key, str(threshold))
        updated_doc = dict(session_doc)
        updated_doc[runtime_key] = threshold
        save_session(rc, session_id, updated_doc)

        return jsonify({
            'success': True,
            'session_id': session_id,
            'symbol': symbol,
            'side': side,
            'opposite_signal_max': threshold,
            'runtime_key': runtime_key,
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@trading_bp.post('/api/trading/xgb_stop_loss')
def update_xgb_stop_loss():
    try:
        data = request.get_json(silent=True) or {}
        rc = get_redis_client()
        session_id = str(data.get('session_id') or '').strip()
        session_doc, err = _get_session_or_400(rc, session_id)
        if err:
            return err
        symbol = str(session_doc.get('symbol') or '').strip().upper()
        side = str(data.get('side') or '').strip().lower()
        if side not in ('long', 'short'):
            return jsonify({'success': False, 'error': 'side must be long or short'}), 400
        model_path_s = _xgb_side_model_path(session_doc, side)
        if not model_path_s or '/models/xgb/' not in model_path_s.replace('\\', '/'):
            return jsonify({'success': False, 'error': f'active XGB {side} model not found for session'}), 400

        _, task_name, default_stop_loss_pct = _read_xgb_runtime_meta(model_path_s)
        runtime_key = f'stop_loss_pct_{side}'

        reset_to_default = bool(data.get('reset') is True)
        if reset_to_default:
            if default_stop_loss_pct in (None, ''):
                return jsonify({'success': False, 'error': 'default stop-loss not found in model meta'}), 400
            delete_runtime_value(rc, session_id, runtime_key)
            if get_runtime_value(rc, session_id, 'risk_management_type') in (None, ''):
                set_runtime_value(rc, session_id, 'risk_management_type', 'exchange_orders')
            return jsonify({
                'success': True,
                'session_id': session_id,
                'symbol': symbol,
                'side': side,
                'xgb_stop_loss_pct': float(default_stop_loss_pct),
                'xgb_stop_loss_pct_default': float(default_stop_loss_pct),
                'xgb_stop_loss_override_active': False,
                'xgb_task': task_name,
            }), 200

        raw_stop_loss = data.get('stop_loss_pct')
        if raw_stop_loss in (None, ''):
            return jsonify({'success': False, 'error': 'stop_loss_pct is required'}), 400

        stop_loss_pct = float(raw_stop_loss)
        if not (0.0 < stop_loss_pct <= 20.0):
            return jsonify({'success': False, 'error': 'stop_loss_pct must be between 0 and 20'}), 400

        set_runtime_value(rc, session_id, runtime_key, str(stop_loss_pct))
        if get_runtime_value(rc, session_id, 'risk_management_type') in (None, ''):
            set_runtime_value(rc, session_id, 'risk_management_type', 'exchange_orders')

        override_active = True
        try:
            if default_stop_loss_pct is not None:
                override_active = abs(float(stop_loss_pct) - float(default_stop_loss_pct)) > 1e-9
        except Exception:
            override_active = True

        return jsonify({
            'success': True,
            'session_id': session_id,
            'symbol': symbol,
            'side': side,
            'xgb_stop_loss_pct': stop_loss_pct,
            'xgb_stop_loss_pct_default': default_stop_loss_pct,
            'xgb_stop_loss_override_active': override_active,
            'xgb_task': task_name,
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@trading_bp.post('/api/trading/xgb_take_profit')
def update_xgb_take_profit():
    try:
        data = request.get_json(silent=True) or {}
        rc = get_redis_client()
        session_id = str(data.get('session_id') or '').strip()
        session_doc, err = _get_session_or_400(rc, session_id)
        if err:
            return err
        symbol = str(session_doc.get('symbol') or '').strip().upper()
        side = str(data.get('side') or '').strip().lower()
        if side not in ('long', 'short'):
            return jsonify({'success': False, 'error': 'side must be long or short'}), 400
        model_path_s = _xgb_side_model_path(session_doc, side)
        if not model_path_s or '/models/xgb/' not in model_path_s.replace('\\', '/'):
            return jsonify({'success': False, 'error': f'active XGB {side} model not found for session'}), 400

        _, task_name, _, default_take_profit = _read_xgb_side_trade_meta(model_path_s)
        runtime_key = f'take_profit_pct_{side}'

        reset_to_default = bool(data.get('reset') is True)
        if reset_to_default:
            if default_take_profit in (None, ''):
                return jsonify({'success': False, 'error': 'default take-profit not found in model meta'}), 400
            delete_runtime_value(rc, session_id, runtime_key)
            if get_runtime_value(rc, session_id, 'risk_management_type') in (None, ''):
                set_runtime_value(rc, session_id, 'risk_management_type', 'exchange_orders')
            return jsonify({
                'success': True,
                'session_id': session_id,
                'symbol': symbol,
                'side': side,
                'xgb_take_profit_pct': float(default_take_profit),
                'xgb_take_profit_pct_default': float(default_take_profit),
                'xgb_take_profit_override_active': False,
                'xgb_task': task_name,
            }), 200

        raw_take_profit = data.get('take_profit_pct')
        if raw_take_profit in (None, ''):
            return jsonify({'success': False, 'error': 'take_profit_pct is required'}), 400

        take_profit_pct = float(raw_take_profit)
        if not (0.0 < take_profit_pct <= 30.0):
            return jsonify({'success': False, 'error': 'take_profit_pct must be between 0 and 30'}), 400

        set_runtime_value(rc, session_id, runtime_key, str(take_profit_pct))
        if get_runtime_value(rc, session_id, 'risk_management_type') in (None, ''):
            set_runtime_value(rc, session_id, 'risk_management_type', 'exchange_orders')

        override_active = True
        try:
            if default_take_profit is not None:
                override_active = abs(float(take_profit_pct) - float(default_take_profit)) > 1e-9
        except Exception:
            override_active = True
        return jsonify({
            'success': True,
            'session_id': session_id,
            'symbol': symbol,
            'side': side,
            'xgb_take_profit_pct': take_profit_pct,
            'xgb_take_profit_pct_default': default_take_profit,
            'xgb_take_profit_override_active': override_active,
            'xgb_task': task_name,
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@trading_bp.post('/api/trading/xgb_hold_steps_extend')
def extend_xgb_hold_steps():
    try:
        data = request.get_json(silent=True) or {}
        rc = get_redis_client()
        session_id = str(data.get('session_id') or '').strip()
        session_doc, err = _get_session_or_400(rc, session_id)
        if err:
            return err

        symbol = str(session_doc.get('symbol') or '').strip().upper()
        model_path_s = str(session_doc.get('model_path') or '').strip() or None
        if not model_path_s or '/models/xgb/' not in model_path_s.replace('\\', '/'):
            return jsonify({'success': False, 'error': 'active XGB agent not found for session'}), 400

        exit_mode = str(get_runtime_value(rc, session_id, 'exit_mode', session_doc.get('exit_mode')) or '').strip().lower()
        if exit_mode != 'hold_steps':
            return jsonify({'success': False, 'error': 'session exit_mode is not hold_steps'}), 400

        raw_increment = data.get('increment_steps')
        if raw_increment in (None, ''):
            return jsonify({'success': False, 'error': 'increment_steps is required'}), 400
        increment = int(float(raw_increment))
        if increment < 5 or increment > 200:
            return jsonify({'success': False, 'error': 'increment_steps must be between 5 and 200'}), 400

        raw_current = get_runtime_value(rc, session_id, 'max_hold_steps', session_doc.get('max_hold_steps'))
        if raw_current in (None, ''):
            _, _, hold_default = _read_xgb_result_meta(model_path_s)
            raw_current = hold_default
        if raw_current in (None, ''):
            return jsonify({'success': False, 'error': 'current max_hold_steps not found'}), 400

        current_steps = int(float(raw_current))
        new_steps = current_steps + increment
        set_runtime_value(rc, session_id, 'max_hold_steps', str(new_steps))
        updated_doc = dict(session_doc)
        updated_doc['max_hold_steps'] = new_steps
        save_session(rc, session_id, updated_doc)

        return jsonify({
            'success': True,
            'session_id': session_id,
            'symbol': symbol,
            'previous_max_hold_steps': current_steps,
            'increment_steps': increment,
            'max_hold_steps': new_steps,
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@trading_bp.post('/api/trading/xgb_leverage')
def update_xgb_leverage():
    try:
        data = request.get_json(silent=True) or {}
        rc = get_redis_client()
        session_id = str(data.get('session_id') or '').strip()
        session_doc, err = _get_session_or_400(rc, session_id)
        if err:
            return err
        symbol = str(session_doc.get('symbol') or '').strip().upper()
        model_path_s = str(session_doc.get('model_path') or '').strip() or None
        if not model_path_s or '/models/xgb/' not in model_path_s.replace('\\', '/'):
            return jsonify({'success': False, 'error': 'active XGB agent not found for session'}), 400

        raw_leverage = data.get('leverage')
        if raw_leverage in (None, ''):
            return jsonify({'success': False, 'error': 'leverage is required'}), 400
        leverage = int(float(raw_leverage))
        if leverage < 1 or leverage > 5:
            return jsonify({'success': False, 'error': 'leverage must be between 1 and 5'}), 400

        set_runtime_value(rc, session_id, 'leverage', str(leverage))
        model_key = _xgb_model_leverage_key(model_path_s)
        if model_key:
            rc.set(model_key, str(leverage))

        return jsonify({
            'success': True,
            'session_id': session_id,
            'symbol': symbol,
            'xgb_leverage': leverage,
            'model_path': model_path_s,
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@trading_bp.post('/api/trading/toggle_client_master')
def toggle_client_master():
    try:
        from utils.settings_store import upsert_setting, ensure_settings_table
        ensure_settings_table()
        data = request.get_json(silent=True) or {}
        session_id = str(data.get('session_id') or '').strip()
        enable = bool(data.get('enable'))
        
        if not session_id:
            return jsonify({'success': False, 'error': 'session_id is required'}), 400
            
        # У нас может быть только ОДНА активная модель для клиентов. 
        # Если включили - сохраняем только её. Если выключили - пустой список.
        master_sessions = [session_id] if enable else []
                
        upsert_setting(
            scope='trading',
            group='master',
            key='MASTER_SESSION_IDS',
            value_type='json',
            label='Master sessions for clients',
            description='List of session IDs that broadcast signals to active clients',
            is_secret=False,
            value=json.dumps(master_sessions)
        )
        return jsonify({'success': True, 'is_client_master': enable}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@trading_bp.post('/api/trading/manual_exit')
def trading_manual_exit():
    try:
        data = request.get_json(silent=True) or {}
        session_id = str(data.get('session_id') or '').strip()
        raw_mode = str(data.get('execution_mode') or 'market').strip().lower()
        execution_mode = 'limit_post_only' if raw_mode == 'limit' else raw_mode
        if execution_mode not in ('market', 'limit_post_only'):
            return jsonify({'success': False, 'error': 'execution_mode must be market or limit'}), 400

        from trading_agent.trading_agent import TradingAgent

        rc = get_redis_client()
        session_doc, err = _get_session_or_400(rc, session_id)
        if err:
            return err
        symbol = str(session_doc.get('symbol') or '').strip().upper()
        model_path = str(session_doc.get('model_path') or '').strip() or None
        if model_path and '/models/xgb/' not in str(model_path).replace('\\', '/'):
            return jsonify({'success': False, 'error': 'active XGB agent not found for session'}), 400

        agent = TradingAgent(
            model_path=model_path,
            symbol=symbol,
            session_id=session_id,
            account_id=str(session_doc.get('account_id') or '').strip(),
        )  # type: ignore
        agent.symbol = symbol
        agent.base_symbol = symbol
        agent._restore_position_from_exchange()

        if not agent.current_position:
            return jsonify({'success': False, 'error': 'no open position for symbol'}), 400

        position_type = str(agent.current_position.get('type') or '').strip().lower()
        if position_type not in ('long', 'short'):
            return jsonify({'success': False, 'error': 'unsupported position type'}), 400

        cancelled_orders = 0
        try:
            open_orders = agent.exchange.fetch_open_orders(symbol) if hasattr(agent.exchange, 'fetch_open_orders') else []
            for order in open_orders or []:
                order_id = order.get('id')
                if not order_id:
                    continue
                try:
                    agent.exchange.cancel_order(order_id, symbol)
                    cancelled_orders += 1
                except Exception:
                    continue
        except Exception:
            pass

        try:
            if rc is not None:
                rc.setex(session_runtime_key(session_id, 'forced_exit_reason'), 3600, 'manual')
        except Exception:
            pass

        if execution_mode == 'market':
            result = agent._execute_cover_short() if position_type == 'short' else agent._execute_sell()
            if not result.get('success'):
                return jsonify({'success': False, 'error': result.get('error') or 'manual exit failed'}), 500
            try:
                if rc is not None:
                    now_ms = int(time.time() * 1000)
                    order_id = None
                    try:
                        order_id = (result.get('order') or {}).get('id')
                    except Exception:
                        order_id = None
                    set_runtime_value(rc, session_id, 'pos_open', '0')
                    set_runtime_value(rc, session_id, 'last_exit_ts_ms', str(now_ms))
                    set_runtime_value(rc, session_id, 'last_exit_reason', 'manual')
                    if order_id:
                        set_runtime_value(rc, session_id, 'last_exit_order_id', str(order_id))
                    delete_runtime_value(rc, session_id, 'forced_exit_reason')
                    delete_runtime_value(rc, session_id, 'pos_entry_ts')
            except Exception:
                pass
            try:
                trade_number = str(result.get('trade_number') or '').strip()
                if trade_number:
                    from orm.database import get_db_session
                    from orm.models import Trade
                    session_db = get_db_session()
                    try:
                        trade = session_db.query(Trade).filter(Trade.trade_number == trade_number).first()
                        if trade:
                            meta = {}
                            try:
                                meta = json.loads(trade.error_message or '{}')
                            except Exception:
                                meta = {}
                            meta['exit_reason'] = 'manual'
                            meta['account_id'] = str(session_doc.get('account_id') or '').strip()
                            meta['model_path'] = model_path
                            meta['model_family'] = 'xgb'
                            closed_pos = result.get('closed_position') or {}
                            if isinstance(closed_pos, dict):
                                meta['entry_price'] = closed_pos.get('entry_price')
                                meta['pos_type_prev'] = closed_pos.get('type')
                            trade.model_prediction = 'exit:manual'
                            trade.error_message = json.dumps(meta, ensure_ascii=False)
                            session_db.commit()
                    finally:
                        session_db.close()
            except Exception:
                pass
            return jsonify({
                'success': True,
                'session_id': session_id,
                'symbol': symbol,
                'execution_mode': 'market',
                'position_type': position_type,
                'cancelled_orders': cancelled_orders,
                'result': result,
            }), 200

        amount = float(agent.current_position.get('amount') or 0.0)
        amount = float(agent._normalize_amount(amount))
        if amount <= 0:
            return jsonify({'success': False, 'error': 'position amount is empty'}), 400

        market = agent.exchange.market(symbol)  # type: ignore
        tick = None
        try:
            info = market.get('info', {})
            pf = info.get('priceFilter', {})
            tick = float(pf.get('tickSize')) if pf.get('tickSize') else None
        except Exception:
            tick = None
        if tick is None:
            precision = market.get('precision', {}).get('price', 2)
            tick = 10 ** (-precision)

        best_bid = None
        best_ask = None
        try:
            book = agent.exchange.fetch_order_book(symbol, 5)  # type: ignore
            bids = book.get('bids') or []
            asks = book.get('asks') or []
            if bids:
                best_bid = float(bids[0][0])
            if asks:
                best_ask = float(asks[0][0])
        except Exception:
            pass

        current_price = float(agent._get_current_price() or 0.0)
        if position_type == 'long':
            limit_price = best_ask if best_ask and best_ask > 0 else max(current_price + float(tick), float(tick))
            order = agent.exchange.create_limit_sell_order(  # type: ignore
                symbol,
                amount,
                float(limit_price),
                {
                    'reduceOnly': True,
                    'timeInForce': 'GTC',
                    'postOnly': True,
                },
            )
            side = 'sell'
        else:
            raw_price = best_bid if best_bid and best_bid > 0 else (current_price - float(tick))
            limit_price = max(float(raw_price), float(tick))
            order = agent.exchange.create_limit_buy_order(  # type: ignore
                symbol,
                amount,
                float(limit_price),
                {
                    'reduceOnly': True,
                    'timeInForce': 'GTC',
                    'postOnly': True,
                },
            )
            side = 'buy'

        return jsonify({
            'success': True,
            'session_id': session_id,
            'symbol': symbol,
            'execution_mode': 'limit',
            'position_type': position_type,
            'side': side,
            'amount': amount,
            'price': float(limit_price),
            'cancelled_orders': cancelled_orders,
            'order': order,
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.get('/api/trading/trade_history')
def trading_trade_history():
    """История закрытых сделок для UI trading/results."""
    try:
        from orm.database import get_db_session
        from orm.models import Trade, Symbol, ModelPrediction, OHLCV
        from sqlalchemy.orm import joinedload
        from datetime import datetime as _dt

        sym_filter = (request.args.get('symbol') or '').strip().upper()
        account_filter = (request.args.get('account_id') or '').strip()
        limit = min(int(request.args.get('limit') or 200), 1000)

        session = get_db_session()
        try:
            q = session.query(Trade).options(joinedload(Trade.symbol)).filter(
                Trade.status == 'executed',
                Trade.is_successful == True,
                Trade.position_pnl.isnot(None),
            )
            if sym_filter:
                sub = session.query(Symbol.id).filter(Symbol.name == sym_filter).scalar()
                if sub:
                    q = q.filter(Trade.symbol_id == sub)
                else:
                    return jsonify({'success': True, 'trades': [], 'total': 0}), 200
            if account_filter:
                q = q.filter(Trade.error_message.like(f'%"account_id": "{account_filter}"%'))
            rows = q.order_by(Trade.executed_at.desc()).limit(limit).all()

            trades_out = []
            rc = None
            try:
                rc = get_redis_client()
            except Exception:
                rc = None
            for r in rows:
                sym = r.symbol.name if r.symbol else '?'
                meta = {}
                if r.error_message:
                    try:
                        meta = json.loads(r.error_message)
                    except Exception:
                        pass
                exit_action = str(r.action or '').strip().lower()
                entry_action = 'buy' if exit_action == 'sell' else 'sell'
                prev_entry = None
                try:
                    exit_dt = r.executed_at or r.created_at
                    if exit_dt is not None:
                        prev_entry = (
                            session.query(Trade)
                            .filter(
                                Trade.symbol_id == r.symbol_id,
                                Trade.status == 'executed',
                                Trade.is_successful == True,
                                Trade.position_pnl.is_(None),
                                Trade.action == entry_action,
                                Trade.executed_at.isnot(None),
                                Trade.executed_at <= exit_dt,
                            )
                            .order_by(Trade.executed_at.desc())
                            .first()
                        )
                except Exception:
                    prev_entry = None

                entry_meta = {}
                if prev_entry is not None and prev_entry.error_message:
                    try:
                        entry_meta = json.loads(prev_entry.error_message)
                    except Exception:
                        entry_meta = {}

                entry_price = meta.get('entry_price')
                if entry_price in (None, '') and prev_entry is not None:
                    entry_price = prev_entry.price

                pos_type = meta.get('pos_type_prev') or ('long' if exit_action == 'sell' else 'short')
                exit_reason = meta.get('exit_reason') or (r.model_prediction or '').replace('exit:', '')
                if str(exit_reason or '').strip().lower() in ('unknown', 'none', 'null'):
                    exit_reason = ''
                if not exit_reason and str(r.model_prediction or '').strip().lower() in ('buy', 'sell'):
                    exit_reason = 'signal'
                leverage = _normalize_leverage(meta.get('leverage') or entry_meta.get('leverage'), 1)
                bal_before = meta.get('bal_before')
                bal_after = meta.get('bal_after') or r.current_balance
                model_path = meta.get('model_path')
                model_family = meta.get('model_family')
                account_id = str(meta.get('account_id') or '').strip() or None
                account_label = None
                try:
                    if not model_path and rc is not None and sym and sym != '?':
                        model_path = rc.get(f'trading:model_path:{sym}') or rc.get('trading:model_path')
                    if model_path not in (None, ''):
                        model_path = str(model_path).strip()
                    if not model_family and model_path:
                        model_family = 'xgb' if '/models/xgb/' in str(model_path).replace('\\', '/') else 'dqn'
                except Exception:
                    model_family = None
                try:
                    if account_id:
                        from utils.settings_store import ensure_settings_table, get_setting_value
                        ensure_settings_table()
                        account_label = get_setting_value('api', 'bybit', f'BYBIT_{account_id}_LABEL') or f'Account {account_id}'
                except Exception:
                    account_label = (f'Account {account_id}' if account_id else None)

                entry_time = None
                entry_ts_ms = meta.get('entry_ts_ms')
                if entry_ts_ms:
                    try:
                        entry_time = _dt.utcfromtimestamp(int(entry_ts_ms) / 1000).isoformat()
                    except Exception:
                        pass
                elif prev_entry is not None and (prev_entry.executed_at or prev_entry.created_at):
                    try:
                        entry_time = (prev_entry.executed_at or prev_entry.created_at).isoformat()
                    except Exception:
                        entry_time = None

                exit_time = (r.executed_at or r.created_at).isoformat() if r.executed_at or r.created_at else None
                pnl = r.position_pnl
                if (entry_price in (None, '') or entry_price is None) and pnl is not None and r.price and r.quantity:
                    try:
                        qty = float(r.quantity)
                        px = float(r.price)
                        pnl_f = float(pnl)
                        if qty > 0 and px > 0:
                            if pos_type == 'short':
                                entry_price = px + (pnl_f / qty)
                            else:
                                entry_price = px - (pnl_f / qty)
                    except Exception:
                        entry_price = entry_price

                pnl_pct = None
                if entry_price and r.price and entry_price > 0:
                    if pos_type == 'short':
                        pnl_pct = round((entry_price / r.price - 1) * 100, 3)
                    else:
                        pnl_pct = round((r.price / entry_price - 1) * 100, 3)
                pnl_pct_leveraged = round(pnl_pct * leverage, 3) if pnl_pct is not None else None
                margin_pnl = round(float(pnl), 4) if pnl is not None else None
                mae_pct = None
                mfe_pct = None
                try:
                    entry_dt_for_excursion = None
                    if entry_ts_ms:
                        entry_dt_for_excursion = _dt.utcfromtimestamp(int(entry_ts_ms) / 1000)
                    elif prev_entry is not None:
                        entry_dt_for_excursion = prev_entry.executed_at or prev_entry.created_at
                    exit_dt_for_excursion = r.executed_at or r.created_at
                    ep_for_excursion = float(entry_price) if entry_price not in (None, '') else None
                    if entry_dt_for_excursion and exit_dt_for_excursion and ep_for_excursion and ep_for_excursion > 0:
                        start_ms = int(entry_dt_for_excursion.timestamp() * 1000)
                        end_ms = int(exit_dt_for_excursion.timestamp() * 1000)
                        bars = (
                            session.query(OHLCV)
                            .filter(
                                OHLCV.symbol_id == r.symbol_id,
                                OHLCV.timeframe == '5m',
                                OHLCV.timestamp >= start_ms,
                                OHLCV.timestamp <= end_ms,
                            )
                            .order_by(OHLCV.timestamp.asc())
                            .all()
                        )
                        if bars:
                            max_high = max(float(b.high or 0.0) for b in bars)
                            min_low = min(float(b.low or 0.0) for b in bars)
                            if pos_type == 'short':
                                mfe_pct = round((ep_for_excursion / min_low - 1) * 100, 4) if min_low > 0 else None
                                mae_pct = round((ep_for_excursion / max_high - 1) * 100, 4) if max_high > 0 else None
                            else:
                                mfe_pct = round((max_high / ep_for_excursion - 1) * 100, 4)
                                mae_pct = round((min_low / ep_for_excursion - 1) * 100, 4)
                except Exception:
                    mae_pct = None
                    mfe_pct = None
                xgb_shap = None
                try:
                    if isinstance(entry_meta.get('xgb_shap'), dict):
                        xgb_shap = entry_meta.get('xgb_shap')
                    elif isinstance(meta.get('xgb_shap'), dict):
                        xgb_shap = meta.get('xgb_shap')
                    if xgb_shap is None:
                        lookup_dt = None
                        if entry_ts_ms:
                            lookup_dt = _dt.utcfromtimestamp(int(entry_ts_ms) / 1000)
                        elif prev_entry is not None:
                            lookup_dt = prev_entry.executed_at or prev_entry.created_at
                        if lookup_dt is not None:
                            mp_q = session.query(ModelPrediction).filter(
                                ModelPrediction.symbol == sym,
                                ModelPrediction.action == entry_action,
                                ModelPrediction.timestamp <= lookup_dt,
                            )
                            if account_id:
                                mp_q = mp_q.filter(ModelPrediction.market_conditions.like(f'%"account_id": "{account_id}"%'))
                            for pred_row in mp_q.order_by(ModelPrediction.timestamp.desc()).limit(50).all():
                                mc = {}
                                try:
                                    mc = json.loads(pred_row.market_conditions or '{}')
                                except Exception:
                                    mc = {}
                                if isinstance(mc.get('xgb_shap'), dict):
                                    xgb_shap = mc.get('xgb_shap')
                                    break
                except Exception:
                    xgb_shap = None
                try:
                    normalized_reason = str(exit_reason or '').strip().lower()
                    if normalized_reason in ('stop_loss', 'sl', 'take_profit', 'tp') and pnl_pct is not None:
                        if normalized_reason in ('stop_loss', 'sl') and float(pnl_pct) > 0:
                            exit_reason = 'take_profit'
                        elif normalized_reason in ('take_profit', 'tp') and float(pnl_pct) < 0:
                            exit_reason = 'stop_loss'
                except Exception:
                    pass

                dur = None
                if entry_ts_ms and r.executed_at:
                    try:
                        dur = round((r.executed_at.timestamp() - int(entry_ts_ms) / 1000) / 60, 1)
                    except Exception:
                        pass
                elif prev_entry is not None and prev_entry.executed_at and r.executed_at:
                    try:
                        dur = round((r.executed_at.timestamp() - prev_entry.executed_at.timestamp()) / 60, 1)
                    except Exception:
                        dur = None

                if not exit_reason:
                    try:
                        tp_pct_default, sl_pct_default, hold_steps_default = _read_xgb_result_meta(model_path)
                        hold_minutes = (hold_steps_default * 5) if hold_steps_default else None
                        if hold_minutes is not None and dur is not None and abs(float(dur) - float(hold_minutes)) <= 45:
                            exit_reason = 'holdstep'
                        elif pnl_pct is not None and tp_pct_default is not None and float(pnl_pct) >= max(tp_pct_default * 0.85, 0.5):
                            exit_reason = 'tp'
                        elif pnl_pct is not None and sl_pct_default is not None and float(pnl_pct) <= -max(sl_pct_default * 0.85, 0.5):
                            exit_reason = 'sl'
                        elif hold_minutes is not None and dur is not None and float(dur) >= float(hold_minutes):
                            exit_reason = 'holdstep'
                        elif str(model_family or '').strip().lower() == 'xgb' and dur is not None and float(dur) >= 600:
                            exit_reason = 'holdstep'
                    except Exception:
                        pass

                if account_filter and str(account_id or '') != account_filter:
                    continue

                item_out = {
                    'symbol': sym,
                    'side': pos_type or 'long',
                    'entry_price': entry_price,
                    'exit_price': r.price,
                    'quantity': r.quantity,
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'pnl': round(pnl, 4) if pnl is not None else None,
                    'pnl_pct': pnl_pct,
                    'leverage': leverage,
                    'pnl_pct_leveraged': pnl_pct_leveraged,
                    'margin_pnl': margin_pnl,
                    'exit_reason': exit_reason,
                    'model_path': model_path,
                    'model_family': model_family,
                    'account_id': account_id,
                    'account_label': account_label,
                    'bal_before': round(bal_before, 2) if bal_before is not None else None,
                    'bal_after': round(bal_after, 2) if bal_after is not None else None,
                    'duration_min': dur,
                    'mae': mae_pct,
                    'mfe': mfe_pct,
                    'xgb_shap': xgb_shap,
                }

                duplicate_idx = None
                try:
                    for idx, prev in enumerate(trades_out):
                        same_position = (
                            prev.get('symbol') == item_out.get('symbol')
                            and prev.get('side') == item_out.get('side')
                            and prev.get('account_id') == item_out.get('account_id')
                            and prev.get('model_path') == item_out.get('model_path')
                            and prev.get('entry_time') == item_out.get('entry_time')
                            and round(float(prev.get('quantity') or 0), 8) == round(float(item_out.get('quantity') or 0), 8)
                        )
                        if not same_position:
                            continue
                        t_prev = _dt.fromisoformat(str(prev.get('exit_time')))
                        t_cur = _dt.fromisoformat(str(item_out.get('exit_time')))
                        if abs((t_prev - t_cur).total_seconds()) <= 600:
                            duplicate_idx = idx
                            break
                except Exception:
                    duplicate_idx = None
                if duplicate_idx is not None:
                    prev_reason = str(trades_out[duplicate_idx].get('exit_reason') or '').strip().lower()
                    cur_reason = str(item_out.get('exit_reason') or '').strip().lower()
                    if cur_reason == 'manual' and prev_reason not in ('manual',):
                        trades_out[duplicate_idx] = item_out
                    continue
                trades_out.append(item_out)
        finally:
            session.close()

        return jsonify({'success': True, 'trades': trades_out, 'total': len(trades_out)}), 200
    except Exception as e:
        logging.error(f"trade_history error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@trading_bp.post('/api/trading/save_shap_csv')
def trading_save_shap_csv():
    """Сохраняет выбранные SHAP-результаты из UI в predict_test/xgb_shap."""
    try:
        data = request.get_json(silent=True) or {}
        csv_text = data.get('csv')
        if not isinstance(csv_text, str) or not csv_text.strip():
            return jsonify({'success': False, 'error': 'csv is required'}), 400

        out_dir = Path('predict_test') / 'xgb_shap'
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime('%Y%m%d_%H%M%SZ')
        rows_count = data.get('rows_count')
        try:
            rows_part = f"_{max(0, int(rows_count))}rows" if rows_count is not None else ""
        except Exception:
            rows_part = ""
        filename = f"trading_shap_results_{stamp}{rows_part}.csv"
        out_path = out_dir / filename
        out_path.write_text(csv_text, encoding='utf-8')

        return jsonify({'success': True, 'path': str(out_path), 'filename': filename}), 200
    except Exception as e:
        logging.error(f"save_shap_csv error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@trading_bp.get('/api/trading/accounts_balances')
def trading_accounts_balances():
    """Баланс по всем Bybit API аккаунтам из БД (UNIFIED/derivatives), чтобы диагностировать 'USDT_free'."""
    try:
        rc = get_redis_client()
        cache_key = 'trading:balances:bybit_accounts'
        force = str(request.args.get('force') or '').strip().lower() in ('1', 'true', 'yes', 'on')
        if rc and not force:
            cached = rc.get(cache_key)
            if cached:
                try:
                    obj = json.loads(cached if isinstance(cached, str) else cached.decode('utf-8'))
                    if isinstance(obj, dict) and obj.get('success'):
                        return jsonify(obj), 200
                except Exception:
                    pass

        # Собираем аккаунты из settings (не отдаём секреты в ответ)
        try:
            from utils.settings_store import ensure_settings_table, list_settings, get_setting_value
            ensure_settings_table()
            rows = list_settings(scope='api', group='bybit')
        except Exception:
            rows = []
            get_setting_value = None

        ids = []
        for r in rows or []:
            try:
                k = str(r.get('key') or '')
                m = re.match(r'^BYBIT_(\d+)_API_KEY$', k)
                if m:
                    ids.append(int(m.group(1)))
            except Exception:
                continue
        ids = sorted(set(ids))

        accounts = []
        for idx in ids:
            try:
                if not get_setting_value:
                    continue
                api_key = get_setting_value('api', 'bybit', f'BYBIT_{idx}_API_KEY')
                secret = get_setting_value('api', 'bybit', f'BYBIT_{idx}_SECRET_KEY')
                label = get_setting_value('api', 'bybit', f'BYBIT_{idx}_LABEL') or f'Account {idx}'
                if api_key and secret:
                    accounts.append({'id': idx, 'label': label})
            except Exception:
                continue

        # Если аккаунтов нет в БД — ответим пустым списком
        if not accounts:
            obj = {'success': True, 'ts': datetime.utcnow().isoformat(), 'accounts': []}
            if rc:
                try:
                    rc.setex(cache_key, 10, json.dumps(obj, ensure_ascii=False))
                except Exception:
                    pass
            return jsonify(obj), 200

        # Запрашиваем баланс по каждому аккаунту
        results = []
        try:
            import ccxt
        except Exception as e:
            return jsonify({'success': False, 'error': f'ccxt import error: {e}'}), 500

        def _extract_usdt(bal: dict) -> dict:
            free = None
            total = None
            try:
                u = bal.get('USDT')
                if isinstance(u, dict):
                    free = u.get('free')
                    total = u.get('total')
                elif u is not None:
                    # иногда ccxt кладёт число/строку
                    free = float(u)
                    total = float(u)
            except Exception:
                pass
            if free is None:
                try:
                    f = bal.get('free') or {}
                    if isinstance(f, dict) and f.get('USDT') is not None:
                        free = float(f.get('USDT'))
                except Exception:
                    pass
            if total is None:
                try:
                    t = bal.get('total') or {}
                    if isinstance(t, dict) and t.get('USDT') is not None:
                        total = float(t.get('USDT'))
                except Exception:
                    pass
            return {'usdt_free': free, 'usdt_total': total}

        from utils.settings_store import get_setting_value as _gsv
        for a in accounts:
            idx = a['id']
            label = a.get('label') or f'Account {idx}'
            api_key = _gsv('api', 'bybit', f'BYBIT_{idx}_API_KEY')
            secret = _gsv('api', 'bybit', f'BYBIT_{idx}_SECRET_KEY')
            err = None
            payload = {'account_id': idx, 'label': label}
            try:
                ex = ccxt.bybit({
                    'apiKey': api_key,
                    'secret': secret,
                    'enableRateLimit': True,
                    'timeout': 30000,
                    'options': {
                        'defaultType': 'swap',
                        'defaultMarginMode': 'isolated',
                        'recv_window': 20000,
                        'recvWindow': 20000,
                        'adjustForTimeDifference': True,
                        'timeDifference': True,
                    }
                })
                # Пытаемся взять именно UNIFIED/деривативный баланс (ccxt-совместимый best effort)
                variants = [
                    {'accountType': 'UNIFIED', 'type': 'swap', 'recv_window': 20000, 'recvWindow': 20000},
                    {'accountType': 'UNIFIED', 'recv_window': 20000, 'recvWindow': 20000},
                    {'type': 'swap', 'recv_window': 20000, 'recvWindow': 20000},
                    {'recv_window': 20000, 'recvWindow': 20000},
                ]
                bal = None
                used = None
                last_e = None
                for v in variants:
                    try:
                        b = ex.fetch_balance(v)
                        if isinstance(b, dict) and b:
                            bal = b
                            used = v
                            break
                    except Exception as ee:
                        last_e = ee
                        continue
                if bal is None:
                    raise RuntimeError(f'fetch_balance failed: {last_e}')
                payload.update(_extract_usdt(bal))
                payload['params_used'] = used
            except Exception as e:
                err = str(e)
            if err:
                payload['error'] = err
            results.append(payload)

        obj = {'success': True, 'ts': datetime.utcnow().isoformat(), 'accounts': results}
        if rc:
            try:
                rc.setex(cache_key, 10, json.dumps(obj, ensure_ascii=False))
            except Exception:
                pass
        return jsonify(obj), 200
    except Exception as e:
        logging.error(f"Ошибка получения accounts_balances: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.get('/api/trading/balance')
def trading_balance():
    """Баланс берём из Redis-кэша trading:current_status"""
    try:
        _rc = get_redis_client()
        cached = _rc.get('trading:current_status') if _rc else None
        resp_obj = None
        if cached:
            try:
                st = json.loads(cached)
                if isinstance(st, dict) and st.get('balance'):
                    resp_obj = {
                        'success': True,
                        'balance': st.get('balance'),
                        'is_trading': st.get('is_trading', False),
                        'is_fresh': st.get('is_fresh', False)
                    }
            except Exception:
                resp_obj = None
        if resp_obj is None:
            resp_obj = {
                'success': True,
                'balance': {},
                'message': 'balance not available (agent not running)'
            }
        return jsonify(resp_obj), 200
    except Exception as e:
        logging.error(f"Ошибка получения баланса: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.post('/api/trading/manual_buy')
def trading_manual_buy_bypass_prediction():
    """Реальная покупка в обход предсказания с установкой TP/SL по текущим настройкам.

    Body JSON: { symbol: 'BTCUSDT', qty?: float, override?: {execution_mode, leverage, limit_config, exit_mode, take_profit_pct, stop_loss_pct, risk_management_type} }
    """
    try:
        try:
            logging.info("[/api/trading/manual_buy] ▶ request received")
        except Exception:
            pass
        data = request.get_json() or {}
        sym = str(data.get('symbol') or 'BTCUSDT').upper().strip()
        if not sym.endswith('USDT'):
            sym = f"{sym}USDT"

        # 1) Создаём TradingAgent
        try:
            from trading_agent.trading_agent import TradingAgent
        except Exception as e:
            return jsonify({'success': False, 'error': f'TradingAgent import error: {e}'}), 500

        # Путь модели из Redis (если есть)
        model_path: Optional[str] = None
        try:
            rc = get_redis_client()
            mp = rc.get('trading:model_path')
            if mp:
                try:
                    model_path = mp.decode('utf-8') if isinstance(mp, (bytes, bytearray)) else str(mp)
                except Exception:
                    model_path = str(mp)
        except Exception:
            rc = None

        agent = TradingAgent(model_path=model_path)  # type: ignore
        agent.symbol = sym
        agent.base_symbol = sym

        # Количество (если не задано — рассчитать)
        qty = data.get('qty')
        try:
            qty = float(qty) if qty is not None else None
        except Exception:
            qty = None
        if qty is None or qty <= 0:
            try:
                qty = float(agent._calculate_trade_amount())  # type: ignore
            except Exception:
                qty = 0.001

        # 2) Эффективные настройки (per-symbol с фолбэком на глобальные)
        take_profit_pct = 1.0
        stop_loss_pct = 1.0
        risk_type = 'exchange_orders'
        exit_mode = 'prediction'
        execution_mode = 'market'
        leverage_val = 1
        limit_config = None
        # Новые параметры ATR‑режима
        risk_stop_mode = 'fixed_pct'  # 'fixed_pct' | 'atr_tp_sl'
        atr_k = 2.5
        atr_m = 1.8
        atr_min_sl_mult = 1.0
        # Параметры трейлинга (для биржевого trailing stop)
        trailing_enabled = False
        trailing_mode = None
        trailing_activate_mode = None
        trailing_activate_value = None
        atr_trail_mult = 1.0
        try:
            if rc is None:
                rc = get_redis_client()

            # Глобальные/пер-символьные ключи
            tp_sym = rc.get(f'trading:take_profit_pct:{sym}')
            sl_sym = rc.get(f'trading:stop_loss_pct:{sym}')
            tp_glob = rc.get('trading:take_profit_pct')
            sl_glob = rc.get('trading:stop_loss_pct')
            rt = rc.get('trading:risk_management_type')
            em = rc.get(f'trading:exit_mode:{sym}')
            ex_mode = rc.get(f'trading:execution_mode:{sym}')
            lev = rc.get(f'trading:leverage:{sym}')
            lc = rc.get(f'trading:limit_config:{sym}')

            # Переопределения из запроса (если пришли)
            override = data.get('override') or {}
            def _pick_num(val, redis_val, default):
                if val is not None:
                    return float(val)
                if redis_val is not None:
                    rv = redis_val.decode('utf-8') if isinstance(redis_val, (bytes, bytearray)) else redis_val
                    try:
                        return float(rv)
                    except Exception:
                        return default
                return default
            def _pick_str(val, redis_val, default):
                if val is not None:
                    return str(val)
                if redis_val is not None:
                    try:
                        return (redis_val.decode('utf-8') if isinstance(redis_val, (bytes, bytearray)) else str(redis_val))
                    except Exception:
                        return default
                return default
            def _pick_bool(val, redis_val, default):
                candidate = val if val is not None else redis_val
                if candidate is None:
                    return default
                if isinstance(candidate, (bytes, bytearray)):
                    try:
                        candidate = candidate.decode('utf-8')
                    except Exception:
                        candidate = str(candidate)
                if isinstance(candidate, str):
                    return candidate.strip().lower() in ('1','true','yes','on')
                try:
                    return bool(candidate)
                except Exception:
                    return default

            take_profit_pct = _pick_num(override.get('take_profit_pct') or data.get('take_profit_pct'), tp_sym or tp_glob, take_profit_pct)
            stop_loss_pct = _pick_num(override.get('stop_loss_pct') or data.get('stop_loss_pct'), sl_sym or sl_glob, stop_loss_pct)

            if override.get('risk_management_type') or data.get('risk_management_type') or rt:
                raw = override.get('risk_management_type') or data.get('risk_management_type') or rt
                risk_type = raw.decode('utf-8') if isinstance(raw, (bytes, bytearray)) else str(raw)

            # ATR режим и параметры
            risk_stop_mode = _pick_str((override.get('risk_stop_mode') or data.get('risk_stop_mode')), (rc.get(f'trading:risk_stop_mode:{sym}') or rc.get('trading:risk_stop_mode')), risk_stop_mode)
            atr_k = _pick_num((override.get('atr_k') or data.get('atr_k')), (rc.get(f'trading:atr_k:{sym}') or rc.get('trading:atr_k')), atr_k)
            atr_m = _pick_num((override.get('atr_m') or data.get('atr_m')), (rc.get(f'trading:atr_m:{sym}') or rc.get('trading:atr_m')), atr_m)
            atr_min_sl_mult = _pick_num((override.get('atr_min_sl_mult') or data.get('atr_min_sl_mult')), (rc.get(f'trading:atr_min_sl_mult:{sym}') or rc.get('trading:atr_min_sl_mult')), atr_min_sl_mult)
            # Трейлинг
            trailing_enabled = _pick_bool((override.get('trailing_enabled') if override.get('trailing_enabled') is not None else data.get('trailing_enabled')),
                                          (rc.get(f'trading:trailing_enabled:{sym}') or rc.get('trading:trailing_enabled')),
                                          trailing_enabled)
            trailing_mode = _pick_str((override.get('trailing_mode') or data.get('trailing_mode')),
                                      (rc.get(f'trading:trailing_mode:{sym}') or rc.get('trading:trailing_mode')),
                                      trailing_mode)
            atr_trail_mult = _pick_num((override.get('atr_trail_mult') or data.get('atr_trail_mult')),
                                       (rc.get(f'trading:atr_trail_mult:{sym}') or rc.get('trading:atr_trail_mult')),
                                       atr_trail_mult)
            trailing_activate_mode = _pick_str((override.get('trailing_activate_mode') or data.get('trailing_activate_mode')),
                                               (rc.get(f'trading:trailing_activate_mode:{sym}') or rc.get('trailing_activate_mode')),
                                               trailing_activate_mode or 'percent')
            trailing_activate_value = _pick_num((override.get('trailing_activate_value') or data.get('trailing_activate_value')),
                                                (rc.get(f'trading:trailing_activate_value:{sym}') or rc.get('trailing_activate_value')),
                                                trailing_activate_value if trailing_activate_value is not None else 0.5)

            # exit_mode / execution_mode / leverage / limit_config
            raw_exit = override.get('exit_mode') or data.get('exit_mode') or em
            if raw_exit:
                exit_mode = raw_exit.decode('utf-8') if isinstance(raw_exit, (bytes, bytearray)) else str(raw_exit)
            raw_exec = override.get('execution_mode') or data.get('execution_mode') or ex_mode
            if raw_exec:
                execution_mode = raw_exec.decode('utf-8') if isinstance(raw_exec, (bytes, bytearray)) else str(raw_exec)
            raw_lev = override.get('leverage') or data.get('leverage') or lev
            if raw_lev is not None:
                try:
                    leverage_val = max(1, min(5, int(raw_lev.decode('utf-8') if isinstance(raw_lev, (bytes, bytearray)) else str(raw_lev))))
                except Exception:
                    leverage_val = 1
            if override.get('limit_config') or data.get('limit_config') or lc:
                try:
                    limit_config = override.get('limit_config') or data.get('limit_config') or json.loads(lc)  # type: ignore
                except Exception:
                    limit_config = None
            try:
                logging.info(
                    f"[/api/trading/manual_buy] effective settings: sym={sym} exec={execution_mode} risk_type={risk_type} tp={take_profit_pct} sl={stop_loss_pct} lev={leverage_val} limit_cfg={'yes' if isinstance(limit_config, dict) else 'no'}"
                )
            except Exception:
                pass
            try:
                logging.info(
                    f"[/api/trading/manual_buy] trailing settings: enabled={trailing_enabled} risk_stop_mode={risk_stop_mode} trailing_mode={trailing_mode} atr_trail_mult={atr_trail_mult} activate_mode={trailing_activate_mode} activate_value={trailing_activate_value}"
                )
            except Exception:
                pass
        except Exception as e:
            logging.warning(f"[manual_buy] Не удалось прочитать настройки: {e}")

        # Вспомогательная функция ATR‑расчёта TP/SL
        def _atr_tp_sl_prices(symbol: str, entry_price: float, k: float, m: float, min_sl: float, side: str):
            try:
                atr_abs, _, _ = get_atr_1h(symbol, length=None)
            except Exception as _e:
                raise RuntimeError(f"ATR not available: {_e}")
            k_eff = max(float(k), float(min_sl))
            if side == 'buy':  # long
                tp = entry_price + float(m) * atr_abs
                sl = entry_price - k_eff * atr_abs
            else:             # short
                tp = entry_price - float(m) * atr_abs
                sl = entry_price + k_eff * atr_abs
            return float(tp), float(sl)

        # Локальный helper ATR TP/SL остаётся здесь; trailing вынесен в доменный модуль

        # 3) Исполнение по execution_mode
        executed_price = 0.0
        risk_orders = {}
        buy_result = None
        try:
            if execution_mode == 'limit_post_only':
                # Запускаем DDD-стратегию с заданным плечом
                from tasks.celery_task_trade import start_execution_strategy
                payload = {
                    'symbol': sym,
                    'execution_mode': 'limit_post_only',
                    'side': 'buy',
                    'qty': float(qty),
                    'limit_config': (limit_config or {}),
                    'leverage': leverage_val
                }
                try:
                    logging.info(f"[/api/trading/manual_buy] enqueue limit_post_only: {payload}")
                except Exception:
                    pass
                start_execution_strategy.apply_async(kwargs=payload, queue='trade')
                try:
                    logging.info(f"[/api/trading/manual_buy] enqueued limit_post_only: symbol={sym} qty={qty} lev={leverage_val} has_limit_cfg={bool(limit_config)}")
                except Exception:
                    pass
                # Инициируем гарантированную постановку TP/SL после возможного fill'а
                try:
                    from tasks.celery_task_trade import ensure_risk_orders as _ensure
                    # Немедленно и отложенно (на случай гонок/задержек)
                    _ensure.apply_async(kwargs={'symbol': sym}, countdown=5, queue='trade')
                    _ensure.apply_async(kwargs={'symbol': sym}, countdown=60, queue='trade')
                    try:
                        logging.info(f"[/api/trading/manual_buy] ensure_risk enqueued for {sym} (5s, 60s)")
                    except Exception:
                        pass
                except Exception:
                    pass
                buy_result = {"success": True, "action": "limit_post_only_enqueued", "side": "buy"}
                return jsonify({'success': True, 'symbol': sym, 'buy': buy_result, 'risk_orders': {'mode': 'pending_ddd'}}), 200
            else:
                # Market: явно ставим плечо, затем покупка
                try:
                    if hasattr(agent, 'exchange') and hasattr(agent.exchange, 'set_leverage'):
                        try:
                            agent.exchange.set_leverage(leverage_val, sym)  # type: ignore
                        except Exception:
                            try:
                                agent.exchange.set_leverage(str(leverage_val), sym, {'buyLeverage': str(leverage_val), 'sellLeverage': str(leverage_val)})  # type: ignore
                            except Exception:
                                pass
                except Exception:
                    pass

                # Предрасчет TP/SL
                m = agent.exchange.market(sym)  # type: ignore
                tick = None
                try:
                    info = m.get('info', {})
                    pf = info.get('priceFilter', {})
                    tick = float(pf.get('tickSize')) if pf.get('tickSize') else None
                except Exception:
                    tick = None
                if tick is None:
                    precision = m.get('precision', {}).get('price', 2)
                    tick = 10 ** (-precision)
                cur_price = float(agent._get_current_price() or 0.0)  # type: ignore
                def _norm_price(p: float) -> float:
                    try:
                        return round((round(p / tick)) * tick, 8)  # type: ignore
                    except Exception:
                        return float(p)
                # Выбор режима стопа
                if risk_stop_mode == 'atr_tp_sl':
                    try:
                        _tp_calc, _sl_calc = _atr_tp_sl_prices(sym, cur_price, atr_k, atr_m, atr_min_sl_mult, side='buy')
                        tp_price_pre = _norm_price(_tp_calc)
                        sl_price_pre = _norm_price(_sl_calc)
                    except Exception:
                        # Fallback на проценты
                        tp_price_pre = _norm_price(cur_price * (1.0 + take_profit_pct / 100.0))
                        sl_price_pre = _norm_price(cur_price * (1.0 - stop_loss_pct / 100.0))
                else:
                    tp_price_pre = _norm_price(cur_price * (1.0 + take_profit_pct / 100.0))
                    sl_price_pre = _norm_price(cur_price * (1.0 - stop_loss_pct / 100.0))

                attach_params = {
                    'leverage': str(leverage_val),
                    'marginMode': 'isolated',
                }
                if risk_type in ('exchange_orders', 'both'):
                    attach_params.update({
                        'takeProfit': tp_price_pre,
                        'stopLoss': sl_price_pre,
                        'tpTriggerBy': 'LastPrice',
                        'slTriggerBy': 'LastPrice',
                    })
                try:
                    logging.info(f"[/api/trading/manual_buy] market attach_params: {attach_params}")
                except Exception:
                    pass

                order = agent.exchange.create_market_buy_order(  # type: ignore
                    sym,
                    float(qty),
                    attach_params,
                )
                buy_result = {
                    'success': True,
                    'symbol': sym,
                    'action': 'buy',
                    'quantity': float(qty),
                    'order': order,
                }
                executed_price = float(order.get('average') or order.get('price') or cur_price or 0.0)
                risk_orders = {'mode': 'attached'}
                try:
                    logging.info(f"[/api/trading/manual_buy] market order placed: id={order.get('id')} avg={executed_price}")
                except Exception:
                    pass
                if trailing_enabled and risk_stop_mode == 'atr_trailing' and executed_price > 0:
                    try:
                        trailing_result = setup_trailing_stop_bybit(
                            agent.exchange,
                            sym,
                            float(qty),
                            executed_price,
                            trailing_mode,
                            trailing_activate_mode,
                            trailing_activate_value,
                            atr_trail_mult,
                        )
                        risk_orders['trailing'] = trailing_result
                        try:
                            resp = trailing_result.get('response') if isinstance(trailing_result, dict) else None
                            rc = resp.get('retCode') if isinstance(resp, dict) else None
                            rm = resp.get('retMsg') if isinstance(resp, dict) else None
                            logging.info(
                                f"[/api/trading/manual_buy] trailing placed: dist={trailing_result.get('trailing_dist')} active={trailing_result.get('active_price')} retCode={rc} retMsg={rm}"
                            )
                        except Exception:
                            pass
                        # Сохраняем последнюю попытку установки трейлинга в Redis (для диагностики)
                        try:
                            import time as _t
                            import json as _json
                            rc0 = get_redis_client()
                            rc0.set(
                                f"trading:trailing:last_setup:{sym}",
                                _json.dumps(
                                    {
                                        "ts": _t.time(),
                                        "ok": True,
                                        "symbol": sym,
                                        "source": "manual_buy",
                                        "result": trailing_result,
                                    },
                                    ensure_ascii=False,
                                ),
                            )
                        except Exception:
                            pass
                    except Exception as e_trail:
                        risk_orders['trailing_error'] = str(e_trail)
                        logging.error(f"[/api/trading/manual_buy] trailing setup failed: {e_trail}")
                        try:
                            import time as _t
                            import json as _json
                            rc0 = get_redis_client()
                            rc0.set(
                                f"trading:trailing:last_setup:{sym}",
                                _json.dumps(
                                    {
                                        "ts": _t.time(),
                                        "ok": False,
                                        "symbol": sym,
                                        "source": "manual_buy",
                                        "error": str(e_trail),
                                    },
                                    ensure_ascii=False,
                                ),
                            )
                        except Exception:
                            pass
                else:
                    # Явный лог, почему трейлинг не пытались выставить
                    try:
                        logging.info(
                            f"[/api/trading/manual_buy] trailing skipped: enabled={trailing_enabled} risk_stop_mode={risk_stop_mode} executed_price={executed_price}"
                        )
                    except Exception:
                        pass
                # Дополнительная гарантированная постановка TP/SL (на случай игнорирования attached параметров биржей)
                try:
                    from tasks.celery_task_trade import ensure_risk_orders as _ensure
                    _ensure.apply_async(kwargs={'symbol': sym}, countdown=2, queue='trade')
                    try:
                        logging.info(f"[/api/trading/manual_buy] ensure_risk enqueued for {sym} (2s)")
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception as _ex:
            # Fallback: обычная покупка + отдельные reduceOnly TP/SL
            try:
                logging.warning(f"[/api/trading/manual_buy] market path failed, fallback to direct order: {str(_ex)}")
            except Exception:
                pass
            buy_result = agent.execute_direct_order('buy', symbol=sym, quantity=qty)  # type: ignore
            if not buy_result or not buy_result.get('success'):
                return jsonify({'success': False, 'error': buy_result.get('error') if isinstance(buy_result, dict) else 'buy failed'}), 500
            try:
                order = buy_result.get('order') or {}
                executed_price = order.get('average') or order.get('price')
                if executed_price is None or float(executed_price) <= 0:
                    executed_price = agent._get_current_price()  # type: ignore
                executed_price = float(executed_price)
            except Exception:
                executed_price = 0.0

            if executed_price and executed_price > 0 and risk_type in ('exchange_orders', 'both'):
                try:
                    # Нормализация цен по tickSize
                    m = agent.exchange.market(sym)  # type: ignore
                    tick = None
                    try:
                        info = m.get('info', {})
                        pf = info.get('priceFilter', {})
                        tick = float(pf.get('tickSize')) if pf.get('tickSize') else None
                    except Exception:
                        tick = None
                    if tick is None:
                        precision = m.get('precision', {}).get('price', 2)
                        tick = 10 ** (-precision)
                    def _norm_price2(p: float) -> float:
                        try:
                            return round((round(p / tick)) * tick, 8)  # type: ignore
                        except Exception:
                            return float(p)
                    # Выбор режима стопа
                    if risk_stop_mode == 'atr_tp_sl':
                        try:
                            _tp_calc, _sl_calc = _atr_tp_sl_prices(sym, executed_price, atr_k, atr_m, atr_min_sl_mult, side='buy')
                            tp_price = _norm_price2(_tp_calc)
                            sl_price = _norm_price2(_sl_calc)
                        except Exception:
                            tp_price = _norm_price2(executed_price * (1.0 + take_profit_pct / 100.0))
                            sl_price = _norm_price2(executed_price * (1.0 - stop_loss_pct / 100.0))
                    else:
                        tp_price = _norm_price2(executed_price * (1.0 + take_profit_pct / 100.0))
                        sl_price = _norm_price2(executed_price * (1.0 - stop_loss_pct / 100.0))
                    amount = float(qty)
                    # TP
                    try:
                        tp_order = agent.exchange.create_limit_sell_order(  # type: ignore
                            sym,
                            amount,
                            tp_price,
                            {
                                'reduceOnly': True,
                                'timeInForce': 'GTC',
                                'postOnly': False,
                            }
                        )
                        risk_orders['take_profit'] = {
                            'order_id': tp_order.get('id'),
                            'price': tp_price,
                            'amount': amount,
                        }
                        try:
                            logging.info(f"[/api/trading/manual_buy] TP placed: id={tp_order.get('id')} price={tp_price} amount={amount}")
                        except Exception:
                            pass
                    except Exception as e:
                        risk_orders['take_profit_error'] = str(e)
                        try:
                            logging.error(f"[/api/trading/manual_buy] TP place failed: {e}")
                        except Exception:
                            pass
                    # SL
                    try:
                        sl_order = agent.exchange.create_stop_market_sell_order(  # type: ignore
                            sym,
                            amount,
                            sl_price,
                            {
                                'reduceOnly': True,
                                'stopPrice': sl_price,
                            }
                        )
                        risk_orders['stop_loss'] = {
                            'order_id': sl_order.get('id'),
                            'stop_price': sl_price,
                            'amount': amount,
                        }
                        try:
                            logging.info(f"[/api/trading/manual_buy] SL placed: id={sl_order.get('id')} stopPrice={sl_price} amount={amount}")
                        except Exception:
                            pass
                    except Exception as e:
                        risk_orders['stop_loss_error'] = str(e)
                        try:
                            logging.error(f"[/api/trading/manual_buy] SL place failed: {e}")
                        except Exception:
                            pass
                except Exception as e:
                    risk_orders['error'] = f'risk setup failed: {e}'
                    try:
                        logging.error(f"[/api/trading/manual_buy] risk setup failed: {e}")
                    except Exception:
                        pass
        if executed_price and executed_price > 0 and trailing_enabled and risk_stop_mode == 'atr_trailing' and 'trailing' not in (risk_orders or {}):
            try:
                trailing_result = setup_trailing_stop_bybit(
                    agent.exchange,
                    sym,
                    float(qty),
                    executed_price,
                    trailing_mode,
                    trailing_activate_mode,
                    trailing_activate_value,
                    atr_trail_mult,
                )
                risk_orders['trailing'] = trailing_result
                try:
                    resp = trailing_result.get('response') if isinstance(trailing_result, dict) else None
                    rc = resp.get('retCode') if isinstance(resp, dict) else None
                    rm = resp.get('retMsg') if isinstance(resp, dict) else None
                    logging.info(
                        f"[/api/trading/manual_buy] trailing placed (post-fallback): dist={trailing_result.get('trailing_dist')} active={trailing_result.get('active_price')} retCode={rc} retMsg={rm}"
                    )
                except Exception:
                    pass
                try:
                    import time as _t
                    import json as _json
                    rc0 = get_redis_client()
                    rc0.set(
                        f"trading:trailing:last_setup:{sym}",
                        _json.dumps(
                            {
                                "ts": _t.time(),
                                "ok": True,
                                "symbol": sym,
                                "source": "manual_buy_post_fallback",
                                "result": trailing_result,
                            },
                            ensure_ascii=False,
                        ),
                    )
                except Exception:
                    pass
            except Exception as e_trail:
                risk_orders['trailing_error'] = str(e_trail)
                logging.error(f"[/api/trading/manual_buy] trailing setup failed (post-fallback): {e_trail}")
                try:
                    import time as _t
                    import json as _json
                    rc0 = get_redis_client()
                    rc0.set(
                        f"trading:trailing:last_setup:{sym}",
                        _json.dumps(
                            {
                                "ts": _t.time(),
                                "ok": False,
                                "symbol": sym,
                                "source": "manual_buy_post_fallback",
                                "error": str(e_trail),
                            },
                            ensure_ascii=False,
                        ),
                    )
                except Exception:
                    pass

        try:
            logging.info(f"[/api/trading/manual_buy] ✓ done: symbol={sym} mode={execution_mode} risk={risk_type} executed_price={executed_price} risk_orders_keys={list(risk_orders.keys()) if isinstance(risk_orders, dict) else 'n/a'}")
        except Exception:
            pass
        return jsonify({'success': True, 'symbol': sym, 'buy': buy_result, 'executed_price': executed_price, 'risk_orders': risk_orders}), 200
    except Exception as e:
        logging.error(f"[manual_buy] Error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.post('/api/trading/test_order')
def trading_test_order():
    """Мгновенное размещение РЕАЛЬНОГО рыночного ордера BUY/SELL"""
    try:
        data = request.get_json() or {}
        action = (data.get('action') or '').lower()
        symbol = data.get('symbol')
        quantity = data.get('quantity')
        if action not in ('buy', 'sell'):
            return jsonify({'success': False, 'error': "action должен быть 'buy' или 'sell'"}), 400
        client = docker.from_env()
        try:
            container = client.containers.get('medoedai')
            if container.status != 'running':
                return jsonify({'success': False, 'error': f'Контейнер medoedai не запущен. Статус: {container.status}'}), 500
            model_path = None
            try:
                rc = get_redis_client()
                mp = rc.get('trading:model_path')
                if mp:
                    model_path = mp.decode('utf-8')
            except Exception:
                pass
            def _py_str_literal(s):
                if s is None:
                    return 'None'
                return repr(str(s))
            cmd = f'python -c "from trading_agent.trading_agent import TradingAgent; agent = TradingAgent(model_path={_py_str_literal(model_path)}); result = agent.execute_direct_order(action={_py_str_literal(action)}, symbol={_py_str_literal(symbol)}, quantity={quantity}); import json; print(\'RESULT: \' + json.dumps(result))"'
            exec_result = container.exec_run(cmd, tty=True)
            if exec_result.exit_code == 0:
                output = exec_result.output.decode('utf-8') if exec_result.output else ""
                if 'RESULT:' in output:
                    result_str = output.split('RESULT:')[1].strip()
                    try:
                        result = json.loads(result_str)
                        return jsonify(result), 200
                    except Exception:
                        return jsonify({'success': True, 'message': 'Ордер выполнен', 'output': output}), 200
                else:
                    return jsonify({'success': True, 'message': 'Ордер выполнен', 'output': output}), 200
            else:
                error_output = exec_result.output.decode('utf-8') if exec_result.output else "No error output"
                return jsonify({'success': False, 'error': f'Ошибка выполнения ордера: {error_output}'}), 500
        except docker.errors.NotFound:
            return jsonify({'success': False, 'error': 'Контейнер medoedai не найден'}), 500
        except Exception as e:
            return jsonify({'success': False, 'error': f'Ошибка Docker: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.get('/api/trading/history')
def trading_history():
    """История торговли"""
    try:
        from utils.trade_utils import get_recent_trades, get_trade_statistics
        recent_trades = get_recent_trades(limit=100)
        statistics = get_trade_statistics()
        return jsonify({
            'success': True,
            'trades': recent_trades,
            'statistics': statistics
        }), 200
    except Exception as e:
        logging.error(f"Ошибка получения истории торговли: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.post('/api/trading/ensure_risk')
def ensure_risk():
    """Ручной запуск идемпотентной постановки TP/SL для символа"""
    try:
        data = request.get_json(silent=True) or {}
        sym = str(data.get('symbol') or '').upper().strip()
        if not sym:
            return jsonify({'success': False, 'error': 'symbol is required'}), 400
        try:
            from tasks.celery_task_trade import ensure_risk_orders as _ensure
            # Для limit_post_only позиция появляется не мгновенно; ранний ensure часто только шумит "no_position_cleanup".
            _ensure.apply_async(kwargs={'symbol': sym}, countdown=20, queue='trade')
            _ensure.apply_async(kwargs={'symbol': sym}, countdown=60, queue='trade')
        except Exception as e:
            return jsonify({'success': False, 'error': f'ensure enqueue failed: {e}'}), 500
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.route('/api/trading/regime_config', methods=['GET', 'POST'])
def regime_config():
    """Конфигурация торговых режимов"""
    try:
        rc = get_redis_client()
        if request.method == 'GET':
            config = rc.get('trading:regime_config')
            if config:
                return jsonify({'success': True, 'config': json.loads(config)}), 200
            else:
                return jsonify({'success': True, 'config': {}}), 200
        else:  # POST
            data = request.get_json() or {}
            rc.set('trading:regime_config', json.dumps(data, ensure_ascii=False))
            return jsonify({'success': True, 'message': 'Конфигурация сохранена'}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@trading_bp.post('/api/trading/manual_periodic')
def manual_periodic_trading():
    try:
        return jsonify({
            'success': False,
            'message': 'Периодическая торговая задача запущена вне расписания',
            'error': 'manual_periodic is not supported in session mode'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500

# ==================== ТОРГОВЫЕ СДЕЛКИ API ====================

@trading_bp.get('/api/trades/recent')
def get_recent_trades_api():
    """Получение последних сделок из базы данных"""
    try:
        limit = request.args.get('limit', 50, type=int)
        trades = get_recent_trades(limit=limit)
        
        # Преобразуем в JSON-совместимый формат
        trades_data = []
        for trade in trades:
            trades_data.append({
                'trade_number': trade.trade_number,
                'symbol': trade.symbol.name if trade.symbol else 'Unknown',
                'action': trade.action,
                'status': trade.status,
                'quantity': trade.quantity,
                'price': trade.price,
                'total_value': trade.total_value,
                'model_prediction': trade.model_prediction,
                'current_balance': trade.current_balance,
                'position_pnl': trade.position_pnl,
                'created_at': trade.created_at.isoformat() if trade.created_at else None,
                'executed_at': trade.executed_at.isoformat() if trade.executed_at else None,
                'is_successful': trade.is_successful,
                'error_message': trade.error_message
            })
        
        return jsonify({
            'success': True,
            'trades': trades_data,
            'total': len(trades_data)
        }), 200
        
    except Exception as e:
        logging.error(f"Ошибка получения последних сделок: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@trading_bp.get('/api/trades/statistics')
def get_trade_statistics_api():
    """Получение статистики торговых сделок"""
    try:
        statistics = get_trade_statistics()
        return jsonify({
            'success': True,
            'statistics': statistics
        }), 200
        
    except Exception as e:
        logging.error(f"Ошибка получения статистики сделок: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@trading_bp.get('/api/trades/by_symbol/<symbol_name>')
def get_trades_by_symbol_api(symbol_name):
    """Получение сделок по конкретному символу"""
    try:
        limit = request.args.get('limit', 50, type=int)
        trades = get_trades_by_symbol(symbol_name, limit=limit)
        
        # Преобразуем в JSON-совместимый формат
        trades_data = []
        for trade in trades:
            trades_data.append({
                'trade_number': trade.trade_number,
                'symbol': trade.symbol.name if trade.symbol else 'Unknown',
                'action': trade.action,
                'status': trade.status,
                'quantity': trade.quantity,
                'price': trade.price,
                'total_value': trade.total_value,
                'model_prediction': trade.model_prediction,
                'current_balance': trade.current_balance,
                'position_pnl': trade.position_pnl,
                'created_at': trade.created_at.isoformat() if trade.created_at else None,
                'executed_at': trade.executed_at.isoformat() if trade.executed_at else None,
                'is_successful': trade.is_successful,
                'error_message': trade.error_message
            })
        
        return jsonify({
            'success': True,
            'symbol': symbol_name,
            'trades': trades_data,
            'total': len(trades_data)
        }), 200
        
    except Exception as e:
        logging.error(f"Ошибка получения сделок по символу {symbol_name}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@trading_bp.get('/api/trades/matched_full')
def get_matched_full_trades():
    """Получение полных совпавших сделок с предсказаниями"""
    try:
        def unify_symbol(s: str) -> str:
            return s.replace('USDT', '').replace('USDC', '').upper()
        
        def to_ms(v):
            try:
                if isinstance(v, str):
                    from datetime import datetime
                    return int(datetime.fromisoformat(v.replace('Z', '+00:00')).timestamp() * 1000)
                return int(v)
            except Exception:
                return 0
        
        def bucket_5m(ms: int):
            return (ms // (5 * 60 * 1000)) * (5 * 60 * 1000)
        
        def pick_pred(bkt: int, sym_u: str, typ: str):
            try:
                rc = get_redis_client()
                key = f'pred:{sym_u}:{bkt}:{typ}'
                raw = rc.get(key)
                if raw:
                    return json.loads(raw.decode('utf-8'))
            except Exception:
                pass
            return None
        
        # Получаем параметры запроса
        symbol = request.args.get('symbol', 'BTCUSDT')
        action = request.args.get('action', 'buy')
        limit = int(request.args.get('limit', 100))
        
        # Получаем сделки
        trades = get_recent_trades(limit=limit * 2)  # Берем больше, чтобы отфильтровать
        
        # Фильтруем по символу и действию
        filtered_trades = []
        for trade in trades:
            if (trade.symbol and trade.symbol.name == symbol and 
                trade.action == action and trade.is_successful):
                filtered_trades.append(trade)
        
        # Ограничиваем результат
        filtered_trades = filtered_trades[:limit]
        
        # Обогащаем данными предсказаний
        enriched_trades = []
        for trade in filtered_trades:
            trade_data = {
                'trade_number': trade.trade_number,
                'symbol': trade.symbol.name if trade.symbol else 'Unknown',
                'action': trade.action,
                'status': trade.status,
                'quantity': trade.quantity,
                'price': trade.price,
                'total_value': trade.total_value,
                'model_prediction': trade.model_prediction,
                'current_balance': trade.current_balance,
                'position_pnl': trade.position_pnl,
                'created_at': trade.created_at.isoformat() if trade.created_at else None,
                'executed_at': trade.executed_at.isoformat() if trade.executed_at else None,
                'is_successful': trade.is_successful,
                'error_message': trade.error_message
            }
            
            # Добавляем данные предсказания
            if trade.created_at:
                ms = to_ms(trade.created_at.isoformat())
                bkt = bucket_5m(ms)
                sym_u = unify_symbol(symbol)
                pred = pick_pred(bkt, sym_u, action)
                
                if pred:
                    trade_data['prediction'] = pred
                    trade_data['prediction_bucket'] = bkt
                    trade_data['prediction_symbol'] = sym_u
            
            enriched_trades.append(trade_data)
        
        return jsonify({
            'success': True,
            'trades': enriched_trades,
            'total': len(enriched_trades),
            'symbol': symbol,
            'action': action
        }), 200
        
    except Exception as e:
        logging.error(f"Ошибка получения полных совпавших сделок: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
@trading_bp.post('/api/trading/toggle_trend_filter')
def toggle_trend_filter():
    try:
        data = request.get_json(silent=True) or {}
        session_id = str(data.get('session_id') or '').strip()
        ignore_trend = bool(data.get('ignore_trend_filter'))
        
        if not session_id:
            return jsonify({'success': False, 'error': 'session_id is required'}), 400
            
        rc = get_redis_client()
        session_doc, err = _get_session_or_400(rc, session_id)
        if err:
            return err
            
        session_doc['ignore_trend_filter'] = ignore_trend
        rc.set(session_config_key(session_id), json.dumps(session_doc, ensure_ascii=False))
        rc.set(session_runtime_key(session_id, 'ignore_trend_filter'), '1' if ignore_trend else '0')
        
        return jsonify({'success': True, 'ignore_trend_filter': ignore_trend}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

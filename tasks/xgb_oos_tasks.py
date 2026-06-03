from __future__ import annotations

import fcntl
import json
import logging
import os
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from celery.signals import task_postrun

from tasks import celery
from utils.db_utils import (
    db_get_or_fetch_ohlcv,
    db_get_ohlcv_only,
)
from agents.xgb.config import XgbConfig
from agents.xgb.features import build_xgb_dataset, uses_1m_features, uses_1d_regime
from agents.xgb.predictor import XgbPredictor
from tasks.xgb_tasks import _build_dfs_from_5m, _filter_1m_to_5m_window, _filter_1d_to_5m_window


logger = logging.getLogger(__name__)

_TF_5M_MS = 5 * 60 * 1000
_OOS_END_LAG_HOURS_DEFAULT = 2.0


def _oos_end_lag_hours() -> float:
    try:
        return max(0.0, float(os.getenv("OOS_END_LAG_HOURS", _OOS_END_LAG_HOURS_DEFAULT)))
    except Exception:
        return _OOS_END_LAG_HOURS_DEFAULT


def _oos_end_ts_ms() -> int:
    """Right edge for OOS: last closed 5m candle at least OOS_END_LAG_HOURS behind now."""
    lag_ms = int(_oos_end_lag_hours() * 3600 * 1000)
    cutoff = int(time.time() * 1000) - lag_ms
    return (cutoff // _TF_5M_MS) * _TF_5M_MS


def _oos_1m_fetch_limit(limit_5m: int) -> int:
    """1m rows to load: 5 per 5m bar + margin for gaps / alignment."""
    return int(limit_5m) * 5 + 5000


def _oos_5m_closed_tail(df_5m: Any, limit: int, end_ts_ms: int | None = None) -> Any:
    """Match run_xgb_oos_test: last `limit` closed 5m bars ending at the OOS cutoff."""
    end_ts = int(end_ts_ms) if end_ts_ms is not None else _oos_end_ts_ms()
    d = df_5m.loc[df_5m["timestamp"] <= end_ts]
    if getattr(d, "empty", False):
        return d
    return d.tail(int(max(1, limit)))


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _append_oos_history(hist_path: Path, item: Dict[str, Any], symbol: str, run_dir: Path) -> None:
    lock_path = hist_path.with_suffix(hist_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            prev = _safe_read_json(hist_path)
            history = prev.get("history") if isinstance(prev.get("history"), list) else []
            history.append(item)
            _atomic_write_json(hist_path, {"symbol": symbol, "run_dir": str(run_dir), "history": history})
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _missing_1m_bucket(df_5m: Any, df_1m: Any, end_ts_ms: int | None = None) -> int | None:
    end_ts = int(end_ts_ms) if end_ts_ms is not None else _oos_end_ts_ms()
    buckets_1m = {int(float(x)) // _TF_5M_MS for x in df_1m["timestamp"].tolist()}
    for raw_ts in df_5m["timestamp"].tolist():
        ts = int(float(raw_ts))
        if ts <= end_ts and ts // _TF_5M_MS not in buckets_1m:
            return ts
    return None


@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={"max_retries": 0}, queue="oos")
def prefetch_xgb_oos_ohlcv(
    self,
    symbol: str,
    limit_candles: int,
) -> Dict[str, Any]:
    """
    Single-shot prefetch task for OOS:
    refresh OHLCV in DB once, then OOS workers read DB-only.
    """
    sym = str(symbol or "").strip().upper()
    limit = int(max(1, int(limit_candles or 1)))
    df, err = db_get_or_fetch_ohlcv(
        symbol_name=sym,
        timeframe="5m",
        limit_candles=limit,
        exchange_id="bybit",
        include_error=True,
        force_fetch=True,
    )
    if df is None or df.empty:
        return {
            "success": False,
            "symbol": sym,
            "limit": limit,
            "error": "prefetch returned empty dataframe",
            "fetch_error": err,
        }
    df_1m, err_1m = db_get_or_fetch_ohlcv(
        symbol_name=sym,
        timeframe="1m",
        limit_candles=_oos_1m_fetch_limit(limit),
        exchange_id="bybit",
        include_error=True,
        force_fetch=True,
    )
    if df_1m is None or df_1m.empty:
        return {
            "success": False,
            "symbol": sym,
            "limit": limit,
            "error": "1m prefetch returned empty dataframe",
            "fetch_error": err_1m,
        }
    missing_1m = _missing_1m_bucket(df, df_1m)
    if missing_1m is not None:
        return {
            "success": False,
            "symbol": sym,
            "limit": limit,
            "error": f"missing 1m candles for 5m timestamp={missing_1m}",
            "fetch_error": err_1m,
            "last_ts": int(df["timestamp"].max()),
        }
    df_1d, err_1d = db_get_or_fetch_ohlcv(
        symbol_name=sym,
        timeframe="1d",
        limit_candles=max(120, int(limit / 288) + 120),
        exchange_id="bybit",
        include_error=True,
    )
    if df_1d is None or df_1d.empty:
        return {
            "success": False,
            "symbol": sym,
            "limit": limit,
            "error": "1d prefetch returned empty dataframe",
            "fetch_error": err_1d,
        }
    last_ts = None
    try:
        last_ts = int(df["timestamp"].max())
    except Exception:
        last_ts = None
    return {
        "success": True,
        "symbol": sym,
        "limit": limit,
        "rows": int(len(df)),
        "rows_1m": int(len(df_1m)),
        "rows_1d": int(len(df_1d)),
        "last_ts": last_ts,
        "fetch_error": err,
        "fetch_error_1m": err_1m,
        "fetch_error_1d": err_1d,
    }


def _is_binary_task(task: str) -> bool:
    t = (task or "").strip().lower()
    return t.startswith("entry") or t.startswith("exit")


def _compute_metrics(y: np.ndarray, pred: np.ndarray, is_binary: bool) -> Dict[str, Any]:
    num_classes = 2 if is_binary else 3
    y = y.astype(np.int64, copy=False)
    pred = pred.astype(np.int64, copy=False)

    acc = float(np.mean(pred == y)) if len(y) else 0.0
    y_counts = np.bincount(y, minlength=num_classes).astype(int).tolist()

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y, pred):
        ti = int(t)
        pi = int(p)
        if 0 <= ti < num_classes and 0 <= pi < num_classes:
            cm[ti, pi] += 1

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    precision = (tp / np.maximum(tp + fp, 1e-12)).tolist()
    recall = (tp / np.maximum(tp + fn, 1e-12)).tolist()
    f1 = (
        (2.0 * (np.array(precision) * np.array(recall)))
        / np.maximum(np.array(precision) + np.array(recall), 1e-12)
    ).tolist()

    f1_buy_sell = None
    if not is_binary:
        try:
            f1_buy_sell = float((float(f1[1]) + float(f1[2])) / 2.0)
        except Exception:
            f1_buy_sell = None

    y_non_hold = float(np.mean(y != 0)) if len(y) else 0.0
    pred_non_hold = float(np.mean(pred != 0)) if len(y) else 0.0

    return {
        "val_acc": acc,
        "val_rows": int(len(y)),
        "y_counts_val": y_counts,
        "cm_val": cm.astype(int).tolist(),
        "precision_val": precision,
        "recall_val": recall,
        "f1_val": f1,
        "f1_buy_sell_val": f1_buy_sell,
        "y_non_hold_rate_val": y_non_hold,
        "pred_non_hold_rate_val": pred_non_hold,
    }


def _trade_return(entry: float, exit_price: float, is_long: bool, fee_frac: float) -> float:
    """Trade return as fraction (e.g. 0.01 = 1%). fee_frac is round-trip."""
    if is_long:
        return (exit_price - entry) / max(entry, 1e-12) - fee_frac
    return (entry - exit_price) / max(entry, 1e-12) - fee_frac


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> np.ndarray:
    """
    Simple ATR (SMA of True Range) to drive ATR-trailing exits.
    Returns array same length as inputs, with NaN for warmup bars (< length).
    """
    n = int(min(len(high), len(low), len(close)))
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    L = int(length) if int(length) > 1 else 14
    h = high[:n].astype(np.float64, copy=False)
    l = low[:n].astype(np.float64, copy=False)
    c = close[:n].astype(np.float64, copy=False)

    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))

    atr = np.full((n,), np.nan, dtype=np.float64)
    if n >= L:
        # rolling mean via cumulative sum
        cs = np.cumsum(tr, dtype=np.float64)
        cs[L:] = cs[L:] - cs[:-L]
        atr[L - 1 :] = cs[L - 1 :] / float(L)
    return atr


def _normalize_signal_exit_start_pct(value: float | None) -> float:
    try:
        pct = float(value) if value is not None else 0.65
    except Exception:
        pct = 0.65
    if pct > 1.0:
        pct /= 100.0
    return float(min(1.0, max(0.0, pct)))


def _resolve_signal_exit_start_step(max_hold: int, window: int, start_pct: float) -> int:
    rounded_hold = max(int(window), int(max_hold), 1)
    proposed = int(round(float(rounded_hold) * float(start_pct)))
    return int(max(int(window), min(int(rounded_hold), proposed)))


def _predict_with_signal_series(
    predictor: XgbPredictor,
    X: np.ndarray,
    *,
    is_binary: bool,
    task_name: str,
    cfg,
    direction_override: str | None = None,
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    thr = float(getattr(cfg, "p_enter_threshold", 0.5) or 0.5)
    direction = str(getattr(cfg, "direction", "long") or "long").strip().lower()
    is_long = direction_override == "long" if direction_override in ("long", "short") else (direction == "long")

    if is_binary:
        proba = predictor.predict_proba(X)[:, 1]
        pred = (proba >= thr).astype(np.int64)
        thresholds = np.full(proba.shape, thr, dtype=np.float64)
        return pred, proba.astype(np.float64, copy=False), thresholds

    pred = predictor.predict_action(X)
    try:
        proba_all = predictor.predict_proba(X)
    except Exception:
        proba_all = None

    if proba_all is None or len(proba_all.shape) != 2 or proba_all.shape[1] < 3:
        return pred, None, None

    task = str(task_name or "").strip().lower()
    signal_idx = 1 if is_long else 2
    if task in ("entry_short", "exit_long"):
        signal_idx = 2
    elif task in ("entry_long", "exit_short"):
        signal_idx = 1
    signal = np.asarray(proba_all[:, signal_idx], dtype=np.float64)
    thresholds = np.full(signal.shape, thr, dtype=np.float64)
    return pred, signal, thresholds


def _run_backtest(
    pred: np.ndarray,
    closes: np.ndarray,
    task_name: str,
    cfg,
    start_capital: float = 10000.0,
    exit_mode: str = "policy",
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
    atr: np.ndarray | None = None,
    atr_mult: float = 2.0,
    direction_override: str | None = None,
    signal_values: np.ndarray | None = None,
    signal_thresholds: np.ndarray | None = None,
    signal_exit_enabled: bool = False,
    signal_exit_window: int = 20,
    signal_exit_start_pct: float | None = None,
    signal_exit_threshold: float | None = None,
) -> Dict[str, Any]:
    """Simple backtest: entry on pred=1, exit on max_hold timeout (entry tasks) or pred=2 (directional)."""
    fee_frac = float(getattr(cfg, "fee_bps", 6.0)) / 10000.0
    max_hold = int(getattr(cfg, "max_hold_steps", 48))
    tp_pct = getattr(cfg, "entry_tp_pct", None)
    sl_pct = getattr(cfg, "entry_sl_pct", None)
    trail_pct = getattr(cfg, "entry_trail_pct", None)
    direction = str(getattr(cfg, "direction", "long") or "long").strip().lower()
    if direction_override in ("long", "short"):
        is_long = direction_override == "long"
    else:
        is_long = "long" in task_name or (direction == "long" and "short" not in task_name)
    is_entry = task_name.startswith("entry")
    is_directional = task_name == "directional"
    is_exit = task_name.startswith("exit")

    mode = str(exit_mode or "policy").strip().lower()
    if mode not in ("policy", "hold_steps", "atr_trail"):
        mode = "policy"

    signal_exit_window = max(1, int(signal_exit_window or 20))
    signal_exit_start_pct = _normalize_signal_exit_start_pct(signal_exit_start_pct)
    signal_exit_start_step = _resolve_signal_exit_start_step(
        max_hold=max_hold,
        window=signal_exit_window,
        start_pct=signal_exit_start_pct,
    )
    signal_exit_active = bool(signal_exit_enabled) and not is_exit and signal_values is not None and signal_thresholds is not None
    signal_exit_thr_fixed = None
    if signal_exit_threshold is not None:
        try:
            signal_exit_thr_fixed = float(signal_exit_threshold)
        except Exception:
            signal_exit_thr_fixed = None

    pnl_total = 0.0
    equity = start_capital
    peak = start_capital
    max_dd = 0.0
    trades_list: list = []
    position = None  # {'entry_price', 'entry_idx', 'peak_price'}
    reenter_after_bar = 0  # for exit: bar index when we can re-enter after exit
    signal_exit_last_signal = None
    signal_exit_last_threshold = None
    signal_exit_last_avg_signal = None
    signal_exit_last_avg_threshold = None
    signal_exit_last_history_size = 0
    signal_exit_ready = False

    n = min(len(pred), len(closes))
    for i in range(n):
        price = float(closes[i])
        if price <= 0:
            continue

        # exits for open position
        if position is not None:
            bars_held = i - position["entry_idx"]
            # exit_*: exit on pred=1 (exit signal)
            if is_exit:
                p = int(pred[i]) if i < len(pred) else 0
                if p == 1:
                    ret = _trade_return(position["entry_price"], price, is_long, fee_frac)
                    trades_list.append({"pnl": ret * start_capital, "ret": ret, "bars": bars_held, "forced": False, "reason": "signal"})
                    pnl_total += ret * start_capital
                    position = None
                    reenter_after_bar = i + 1
                    # skip rest of position block
                else:
                    pass  # fall through to peak update
            if position is not None:
                try:
                    ph = price
                    if is_long:
                        if highs is not None and i < len(highs):
                            try:
                                ph = float(highs[i])
                            except Exception:
                                ph = price
                        position["peak_price"] = max(float(position.get("peak_price") or position["entry_price"]), ph)
                    else:
                        if lows is not None and i < len(lows):
                            try:
                                ph = float(lows[i])
                            except Exception:
                                ph = price
                        position["peak_price"] = min(float(position.get("peak_price") or position["entry_price"]), ph)
                except Exception:
                    position["peak_price"] = price

            # entry_* exit policy: trailing / tp / sl (optional)
            if is_entry:
                if mode == "policy":
                    raw_ret = (price - position["entry_price"]) / max(position["entry_price"], 1e-12) if is_long else (position["entry_price"] - price) / max(position["entry_price"], 1e-12)
                    # trailing from peak
                    if trail_pct is not None:
                        peak_p = float(position.get("peak_price") or position["entry_price"])
                        if is_long:
                            dd = (peak_p - price) / max(peak_p, 1e-12)
                        else:
                            dd = (price - peak_p) / max(peak_p, 1e-12)
                        if dd >= float(trail_pct):
                            ret = _trade_return(position["entry_price"], price, is_long, fee_frac)
                            trades_list.append({"pnl": ret * start_capital, "ret": ret, "bars": bars_held, "forced": False, "reason": "trail"})
                            pnl_total += ret * start_capital
                            position = None
                    if position is not None and tp_pct is not None and raw_ret >= float(tp_pct):
                        ret = _trade_return(position["entry_price"], price, is_long, fee_frac)
                        trades_list.append({"pnl": ret * start_capital, "ret": ret, "bars": bars_held, "forced": False, "reason": "tp"})
                        pnl_total += ret * start_capital
                        position = None
                    if position is not None and sl_pct is not None and raw_ret <= float(sl_pct):
                        ret = _trade_return(position["entry_price"], price, is_long, fee_frac)
                        trades_list.append({"pnl": ret * start_capital, "ret": ret, "bars": bars_held, "forced": False, "reason": "sl"})
                        pnl_total += ret * start_capital
                        position = None
                elif mode == "atr_trail" and atr is not None and i < len(atr):
                    try:
                        a = float(atr[i])
                    except Exception:
                        a = float("nan")
                    if a == a and a > 0:  # not NaN
                        peak_p = float(position.get("peak_price") or position["entry_price"])
                        if is_long:
                            stop = peak_p - float(atr_mult) * a
                            triggered = price <= stop
                        else:
                            stop = peak_p + float(atr_mult) * a
                            triggered = price >= stop
                        
                        if triggered:
                            ret = _trade_return(position["entry_price"], price, is_long, fee_frac)
                            trades_list.append({"pnl": ret * start_capital, "ret": ret, "bars": bars_held, "forced": False, "reason": "atr_trail"})
                            pnl_total += ret * start_capital
                            position = None

                if position is not None and signal_exit_active and i < len(signal_values) and i < len(signal_thresholds):
                    try:
                        sig = float(signal_values[i])
                        if signal_exit_thr_fixed is not None:
                            sig_thr = float(signal_exit_thr_fixed)
                        else:
                            sig_thr = float(signal_thresholds[i])
                    except Exception:
                        sig = None
                        sig_thr = None
                    if sig is not None and sig_thr is not None:
                        history = position.setdefault("signal_history", [])
                        history.append({"signal": sig, "threshold": sig_thr})
                        if len(history) > signal_exit_window:
                            del history[:-signal_exit_window]
                        signal_exit_last_signal = sig
                        signal_exit_last_threshold = sig_thr
                        signal_exit_last_history_size = len(history)
                        if bars_held >= signal_exit_start_step and len(history) >= signal_exit_window:
                            signal_exit_ready = True
                            signal_exit_last_avg_signal = float(sum(float(x["signal"]) for x in history) / len(history))
                            signal_exit_last_avg_threshold = float(sum(float(x["threshold"]) for x in history) / len(history))
                            if signal_exit_last_avg_signal < signal_exit_last_avg_threshold:
                                ret = _trade_return(position["entry_price"], price, is_long, fee_frac)
                                trades_list.append({"pnl": ret * start_capital, "ret": ret, "bars": bars_held, "forced": False, "reason": "weak_signal_avg"})
                                pnl_total += ret * start_capital
                                position = None

        # forced exit after max_hold
        if position is not None:
            bars_held = i - position["entry_idx"]
            if bars_held >= max_hold:
                ret = _trade_return(position["entry_price"], price, is_long, fee_frac)
                trades_list.append({"pnl": ret * start_capital, "ret": ret, "bars": bars_held, "forced": True, "reason": "timeout"})
                pnl_total += ret * start_capital
                position = None
                if is_exit:
                    reenter_after_bar = i + 1

        # signals
        p = int(pred[i])
        if is_exit:
            if position is None and i >= reenter_after_bar:
                pk = price
                if is_long:
                    if highs is not None and i < len(highs):
                        try:
                            pk = float(highs[i])
                        except Exception:
                            pk = price
                else:
                    if lows is not None and i < len(lows):
                        try:
                            pk = float(lows[i])
                        except Exception:
                            pk = price
                position = {"entry_price": price, "entry_idx": i, "peak_price": pk}
        elif is_entry:
            if p == 1 and position is None:
                pk = price
                if is_long:
                    if highs is not None and i < len(highs):
                        try:
                            pk = float(highs[i])
                        except Exception:
                            pk = price
                else:
                    if lows is not None and i < len(lows):
                        try:
                            pk = float(lows[i])
                        except Exception:
                            pk = price
                position = {"entry_price": price, "entry_idx": i, "peak_price": pk, "signal_history": []}
        elif is_directional:
            if p == 1 and position is None:
                position = {"entry_price": price, "entry_idx": i, "peak_price": price, "signal_history": []}
            elif p == 2 and position is not None:
                ret = _trade_return(position["entry_price"], price, is_long, fee_frac)
                bars_held = i - position["entry_idx"]
                trades_list.append({"pnl": ret * start_capital, "ret": ret, "bars": bars_held, "forced": False, "reason": "signal"})
                pnl_total += ret * start_capital
                position = None

        # equity & DD
        unr = 0.0
        if position is not None:
            unr = _trade_return(position["entry_price"], price, is_long, fee_frac) * start_capital
        equity = start_capital + pnl_total + unr
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)

    # force close at end
    if position is not None:
        price = float(closes[n - 1])
        ret = _trade_return(position["entry_price"], price, is_long, fee_frac)
        bars_held = (n - 1) - position["entry_idx"]
        trades_list.append({"pnl": ret * start_capital, "ret": ret, "bars": bars_held, "forced": True, "reason": "eof"})
        pnl_total += ret * start_capital

    tc = len(trades_list)
    wins = sum(1 for t in trades_list if t["pnl"] > 0)
    gp = sum(max(0, t["pnl"]) for t in trades_list)
    gl = abs(sum(min(0, t["pnl"]) for t in trades_list))
    pf = (gp / gl) if gl > 1e-12 else (999.99 if gp > 0 else None)
    roi = (pnl_total / start_capital) * 100.0 if start_capital > 0 else 0.0

    # Exit reason stats (to understand HOW exits happened)
    reason_counts: Dict[str, int] = {}
    reason_pnl: Dict[str, float] = {}
    reason_wins: Dict[str, int] = {}
    reason_losses: Dict[str, int] = {}
    forced_counts: Dict[str, int] = {}
    try:
        for t in trades_list:
            r = str(t.get("reason") or "unknown")
            pnl = float(t.get("pnl") or 0.0)
            reason_counts[r] = int(reason_counts.get(r, 0)) + 1
            reason_pnl[r] = float(reason_pnl.get(r, 0.0)) + pnl
            if pnl > 0:
                reason_wins[r] = int(reason_wins.get(r, 0)) + 1
            else:
                reason_losses[r] = int(reason_losses.get(r, 0)) + 1
            if bool(t.get("forced")):
                forced_counts[r] = int(forced_counts.get(r, 0)) + 1
    except Exception:
        reason_counts, reason_pnl, reason_wins, reason_losses, forced_counts = {}, {}, {}, {}, {}

    reason_share: Dict[str, float] = {}
    try:
        denom = float(tc) if tc else 0.0
        if denom > 0:
            for k, v in reason_counts.items():
                reason_share[str(k)] = float(v) / denom
    except Exception:
        reason_share = {}

    return {
        "pnl_total": round(pnl_total, 4),
        "roi_pct": round(roi, 4),
        "winrate": round(wins / tc, 4) if tc else None,
        "profit_factor": round(pf, 4) if pf is not None else None,
        "max_dd": round(max_dd, 6),
        "trades_count": tc,
        "wins": wins,
        "losses": tc - wins,
        "avg_trade_pnl": round(pnl_total / tc, 4) if tc else None,
        "avg_bars_held": round(sum(t["bars"] for t in trades_list) / tc, 1) if tc else None,
        "start_capital": start_capital,
        "equity_end": round(start_capital + pnl_total, 2),
        "reason_counts": reason_counts,
        "reason_share": reason_share,
        "reason_pnl": {k: round(float(v), 4) for k, v in reason_pnl.items()},
        "reason_wins": reason_wins,
        "reason_losses": reason_losses,
        "reason_forced_counts": forced_counts,
        "signal_exit": {
            "enabled": bool(signal_exit_enabled),
            "active": bool(signal_exit_active),
            "window": int(signal_exit_window),
            "start_step": int(signal_exit_start_step),
            "start_pct": round(float(signal_exit_start_pct), 6),
            "exit_threshold": round(float(signal_exit_thr_fixed), 6) if signal_exit_thr_fixed is not None else None,
            "history_size": int(signal_exit_last_history_size),
            "last_signal": round(float(signal_exit_last_signal), 6) if signal_exit_last_signal is not None else None,
            "last_threshold": round(float(signal_exit_last_threshold), 6) if signal_exit_last_threshold is not None else None,
            "avg_signal": round(float(signal_exit_last_avg_signal), 6) if signal_exit_last_avg_signal is not None else None,
            "avg_threshold": round(float(signal_exit_last_avg_threshold), 6) if signal_exit_last_avg_threshold is not None else None,
            "ready": bool(signal_exit_ready),
        },
    }


def _bars_for_days(days: int) -> int:
    # 5m bars/day = 24*60/5 = 288
    return int(max(1, days) * 288)


def _resolve_safe_run_dir(result_dir: str) -> Path:
    target = Path(str(result_dir or "")).resolve()
    base = (Path("result") / "xgb").resolve()
    if base not in target.parents:
        raise ValueError("result_dir outside result/xgb")
    if target.parent.name != "runs":
        raise ValueError("result_dir is not a run directory")
    if not target.exists() or not target.is_dir():
        raise ValueError("result_dir not found")
    return target


def _normalize_symbol_for_db(v: str) -> str:
    s = str(v or "").strip().upper().replace("/", "")
    if not s:
        return ""
    # If symbol is code-like folder name (BTC), assume USDT market.
    if not s.endswith(("USDT", "USD", "USDC", "BUSD", "USDP")):
        s = f"{s}USDT"
    return s


def _resolve_oos_symbol(manifest: Dict[str, Any], run_dir: Path) -> str:
    # 1) Primary source: manifest.symbol
    sym = _normalize_symbol_for_db(manifest.get("symbol") or "")
    if sym:
        return sym
    # 2) Fallback: manifest.symbol_code
    sym = _normalize_symbol_for_db(manifest.get("symbol_code") or "")
    if sym:
        return sym
    # 3) Fallback: result/xgb/<SYMBOL_CODE>/runs/<RUN>
    try:
        # run_dir.parent -> runs, run_dir.parent.parent -> <SYMBOL_CODE>
        sym = _normalize_symbol_for_db(run_dir.parent.parent.name)
    except Exception:
        sym = ""
    return sym


@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={"max_retries": 0}, queue="oos")
def run_xgb_oos_test(
    self,
    result_dir: str,
    days: int = 30,
    ts: str | None = None,
    exit_mode: str | None = None,
    atr_len: int | None = None,
    atr_mult: float | None = None,
    override_exit_policy: bool = False,
    entry_tp_pct: float | None = None,
    entry_sl_pct: float | None = None,
    entry_trail_pct: float | None = None,
    p_enter_threshold: float | None = None,
    direction_override: str | None = None,
    signal_exit_enabled: bool = False,
    signal_exit_window: int | None = None,
    signal_exit_start_pct: float | None = None,
    signal_exit_threshold: float | None = None,
    batch_id: str | None = None,
    oos_end_ts_ms: int | None = None,
) -> Dict[str, Any]:
    """
    OOS evaluation for a saved XGB run on the last N days of candles.
    Stores history in <run_dir>/oos_xgb_results.json.
    """
    run_dir = _resolve_safe_run_dir(result_dir)
    manifest = _safe_read_json(run_dir / "manifest.json")
    meta = _safe_read_json(run_dir / "meta.json")
    cfg_snap = meta.get("cfg_snapshot") if isinstance(meta.get("cfg_snapshot"), dict) else {}

    symbol = _resolve_oos_symbol(manifest=manifest, run_dir=run_dir)
    if not symbol:
        raise ValueError("symbol missing in manifest and run_dir fallback")

    cfg = XgbConfig()
    # apply cfg snapshot fields that exist
    for k, v in cfg_snap.items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, v)
            except Exception:
                pass

    # Optional override for entry_* exit policy (affects backtest ONLY).
    if bool(override_exit_policy):
        try:
            cfg.entry_tp_pct = float(entry_tp_pct) if entry_tp_pct is not None else None
        except Exception:
            cfg.entry_tp_pct = None
        try:
            cfg.entry_sl_pct = float(entry_sl_pct) if entry_sl_pct is not None else None
        except Exception:
            cfg.entry_sl_pct = None
        try:
            cfg.entry_trail_pct = float(entry_trail_pct) if entry_trail_pct is not None else None
        except Exception:
            cfg.entry_trail_pct = None

        # Normalize sign conventions to avoid accidental 1-bar exits:
        # - TP must be >= 0
        # - SL must be <= 0
        # - Trail must be >= 0
        try:
            if cfg.entry_tp_pct is not None and float(cfg.entry_tp_pct) < 0:
                cfg.entry_tp_pct = abs(float(cfg.entry_tp_pct))
        except Exception:
            cfg.entry_tp_pct = None
        try:
            if cfg.entry_sl_pct is not None and float(cfg.entry_sl_pct) > 0:
                cfg.entry_sl_pct = -abs(float(cfg.entry_sl_pct))
        except Exception:
            cfg.entry_sl_pct = None
        try:
            if cfg.entry_trail_pct is not None and float(cfg.entry_trail_pct) < 0:
                cfg.entry_trail_pct = abs(float(cfg.entry_trail_pct))
        except Exception:
            cfg.entry_trail_pct = None

    if p_enter_threshold is not None:
        try:
            cfg.p_enter_threshold = float(p_enter_threshold)
        except Exception:
            pass

    task_name = str(getattr(cfg, "task", "") or manifest.get("task") or "").strip().lower()
    is_binary = _is_binary_task(task_name)

    bars = _bars_for_days(int(days))
    # Need some buffer for label lookahead
    lookahead = 0
    try:
        lookahead = int(getattr(cfg, "max_hold_steps", 0) or 0)
    except Exception:
        lookahead = 0
    try:
        lookahead = max(lookahead, int(getattr(cfg, "horizon_steps", 0) or 0))
    except Exception:
        pass

    limit = int(bars + lookahead + 10)
    oos_end_ts = int(oos_end_ts_ms) if oos_end_ts_ms is not None else _oos_end_ts_ms()
    oos_lag_hours = _oos_end_lag_hours()
    # OOS reads strictly from DB; API prefetch is done before batch launch.
    df_5min = db_get_ohlcv_only(
        symbol_name=symbol,
        timeframe="5m",
        limit_candles=limit,
        freshness_max_age_sec=3600,
        allow_api_fallback=False,
        end_ts_ms=oos_end_ts,
    )
    if df_5min is None or df_5min.empty:
        return {"success": False, "error": f"No DB candles for {symbol}. Run prefetch first."}
    try:
        df_5min = df_5min[df_5min["timestamp"] <= int(oos_end_ts)].copy()
    except Exception:
        pass
    if df_5min is None or df_5min.empty:
        return {"success": False, "error": f"No closed 5m DB candles for {symbol}."}
    try:
        last_ts = int(df_5min["timestamp"].max())
        age_sec = max(0.0, (time.time() * 1000.0 - float(last_ts)) / 1000.0)
        if last_ts < int(oos_end_ts):
            return {
                "success": False,
                "error": f"DB candles do not cover OOS cutoff for {symbol}: last_ts={last_ts}, oos_end_ts={int(oos_end_ts)}. Run prefetch first.",
                "db_last_ts": last_ts,
                "oos_end_ts": int(oos_end_ts),
            }
    except Exception:
        pass

    df_5min_oos = df_5min.tail(limit).copy()

    df_1min_oos = None
    df_1d_oos = None
    if uses_1m_features(cfg):
        lim_1m = _oos_1m_fetch_limit(limit)
        df_1min_raw = db_get_ohlcv_only(
            symbol_name=symbol,
            timeframe="1m",
            limit_candles=lim_1m,
            freshness_max_age_sec=3600,
            allow_api_fallback=False,
            end_ts_ms=int(oos_end_ts) + (_TF_5M_MS - 60_000),
        )
        if df_1min_raw is None or df_1min_raw.empty:
            return {"success": False, "error": f"No DB 1m candles for {symbol}. Run prefetch first."}
        try:
            df_1min_oos = _filter_1m_to_5m_window(df_1min_raw, df_5min_oos)
        except Exception as exc:
            return {"success": False, "error": f"1m candles do not cover OOS window: {exc}"}
        miss_ts = _missing_1m_bucket(df_5min_oos, df_1min_oos)
        if miss_ts is not None:
            df_1min_raw2, err_1m2 = db_get_or_fetch_ohlcv(
                symbol_name=symbol,
                timeframe="1m",
                limit_candles=lim_1m,
                exchange_id="bybit",
                include_error=True,
                force_fetch=True,
            )
            if df_1min_raw2 is None or df_1min_raw2.empty:
                return {
                    "success": False,
                    "error": f"1m gap at 5m ts={miss_ts} for {symbol}; repair fetch failed: {err_1m2}",
                }
            try:
                df_1min_oos = _filter_1m_to_5m_window(df_1min_raw2, df_5min_oos)
            except Exception as exc:
                return {"success": False, "error": f"1m repair window failed: {exc}"}
            miss_ts = _missing_1m_bucket(df_5min_oos, df_1min_oos)
            if miss_ts is not None:
                return {
                    "success": False,
                    "error": f"missing 1m for 5m timestamp={miss_ts} after repair; check DB/API gaps",
                }
    if uses_1d_regime(cfg):
        df_1d_raw = db_get_ohlcv_only(
            symbol_name=symbol,
            timeframe="1d",
            limit_candles=max(120, int(limit / 288) + 120),
            freshness_max_age_sec=24 * 3600,
            allow_api_fallback=False,
            end_ts_ms=int(oos_end_ts),
        )
        if df_1d_raw is None or df_1d_raw.empty:
            return {"success": False, "error": f"No DB 1d candles for {symbol}. Run prefetch first."}
        try:
            df_1d_oos = _filter_1d_to_5m_window(df_1d_raw, df_5min_oos)
        except Exception as exc:
            return {"success": False, "error": f"1d candles do not cover OOS window: {exc}"}

    dfs = _build_dfs_from_5m(df_5min_oos, df_1min=df_1min_oos, df_1d=df_1d_oos)

    X, y, meta2, aux = build_xgb_dataset(dfs, cfg)
    if X is None or len(y) == 0:
        return {"success": False, "error": "Empty dataset for OOS"}

    model_path_raw = str(manifest.get("model_path") or "").strip()
    model_candidates: list[Path] = []
    if model_path_raw:
        mp = Path(model_path_raw)
        model_candidates.append(mp if mp.is_absolute() else Path.cwd() / mp)
    model_candidates.append(run_dir / "model.json")
    model_path_obj = next((p for p in model_candidates if p.exists() and p.is_file()), None)
    if model_path_obj is None:
        return {
            "success": False,
            "error": "model file not found for run",
            "run_dir": str(run_dir),
            "model_path_manifest": model_path_raw or None,
        }
    model_path = str(model_path_obj)
    predictor = XgbPredictor(model_path=model_path)
    pred, signal_values, signal_thresholds = _predict_with_signal_series(
        predictor=predictor,
        X=X,
        is_binary=is_binary,
        task_name=task_name,
        cfg=cfg,
        direction_override=direction_override,
    )

    metrics = _compute_metrics(y=y, pred=pred, is_binary=is_binary)

    # Backtest simulation
    closes = aux.get("closes")
    closes_mode = aux.get("closes_mode", "")

    # For exit models: build_xgb_dataset returns exit_rows (sampled trade windows),
    # not a timeseries. For backtest we need per-bar predictions on full timeseries.
    # Re-predict using directional-like features (X_base + zero trade_feats).
    if (closes_mode == "exit_rows" or task_name.startswith("exit")) and closes is None:
        try:
            from agents.xgb.features import _build_base_features
            cfg_ts = XgbConfig()
            for k, v in cfg_snap.items():
                if hasattr(cfg_ts, k):
                    try:
                        setattr(cfg_ts, k, v)
                    except Exception:
                        pass
            X_base, closes_ts = _build_base_features(dfs, cfg_ts)
            W = 200
            if len(closes_ts) > W + 100:
                X_base = X_base[W:]
                closes_ts = closes_ts[W:]
            trade_feats = np.zeros((len(X_base), 4), dtype=np.float32)
            X_ts = np.concatenate([X_base.astype(np.float32), trade_feats], axis=1)
            pred, signal_values, signal_thresholds = _predict_with_signal_series(
                predictor=predictor,
                X=X_ts,
                is_binary=is_binary,
                task_name=task_name,
                cfg=cfg,
                direction_override=direction_override,
            )
            closes = closes_ts
            closes_mode = "timeseries"
        except Exception:
            closes = None

    backtest: Dict[str, Any] = {}
    if closes is not None and len(closes) > 0 and closes_mode == "timeseries":
        highs = None
        lows = None
        atr = None
        try:
            df5 = dfs.get("df_5min")
            if hasattr(df5, "__getitem__") and ("high" in df5.columns) and ("low" in df5.columns) and ("close" in df5.columns):
                h = np.asarray(df5["high"].to_numpy(), dtype=np.float64)
                l = np.asarray(df5["low"].to_numpy(), dtype=np.float64)
                c = np.asarray(df5["close"].to_numpy(), dtype=np.float64)
                W = 200
                start = W if len(c) > (W + 100) else 0
                n2 = int(len(closes))
                highs = h[start : start + n2]
                lows = l[start : start + n2]
                c2 = c[start : start + n2]
                if atr_len is not None or atr_mult is not None or str(exit_mode or "").strip().lower() == "atr_trail":
                    L = int(atr_len) if atr_len is not None else 14
                    atr = _compute_atr(highs, lows, c2, length=L)
        except Exception:
            highs = None
            lows = None
            atr = None

        em = str(exit_mode or "policy").strip().lower()
        am = float(atr_mult) if atr_mult is not None else 2.0
        backtest = _run_backtest(
            pred=pred,
            closes=closes,
            task_name=task_name,
            cfg=cfg,
            exit_mode=em,
            highs=highs,
            lows=lows,
            atr=atr,
            atr_mult=am,
            direction_override=direction_override,
            signal_values=signal_values,
            signal_thresholds=signal_thresholds,
            signal_exit_enabled=bool(signal_exit_enabled),
            signal_exit_window=int(signal_exit_window) if signal_exit_window is not None else 20,
            signal_exit_start_pct=signal_exit_start_pct,
            signal_exit_threshold=signal_exit_threshold,
        )
    else:
        backtest = {"skip": True, "reason": "no timeseries closes for backtest"}

    out = {
        "success": True,
        "symbol": symbol,
        "task": task_name,
        "direction": direction_override if direction_override in ("long", "short") else str(getattr(cfg, "direction", "") or manifest.get("direction") or ""),
        "days": int(days),
        "bars": int(bars),
        "run_dir": str(run_dir),
        "model_path": model_path,
        "cfg_snapshot": asdict(cfg),
        "oos_metrics": metrics,
        "backtest": backtest,
        "meta": meta2,
        "exit_mode": str(exit_mode or "policy"),
        "atr_len": int(atr_len) if atr_len is not None else None,
        "atr_mult": float(atr_mult) if atr_mult is not None else None,
        "signal_exit_enabled": bool(signal_exit_enabled),
        "signal_exit_window": int(signal_exit_window) if signal_exit_window is not None else 20,
        "signal_exit_start_pct": round(float(_normalize_signal_exit_start_pct(signal_exit_start_pct)), 6),
        "signal_exit_threshold": (
            round(float(signal_exit_threshold), 6) if signal_exit_threshold is not None else None
        ),
        "oos_end_ts": int(oos_end_ts),
        "oos_lag_hours": float(oos_lag_hours),
        "signal_exit_start_step": (
            int((backtest.get("signal_exit") or {}).get("start_step"))
            if isinstance(backtest.get("signal_exit"), dict) and (backtest.get("signal_exit") or {}).get("start_step") is not None
            else None
        ),
        "ts": ts or (datetime.utcnow().isoformat() + "Z"),
    }

    # Persist history
    hist_path = run_dir / "oos_xgb_results.json"
    _append_oos_history(
        hist_path,
        {"ts": out["ts"], "days": int(days), "metrics": metrics, "backtest": backtest},
        symbol,
        run_dir,
    )

    return out


@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={"max_retries": 0}, queue="oos")
def finalize_xgb_oos_batch_csv(self, batch_id: str) -> Dict[str, Any]:
    from utils.xgb_oos_batch_csv import finalize_batch_csv

    return finalize_batch_csv(str(batch_id))


@task_postrun.connect
def _xgb_oos_batch_task_postrun(
    sender=None,
    task_id=None,
    kwargs=None,
    retval=None,
    state=None,
    **extra,
) -> None:
    try:
        name = getattr(sender, "name", "") if sender is not None else ""
        if name != "tasks.xgb_oos_tasks.run_xgb_oos_test":
            return
        batch_id = (kwargs or {}).get("batch_id")
        if not batch_id:
            return
        from utils.xgb_oos_batch_csv import on_batch_child_task_done

        if state == "SUCCESS" and isinstance(retval, dict):
            result = retval
        else:
            result = {
                "success": False,
                "error": f"task {state}",
                "run_dir": (kwargs or {}).get("result_dir"),
                "days": (kwargs or {}).get("days"),
                "exit_mode": (kwargs or {}).get("exit_mode"),
            }
        on_batch_child_task_done(str(batch_id), result)
    except Exception:
        pass


@celery.task(bind=True, queue="oos")
def watchdog_xgb_oos_batches(self) -> None:
    from utils.redis_utils import get_redis_client
    from utils.xgb_oos_batch_csv import finalize_batch_csv
    import time
    
    r = get_redis_client()
    for k in r.scan_iter("xgb_oos:batch:*:expected", count=100):
        try:
            batch_id = k.split(":")[2] if isinstance(k, str) else k.decode('utf-8').split(":")[2]
        except Exception:
            continue
        
        expected = int(r.get(f"xgb_oos:batch:{batch_id}:expected") or "0")
        done = int(r.get(f"xgb_oos:batch:{batch_id}:done") or "0")
        
        # Don't close if already done
        if expected > 0 and done < expected:
            last_activity = r.get(f"xgb_oos:batch:{batch_id}:last_activity")
            if last_activity:
                try:
                    last_ts = float(last_activity)
                    # if more than 20 mins since last activity
                    if time.time() - last_ts > 1200:
                        logger.warning(
                            "Watchdog: closing stalled batch %s (done=%s/%s)",
                            batch_id,
                            done,
                            expected,
                        )
                        finalize_batch_csv(batch_id, error_status="partial_timeout")
                except Exception:
                    logger.exception("Watchdog: failed to finalize batch %s", batch_id)


from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from tasks import celery
from utils.db_utils import db_get_or_fetch_ohlcv
from agents.xgb.config import XgbConfig
from agents.xgb.features import build_xgb_dataset
from agents.xgb.predictor import XgbPredictor


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
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


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

    pnl_total = 0.0
    equity = start_capital
    peak = start_capital
    max_dd = 0.0
    trades_list: list = []
    position = None  # {'entry_price', 'entry_idx', 'peak_price'}
    reenter_after_bar = 0  # for exit: bar index when we can re-enter after exit

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
                    if highs is not None and i < len(highs):
                        try:
                            ph = float(highs[i])
                        except Exception:
                            ph = price
                    position["peak_price"] = max(float(position.get("peak_price") or position["entry_price"]), ph)
                except Exception:
                    position["peak_price"] = price

            # entry_* exit policy: trailing / tp / sl (optional)
            if is_entry:
                if mode == "policy":
                    raw_ret = (price - position["entry_price"]) / max(position["entry_price"], 1e-12) if is_long else (position["entry_price"] - price) / max(position["entry_price"], 1e-12)
                    # trailing from peak (in return units, long-only peak; for short keep disabled unless explicitly set)
                    if trail_pct is not None and is_long:
                        peak_p = float(position.get("peak_price") or position["entry_price"])
                        dd = (peak_p - price) / max(peak_p, 1e-12)
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
                elif mode == "atr_trail" and is_long and atr is not None and i < len(atr):
                    try:
                        a = float(atr[i])
                    except Exception:
                        a = float("nan")
                    if a == a and a > 0:  # not NaN
                        peak_p = float(position.get("peak_price") or position["entry_price"])
                        stop = peak_p - float(atr_mult) * a
                        if price <= stop:
                            ret = _trade_return(position["entry_price"], price, is_long, fee_frac)
                            trades_list.append({"pnl": ret * start_capital, "ret": ret, "bars": bars_held, "forced": False, "reason": "atr_trail"})
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
                if highs is not None and i < len(highs):
                    try:
                        pk = float(highs[i])
                    except Exception:
                        pk = price
                position = {"entry_price": price, "entry_idx": i, "peak_price": pk}
        elif is_entry:
            if p == 1 and position is None:
                pk = price
                if highs is not None and i < len(highs):
                    try:
                        pk = float(highs[i])
                    except Exception:
                        pk = price
                position = {"entry_price": price, "entry_idx": i, "peak_price": pk}
        elif is_directional:
            if p == 1 and position is None:
                position = {"entry_price": price, "entry_idx": i, "peak_price": price}
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
    return target


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
) -> Dict[str, Any]:
    """
    OOS evaluation for a saved XGB run on the last N days of candles.
    Stores history in <run_dir>/oos_xgb_results.json.
    """
    run_dir = _resolve_safe_run_dir(result_dir)
    manifest = _safe_read_json(run_dir / "manifest.json")
    meta = _safe_read_json(run_dir / "meta.json")
    cfg_snap = meta.get("cfg_snapshot") if isinstance(meta.get("cfg_snapshot"), dict) else {}

    symbol = str(manifest.get("symbol") or "").strip().upper()
    if not symbol:
        raise ValueError("symbol missing in manifest")

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
    df_5min = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe="5m", limit_candles=limit, exchange_id="bybit")
    if df_5min is None or df_5min.empty:
        return {"success": False, "error": f"No candles for {symbol}"}

    df_5min_oos = df_5min.tail(limit).copy()

    # Build dfs like training task does
    import pandas as pd  # type: ignore

    df_5min_oos = df_5min_oos.copy()
    df_5min_oos["datetime"] = pd.to_datetime(df_5min_oos["timestamp"], unit="ms")
    df_5min_oos.set_index("datetime", inplace=True)
    df_15min = (
        df_5min_oos.resample("15min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    df_1h = (
        df_5min_oos.resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    df_5min_oos.reset_index(drop=False, inplace=True)
    dfs = {"df_5min": df_5min_oos, "df_15min": df_15min, "df_1h": df_1h}

    X, y, meta2, aux = build_xgb_dataset(dfs, cfg)
    if X is None or len(y) == 0:
        return {"success": False, "error": "Empty dataset for OOS"}

    model_path = str(manifest.get("model_path") or (run_dir / "model.json"))
    predictor = XgbPredictor(model_path=model_path)
    if is_binary:
        thr = float(getattr(cfg, "p_enter_threshold", 0.5) or 0.5)
        proba = predictor.predict_proba(X)[:, 1]
        pred = (proba >= thr).astype(np.int64)
    else:
        pred = predictor.predict_action(X)

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
            if is_binary:
                proba_ts = predictor.predict_proba(X_ts)[:, 1]
                pred = (proba_ts >= thr).astype(np.int64)
            else:
                pred = predictor.predict_action(X_ts)
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
        backtest = _run_backtest(pred=pred, closes=closes, task_name=task_name, cfg=cfg, exit_mode=em, highs=highs, lows=lows, atr=atr, atr_mult=am, direction_override=direction_override)
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
        "ts": ts or (datetime.utcnow().isoformat() + "Z"),
    }

    # Persist history
    hist_path = run_dir / "oos_xgb_results.json"
    prev = _safe_read_json(hist_path)
    history = prev.get("history") if isinstance(prev.get("history"), list) else []
    history.append({"ts": out["ts"], "days": int(days), "metrics": metrics, "backtest": backtest})
    _atomic_write_json(hist_path, {"symbol": symbol, "run_dir": str(run_dir), "history": history})

    return out


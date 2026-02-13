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


def _run_backtest(
    pred: np.ndarray, closes: np.ndarray, task_name: str, cfg, start_capital: float = 10000.0,
) -> Dict[str, Any]:
    """Simple backtest: entry on pred=1, exit on max_hold timeout (entry tasks) or pred=2 (directional)."""
    fee_frac = float(getattr(cfg, "fee_bps", 6.0)) / 10000.0
    max_hold = int(getattr(cfg, "max_hold_steps", 48))
    direction = str(getattr(cfg, "direction", "long") or "long").strip().lower()
    is_long = "long" in task_name or (direction == "long" and "short" not in task_name)
    is_entry = task_name.startswith("entry")
    is_directional = task_name == "directional"
    is_exit = task_name.startswith("exit")

    if is_exit:
        return {"skip": True, "reason": "exit-only model, standalone backtest N/A"}

    pnl_total = 0.0
    equity = start_capital
    peak = start_capital
    max_dd = 0.0
    trades_list: list = []
    position = None  # {'entry_price', 'entry_idx'}

    n = min(len(pred), len(closes))
    for i in range(n):
        price = float(closes[i])
        if price <= 0:
            continue

        # forced exit after max_hold
        if position is not None:
            bars_held = i - position["entry_idx"]
            if bars_held >= max_hold:
                ret = _trade_return(position["entry_price"], price, is_long, fee_frac)
                trades_list.append({"pnl": ret * start_capital, "ret": ret, "bars": bars_held, "forced": True})
                pnl_total += ret * start_capital
                position = None

        # signals
        p = int(pred[i])
        if is_entry:
            if p == 1 and position is None:
                position = {"entry_price": price, "entry_idx": i}
        elif is_directional:
            if p == 1 and position is None:
                position = {"entry_price": price, "entry_idx": i}
            elif p == 2 and position is not None:
                ret = _trade_return(position["entry_price"], price, is_long, fee_frac)
                bars_held = i - position["entry_idx"]
                trades_list.append({"pnl": ret * start_capital, "ret": ret, "bars": bars_held, "forced": False})
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
        trades_list.append({"pnl": ret * start_capital, "ret": ret, "bars": bars_held, "forced": True})
        pnl_total += ret * start_capital

    tc = len(trades_list)
    wins = sum(1 for t in trades_list if t["pnl"] > 0)
    gp = sum(max(0, t["pnl"]) for t in trades_list)
    gl = abs(sum(min(0, t["pnl"]) for t in trades_list))
    pf = (gp / gl) if gl > 1e-12 else (999.99 if gp > 0 else None)
    roi = (pnl_total / start_capital) * 100.0 if start_capital > 0 else 0.0

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
def run_xgb_oos_test(self, result_dir: str, days: int = 30, ts: str | None = None) -> Dict[str, Any]:
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
    pred = predictor.predict_action(X)

    metrics = _compute_metrics(y=y, pred=pred, is_binary=is_binary)

    # Backtest simulation
    closes = aux.get("closes")
    backtest: Dict[str, Any] = {}
    if closes is not None and len(closes) > 0 and aux.get("closes_mode") == "timeseries":
        backtest = _run_backtest(pred=pred, closes=closes, task_name=task_name, cfg=cfg)
    else:
        backtest = {"skip": True, "reason": "no timeseries closes for backtest"}

    out = {
        "success": True,
        "symbol": symbol,
        "task": task_name,
        "direction": str(getattr(cfg, "direction", "") or manifest.get("direction") or ""),
        "days": int(days),
        "bars": int(bars),
        "run_dir": str(run_dir),
        "model_path": model_path,
        "cfg_snapshot": asdict(cfg),
        "oos_metrics": metrics,
        "backtest": backtest,
        "meta": meta2,
        "ts": ts or (datetime.utcnow().isoformat() + "Z"),
    }

    # Persist history
    hist_path = run_dir / "oos_xgb_results.json"
    prev = _safe_read_json(hist_path)
    history = prev.get("history") if isinstance(prev.get("history"), list) else []
    history.append({"ts": out["ts"], "days": int(days), "metrics": metrics, "backtest": backtest})
    _atomic_write_json(hist_path, {"symbol": symbol, "run_dir": str(run_dir), "history": history})

    return out


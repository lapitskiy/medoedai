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


@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={"max_retries": 0}, queue="celery")
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

    X, y, meta2, _aux = build_xgb_dataset(dfs, cfg)
    if X is None or len(y) == 0:
        return {"success": False, "error": "Empty dataset for OOS"}

    model_path = str(manifest.get("model_path") or (run_dir / "model.json"))
    predictor = XgbPredictor(model_path=model_path)
    pred = predictor.predict_action(X)

    metrics = _compute_metrics(y=y, pred=pred, is_binary=is_binary)
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
        "meta": meta2,
        "ts": ts or (datetime.utcnow().isoformat() + "Z"),
    }

    # Persist history
    hist_path = run_dir / "oos_xgb_results.json"
    prev = _safe_read_json(hist_path)
    history = prev.get("history") if isinstance(prev.get("history"), list) else []
    history.append({"ts": out["ts"], "days": int(days), "metrics": metrics})
    _atomic_write_json(hist_path, {"symbol": symbol, "run_dir": str(run_dir), "history": history})

    return out


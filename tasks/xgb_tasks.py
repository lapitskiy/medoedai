"""Celery задачи для обучения XGB (XGBoost) моделей."""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd  # type: ignore

from tasks import celery
from utils.db_utils import db_get_or_fetch_ohlcv, load_latest_candles_from_csv_to_db
from utils.parser import parser_download_and_combine_with_library
from utils.redis_utils import get_redis_client

from agents.xgb.config import XgbConfig
from agents.xgb.trainer import XgbTrainer


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _safe_patch_manifest(run_dir: str, patch: Dict[str, Any]) -> None:
    try:
        mf = os.path.join(run_dir, "manifest.json")
        if not os.path.exists(mf):
            return
        with open(mf, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return
        data.update(patch)
        _atomic_write_json(mf, data)
    except Exception:
        return


def _safe_rmtree(path: str) -> bool:
    """
    Safety delete: only allows removing run directories under result/xgb/*/runs/*.
    """
    try:
        from pathlib import Path

        target = Path(str(path or "")).resolve()
        base = (Path("result") / "xgb").resolve()
        if base not in target.parents:
            return False
        # expected .../result/xgb/<SYM>/runs/<RUN>
        if target.parent.name != "runs":
            return False
        shutil.rmtree(str(target), ignore_errors=True)
        return True
    except Exception:
        return False


def _env_flag(name: str, default: bool = False) -> bool:
    v = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return v in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, None)
    if raw is None:
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, None)
    if raw is None:
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _env_opt_float(name: str) -> float | None:
    raw = os.environ.get(name, None)
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _passes_trading_filters_entry_exit(metrics: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
    """
    Hard filters for selecting best config in entry/exit grids (rare-but-strong entries).

    Env controls:
    - XGB_GRID_TRADING_FILTER=1: enable filters (default: 0)
    - XGB_GRID_TARGET_TRADES_PER_MONTH: target trades per month (optional)
    - XGB_GRID_BARS_PER_MONTH: bars/month for converting to pred_non_hold (default: 8640 for 5m)
    - XGB_GRID_TARGET_TOLERANCE: keep if in [target/tol, target*tol] (default: 2.0)
    - XGB_GRID_PRED_NON_HOLD_MIN / XGB_GRID_PRED_NON_HOLD_MAX: explicit bounds (optional)
    - XGB_GRID_Y_NON_HOLD_MIN / XGB_GRID_Y_NON_HOLD_MAX: label coverage bounds (defaults: 0.01..0.70)
    """
    enabled = _env_flag("XGB_GRID_TRADING_FILTER", default=False)
    if not enabled:
        return True, {"enabled": False}

    # Extract rates
    try:
        pred_non_hold = float(metrics.get("pred_non_hold_rate_val", 0.0) or 0.0)
    except Exception:
        pred_non_hold = 0.0
    try:
        y_non_hold = float(metrics.get("y_non_hold_rate_val", 0.0) or 0.0)
    except Exception:
        y_non_hold = 0.0

    # y_non_hold bounds (avoid degenerate labels)
    y_min = _env_float("XGB_GRID_Y_NON_HOLD_MIN", 0.01)
    y_max = _env_float("XGB_GRID_Y_NON_HOLD_MAX", 0.70)
    if y_non_hold < y_min or y_non_hold > y_max:
        return False, {
            "enabled": True,
            "reason": "y_non_hold_out_of_bounds",
            "y_non_hold": y_non_hold,
            "y_min": y_min,
            "y_max": y_max,
        }

    # pred_non_hold bounds: explicit OR derived from target trades/month
    pred_min = _env_opt_float("XGB_GRID_PRED_NON_HOLD_MIN")
    pred_max = _env_opt_float("XGB_GRID_PRED_NON_HOLD_MAX")

    if pred_min is None or pred_max is None:
        target_trades = _env_opt_float("XGB_GRID_TARGET_TRADES_PER_MONTH")
        if target_trades is not None:
            bars_per_month = float(_env_int("XGB_GRID_BARS_PER_MONTH", 8640))
            tol = max(1.01, float(_env_float("XGB_GRID_TARGET_TOLERANCE", 2.0)))
            target_rate = float(target_trades) / max(bars_per_month, 1.0)
            pred_min = target_rate / tol
            pred_max = target_rate * tol

    if pred_min is not None and pred_non_hold < float(pred_min):
        return False, {"enabled": True, "reason": "pred_non_hold_too_low", "pred_non_hold": pred_non_hold, "pred_min": float(pred_min)}
    if pred_max is not None and pred_non_hold > float(pred_max):
        return False, {"enabled": True, "reason": "pred_non_hold_too_high", "pred_non_hold": pred_non_hold, "pred_max": float(pred_max)}

    return True, {
        "enabled": True,
        "pred_non_hold": pred_non_hold,
        "y_non_hold": y_non_hold,
        "pred_min": pred_min,
        "pred_max": pred_max,
        "y_min": y_min,
        "y_max": y_max,
    }

def _build_dfs_from_5m(df_5min: pd.DataFrame) -> Dict[str, Any]:
    df_5min = df_5min.copy()
    df_5min["datetime"] = pd.to_datetime(df_5min["timestamp"], unit="ms")
    df_5min.set_index("datetime", inplace=True)
    df_15min = (
        df_5min.resample("15min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    df_1h = (
        df_5min.resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    df_5min.reset_index(drop=False, inplace=True)
    return {"df_5min": df_5min, "df_15min": df_15min, "df_1h": df_1h}


@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={"max_retries": 0}, queue="celery")
def train_xgb_symbol(
    self,
    symbol: str,
    direction: str = "long",
    horizon_steps: int | None = None,
    threshold: float | None = None,
    limit_candles: int | None = None,
    task: str | None = None,
    fee_bps: float | None = None,
    max_hold_steps: int | None = None,
    min_profit: float | None = None,
    label_delta: float | None = None,
    entry_stride: int | None = None,
    max_trades: int | None = None,
) -> Dict[str, Any]:
    symbol = (symbol or "BTCUSDT").upper()
    direction = (direction or "long").strip().lower()
    rc = get_redis_client()
    logs: List[str] = []

    running_key = f"celery:train:xgb:task:{symbol}:{direction}"

    def push_log(message: str) -> None:
        ts = datetime.utcnow().strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        logs.append(entry)
        self.update_state(state="PROGRESS", meta={"logs": list(logs), "symbol": symbol, "direction": direction})

    # Dedup: if already running, fail fast
    try:
        if rc.exists(running_key):
            return {"success": False, "error": f"XGB training for {symbol} ({direction}) already running"}
        rc.setex(running_key, 6 * 3600, getattr(getattr(self, "request", None), "id", "1"))
    except Exception:
        pass

    try:
        push_log(f"🚀 XGB train start: {symbol} direction={direction}")

        # Load 5m data
        push_log("📥 Load 5m candles from DB...")
        limit_candles = int(limit_candles) if (limit_candles is not None) else 100000
        df_5min = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe="5m", limit_candles=limit_candles, exchange_id="bybit")
        if df_5min is None or df_5min.empty:
            push_log("📥 No data in DB, trying to download and load...")
            try:
                csv_path = parser_download_and_combine_with_library(symbol=symbol, interval="5m", limit=limit_candles)
                if csv_path:
                    loaded = load_latest_candles_from_csv_to_db(file_path=csv_path, symbol_name=symbol, timeframe="5m")
                    push_log(f"✅ Loaded to DB: {loaded} candles")
                df_5min = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe="5m", limit_candles=limit_candles, exchange_id="bybit")
            except Exception as e:
                push_log(f"❌ Auto-download failed: {e}")
                df_5min = None

        if df_5min is None or df_5min.empty:
            return {"success": False, "error": f"No candles for {symbol}"}

        # Build 15m and 1h from 5m (matches existing DQN task behavior)
        dfs = _build_dfs_from_5m(df_5min)

        cfg = XgbConfig(direction=direction)
        if task is not None:
            cfg.task = str(task).strip().lower()
        if horizon_steps is not None:
            cfg.horizon_steps = int(horizon_steps)
        if threshold is not None:
            cfg.threshold = float(threshold)
        if fee_bps is not None:
            cfg.fee_bps = float(fee_bps)
        if max_hold_steps is not None:
            cfg.max_hold_steps = int(max_hold_steps)
        if min_profit is not None:
            cfg.min_profit = float(min_profit)
        if label_delta is not None:
            cfg.label_delta = float(label_delta)
        if entry_stride is not None:
            cfg.entry_stride = int(entry_stride)
        if max_trades is not None:
            cfg.max_trades = int(max_trades)

        push_log(f"🧠 Train: task={cfg.task}, dir={cfg.direction}, horizon={cfg.horizon_steps}, thr={cfg.threshold}, max_hold={cfg.max_hold_steps}, fee_bps={cfg.fee_bps}")
        trainer = XgbTrainer(cfg)
        result = trainer.train(symbol=symbol, dfs=dfs, result_root="result")

        push_log(f"✅ Done: val_acc={result.get('metrics', {}).get('val_acc')}")
        self.update_state(state="SUCCESS", meta={"logs": list(logs), "symbol": symbol, "direction": direction, "result": result})
        return result
    finally:
        try:
            rc.delete(running_key)
        except Exception:
            pass


@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={"max_retries": 0}, queue="celery")
def train_xgb_grid(
    self,
    symbol: str,
    limit_candles_final: int | None = None,
    limit_candles_quick: int | None = None,
    quick_n_estimators: int = 200,
    final_n_estimators: int = 600,
    base_max_hold_steps: int | None = None,
    base_min_profit: float | None = None,
) -> Dict[str, Any]:
    """
    Быстрый grid (horizon/threshold/direction + max_hold_steps/min_profit) на ограниченном окне и малом числе деревьев,
    затем финальный train лучшего конфига на полном окне.
    """
    symbol = (symbol or "BTCUSDT").upper()
    rc = get_redis_client()
    logs: List[str] = []
    running_key = f"celery:train:xgb:grid:{symbol}"

    def push_log(message: str) -> None:
        ts = datetime.utcnow().strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        logs.append(entry)
        self.update_state(state="PROGRESS", meta={"logs": list(logs), "symbol": symbol})

    try:
        if rc.exists(running_key):
            return {"success": False, "error": f"XGB grid for {symbol} already running"}
        rc.setex(running_key, 6 * 3600, getattr(getattr(self, "request", None), "id", "1"))
    except Exception:
        pass

    try:
        thresholds = [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.006, 0.008, 0.01]
        horizons = [24, 48, 64, 96, 128]
        directions = ["long", "short"]

        # Optional: also sweep "position-aware" labeling knobs.
        # NOTE: for task=directional they don't affect labels, but are still useful to unify configs/manifest fields.
        base_max_hold_steps = int(base_max_hold_steps) if base_max_hold_steps is not None else 48
        base_min_profit = float(base_min_profit) if base_min_profit is not None else 0.0

        def _uniq_sorted(vals):
            out = []
            for v in vals:
                if v not in out:
                    out.append(v)
            return sorted(out)

        max_hold_steps_list = _uniq_sorted([max(2, base_max_hold_steps // 2), base_max_hold_steps, base_max_hold_steps * 2])
        if base_min_profit > 0:
            min_profit_list = _uniq_sorted([max(0.0, base_min_profit * 0.5), base_min_profit, base_min_profit * 2.0])
        else:
            # fractions (0.002 = 0.2%)
            min_profit_list = [0.004, 0.006, 0.008, 0.01, 0.012, 0.016, 0.018, 0.02, 0.025]

        limit_candles_final = int(limit_candles_final) if limit_candles_final is not None else 100000
        limit_candles_quick = int(limit_candles_quick) if limit_candles_quick is not None else min(50000, limit_candles_final)

        push_log(f"📥 Load 5m candles for grid: final={limit_candles_final}, quick={limit_candles_quick}")
        df_5min_full = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe="5m", limit_candles=limit_candles_final, exchange_id="bybit")
        if df_5min_full is None or df_5min_full.empty:
            return {"success": False, "error": f"No candles for {symbol}"}

        df_5min_quick = df_5min_full.tail(limit_candles_quick).copy()
        dfs_quick = _build_dfs_from_5m(df_5min_quick)

        total_runs = len(directions) * len(horizons) * len(thresholds) * len(max_hold_steps_list) * len(min_profit_list)
        push_log(f"🧪 Quick grid: {total_runs} runs, n_estimators={quick_n_estimators}")
        results: List[Dict[str, Any]] = []
        best = None
        best_score = -1e9
        grid_id = f"grid-{str(uuid.uuid4())[:6].lower()}"

        for direction in directions:
            for H in horizons:
                for thr in thresholds:
                    for mh in max_hold_steps_list:
                        for mp in min_profit_list:
                            cfg = XgbConfig(direction=direction)
                            cfg.horizon_steps = int(H)
                            cfg.threshold = float(thr)
                            cfg.max_hold_steps = int(mh)
                            cfg.min_profit = float(mp)
                            cfg.n_estimators = int(quick_n_estimators)
                            trainer = XgbTrainer(cfg)
                            r = trainer.train(symbol=symbol, dfs=dfs_quick, result_root="result")
                            try:
                                if isinstance(r, dict) and r.get("result_dir"):
                                    _safe_patch_manifest(str(r["result_dir"]), {"source": "grid", "grid_id": grid_id})
                            except Exception:
                                pass
                            m = r.get("metrics") if isinstance(r.get("metrics"), dict) else {}
                            score = float(m.get("f1_buy_sell_val") if m.get("f1_buy_sell_val") is not None else m.get("val_acc", 0.0))
                            item = {
                                "direction": direction,
                                "horizon_steps": H,
                                "threshold": thr,
                                "max_hold_steps": mh,
                                "min_profit": mp,
                                "score": score,
                                "metrics": m,
                                "run_name": r.get("run_name"),
                                "result_dir": r.get("result_dir"),
                            }
                            results.append(item)
                            if score > best_score:
                                best_score = score
                                best = item
                            push_log(f"grid: dir={direction} H={H} thr={thr} mh={mh} mp={mp} score={score:.4f}")

        if not best:
            return {"success": False, "error": "Grid produced no results"}

        push_log(f"🏁 Best quick: dir={best['direction']} H={best['horizon_steps']} thr={best['threshold']} score={best_score:.4f}")

        # Final train on full window with best config
        dfs_final = _build_dfs_from_5m(df_5min_full)
        cfg_final = XgbConfig(direction=str(best["direction"]))
        cfg_final.horizon_steps = int(best["horizon_steps"])
        cfg_final.threshold = float(best["threshold"])
        cfg_final.max_hold_steps = int(best.get("max_hold_steps", base_max_hold_steps))
        cfg_final.min_profit = float(best.get("min_profit", base_min_profit))
        cfg_final.n_estimators = int(final_n_estimators)
        trainer_final = XgbTrainer(cfg_final)
        final_res = trainer_final.train(symbol=symbol, dfs=dfs_final, result_root="result")

        # Optional cleanup: keep only grid_final by default (disk-friendly).
        # - XGB_GRID_KEEP_ALL=1  -> keep everything
        # - XGB_GRID_KEEP_BEST=1 -> also keep best quick run directory
        keep_all = str(os.environ.get("XGB_GRID_KEEP_ALL", "0")).strip().lower() in ("1", "true", "yes", "on")
        keep_best = str(os.environ.get("XGB_GRID_KEEP_BEST", "0")).strip().lower() in ("1", "true", "yes", "on")
        if not keep_all:
            keep_dirs = {str(final_res.get("result_dir") or "").strip()}
            if keep_best:
                keep_dirs.add(str(best.get("result_dir") or "").strip())
            deleted = 0
            for it in results:
                rd = str(it.get("result_dir") or "").strip()
                if not rd or rd in keep_dirs:
                    continue
                if _safe_rmtree(rd):
                    deleted += 1
            if deleted:
                msg = f"🧹 Cleanup: deleted {deleted} quick grid runs"
                if keep_best:
                    msg += " (kept best quick via XGB_GRID_KEEP_BEST=1)"
                push_log(msg)

        sym_code = (symbol or "").upper().replace("USDT", "").replace("USD", "").replace("USDC", "").replace("BUSD", "").replace("USDP", "")
        grid_dir = os.path.join("result", "xgb", sym_code or "UNKNOWN", "grids", grid_id)
        os.makedirs(grid_dir, exist_ok=True)
        _atomic_write_json(os.path.join(grid_dir, "grid_results.json"), {"symbol": symbol, "results": results, "best": best})
        _atomic_write_json(os.path.join(grid_dir, "final_result.json"), final_res)
        try:
            if isinstance(final_res, dict) and final_res.get("result_dir"):
                _safe_patch_manifest(str(final_res["result_dir"]), {"source": "grid_final", "grid_id": grid_id})
        except Exception:
            pass

        out = {
            "success": True,
            "symbol": symbol,
            "grid_id": grid_id,
            "best": best,
            "final": final_res,
            "grid_dir": grid_dir,
        }
        self.update_state(state="SUCCESS", meta={"logs": list(logs), "symbol": symbol, "result": out})
        return out
    finally:
        try:
            rc.delete(running_key)
        except Exception:
            pass


@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={"max_retries": 0}, queue="celery")
def train_xgb_grid_entry_exit(
    self,
    symbol: str,
    task: str,
    direction: str = "long",
    limit_candles_final: int | None = None,
    limit_candles_quick: int | None = None,
    quick_n_estimators: int = 200,
    final_n_estimators: int = 600,
    base_max_hold_steps: int | None = None,
    base_fee_bps: float | None = None,
    base_min_profit: float | None = None,
    base_label_delta: float | None = None,
) -> Dict[str, Any]:
    """
    Grid for entry_* / exit_* tasks.
    Chooses best by f1_val[1] (class=1) on quick window, then trains final on full window.
    """
    symbol = (symbol or "BTCUSDT").upper()
    direction = (direction or "long").strip().lower()
    task = (task or "").strip().lower()
    rc = get_redis_client()
    logs: List[str] = []
    running_key = f"celery:train:xgb:grid:{symbol}:{task}"

    def push_log(message: str) -> None:
        ts = datetime.utcnow().strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        logs.append(entry)
        self.update_state(state="PROGRESS", meta={"logs": list(logs), "symbol": symbol, "task": task})

    try:
        if rc.exists(running_key):
            return {"success": False, "error": f"XGB grid for {symbol} ({task}) already running"}
        rc.setex(running_key, 6 * 3600, getattr(getattr(self, "request", None), "id", "1"))
    except Exception:
        pass

    try:
        if not (task.startswith("entry") or task.startswith("exit")):
            return {"success": False, "error": f"Unsupported task for this grid: {task}"}

        limit_candles_final = int(limit_candles_final) if limit_candles_final is not None else 100000
        limit_candles_quick = int(limit_candles_quick) if limit_candles_quick is not None else min(50000, limit_candles_final)

        base_max_hold_steps = int(base_max_hold_steps) if base_max_hold_steps is not None else 48
        base_fee_bps = float(base_fee_bps) if base_fee_bps is not None else 6.0
        base_min_profit = float(base_min_profit) if base_min_profit is not None else 0.0
        base_label_delta = float(base_label_delta) if base_label_delta is not None else 0.0005

        def _uniq_sorted(vals):
            out = []
            for v in vals:
                if v not in out:
                    out.append(v)
            return sorted(out)

        # Build grid values around base (smaller -> base -> bigger)
        hold_steps = _uniq_sorted([max(2, base_max_hold_steps // 2), base_max_hold_steps, base_max_hold_steps * 2])
        fee_bps_list = _uniq_sorted([max(0.0, base_fee_bps * 0.5), base_fee_bps, base_fee_bps * 1.5])
        if base_min_profit > 0:
            min_profit_list = _uniq_sorted([max(0.0, base_min_profit * 0.5), base_min_profit, base_min_profit * 2.0])
        else:
            min_profit_list = [0.0, 0.002, 0.005]
        if task.startswith("exit"):
            if base_label_delta > 0:
                label_delta_list = _uniq_sorted([max(0.0, base_label_delta * 0.5), base_label_delta, base_label_delta * 2.0])
            else:
                label_delta_list = [0.0005, 0.001, 0.002]
        else:
            label_delta_list = [base_label_delta]

        grid_id = f"grid-{str(uuid.uuid4())[:6].lower()}"
        push_log(f"📥 Load 5m candles for grid: final={limit_candles_final}, quick={limit_candles_quick}")
        df_5min_full = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe="5m", limit_candles=limit_candles_final, exchange_id="bybit")
        if df_5min_full is None or df_5min_full.empty:
            return {"success": False, "error": f"No candles for {symbol}"}

        df_5min_quick = df_5min_full.tail(limit_candles_quick).copy()
        dfs_quick = _build_dfs_from_5m(df_5min_quick)
        dfs_final = _build_dfs_from_5m(df_5min_full)

        combos = len(hold_steps) * len(fee_bps_list) * len(min_profit_list) * len(label_delta_list)
        push_log(f"🧪 Quick grid {task}: combos={combos}, n_estimators={quick_n_estimators}")

        results: List[Dict[str, Any]] = []
        best = None
        best_score = -1e9
        best_any = None
        best_any_score = -1e9
        passed = 0
        filtered_out = 0
        trading_filter_enabled = _env_flag("XGB_GRID_TRADING_FILTER", default=False)

        for mh in hold_steps:
            for fb in fee_bps_list:
                for mp in min_profit_list:
                    for ld in label_delta_list:
                        cfg = XgbConfig(direction=direction)
                        cfg.task = task
                        cfg.max_hold_steps = int(mh)
                        cfg.fee_bps = float(fb)
                        cfg.min_profit = float(mp)
                        cfg.label_delta = float(ld)
                        cfg.n_estimators = int(quick_n_estimators)
                        trainer = XgbTrainer(cfg)
                        r = trainer.train(symbol=symbol, dfs=dfs_quick, result_root="result")
                        try:
                            if isinstance(r, dict) and r.get("result_dir"):
                                _safe_patch_manifest(str(r["result_dir"]), {"source": "grid", "grid_id": grid_id})
                        except Exception:
                            pass
                        m = r.get("metrics") if isinstance(r.get("metrics"), dict) else {}
                        f1v = m.get("f1_val")
                        score = None
                        try:
                            if isinstance(f1v, list) and len(f1v) > 1:
                                score = float(f1v[1])
                        except Exception:
                            score = None
                        if score is None:
                            score = float(m.get("val_acc", 0.0))
                        # Track best-by-metric for fallback
                        if float(score) > best_any_score:
                            best_any_score = float(score)
                            best_any = {
                                "task": task,
                                "direction": direction,
                                "max_hold_steps": mh,
                                "fee_bps": fb,
                                "min_profit": mp,
                                "label_delta": ld,
                                "score": float(score),
                                "metrics": m,
                                "run_name": r.get("run_name"),
                                "result_dir": r.get("result_dir"),
                            }

                        ok, info = _passes_trading_filters_entry_exit(m)
                        if trading_filter_enabled and info.get("enabled") and not ok:
                            filtered_out += 1
                            item = {
                                "task": task,
                                "direction": direction,
                                "max_hold_steps": mh,
                                "fee_bps": fb,
                                "min_profit": mp,
                                "label_delta": ld,
                                "score": float(score),
                                "metrics": m,
                                "run_name": r.get("run_name"),
                                "result_dir": r.get("result_dir"),
                                "filtered_out": True,
                                "filter_info": info,
                            }
                            results.append(item)
                            push_log(f"grid: mh={mh} fee={fb:.1f} mp={mp:.4f} ld={ld:.4f} filtered ({info.get('reason','?')})")
                            continue
                        if trading_filter_enabled and info.get("enabled") and ok:
                            passed += 1
                        item = {
                            "task": task,
                            "direction": direction,
                            "max_hold_steps": mh,
                            "fee_bps": fb,
                            "min_profit": mp,
                            "label_delta": ld,
                            "score": float(score),
                            "metrics": m,
                            "run_name": r.get("run_name"),
                            "result_dir": r.get("result_dir"),
                        }
                        results.append(item)
                        if float(score) > best_score:
                            best_score = float(score)
                            best = item
                        push_log(f"grid: mh={mh} fee={fb:.1f} mp={mp:.4f} ld={ld:.4f} score={float(score):.4f}")

        if trading_filter_enabled:
            if passed == 0 and filtered_out > 0 and best_any is not None:
                best = best_any
                best_score = best_any_score
                push_log("⚠️ Trading filter enabled but no candidates passed; fallback to best-by-metric")
            else:
                push_log(f"🎯 Trading filter enabled: passed={passed} filtered_out={filtered_out}")

        if not best and best_any is not None:
            best = best_any
            best_score = best_any_score

        if not best:
            return {"success": False, "error": "Grid produced no results"}

        push_log(f"🏁 Best quick {task}: score={best_score:.4f} mh={best['max_hold_steps']} fee={best['fee_bps']} mp={best['min_profit']} ld={best['label_delta']}")

        cfg_final = XgbConfig(direction=direction)
        cfg_final.task = task
        cfg_final.max_hold_steps = int(best["max_hold_steps"])
        cfg_final.fee_bps = float(best["fee_bps"])
        cfg_final.min_profit = float(best["min_profit"])
        cfg_final.label_delta = float(best["label_delta"])
        cfg_final.n_estimators = int(final_n_estimators)
        trainer_final = XgbTrainer(cfg_final)
        final_res = trainer_final.train(symbol=symbol, dfs=dfs_final, result_root="result")

        # Cleanup quick grid runs: keep only grid_final by default.
        keep_all = str(os.environ.get("XGB_GRID_KEEP_ALL", "0")).strip().lower() in ("1", "true", "yes", "on")
        keep_best = str(os.environ.get("XGB_GRID_KEEP_BEST", "0")).strip().lower() in ("1", "true", "yes", "on")
        if not keep_all:
            keep_dirs = {str(final_res.get("result_dir") or "").strip()}
            if keep_best and best:
                keep_dirs.add(str(best.get("result_dir") or "").strip())
            deleted = 0
            for it in results:
                rd = str(it.get("result_dir") or "").strip()
                if not rd or rd in keep_dirs:
                    continue
                if _safe_rmtree(rd):
                    deleted += 1
            if deleted:
                msg = f"🧹 Cleanup: deleted {deleted} quick grid runs"
                if keep_best:
                    msg += " (kept best quick via XGB_GRID_KEEP_BEST=1)"
                push_log(msg)

        sym_code = (symbol or "").upper().replace("USDT", "").replace("USD", "").replace("USDC", "").replace("BUSD", "").replace("USDP", "")
        grid_dir = os.path.join("result", "xgb", sym_code or "UNKNOWN", "grids", grid_id)
        os.makedirs(grid_dir, exist_ok=True)
        _atomic_write_json(os.path.join(grid_dir, "grid_results.json"), {"symbol": symbol, "task": task, "results": results, "best": best})
        _atomic_write_json(os.path.join(grid_dir, "final_result.json"), final_res)
        try:
            if isinstance(final_res, dict) and final_res.get("result_dir"):
                _safe_patch_manifest(str(final_res["result_dir"]), {"source": "grid_final", "grid_id": grid_id})
        except Exception:
            pass

        out = {
            "success": True,
            "symbol": symbol,
            "task": task,
            "grid_id": grid_id,
            "best": best,
            "final": final_res,
            "grid_dir": grid_dir,
        }
        self.update_state(state="SUCCESS", meta={"logs": list(logs), "symbol": symbol, "task": task, "result": out})
        return out
    finally:
        try:
            rc.delete(running_key)
        except Exception:
            pass


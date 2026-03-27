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
from utils.db_utils import db_get_or_fetch_ohlcv, load_latest_candles_from_csv_to_db, _discover_bybit_api_keys, _discover_all_bybit_api_keys, _key_prefix_4
from utils.parser import parser_download_and_combine_with_library
from utils.redis_utils import get_redis_client
from utils.settings_store import get_setting_value as _get_setting_value


def _api_hint(symbol: str) -> str:
    """Короткая строка: какой Bybit API ключ используется для символа."""
    try:
        ak, _sk, var = _discover_bybit_api_keys(symbol)
        prefix = _key_prefix_4(ak) if ak else None
        label = var or 'none'
        if prefix:
            return f"[API: {label} ({prefix}…)]"
        return f"[API: {label} (no key)]"
    except Exception:
        return "[API: unknown]"


def _fetch_ohlcv_with_fallback(symbol: str, fetch_limit: int, push_log, max_attempts: int = 3, pause_sec: int = 60):
    """Fetch OHLCV with retry + fallback to other API keys on rate limit."""
    all_keys = _discover_all_bybit_api_keys()
    primary_ak, _, primary_var = _discover_bybit_api_keys(symbol)
    primary_prefix = _key_prefix_4(primary_ak) if primary_ak else None

    keys_order = []
    for ak, sk, var in all_keys:
        if _key_prefix_4(ak) == primary_prefix:
            keys_order.insert(0, (ak, sk, var))
        else:
            keys_order.append((ak, sk, var))
    if not keys_order:
        keys_order = [(None, None, 'none')]

    last_err = None
    for attempt in range(max_attempts):
        key_idx = min(attempt, len(keys_order) - 1)
        ak, sk, var = keys_order[key_idx]
        hint = f"[API: {var} ({_key_prefix_4(ak)}…)]" if ak else "[API: no key]"

        try:
            df, err = db_get_or_fetch_ohlcv(
                symbol_name=symbol, timeframe="5m",
                limit_candles=fetch_limit, exchange_id="bybit",
                include_error=True,
                override_api_key=ak, override_secret_key=sk,
            )
            last_err = err
        except Exception as _e:
            last_err = str(_e)
            df = None
            push_log(f"⚠ Attempt {attempt+1}/{max_attempts} failed: {_e} {hint}")

        if df is not None and not df.empty:
            return df, None

        if attempt < max_attempts - 1:
            next_key_idx = min(attempt + 1, len(keys_order) - 1)
            next_var = keys_order[next_key_idx][2]
            next_hint = f" → fallback: {next_var}" if next_key_idx != key_idx else ""
            push_log(f"⚠ No candles (attempt {attempt+1}/{max_attempts}), reason={last_err} {hint}{next_hint}, retrying in {pause_sec}s…")
            time.sleep(pause_sec)

    return None, last_err

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


def _setting_int(scope: str, group: str | None, key: str, default: int) -> int:
    try:
        raw = _get_setting_value(scope, group, key)
        if raw is None:
            return int(default)
        return int(float(str(raw).strip()))
    except Exception:
        return int(default)


def _setting_bool(scope: str, group: str | None, key: str, default: bool) -> bool:
    try:
        raw = _get_setting_value(scope, group, key)
        if raw is None:
            return bool(default)
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        return bool(default)


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


_BARS_PER_DAY_5M = 288  # 24*60/5


def _apply_cutoff_and_tail(df_5min: pd.DataFrame, train_window: int, cutoff_days: int | None) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Keeps ONLY candles <= (max_ts - cutoff_days) and then takes last train_window candles.
    Returns (df_train, info).
    """
    if train_window <= 0:
        raise ValueError("train_window must be > 0")
    cd = int(cutoff_days or 0)
    if cd <= 0:
        out = df_5min.tail(int(train_window)).copy()
        return out, {"cutoff_days": 0, "cutoff_ts_ms": None, "cutoff_dt": None, "train_window": int(train_window)}
    if "timestamp" not in df_5min.columns:
        raise ValueError("df_5min has no timestamp column for cutoff")
    try:
        max_ts = int(float(df_5min["timestamp"].max()))
    except Exception as e:
        raise ValueError(f"cannot compute max timestamp: {e}")
    cutoff_ts = int(max_ts - cd * 24 * 3600 * 1000)
    df_cut = df_5min[df_5min["timestamp"] <= cutoff_ts].copy()
    if len(df_cut) < int(train_window):
        raise ValueError(f"not enough candles for cutoff_days={cd}: have={len(df_cut)} need={int(train_window)}")
    out = df_cut.tail(int(train_window)).copy()
    try:
        cutoff_dt = pd.to_datetime(cutoff_ts, unit="ms").isoformat()
    except Exception:
        cutoff_dt = None
    return out, {"cutoff_days": cd, "cutoff_ts_ms": cutoff_ts, "cutoff_dt": cutoff_dt, "train_window": int(train_window)}


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
    cutoff_days: int | None = None,
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
        push_log(f"🚀 XGB train start: {symbol} direction={direction} {_api_hint(symbol)}")

        # Load 5m data
        push_log("📥 Load 5m candles from DB...")
        train_window = int(limit_candles) if (limit_candles is not None) else 100000
        cd = int(cutoff_days or 0)
        fetch_limit = int(train_window + max(0, cd) * _BARS_PER_DAY_5M + 2000)
        df_5min = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe="5m", limit_candles=fetch_limit, exchange_id="bybit")
        if df_5min is None or df_5min.empty:
            push_log("📥 No data in DB, trying to download and load...")
            try:
                csv_path = parser_download_and_combine_with_library(symbol=symbol, interval="5m", limit=fetch_limit)
                if csv_path:
                    loaded = load_latest_candles_from_csv_to_db(file_path=csv_path, symbol_name=symbol, timeframe="5m")
                    push_log(f"✅ Loaded to DB: {loaded} candles")
                df_5min = db_get_or_fetch_ohlcv(symbol_name=symbol, timeframe="5m", limit_candles=fetch_limit, exchange_id="bybit")
            except Exception as e:
                push_log(f"❌ Auto-download failed: {e}")
                df_5min = None

        if df_5min is None or df_5min.empty:
            return {"success": False, "error": f"No candles for {symbol} {_api_hint(symbol)}"}

        df_5min, cutoff_info = _apply_cutoff_and_tail(df_5min, train_window=train_window, cutoff_days=cd)
        push_log(f"🗓 cutoff_days={cutoff_info.get('cutoff_days')} cutoff_dt={cutoff_info.get('cutoff_dt')} train_window={train_window} fetched={fetch_limit}")

        # Build 15m and 1h from 5m (matches existing DQN task behavior)
        dfs = _build_dfs_from_5m(df_5min)

        cfg = XgbConfig(direction=direction)
        # Disable ATR feature for NEW training runs (keeps shape stable; ATR column is zeroed).
        cfg.use_atr_feature = False
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
    cutoff_days: int | None = None,
) -> Dict[str, Any]:
    """
    Быстрый grid (horizon/threshold/direction + max_hold_steps/min_profit) на ограниченном окне и малом числе деревьев,
    затем финальный train лучшего конфига на полном окне.
    """
    symbol = (symbol or "BTCUSDT").upper()
    rc = get_redis_client()
    logs: List[str] = []
    running_key = f"celery:train:xgb:grid:{symbol}"
    started_at = time.time()

    def push_log(message: str) -> None:
        ts = datetime.utcnow().strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        logs.append(entry)
        self.update_state(state="PROGRESS", meta={"logs": list(logs), "symbol": symbol, "started_at": started_at})

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

        cd = int(cutoff_days or 0)
        fetch_limit = int(limit_candles_final + max(0, cd) * _BARS_PER_DAY_5M + 2000)
        push_log(f"📥 Load 5m candles for grid: final={limit_candles_final}, quick={limit_candles_quick}, cutoff_days={cd}, fetch={fetch_limit} {_api_hint(symbol)}")
        df_5min_full, _fetch_err = _fetch_ohlcv_with_fallback(symbol, fetch_limit, push_log)
        if df_5min_full is None or df_5min_full.empty:
            return {"success": False, "error": f"No candles for {symbol} after 3 attempts. reason={_fetch_err} {_api_hint(symbol)}"}

        df_5min_full, cutoff_info = _apply_cutoff_and_tail(df_5min_full, train_window=int(limit_candles_final), cutoff_days=cd)
        push_log(f"🗓 cutoff_dt={cutoff_info.get('cutoff_dt')} train_window={limit_candles_final}")

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
        cfg_final.use_atr_feature = False
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
    cutoff_days: int | None = None,
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
    started_at = time.time()

    def push_log(message: str) -> None:
        ts = datetime.utcnow().strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        logs.append(entry)
        self.update_state(state="PROGRESS", meta={"logs": list(logs), "symbol": symbol, "task": task, "started_at": started_at})

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
        grid_id = f"grid-{str(uuid.uuid4())[:6].lower()}"
        cd = int(cutoff_days or 0)
        fetch_limit = int(limit_candles_final + max(0, cd) * _BARS_PER_DAY_5M + 2000)
        push_log(f"📥 Load 5m candles for grid: final={limit_candles_final}, quick={limit_candles_quick}, cutoff_days={cd}, fetch={fetch_limit} {_api_hint(symbol)}")
        df_5min_full, _fetch_err = _fetch_ohlcv_with_fallback(symbol, fetch_limit, push_log)
        if df_5min_full is None or df_5min_full.empty:
            return {"success": False, "error": f"No candles for {symbol} after 3 attempts. reason={_fetch_err} {_api_hint(symbol)}"}

        df_5min_full, cutoff_info = _apply_cutoff_and_tail(df_5min_full, train_window=int(limit_candles_final), cutoff_days=cd)
        push_log(f"🗓 cutoff_dt={cutoff_info.get('cutoff_dt')} train_window={limit_candles_final}")

        df_5min_quick = df_5min_full.tail(limit_candles_quick).copy()
        dfs_quick = _build_dfs_from_5m(df_5min_quick)
        dfs_final = _build_dfs_from_5m(df_5min_full)

        combos = len(hold_steps) * len(fee_bps_list) * len(min_profit_list)
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
                        cfg = XgbConfig(direction=direction)
                        cfg.use_atr_feature = False
                        cfg.task = task
                        cfg.max_hold_steps = int(mh)
                        cfg.fee_bps = float(fb)
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
                                "score": float(score),
                                "metrics": m,
                                "run_name": r.get("run_name"),
                                "result_dir": r.get("result_dir"),
                                "filtered_out": True,
                                "filter_info": info,
                            }
                            results.append(item)
                            push_log(f"grid: mh={mh} fee={fb:.1f} mp={mp:.4f} filtered ({info.get('reason','?')})")
                            continue
                        if trading_filter_enabled and info.get("enabled") and ok:
                            passed += 1
                        item = {
                            "task": task,
                            "direction": direction,
                            "max_hold_steps": mh,
                            "fee_bps": fb,
                            "min_profit": mp,
                            "score": float(score),
                            "metrics": m,
                            "run_name": r.get("run_name"),
                            "result_dir": r.get("result_dir"),
                        }
                        results.append(item)
                        if float(score) > best_score:
                            best_score = float(score)
                            best = item
                        push_log(f"grid: mh={mh} fee={fb:.1f} mp={mp:.4f} score={float(score):.4f}")

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

        push_log(f"🏁 Best quick {task}: score={best_score:.4f} mh={best['max_hold_steps']} fee={best['fee_bps']} mp={best['min_profit']}")

        cfg_final = XgbConfig(direction=direction)
        cfg_final.use_atr_feature = False
        cfg_final.task = task
        cfg_final.max_hold_steps = int(best["max_hold_steps"])
        cfg_final.fee_bps = float(best["fee_bps"])
        cfg_final.min_profit = float(best["min_profit"])
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


# ---------------------------------------------------------------------------
# Full hyper-parameter grid: labeling + model params
# ---------------------------------------------------------------------------

def _parse_list(raw, cast=float) -> list:
    """Parse comma-separated string or list into typed list."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [cast(v) for v in raw]
    if isinstance(raw, str):
        return [cast(v.strip()) for v in raw.split(",") if v.strip()]
    return [cast(raw)]


def _parse_opt_float_list(raw) -> list[float | None]:
    """
    Parse list/csv into list[float|None].
    Supported null tokens: '', 'none', 'null'.
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    elif isinstance(raw, str):
        items = [v.strip() for v in raw.split(",")]
    else:
        items = [raw]
    out: list[float | None] = []
    for v in items:
        if v is None:
            out.append(None)
            continue
        s = str(v).strip()
        if s == "" or s.lower() in ("none", "null"):
            out.append(None)
            continue
        try:
            out.append(float(s))
        except Exception:
            # ignore junk token
            continue
    return out


@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={"max_retries": 0}, queue="celery")
def train_xgb_grid_full(
    self,
    symbol: str,
    direction: str = "long",
    task: str = "entry_long",
    limit_candles: int | None = None,
    cutoff_days: int | None = None,
    # entry_* policy (inference threshold + simple exits for entry backtest/OOS)
    p_enter_threshold_list: list | str | None = None,
    entry_tp_pct: float | None = None,
    entry_sl_pct: float | None = None,
    entry_trail_pct: float | None = None,
    # optional: grid over exit-policy for entry_* (values are FRACTIONS, e.g. 0.012 = 1.2%)
    entry_tp_pct_list: list | str | None = None,
    entry_sl_pct_list: list | str | None = None,
    entry_trail_pct_list: list | str | None = None,
    # labeling ranges (lists)
    horizon_steps_list: list | str | None = None,
    threshold_list: list | str | None = None,
    max_hold_steps_list: list | str | None = None,
    min_profit_list: list | str | None = None,
    fee_bps_list: list | str | None = None,
    # model hyper-param ranges (lists)
    max_depth_list: list | str | None = None,
    learning_rate_list: list | str | None = None,
    n_estimators_list: list | str | None = None,
    subsample_list: list | str | None = None,
    colsample_bytree_list: list | str | None = None,
    reg_lambda_list: list | str | None = None,
    min_child_weight_list: list | str | None = None,
    gamma_list: list | str | None = None,
    scale_pos_weight_list: list | str | None = None,
    early_stopping_rounds: int = 50,
    keep_top_n: int = 20,
) -> Dict[str, Any]:
    """Full grid over BOTH labeling and model hyper-params. Saves top-N runs."""
    import itertools
    import numpy as np  # noqa: F811

    symbol = (symbol or "BTCUSDT").upper()
    direction = (direction or "long").strip().lower()
    task_name = (task or "entry_long").strip().lower()
    rc = get_redis_client()
    logs: List[str] = []
    running_key = f"celery:train:xgb:gridfull:{symbol}:{task_name}"
    # progress counters (also exposed to UI via task meta)
    total = 0
    done = 0
    topn = max(1, int(keep_top_n or 1))
    started_at = time.time()

    def push_log(msg: str) -> None:
        ts = datetime.utcnow().strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        logs.append(entry)
        self.update_state(
            state="PROGRESS",
            meta={
                "logs": list(logs),
                "symbol": symbol,
                "task": task_name,
                "done": int(done),
                "total": int(total),
                "keep_top_n": int(topn),
                "started_at": started_at,
            },
        )

    try:
        if rc.exists(running_key):
            return {"success": False, "error": f"XGB grid-full for {symbol}/{task_name} already running"}
        rc.setex(running_key, 12 * 3600, getattr(getattr(self, "request", None), "id", "1"))
    except Exception:
        pass

    try:
        # --- defaults ---
        is_binary = task_name.startswith("entry") or task_name.startswith("exit")

        hs_list = _parse_list(horizon_steps_list, int) or [12]
        thr_list = _parse_list(threshold_list, float) or [0.002]
        mh_list = _parse_list(max_hold_steps_list, int) or [48]
        mp_list = _parse_list(min_profit_list, float) or [0.0]
        fb_list = _parse_list(fee_bps_list, float) or [6.0]

        md_list = _parse_list(max_depth_list, int) or [6]
        lr_list = _parse_list(learning_rate_list, float) or [0.05]
        ne_list = _parse_list(n_estimators_list, int) or [600]
        ss_list = _parse_list(subsample_list, float) or [0.9]
        cb_list = _parse_list(colsample_bytree_list, float) or [0.9]
        rl_list = _parse_list(reg_lambda_list, float) or [1.0]
        mcw_list = _parse_list(min_child_weight_list, float) or [1.0]
        gm_list = _parse_list(gamma_list, float) or [0.0]
        spw_list = _parse_list(scale_pos_weight_list, float) or [1.0]
        pe_list = _parse_list(p_enter_threshold_list, float) or [0.5]

        # Exit-policy lists (used ONLY for entry_* tasks).
        tp_list = _parse_opt_float_list(entry_tp_pct_list)
        sl_list = _parse_opt_float_list(entry_sl_pct_list)
        tr_list = _parse_opt_float_list(entry_trail_pct_list)
        # Backward compatible: if lists are not provided, take single values.
        if not tp_list and entry_tp_pct is not None:
            tp_list = [float(entry_tp_pct)]
        if not sl_list and entry_sl_pct is not None:
            sl_list = [float(entry_sl_pct)]
        if not tr_list and entry_trail_pct is not None:
            tr_list = [float(entry_trail_pct)]
        # If still empty, interpret as disabled (None)
        if not tp_list:
            tp_list = [None]
        if not sl_list:
            sl_list = [None]
        if not tr_list:
            tr_list = [None]

        # If task is directional, labeling grid uses horizon/threshold; entry/exit uses max_hold/min_profit/fee/delta.
        if task_name == "directional":
            label_combos = list(itertools.product(hs_list, thr_list))
            label_keys = ("horizon_steps", "threshold")
        else:
            label_combos = list(itertools.product(mh_list, mp_list, fb_list))
            label_keys = ("max_hold_steps", "min_profit", "fee_bps")

        # For binary tasks, include p_enter_threshold in the grid (affects val metrics + OOS behavior).
        # For entry_* tasks, also include exit-policy grid (tp/sl/trail) so labels and proxy_pnl match it.
        if is_binary and task_name.startswith("entry"):
            model_combos = list(
                itertools.product(
                    md_list, lr_list, ne_list, ss_list, cb_list, rl_list, mcw_list, gm_list, spw_list, pe_list,
                    tp_list, sl_list, tr_list,
                )
            )
            model_keys = (
                "max_depth", "learning_rate", "n_estimators", "subsample", "colsample_bytree",
                "reg_lambda", "min_child_weight", "gamma", "scale_pos_weight", "p_enter_threshold",
                "entry_tp_pct", "entry_sl_pct", "entry_trail_pct",
            )
        elif is_binary:
            model_combos = list(itertools.product(md_list, lr_list, ne_list, ss_list, cb_list, rl_list, mcw_list, gm_list, spw_list, pe_list))
            model_keys = ("max_depth", "learning_rate", "n_estimators", "subsample", "colsample_bytree", "reg_lambda", "min_child_weight", "gamma", "scale_pos_weight", "p_enter_threshold")
        else:
            model_combos = list(itertools.product(md_list, lr_list, ne_list, ss_list, cb_list, rl_list, mcw_list, gm_list, spw_list))
            model_keys = ("max_depth", "learning_rate", "n_estimators", "subsample", "colsample_bytree", "reg_lambda", "min_child_weight", "gamma", "scale_pos_weight")

        total = len(label_combos) * len(model_combos)
        push_log(f"📊 Grid-full: label_combos={len(label_combos)} × model_combos={len(model_combos)} = {total} runs {_api_hint(symbol)}")

        limit_candles = int(limit_candles) if limit_candles is not None else 100000
        cd = int(cutoff_days or 0)
        fetch_limit = int(limit_candles + max(0, cd) * _BARS_PER_DAY_5M + 2000)
        push_log(f"📥 Loading {limit_candles} 5m candles… cutoff_days={cd} fetch={fetch_limit} {_api_hint(symbol)}")
        df_5min, _fetch_err = _fetch_ohlcv_with_fallback(symbol, fetch_limit, push_log)
        if df_5min is None or df_5min.empty:
            return {"success": False, "error": f"No candles for {symbol} after 3 attempts. reason={_fetch_err} {_api_hint(symbol)}"}
        df_5min, cutoff_info = _apply_cutoff_and_tail(df_5min, train_window=int(limit_candles), cutoff_days=cd)
        push_log(f"🗓 cutoff_dt={cutoff_info.get('cutoff_dt')} train_window={limit_candles}")
        dfs = _build_dfs_from_5m(df_5min)

        grid_id = f"gridfull-{str(uuid.uuid4())[:6].lower()}"
        results: List[Dict[str, Any]] = []
        rank_by_proxy = bool(
            is_binary and _setting_bool(
                "xgb",
                "grid_full",
                "XGB_GRID_FULL_RANK_BY_PROXY_PNL",
                _env_flag("XGB_GRID_FULL_RANK_BY_PROXY_PNL", default=True),
            )
        )
        proxy_trades_min = max(
            0,
            int(
                _setting_int(
                    "xgb",
                    "grid_full",
                    "XGB_GRID_FULL_PROXY_TRADES_MIN",
                    _env_int("XGB_GRID_FULL_PROXY_TRADES_MIN", 1),
                )
            ),
        )
        proxy_trades_max = max(
            proxy_trades_min,
            int(
                _setting_int(
                    "xgb",
                    "grid_full",
                    "XGB_GRID_FULL_PROXY_TRADES_MAX",
                    _env_int("XGB_GRID_FULL_PROXY_TRADES_MAX", 10**9),
                )
            ),
        )

        def _grid_full_sort_key(item: Dict[str, Any]):
            """
            For binary tasks prefer runs with valid proxy_pnl and trades bounds:
            1) in-bounds proxy candidate
            2) higher proxy pnl_sum
            3) fallback to classification score / f1(1)
            """
            m = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
            try:
                score = float(item.get("score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            f1v = m.get("f1_val")
            try:
                f1_1 = float(f1v[1]) if isinstance(f1v, list) and len(f1v) > 1 else 0.0
            except Exception:
                f1_1 = 0.0
            if not rank_by_proxy:
                return (score, f1_1)

            proxy = m.get("proxy_pnl_val") if isinstance(m.get("proxy_pnl_val"), dict) else {}
            try:
                enabled = bool(proxy.get("enabled"))
            except Exception:
                enabled = False
            try:
                trades = int(proxy.get("trades", 0) or 0)
            except Exception:
                trades = 0
            try:
                pnl_sum = float(proxy.get("pnl_sum", 0.0) or 0.0)
            except Exception:
                pnl_sum = 0.0
            in_bounds = bool(enabled and (proxy_trades_min <= trades <= proxy_trades_max))
            # reverse-sort tuple
            return (1 if in_bounds else 0, pnl_sum if in_bounds else float("-inf"), score, f1_1)

        def _periodic_cleanup(force: bool = False) -> None:
            """
            Prevents unbounded growth of result/xgb/*/runs during big grids.
            Strategy: when accumulated successful runs reach 2×keep_top_n,
            delete everything except current top-N by score and continue.
            """
            nonlocal results
            try:
                if topn <= 0:
                    return
                trigger = max(2, int(topn) * 2)
                if (not force) and (len(results) < trigger):
                    return
                results.sort(key=_grid_full_sort_key, reverse=True)
                keep = results[:topn]
                keep_dirs = set()
                for it in keep:
                    rd = str(it.get("result_dir") or "").strip()
                    if rd:
                        keep_dirs.add(rd)
                deleted = 0
                for it in results[topn:]:
                    rd = str(it.get("result_dir") or "").strip()
                    if not rd or rd in keep_dirs:
                        continue
                    if _safe_rmtree(rd):
                        deleted += 1
                # trim in-memory list too (so next trigger happens after another +topn runs)
                results = keep
                if deleted:
                    push_log(f"🧹 Cleanup (periodic): kept top-{topn}, deleted {deleted} runs")
            except Exception as exc:
                push_log(f"⚠ Cleanup error: {exc}")

        for lc in label_combos:
            lc_dict = dict(zip(label_keys, lc))
            for mc in model_combos:
                mc_dict = dict(zip(model_keys, mc))
                cfg = XgbConfig(direction=direction)
                cfg.use_atr_feature = False
                cfg.task = task_name
                cfg.early_stopping_rounds = int(early_stopping_rounds)
                cfg.n_jobs = 2  # lower for grid parallelism
                # labeling params
                if task_name == "directional":
                    cfg.horizon_steps = int(lc_dict["horizon_steps"])
                    cfg.threshold = float(lc_dict["threshold"])
                else:
                    cfg.max_hold_steps = int(lc_dict["max_hold_steps"])
                    cfg.min_profit = float(lc_dict["min_profit"])
                    cfg.fee_bps = float(lc_dict["fee_bps"])
                # model params
                cfg.max_depth = int(mc_dict["max_depth"])
                cfg.learning_rate = float(mc_dict["learning_rate"])
                cfg.n_estimators = int(mc_dict["n_estimators"])
                cfg.subsample = float(mc_dict["subsample"])
                cfg.colsample_bytree = float(mc_dict["colsample_bytree"])
                cfg.reg_lambda = float(mc_dict["reg_lambda"])
                cfg.min_child_weight = float(mc_dict["min_child_weight"])
                cfg.gamma = float(mc_dict["gamma"])
                cfg.scale_pos_weight = float(mc_dict["scale_pos_weight"])
                if is_binary and "p_enter_threshold" in mc_dict:
                    cfg.p_enter_threshold = float(mc_dict["p_enter_threshold"])

                # Exit policy snapshot for entry_* (used by labeling + proxy_pnl + OOS backtest for entry tasks)
                if task_name.startswith("entry"):
                    try:
                        cfg.entry_tp_pct = float(mc_dict.get("entry_tp_pct")) if mc_dict.get("entry_tp_pct") is not None else None
                    except Exception:
                        cfg.entry_tp_pct = None
                    try:
                        cfg.entry_sl_pct = float(mc_dict.get("entry_sl_pct")) if mc_dict.get("entry_sl_pct") is not None else None
                    except Exception:
                        cfg.entry_sl_pct = None
                    try:
                        cfg.entry_trail_pct = float(mc_dict.get("entry_trail_pct")) if mc_dict.get("entry_trail_pct") is not None else None
                    except Exception:
                        cfg.entry_trail_pct = None
                else:
                    # Non-entry tasks: keep as provided (single values only)
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

                # Normalize signs (same as OOS route behavior):
                # - tp should be positive
                # - sl should be negative
                # - trail should be positive
                try:
                    if cfg.entry_tp_pct is not None and float(cfg.entry_tp_pct) < 0:
                        cfg.entry_tp_pct = abs(float(cfg.entry_tp_pct))
                except Exception:
                    pass
                try:
                    if cfg.entry_sl_pct is not None and float(cfg.entry_sl_pct) > 0:
                        cfg.entry_sl_pct = -abs(float(cfg.entry_sl_pct))
                except Exception:
                    pass
                try:
                    if cfg.entry_trail_pct is not None and float(cfg.entry_trail_pct) < 0:
                        cfg.entry_trail_pct = abs(float(cfg.entry_trail_pct))
                except Exception:
                    pass

                try:
                    trainer = XgbTrainer(cfg)
                    r = trainer.train(symbol=symbol, dfs=dfs, result_root="result")
                    try:
                        if isinstance(r, dict) and r.get("result_dir"):
                            _safe_patch_manifest(str(r["result_dir"]), {"source": "grid_full", "grid_id": grid_id})
                    except Exception:
                        pass
                    m = r.get("metrics") if isinstance(r.get("metrics"), dict) else {}
                    # Score: f1(1) for binary, f1_buy_sell for directional
                    if is_binary:
                        f1v = m.get("f1_val")
                        score = float(f1v[1]) if isinstance(f1v, list) and len(f1v) > 1 else float(m.get("val_acc", 0.0))
                    else:
                        score = float(m.get("f1_buy_sell_val") if m.get("f1_buy_sell_val") is not None else m.get("val_acc", 0.0))
                    item = {**lc_dict, **mc_dict,
                            "entry_tp_pct": cfg.entry_tp_pct,
                            "entry_sl_pct": cfg.entry_sl_pct,
                            "entry_trail_pct": cfg.entry_trail_pct,
                            "score": score, "metrics": m,
                            "run_name": r.get("run_name"), "result_dir": r.get("result_dir")}
                    if is_binary:
                        pp = m.get("proxy_pnl_val") if isinstance(m.get("proxy_pnl_val"), dict) else {}
                        try:
                            item["proxy_pnl_sum"] = float(pp.get("pnl_sum", 0.0) or 0.0)
                        except Exception:
                            item["proxy_pnl_sum"] = 0.0
                        try:
                            item["proxy_trades"] = int(pp.get("trades", 0) or 0)
                        except Exception:
                            item["proxy_trades"] = 0
                    results.append(item)
                except Exception as exc:
                    push_log(f"⚠ Error: {exc}")

                done += 1
                if done % 10 == 0 or done == total:
                    push_log(f"⏳ {done}/{total} done")
                # periodic cleanup for long grids (delete early, not only at the end)
                if topn > 0 and len(results) >= max(2, topn * 2):
                    _periodic_cleanup()

        # Final cleanup + sort (ensure only top-N remains even if grid ended early)
        _periodic_cleanup(force=True)
        results.sort(key=_grid_full_sort_key, reverse=True)

        # Save grid results
        sym_code = (symbol or "").upper().replace("USDT", "").replace("USD", "").replace("USDC", "").replace("BUSD", "").replace("USDP", "")
        grid_dir = os.path.join("result", "xgb", sym_code or "UNKNOWN", "grids", grid_id)
        os.makedirs(grid_dir, exist_ok=True)
        _atomic_write_json(os.path.join(grid_dir, "grid_results.json"), {
            "symbol": symbol, "task": task_name, "direction": direction,
            "total_runs": total, "results": results[:topn],
            "best": results[0] if results else None,
        })

        best = results[0] if results else None
        push_log(f"🏁 Best score={best['score']:.4f}" if best else "🏁 No results")

        out = {"success": True, "symbol": symbol, "task": task_name, "grid_id": grid_id,
               "total_runs": total, "best": best, "grid_dir": grid_dir}
        self.update_state(
            state="SUCCESS",
            meta={
                "logs": list(logs),
                "symbol": symbol,
                "task": task_name,
                "done": int(total),
                "total": int(total),
                "keep_top_n": int(topn),
                "result": out,
            },
        )
        return out
    finally:
        try:
            rc.delete(running_key)
        except Exception:
            pass


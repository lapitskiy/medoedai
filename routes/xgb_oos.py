from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

from celery.result import AsyncResult  # type: ignore
from flask import Blueprint, jsonify, request, send_file, abort, render_template  # type: ignore

from tasks import celery  # type: ignore
from tasks.xgb_oos_tasks import _missing_1m_bucket, _oos_1m_fetch_limit, _oos_5m_closed_tail, _oos_end_lag_hours, _oos_end_ts_ms
from utils.db_utils import db_get_ohlcv_only
from utils.redis_utils import get_redis_client
from utils.xgb_oos_batch_csv import (
    get_batch_results,
    get_batch_status,
    force_finalize_batch_csv,
    new_batch_id,
    register_batch,
    write_oos_batch_csv,
)

xgb_oos_bp = Blueprint("xgb_oos", __name__)
logger = logging.getLogger(__name__)

_CSV_DIR = Path("predict_test") / "xgb_oos"
_WF_DIR = Path("result") / "wf" / "xgb"
_XGB_HYPO_DIR = Path("predict_test") / "xgb_hypo"
_ACTIVE_XGB_PRESET = _XGB_HYPO_DIR / "xgb_grid_full_preset.json"
_EXPERIMENTS_DIR = _XGB_HYPO_DIR / "experiments"
_OOS_PREFETCH_WAIT_SEC = int(os.getenv("OOS_PREFETCH_WAIT_SEC", "75") or "75")


def _now_utc_msk_str() -> Tuple[str, str]:
    now_utc = datetime.utcnow()
    now_msk = now_utc + timedelta(hours=3)
    return now_utc.strftime("%Y-%m-%d %H:%M:%S"), now_msk.strftime("%Y-%m-%d %H:%M:%S")


def _copy_xgb_run_to_prod(run_dir: Path, ensemble: str) -> Dict[str, Any]:
    symbol_dir = str(run_dir.parent.parent.name or "").strip()
    run_id = str(run_dir.name or "").strip()
    if not symbol_dir or not run_id:
        raise ValueError("bad run_dir structure")

    run_manifest = _safe_read_json(run_dir / "manifest.json")
    models_root = Path("models") / "xgb" / symbol_dir.lower()
    ensemble_dir = models_root / ensemble
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    existing: List[int] = []
    for p in ensemble_dir.iterdir():
        if p.is_dir() and p.name.startswith("v"):
            try:
                existing.append(int(p.name[1:]))
            except ValueError:
                pass
    next_num = (max(existing) + 1) if existing else 1
    version_name = f"v{next_num}"
    version_dir = ensemble_dir / version_name
    version_dir.mkdir()

    copied_files: List[str] = []
    for f in run_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, version_dir / f.name)
            copied_files.append(f.name)

    yaml_lines = [
        f'id: "{run_id}"',
        f'symbol: "{symbol_dir}"',
        f'ensemble: "{ensemble}"',
        f'version: "{version_name}"',
        f'created_at: "{datetime.utcnow().isoformat()}"',
        f'run_id: "{run_id}"',
        'model_type: "xgb"',
    ]
    for key in ("direction", "task"):
        val = run_manifest.get(key)
        if val:
            yaml_lines.append(f'{key}: "{val}"')
    yaml_lines.append(f'source_run_path: "{run_dir.as_posix()}"')
    yaml_lines.append("files:")
    for fn in copied_files:
        yaml_lines.append(f'  - "{fn}"')
    (version_dir / "manifest.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    current_link = ensemble_dir / "current"
    try:
        if current_link.exists() or current_link.is_symlink():
            if current_link.is_symlink() or current_link.is_file():
                current_link.unlink()
            elif current_link.is_dir():
                shutil.rmtree(current_link)
        os.symlink(version_name, current_link)
    except Exception:
        try:
            current_link.write_text(version_name, encoding="utf-8")
        except Exception:
            pass

    return {
        "run_id": run_id,
        "symbol": str(run_manifest.get("symbol") or symbol_dir),
        "symbol_dir": symbol_dir,
        "ensemble": ensemble,
        "version": version_name,
        "path": version_dir.as_posix(),
        "files": copied_files,
    }


def _copy_xgb_run_to_wf(run_dir: Path) -> Dict[str, Any]:
    symbol_dir = str(run_dir.parent.parent.name or "").strip().lower()
    run_id = str(run_dir.name or "").strip()
    if not symbol_dir or not run_id:
        raise ValueError("bad run_dir structure")

    target_dir = _WF_DIR / symbol_dir / run_id
    if target_dir.exists():
        raise ValueError(f"WF candidate already exists: {target_dir.as_posix()}")
    target_dir.mkdir(parents=True, exist_ok=False)

    copied_files: List[str] = []
    for f in run_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, target_dir / f.name)
            copied_files.append(f.name)

    wf_meta = {
        "symbol": symbol_dir,
        "run_id": run_id,
        "source_run_path": run_dir.as_posix(),
        "copied_at": datetime.utcnow().isoformat() + "Z",
        "files": copied_files,
    }
    (target_dir / "wf_manifest.json").write_text(
        json.dumps(wf_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "run_id": run_id,
        "symbol_dir": symbol_dir,
        "path": target_dir.as_posix(),
        "files": copied_files,
    }


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _safe_slug(value: str, fallback: str = "experiment") -> str:
    raw = str(value or "").strip().lower()
    slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)
    slug = "_".join(part for part in slug.split("_") if part)
    return (slug[:80] or fallback)


def _symbol_dir_name(symbol: str) -> str:
    value = "".join(ch for ch in str(symbol or "").strip().lower() if ch.isalnum())
    for suffix in ("usdt", "usdc", "busd", "usdp", "usd"):
        if value.endswith(suffix) and len(value) > len(suffix):
            value = value[: -len(suffix)]
            break
    return value or "unknown"


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    result = _float_or_none(value)
    return int(result) if result is not None else None


def _serialize_oos_experiment_model(result: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = str(result.get("run_dir") or "")
    cfg = result.get("cfg_snapshot") if isinstance(result.get("cfg_snapshot"), dict) else {}
    metrics = result.get("oos_metrics") if isinstance(result.get("oos_metrics"), dict) else {}
    backtest = result.get("backtest") if isinstance(result.get("backtest"), dict) else {}
    f1_val = metrics.get("f1_val") if isinstance(metrics.get("f1_val"), list) else []
    precision_val = metrics.get("precision_val") if isinstance(metrics.get("precision_val"), list) else []
    recall_val = metrics.get("recall_val") if isinstance(metrics.get("recall_val"), list) else []

    return {
        "uuid": Path(run_dir).name or None,
        "run_dir": run_dir,
        "symbol": result.get("symbol"),
        "task": result.get("task"),
        "direction": result.get("direction"),
        "days": result.get("days"),
        "exit_mode": result.get("exit_mode"),
        "pnl_total": backtest.get("pnl_total"),
        "roi_pct": backtest.get("roi_pct"),
        "trades_count": backtest.get("trades_count"),
        "profit_factor": backtest.get("profit_factor"),
        "max_dd": backtest.get("max_dd"),
        "winrate": backtest.get("winrate"),
        "avg_trade_pnl": backtest.get("avg_trade_pnl"),
        "avg_bars_held": backtest.get("avg_bars_held"),
        "equity_end": backtest.get("equity_end"),
        "val_acc": metrics.get("val_acc"),
        "f1_buy_sell": metrics.get("f1_buy_sell_val"),
        "f1_1": f1_val[1] if len(f1_val) > 1 else None,
        "prec_1": precision_val[1] if len(precision_val) > 1 else None,
        "recall_1": recall_val[1] if len(recall_val) > 1 else None,
        "y_non_hold": metrics.get("y_non_hold_rate_val"),
        "pred_non_hold": metrics.get("pred_non_hold_rate_val"),
        "cfg": {
            "horizon_steps": cfg.get("horizon_steps"),
            "threshold": cfg.get("threshold"),
            "p_enter_threshold": cfg.get("p_enter_threshold"),
            "max_hold_steps": cfg.get("max_hold_steps"),
            "min_profit": cfg.get("min_profit"),
            "fee_bps": cfg.get("fee_bps"),
            "n_estimators": cfg.get("n_estimators"),
            "max_depth": cfg.get("max_depth"),
            "learning_rate": cfg.get("learning_rate"),
            "subsample": cfg.get("subsample"),
            "colsample_bytree": cfg.get("colsample_bytree"),
            "reg_lambda": cfg.get("reg_lambda"),
            "min_child_weight": cfg.get("min_child_weight"),
            "gamma": cfg.get("gamma"),
            "scale_pos_weight": cfg.get("scale_pos_weight"),
        },
    }


def _serialize_oos_experiment_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = str(row.get("run_dir") or "")
    return {
        "uuid": Path(run_dir).name or None,
        "run_dir": run_dir,
        "symbol": row.get("symbol"),
        "task": row.get("task"),
        "direction": row.get("direction"),
        "days": _int_or_none(row.get("days")),
        "exit_mode": row.get("exit_mode"),
        "pnl_total": _float_or_none(row.get("pnl_total")),
        "roi_pct": _float_or_none(row.get("roi_pct")),
        "trades_count": _int_or_none(row.get("trades_count")),
        "profit_factor": _float_or_none(row.get("profit_factor")),
        "max_dd": _float_or_none(row.get("max_dd")),
        "winrate": _float_or_none(row.get("winrate")),
        "avg_trade_pnl": _float_or_none(row.get("avg_trade_pnl")),
        "avg_bars_held": _float_or_none(row.get("avg_bars_held")),
        "equity_end": _float_or_none(row.get("equity_end")),
        "val_acc": _float_or_none(row.get("val_acc")),
        "f1_buy_sell": _float_or_none(row.get("f1_buy_sell")),
        "f1_1": _float_or_none(row.get("f1_1")),
        "prec_1": _float_or_none(row.get("prec_1")),
        "recall_1": _float_or_none(row.get("recall_1")),
        "y_non_hold": _float_or_none(row.get("y_non_hold")),
        "pred_non_hold": _float_or_none(row.get("pred_non_hold")),
        "cfg": {
            "horizon_steps": _int_or_none(row.get("horizon_steps")),
            "threshold": _float_or_none(row.get("threshold")),
            "p_enter_threshold": _float_or_none(row.get("p_enter_thr")),
            "max_hold_steps": _int_or_none(row.get("max_hold_steps")),
            "min_profit": _float_or_none(row.get("min_profit")),
            "fee_bps": _float_or_none(row.get("fee_bps")),
            "n_estimators": _int_or_none(row.get("n_estimators")),
            "max_depth": _int_or_none(row.get("max_depth")),
            "learning_rate": _float_or_none(row.get("learning_rate")),
            "subsample": _float_or_none(row.get("subsample")),
            "colsample_bytree": _float_or_none(row.get("colsample_bytree")),
            "reg_lambda": _float_or_none(row.get("reg_lambda")),
            "min_child_weight": _float_or_none(row.get("min_child_weight")),
            "gamma": _float_or_none(row.get("gamma")),
            "scale_pos_weight": _float_or_none(row.get("scale_pos_weight")),
        },
    }


def _serialize_oos_experiment_summary(summary_path: Path) -> Dict[str, Any]:
    summary = _safe_read_json(summary_path)
    models = summary.get("selected_models") if isinstance(summary.get("selected_models"), list) else []
    model_items = [item for item in models if isinstance(item, dict)]
    roi_values = [
        value for value in (_float_or_none(item.get("roi_pct")) for item in model_items)
        if value is not None
    ]
    pf_values = [
        value for value in (_float_or_none(item.get("profit_factor")) for item in model_items)
        if value is not None
    ]
    dd_values = [
        value for value in (_float_or_none(item.get("max_dd")) for item in model_items)
        if value is not None
    ]
    trades_sum = sum(_int_or_none(item.get("trades_count")) or 0 for item in model_items)
    top_model = model_items[0] if model_items else {}

    return {
        "symbol": summary.get("symbol"),
        "experiment_name": summary.get("experiment_name") or summary_path.parent.name,
        "created_at": summary.get("created_at"),
        "path": summary_path.parent.as_posix(),
        "summary_path": summary_path.as_posix(),
        "preset_path": (summary_path.parent / "preset.json").as_posix(),
        "oos_csv": summary.get("oos_csv"),
        "selection_metric": summary.get("selection_metric"),
        "selected_count": len(model_items),
        "top_uuid": top_model.get("uuid"),
        "top_roi_pct": _float_or_none(top_model.get("roi_pct")),
        "top_pnl_total": _float_or_none(top_model.get("pnl_total")),
        "top_profit_factor": _float_or_none(top_model.get("profit_factor")),
        "top_trades_count": _int_or_none(top_model.get("trades_count")),
        "top3_uuids": [str(item.get("uuid")) for item in model_items[:3] if item.get("uuid")],
        "avg_roi_top": (sum(roi_values) / len(roi_values)) if roi_values else None,
        "avg_profit_factor_top": (sum(pf_values) / len(pf_values)) if pf_values else None,
        "worst_max_dd_top": max(dd_values) if dd_values else None,
        "trades_sum": trades_sum,
    }


def _normalize_symbol_for_db(v: str) -> str:
    s = str(v or "").strip().upper().replace("/", "")
    if not s:
        return ""
    if not s.endswith(("USDT", "USD", "USDC", "BUSD", "USDP")):
        s = f"{s}USDT"
    return s


def _cfg_uses_1m_features(cfg: Dict[str, Any]) -> bool:
    return bool(
        cfg.get("use_1m_microvol")
        or cfg.get("use_1m_momentum")
        or cfg.get("use_1m_candle_structure")
        or cfg.get("use_1m_volume")
    )


def _cfg_uses_1d_regime(cfg: Dict[str, Any]) -> bool:
    return bool(cfg.get("use_1d_regime"))


def _run_symbol_and_lookahead(run_dir: Path) -> Tuple[str, int, bool, bool]:
    manifest = _safe_read_json(run_dir / "manifest.json")
    meta = _safe_read_json(run_dir / "meta.json")
    cfg = meta.get("cfg_snapshot") if isinstance(meta.get("cfg_snapshot"), dict) else {}
    sym = _normalize_symbol_for_db(
        str(manifest.get("symbol") or manifest.get("symbol_code") or run_dir.parent.parent.name or "")
    )
    lookahead = 0
    try:
        lookahead = max(lookahead, int(cfg.get("max_hold_steps") or 0))
    except Exception:
        pass
    try:
        lookahead = max(lookahead, int(cfg.get("horizon_steps") or 0))
    except Exception:
        pass
    return sym, int(lookahead), _cfg_uses_1m_features(cfg), _cfg_uses_1d_regime(cfg)


def _start_oos_prefetch_task(sym: str, limit: int, lock_suffix: str) -> str | None:
    prefetch_task_id = None
    prefetch_lock_key = f"oos:prefetch:{lock_suffix}:{sym}"
    try:
        rc = get_redis_client()
        if rc:
            existing = rc.get(prefetch_lock_key)
            if existing:
                tid = existing.decode("utf-8") if isinstance(existing, bytes) else str(existing)
                ar = AsyncResult(tid, app=celery)
                if ar.state in ("PENDING", "STARTED", "RETRY"):
                    return tid
                try:
                    rc.delete(prefetch_lock_key)
                except Exception:
                    pass
            t = celery.send_task(
                "tasks.xgb_oos_tasks.prefetch_xgb_oos_ohlcv",
                kwargs={"symbol": sym, "limit_candles": int(limit)},
                queue="oos",
            )
            prefetch_task_id = t.id
            try:
                rc.setex(prefetch_lock_key, 1800, prefetch_task_id)
            except Exception:
                pass
    except Exception:
        prefetch_task_id = None
    return prefetch_task_id


def _wait_for_oos_prefetch_task(task_id: str | None) -> Dict[str, Any]:
    if not task_id or _OOS_PREFETCH_WAIT_SEC <= 0:
        return {"state": None, "ready": False}
    deadline = time.time() + float(_OOS_PREFETCH_WAIT_SEC)
    ar = AsyncResult(task_id, app=celery)
    while time.time() < deadline:
        state = ar.state
        if state in ("SUCCESS", "FAILURE", "REVOKED"):
            result = None
            try:
                result = ar.result
            except Exception:
                result = None
            if not isinstance(result, (dict, list, str, int, float, bool, type(None))):
                result = str(result)
            return {"state": state, "ready": True, "result": result}
        time.sleep(1.0)
    return {"state": ar.state, "ready": False}


def _prefetch_oos_to_db(result_dirs: List[str], days_list: List[int]) -> Dict[str, Any]:
    by_symbol: Dict[str, Dict[str, Any]] = {}
    for raw in result_dirs:
        rd = Path(str(raw or "").strip()).resolve()
        if not rd.exists() or not rd.is_dir() or rd.parent.name != "runs":
            continue
        sym, look, needs_1m, needs_1d = _run_symbol_and_lookahead(rd)
        if not sym:
            continue
        info = by_symbol.setdefault(sym, {"lookahead": 0, "needs_1m": False, "needs_1d": False})
        info["lookahead"] = max(int(info.get("lookahead", 0)), int(look))
        info["needs_1m"] = bool(info.get("needs_1m")) or bool(needs_1m)
        info["needs_1d"] = bool(info.get("needs_1d")) or bool(needs_1d)

    if not by_symbol:
        return {"success": False, "error": "No valid symbols resolved for prefetch"}

    max_days = max([int(d) for d in days_list if int(d) > 0], default=30)
    bars = int(max_days) * 288
    prefetch: List[Dict[str, Any]] = []
    now_ms = int(datetime.utcnow().timestamp() * 1000)
    oos_end_ts = _oos_end_ts_ms()
    oos_lag_hours = _oos_end_lag_hours()

    for sym, info in by_symbol.items():
        utc_s, msk_s = _now_utc_msk_str()
        look = int(info.get("lookahead", 0))
        limit = int(bars + int(look) + 10)
        logger.info(
            f"[OOS_PREFETCH] start sym={sym} days={max_days} lookahead={look} limit={limit} "
            f"oos_end_ts={oos_end_ts} lag_hours={oos_lag_hours} now_utc='{utc_s}' now_msk='{msk_s}'"
        )
        df = db_get_ohlcv_only(
            symbol_name=sym,
            timeframe="5m",
            limit_candles=limit,
            freshness_max_age_sec=24 * 3600,
            allow_api_fallback=False,
            end_ts_ms=oos_end_ts,
        )
        err = None
        if df is None or df.empty:
            return {
                "success": False,
                "error": f"Prefetch failed for {sym}",
                "symbol": sym,
                "limit": limit,
                "fetch_error": "DB-only mode: no candles in DB",
            }
        last_ts = None
        age_sec = None
        try:
            last_ts = int(df["timestamp"].max())
            age_sec = int(max(0, (now_ms - last_ts) // 1000))
        except Exception:
            last_ts = None
            age_sec = None
        logger.info(
            f"[OOS_PREFETCH] done sym={sym} rows={0 if df is None else len(df)} "
            f"last_ts={last_ts} age_sec={age_sec} fetch_error={err} "
            f"oos_end_ts={oos_end_ts} now_utc='{utc_s}' now_msk='{msk_s}'"
        )
        # Prefetch is considered successful if DB covers the fixed OOS cutoff.
        if last_ts is None or int(last_ts) < int(oos_end_ts):
            # DB-only OOS: request one shared refresh task (deduplicated by Redis key).
            prefetch_task_id = _start_oos_prefetch_task(sym, limit, "5m")
            wait_info = _wait_for_oos_prefetch_task(prefetch_task_id)
            if wait_info.get("state") == "SUCCESS":
                now_ms = int(datetime.utcnow().timestamp() * 1000)
                df = db_get_ohlcv_only(
                    symbol_name=sym,
                    timeframe="5m",
                    limit_candles=limit,
                    freshness_max_age_sec=24 * 3600,
                    allow_api_fallback=False,
                    end_ts_ms=oos_end_ts,
                )
                if df is not None and not df.empty:
                    last_ts = int(df["timestamp"].max())
                    age_sec = int(max(0, (now_ms - last_ts) // 1000))
                    if int(last_ts) >= int(oos_end_ts):
                        logger.info(
                            f"[OOS_PREFETCH] cutoff covered after wait sym={sym} "
                            f"last_ts={last_ts} oos_end_ts={oos_end_ts} age_sec={age_sec}"
                        )
                    else:
                        wait_info["ready"] = False
            if last_ts is None or int(last_ts) < int(oos_end_ts):
                logger.warning(
                    f"[OOS_PREFETCH] stale sym={sym} last_ts={last_ts} age_sec={age_sec} "
                    f"oos_end_ts={oos_end_ts} lag_hours={oos_lag_hours} fetch_error={err} "
                    f"prefetch_task_id={prefetch_task_id} wait_state={wait_info.get('state')} "
                    f"now_utc='{utc_s}' now_msk='{msk_s}'"
                )
                return {
                    "success": False,
                    "error": f"Prefetch stale for {sym}: DB does not cover OOS cutoff",
                    "symbol": sym,
                    "limit": limit,
                    "fetch_error": err,
                    "db_last_ts": last_ts,
                    "oos_end_ts": int(oos_end_ts),
                    "oos_lag_hours": float(oos_lag_hours),
                    "age_sec": age_sec,
                    "prefetch_task_started": bool(prefetch_task_id),
                    "prefetch_task_id": prefetch_task_id,
                    "prefetch_wait": wait_info,
                }
        aux_error = None
        if bool(info.get("needs_1m")):
            lim_1m = _oos_1m_fetch_limit(limit)
            df_1m = db_get_ohlcv_only(
                symbol_name=sym,
                timeframe="1m",
                limit_candles=lim_1m,
                freshness_max_age_sec=3600,
                allow_api_fallback=False,
                end_ts_ms=int(oos_end_ts) + (5 * 60 * 1000 - 60_000),
            )
            if df_1m is None or df_1m.empty:
                aux_error = "missing fresh 1m candles"
            else:
                try:
                    df_5m_check = _oos_5m_closed_tail(df, int(limit), end_ts_ms=oos_end_ts)
                    if df_5m_check is None or getattr(df_5m_check, "empty", False):
                        aux_error = "no closed 5m slice for 1m coverage check"
                    else:
                        missing_5m = _missing_1m_bucket(df_5m_check, df_1m, end_ts_ms=oos_end_ts)
                        if missing_5m is not None:
                            aux_error = f"missing 1m candles for 5m timestamp={missing_5m}"
                except Exception as exc:
                    aux_error = f"bad 1m coverage check: {exc}"
        if aux_error is None and bool(info.get("needs_1d")):
            df_1d = db_get_ohlcv_only(
                symbol_name=sym,
                timeframe="1d",
                limit_candles=max(120, int(limit / 288) + 120),
                freshness_max_age_sec=24 * 3600,
                allow_api_fallback=False,
                end_ts_ms=oos_end_ts,
            )
            if df_1d is None or df_1d.empty:
                aux_error = "missing fresh 1d candles"
        if aux_error:
            prefetch_task_id = _start_oos_prefetch_task(sym, limit, "aux")
            wait_info = _wait_for_oos_prefetch_task(prefetch_task_id)
            if wait_info.get("state") == "SUCCESS":
                aux_error = None
                if bool(info.get("needs_1m")):
                    df_1m = db_get_ohlcv_only(
                        symbol_name=sym,
                        timeframe="1m",
                        limit_candles=_oos_1m_fetch_limit(limit),
                        freshness_max_age_sec=3600,
                        allow_api_fallback=False,
                        end_ts_ms=int(oos_end_ts) + (5 * 60 * 1000 - 60_000),
                    )
                    if df_1m is None or df_1m.empty:
                        aux_error = "missing fresh 1m candles"
                    else:
                        missing_5m = _missing_1m_bucket(_oos_5m_closed_tail(df, int(limit), end_ts_ms=oos_end_ts), df_1m, end_ts_ms=oos_end_ts)
                        if missing_5m is not None:
                            aux_error = f"missing 1m candles for 5m timestamp={missing_5m}"
                if aux_error is None and bool(info.get("needs_1d")):
                    df_1d = db_get_ohlcv_only(
                        symbol_name=sym,
                        timeframe="1d",
                        limit_candles=max(120, int(limit / 288) + 120),
                        freshness_max_age_sec=24 * 3600,
                        allow_api_fallback=False,
                        end_ts_ms=oos_end_ts,
                    )
                    if df_1d is None or df_1d.empty:
                        aux_error = "missing fresh 1d candles"
                if aux_error is None:
                    logger.info(
                        f"[OOS_PREFETCH] aux fresh after wait sym={sym} "
                        f"prefetch_task_id={prefetch_task_id}"
                    )
            if aux_error:
                return {
                    "success": False,
                    "error": f"Prefetch stale for {sym}: {aux_error}",
                    "symbol": sym,
                    "limit": limit,
                    "fetch_error": aux_error,
                    "db_last_ts": last_ts,
                    "oos_end_ts": int(oos_end_ts),
                    "oos_lag_hours": float(oos_lag_hours),
                    "age_sec": age_sec,
                    "prefetch_task_started": bool(prefetch_task_id),
                    "prefetch_task_id": prefetch_task_id,
                    "prefetch_wait": wait_info,
                }
        prefetch.append({
            "symbol": sym,
            "rows": int(len(df)),
            "limit": int(limit),
            "last_ts": last_ts,
            "oos_end_ts": int(oos_end_ts),
            "oos_lag_hours": float(oos_lag_hours),
        })

    logger.info(f"[OOS_PREFETCH] ok symbols={sorted(by_symbol.keys())} items={len(prefetch)}")
    return {"success": True, "prefetch": prefetch, "symbols": sorted(by_symbol.keys()), "days": int(max_days)}


def _scan_xgb_runs() -> List[Dict[str, Any]]:
    """
    Сканирует result/xgb/*/runs/* и возвращает список запусков с метриками.
    """
    base = Path("result") / "xgb"
    out: List[Dict[str, Any]] = []
    if not base.exists():
        return out
    for sym_dir in base.iterdir():
        if not sym_dir.is_dir():
            continue
        runs_root = sym_dir / "runs"
        if not runs_root.exists():
            continue
        for run_dir in runs_root.iterdir():
            if not run_dir.is_dir():
                continue
            manifest = _safe_read_json(run_dir / "manifest.json")
            meta = _safe_read_json(run_dir / "meta.json")
            metrics = _safe_read_json(run_dir / "metrics.json")
            cfg = meta.get("cfg_snapshot") if isinstance(meta.get("cfg_snapshot"), dict) else {}
            try:
                mtime = float(run_dir.stat().st_mtime)
            except Exception:
                mtime = 0.0

            model_path = str(manifest.get("model_path") or (run_dir / "model.json"))
            out.append(
                {
                    "symbol": (manifest.get("symbol") or sym_dir.name),
                    "run_name": (manifest.get("run_name") or run_dir.name),
                    "direction": (manifest.get("direction") or cfg.get("direction") or ""),
                    "task": (manifest.get("task") or cfg.get("task") or ""),
                    "source": (manifest.get("source") or ""),
                    "grid_id": (manifest.get("grid_id") or ""),
                    "result_dir": str(run_dir),
                    "model_path": model_path,
                    "mtime": mtime,
                    "metrics": metrics,
                    "cfg": cfg,
                }
            )
    out.sort(key=lambda r: float(r.get("mtime") or 0.0), reverse=True)
    return out


def _scan_wf_xgb_runs() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not _WF_DIR.exists():
        return out
    for symbol_dir in _WF_DIR.iterdir():
        if not symbol_dir.is_dir():
            continue
        for run_dir in symbol_dir.iterdir():
            if not run_dir.is_dir():
                continue
            wf_manifest = _safe_read_json(run_dir / "wf_manifest.json")
            manifest = _safe_read_json(run_dir / "manifest.json")
            metrics = _safe_read_json(run_dir / "metrics.json")
            try:
                mtime = float(run_dir.stat().st_mtime)
            except Exception:
                mtime = 0.0
            out.append(
                {
                    "symbol": str(wf_manifest.get("symbol") or symbol_dir.name).upper(),
                    "run_name": str(wf_manifest.get("run_id") or run_dir.name),
                    "result_dir": run_dir.as_posix(),
                    "source_run_path": str(wf_manifest.get("source_run_path") or ""),
                    "copied_at": wf_manifest.get("copied_at"),
                    "direction": manifest.get("direction"),
                    "task": manifest.get("task"),
                    "metrics": metrics if isinstance(metrics, dict) else {},
                    "mtime": mtime,
                }
            )
    out.sort(key=lambda r: float(r.get("mtime") or 0.0), reverse=True)
    return out


def _serialize_xgb_run(run: Dict[str, Any]) -> Dict[str, Any]:
    metrics = run.get("metrics") if isinstance(run.get("metrics"), dict) else {}
    cfg = run.get("cfg") if isinstance(run.get("cfg"), dict) else {}
    return {
        "symbol": run.get("symbol"),
        "run_name": run.get("run_name"),
        "direction": run.get("direction"),
        "task": run.get("task"),
        "source": run.get("source"),
        "grid_id": run.get("grid_id"),
        "result_dir": run.get("result_dir"),
        "model_path": run.get("model_path"),
        "mtime": run.get("mtime"),
        "metrics": metrics,
        "cfg": cfg,
    }


def _serialize_wf_xgb_run(run: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "symbol": run.get("symbol"),
        "run_name": run.get("run_name"),
        "result_dir": run.get("result_dir"),
        "source_run_path": run.get("source_run_path"),
        "copied_at": run.get("copied_at"),
        "direction": run.get("direction"),
        "task": run.get("task"),
        "metrics": run.get("metrics") if isinstance(run.get("metrics"), dict) else {},
        "mtime": run.get("mtime"),
    }


def _is_binary_task(task: str) -> bool:
    t = str(task or "").strip().lower()
    return t.startswith("entry") or t.startswith("exit")


def _score_run(metrics: Dict[str, Any], manifest: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Score run for pruning BEFORE OOS.
    Primary: F1 for class=1 on binary tasks, else f1_buy_sell/val_acc.
    Soft sanity: deprioritize runs with 0 proxy trades or collapsed predictions.
    """
    task = str(metrics.get("task") or manifest.get("task") or cfg.get("task") or "").strip().lower()
    is_binary = _is_binary_task(task)

    # base metric
    base = None
    if is_binary:
        f1v = metrics.get("f1_val")
        if isinstance(f1v, list) and len(f1v) > 1:
            try:
                base = float(f1v[1])
            except Exception:
                base = None
    if base is None:
        try:
            fbs = metrics.get("f1_buy_sell_val")
            base = float(fbs) if fbs is not None else None
        except Exception:
            base = None
    if base is None:
        try:
            base = float(metrics.get("val_acc") or 0.0)
        except Exception:
            base = 0.0

    # penalties (very light; still keep mostly by metric)
    penalty = 0.0
    try:
        pred_non_hold = float(metrics.get("pred_non_hold_rate_val") or 0.0)
        if pred_non_hold <= 1e-6:
            penalty += 0.50  # collapsed → put to the bottom
    except Exception:
        pass
    try:
        pp = metrics.get("proxy_pnl_val") if isinstance(metrics.get("proxy_pnl_val"), dict) else {}
        trades = int(pp.get("trades") or 0)
        if trades <= 0:
            penalty += 0.25
    except Exception:
        pass

    info = {"task": task, "base": base, "penalty": penalty, "pred_non_hold_rate_val": metrics.get("pred_non_hold_rate_val"), "proxy_pnl_val": metrics.get("proxy_pnl_val")}
    return float(base) - float(penalty), info


@xgb_oos_bp.get("/oos_xgb")
def xgb_oos_page():
    try:
        runs = _scan_xgb_runs()
        return render_template("oos/xgb_oos.html", xgb_runs=runs)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@xgb_oos_bp.get("/api/xgb/oos/runs")
def xgb_oos_runs():
    runs = _scan_xgb_runs()
    return jsonify({"success": True, "runs": [_serialize_xgb_run(run) for run in runs]})


@xgb_oos_bp.get("/api/xgb/wf/runs")
def xgb_wf_runs():
    runs = _scan_wf_xgb_runs()
    return jsonify({"success": True, "runs": [_serialize_wf_xgb_run(run) for run in runs]})


@xgb_oos_bp.get("/api/xgb/oos/experiments")
def xgb_oos_experiments():
    experiments: List[Dict[str, Any]] = []
    if _EXPERIMENTS_DIR.exists():
        for summary_path in _EXPERIMENTS_DIR.glob("*/*/summary.json"):
            try:
                experiments.append(_serialize_oos_experiment_summary(summary_path))
            except Exception as e:
                logger.warning("Failed to read XGB OOS experiment %s: %s", summary_path, e)

    experiments.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return jsonify({"success": True, "experiments": experiments})


@xgb_oos_bp.get("/api/xgb/hypo/short_anchors")
def xgb_hypo_short_anchors():
    anchors_path = _XGB_HYPO_DIR / "xgb_short_model_anchors.json"
    if not anchors_path.exists():
        return jsonify({"success": False, "error": "File not found"}), 404
    try:
        with open(anchors_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify({"success": True, "data": data})
    except Exception as e:
        logger.error("Failed to read anchors json: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@xgb_oos_bp.post("/xgb_oos_test_async")
def xgb_oos_test_async():
    """
    Запускает XGB OOS evaluation в Celery (очередь 'celery' — обычный воркер).
    Body:
      { result_dir: 'result/xgb/<SYM>/runs/<RUN>', days: 30 }
    """
    try:
        data = request.get_json(silent=True) or {}
        result_dir = str(data.get("result_dir") or "").strip()
        days = int(data.get("days") or 30)
        exit_mode = str(data.get("exit_mode") or "policy").strip().lower()
        atr_len = data.get("atr_len", None)
        atr_mult = data.get("atr_mult", None)
        direction_override_raw = str(data.get("direction_override") or "").strip().lower()
        direction_override = direction_override_raw if direction_override_raw in ("long", "short") else None
        override_exit_policy = bool(data.get("override_exit_policy")) if isinstance(data, dict) else False
        entry_tp_pct = data.get("entry_tp_pct", None)
        entry_sl_pct = data.get("entry_sl_pct", None)
        entry_trail_pct = data.get("entry_trail_pct", None)
        signal_exit_enabled = bool(data.get("signal_exit_enabled")) if isinstance(data, dict) else False
        signal_exit_window = data.get("signal_exit_window", None)
        signal_exit_start_pct = data.get("signal_exit_start_pct", None)
        p_enter_threshold_raw = data.get("p_enter_threshold", None)
        try:
            p_enter_threshold = float(p_enter_threshold_raw) if p_enter_threshold_raw not in (None, "") else None
        except Exception:
            p_enter_threshold = None
        days = max(1, min(days, 365))
        if not result_dir:
            return jsonify({"success": False, "error": "result_dir required"}), 400
        if exit_mode not in ("policy", "hold_steps", "atr_trail"):
            exit_mode = "policy"
        try:
            atr_len = int(atr_len) if atr_len not in (None, "") else None
        except Exception:
            atr_len = None
        try:
            atr_mult = float(atr_mult) if atr_mult not in (None, "") else None
        except Exception:
            atr_mult = None
        try:
            entry_tp_pct = float(entry_tp_pct) if (override_exit_policy and entry_tp_pct not in (None, "")) else None
        except Exception:
            entry_tp_pct = None
        try:
            entry_sl_pct = float(entry_sl_pct) if (override_exit_policy and entry_sl_pct not in (None, "")) else None
        except Exception:
            entry_sl_pct = None
        try:
            entry_trail_pct = float(entry_trail_pct) if (override_exit_policy and entry_trail_pct not in (None, "")) else None
        except Exception:
            entry_trail_pct = None
        try:
            signal_exit_window = int(signal_exit_window) if (signal_exit_enabled and signal_exit_window not in (None, "")) else 20
        except Exception:
            signal_exit_window = 20
        signal_exit_window = max(1, min(int(signal_exit_window or 20), 500))
        try:
            signal_exit_start_pct = float(signal_exit_start_pct) if (signal_exit_enabled and signal_exit_start_pct not in (None, "")) else 0.65
        except Exception:
            signal_exit_start_pct = 0.65
        if signal_exit_start_pct > 1.0:
            signal_exit_start_pct = signal_exit_start_pct / 100.0
        signal_exit_start_pct = max(0.0, min(float(signal_exit_start_pct or 0.65), 1.0))
        signal_exit_threshold = None
        signal_exit_threshold_raw = data.get("signal_exit_threshold", None)
        if signal_exit_enabled and signal_exit_threshold_raw not in (None, ""):
            try:
                signal_exit_threshold = float(signal_exit_threshold_raw)
            except Exception:
                signal_exit_threshold = None
            if signal_exit_threshold is not None and not (0.0 < float(signal_exit_threshold) < 1.0):
                return jsonify({"success": False, "error": "signal_exit_threshold must be between 0 and 1"}), 400
        # Prefetch once before launching OOS task; worker then reads from DB-only.
        pre = _prefetch_oos_to_db(result_dirs=[result_dir], days_list=[days])
        if not bool(pre.get("success")):
            return jsonify({"success": False, "error": pre.get("error") or "prefetch failed", "prefetch": pre}), 400
        task = celery.send_task(
            "tasks.xgb_oos_tasks.run_xgb_oos_test",
            kwargs={
                "result_dir": result_dir,
                "days": days,
                "ts": datetime.utcnow().isoformat() + "Z",
                "exit_mode": exit_mode,
                "atr_len": atr_len,
                "atr_mult": atr_mult,
                "override_exit_policy": bool(override_exit_policy),
                "entry_tp_pct": entry_tp_pct,
                "entry_sl_pct": entry_sl_pct,
                "entry_trail_pct": entry_trail_pct,
                "p_enter_threshold": p_enter_threshold,
                "direction_override": direction_override,
                "signal_exit_enabled": bool(signal_exit_enabled),
                "signal_exit_window": int(signal_exit_window),
                "signal_exit_start_pct": float(signal_exit_start_pct),
                "signal_exit_threshold": signal_exit_threshold,
            },
            queue="oos",
        )
        return jsonify({"success": True, "task_id": task.id, "prefetch": pre.get("prefetch", [])})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@xgb_oos_bp.get("/xgb_oos_test_status")
def xgb_oos_test_status():
    try:
        task_id = (request.args.get("task_id") or "").strip()
        if not task_id:
            return jsonify({"success": False, "error": "task_id required"}), 400
        ar = AsyncResult(task_id, app=celery)
        resp: Dict[str, Any] = {"success": True, "task_id": task_id, "state": ar.state}
        if ar.state == "SUCCESS":
            try:
                res = ar.result
                if isinstance(res, dict):
                    resp["result"] = res
            except Exception:
                pass
        elif ar.state == "FAILURE":
            try:
                res = ar.result
                if isinstance(res, dict):
                    resp["result"] = res
                    if res.get("error"):
                        resp["error"] = str(res.get("error"))
                else:
                    resp["error"] = str(res)
            except Exception:
                pass
            try:
                tb = ar.traceback
                if tb:
                    # Keep payload compact for polling endpoint.
                    resp["traceback"] = str(tb)[-4000:]
            except Exception:
                pass
        return jsonify(resp)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@xgb_oos_bp.post("/xgb_oos_delete_runs")
def xgb_oos_delete_runs():
    """Удаляет выбранные XGB run директории."""
    data = request.get_json(silent=True) or {}
    paths = data.get("paths") if isinstance(data, dict) else None
    if not isinstance(paths, list) or not paths:
        return jsonify({"success": False, "error": "paths[] required"}), 400

    base = (Path("result") / "xgb").resolve()
    deleted: List[str] = []
    errors: List[Dict[str, Any]] = []

    for p in paths:
        try:
            raw = str(p or "").strip()
            if not raw:
                continue
            target = Path(raw).resolve()
            if base not in target.parents:
                raise ValueError("path outside result/xgb")
            if target.name == "runs" or target.parent.name != "runs":
                raise ValueError("not a run directory")
            if not target.exists() or not target.is_dir():
                raise ValueError("not found")
            shutil.rmtree(target)
            deleted.append(str(target))
        except Exception as e:
            errors.append({"path": str(p), "error": str(e)})

    return jsonify({"success": True, "deleted": deleted, "errors": errors, "deleted_count": len(deleted)})


@xgb_oos_bp.post("/xgb_oos_copy_to_prod")
def xgb_oos_copy_to_prod():
    data = request.get_json(silent=True) or {}
    paths = data.get("paths") if isinstance(data, dict) else None
    ensemble = str(data.get("ensemble") or "ensemble-a").strip() if isinstance(data, dict) else "ensemble-a"
    if not isinstance(paths, list) or not paths:
        return jsonify({"success": False, "error": "paths[] required"}), 400
    if ensemble not in ("ensemble-a", "ensemble-b", "ensemble-c"):
        return jsonify({"success": False, "error": "bad ensemble"}), 400

    base = (Path("result") / "xgb").resolve()
    copied: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for p in paths:
        try:
            raw = str(p or "").strip()
            if not raw:
                continue
            target = Path(raw).resolve()
            if base not in target.parents:
                raise ValueError("path outside result/xgb")
            if target.name == "runs" or target.parent.name != "runs":
                raise ValueError("not a run directory")
            if not target.exists() or not target.is_dir():
                raise ValueError("not found")
            copied.append(_copy_xgb_run_to_prod(run_dir=target, ensemble=ensemble))
        except Exception as e:
            errors.append({"path": str(p), "error": str(e)})

    return jsonify({
        "success": bool(copied),
        "copied": copied,
        "errors": errors,
        "copied_count": len(copied),
        "ensemble": ensemble,
    })


@xgb_oos_bp.post("/xgb_oos_copy_to_wf")
def xgb_oos_copy_to_wf():
    data = request.get_json(silent=True) or {}
    paths = data.get("paths") if isinstance(data, dict) else None
    if not isinstance(paths, list) or not paths:
        return jsonify({"success": False, "error": "paths[] required"}), 400

    base = (Path("result") / "xgb").resolve()
    copied: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for p in paths:
        try:
            raw = str(p or "").strip()
            if not raw:
                continue
            target = Path(raw).resolve()
            if base not in target.parents:
                raise ValueError("path outside result/xgb")
            if target.name == "runs" or target.parent.name != "runs":
                raise ValueError("not a run directory")
            if not target.exists() or not target.is_dir():
                raise ValueError("not found")
            copied.append(_copy_xgb_run_to_wf(run_dir=target))
        except Exception as e:
            errors.append({"path": str(p), "error": str(e)})

    return jsonify({
        "success": bool(copied),
        "copied": copied,
        "errors": errors,
        "copied_count": len(copied),
    })


@xgb_oos_bp.post("/xgb_oos_prune_runs")
def xgb_oos_prune_runs():
    """
    Deletes worst runs among selected, keeping top keep_pct (default 10%).
    Uses training/val metrics from <run>/metrics.json (BEFORE OOS).
    """
    data = request.get_json(silent=True) or {}
    paths = data.get("paths") if isinstance(data, dict) else None
    keep_pct = data.get("keep_pct", 10)
    dry_run = bool(data.get("dry_run")) if isinstance(data, dict) else False
    if not isinstance(paths, list) or not paths:
        return jsonify({"success": False, "error": "paths[] required"}), 400
    try:
        keep_pct = float(keep_pct)
    except Exception:
        return jsonify({"success": False, "error": "keep_pct must be a number"}), 400
    keep_pct = max(1.0, min(100.0, keep_pct))

    base = (Path("result") / "xgb").resolve()
    scored: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    groups: Dict[str, List[Dict[str, Any]]] = {}

    for p in paths:
        try:
            raw = str(p or "").strip()
            if not raw:
                continue
            run_dir = Path(raw).resolve()
            if base not in run_dir.parents:
                raise ValueError("path outside result/xgb")
            if run_dir.name == "runs" or run_dir.parent.name != "runs":
                raise ValueError("not a run directory")
            if not run_dir.exists() or not run_dir.is_dir():
                raise ValueError("not found")
            manifest = _safe_read_json(run_dir / "manifest.json")
            meta = _safe_read_json(run_dir / "meta.json")
            metrics = _safe_read_json(run_dir / "metrics.json")
            cfg = meta.get("cfg_snapshot") if isinstance(meta.get("cfg_snapshot"), dict) else {}
            score, info = _score_run(metrics=metrics, manifest=manifest, cfg=cfg)
            
            direction = str(manifest.get("direction") or cfg.get("direction") or "unknown").lower()
            if direction not in groups:
                groups[direction] = []
            
            groups[direction].append({"path": str(run_dir), "score": float(score), "info": info})
            scored.append({"path": str(run_dir), "score": float(score), "info": info})
        except Exception as e:
            errors.append({"path": str(p), "error": str(e)})

    if not scored:
        return jsonify({"success": False, "error": "No valid run dirs to prune", "errors": errors}), 400

    keep = []
    drop = []
    for direction, items in groups.items():
        items.sort(key=lambda x: float(x.get("score") or -1e9), reverse=True)
        n = len(items)
        keep_n = max(1, int(math.ceil(n * (keep_pct / 100.0))))
        keep.extend(items[:keep_n])
        drop.extend(items[keep_n:])

    deleted: List[str] = []
    del_errors: List[Dict[str, Any]] = []
    if not dry_run:
        for it in drop:
            try:
                target = Path(str(it.get("path") or "")).resolve()
                if not target.exists() or not target.is_dir():
                    continue
                shutil.rmtree(target)
                deleted.append(str(target))
            except Exception as e:
                del_errors.append({"path": str(it.get("path")), "error": str(e)})

    return jsonify(
        {
            "success": True,
            "selected": int(n),
            "keep_pct": float(keep_pct),
            "dry_run": bool(dry_run),
            "kept_count": int(len(keep)),
            "deleted_count": int(len(deleted)),
            "kept": keep,
            "deleted": deleted,
            "errors": errors + del_errors,
            "cutoff_score": float(keep[-1]["score"]) if keep else None,
        }
    )


@xgb_oos_bp.post("/xgb_oos_prune_duplicates")
def xgb_oos_prune_duplicates():
    """
    Группирует все XGB run по (symbol, task, direction, H, Thr, max_hold, min_profit, fee_bps).
    Оставляет top-1 по f1(1) (или f1_buy_sell) в каждой группе, остальные удаляет.
    """
    runs = _scan_xgb_runs()
    base = (Path("result") / "xgb").resolve()

    def _config_key(r: Dict[str, Any]) -> Tuple[Any, ...]:
        c = r.get("cfg") or {}
        return (
            r.get("symbol", ""),
            r.get("task", ""),
            r.get("direction", ""),
            c.get("horizon_steps"),
            c.get("threshold"),
            c.get("max_hold_steps"),
            c.get("min_profit"),
            c.get("fee_bps"),
        )

    def _get_score(r: Dict[str, Any]) -> float:
        m = r.get("metrics") or {}
        task = str(r.get("task") or "").strip().lower()
        if _is_binary_task(task):
            f1v = m.get("f1_val")
            if isinstance(f1v, list) and len(f1v) > 1:
                try:
                    return float(f1v[1])
                except Exception:
                    pass
        fbs = m.get("f1_buy_sell_val")
        if fbs is not None:
            try:
                return float(fbs)
            except Exception:
                pass
        return float(m.get("val_acc") or 0.0)

    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in runs:
        k = _config_key(r)
        if k not in groups:
            groups[k] = []
        groups[k].append(r)

    to_delete: List[str] = []
    kept_count = 0
    for k, group in groups.items():
        if len(group) <= 1:
            kept_count += len(group)
            continue
        group.sort(key=_get_score, reverse=True)
        best = group[0]
        kept_count += 1
        for r in group[1:]:
            to_delete.append(r["result_dir"])

    deleted: List[str] = []
    for p in to_delete:
        try:
            target = Path(p).resolve()
            if base not in target.parents or not target.exists() or not target.is_dir():
                continue
            if target.parent.name != "runs":
                continue
            shutil.rmtree(target)
            deleted.append(p)
        except Exception:
            pass

    return jsonify({
        "success": True,
        "kept_count": kept_count,
        "deleted_count": len(deleted),
        "deleted": deleted,
    })


# ── Batch OOS ──────────────────────────────────────────────────────────

@xgb_oos_bp.post("/xgb_oos_batch_async")
def xgb_oos_batch_async():
    """Запускает OOS для нескольких run'ов параллельно через Celery."""
    data = request.get_json(silent=True) or {}
    result_dirs = data.get("result_dirs", [])
    days_grid = data.get("days_grid", None)
    p_enter_grid_enabled = bool(data.get("p_enter_grid_enabled")) if isinstance(data, dict) else False
    p_enter_thresholds_raw = data.get("p_enter_thresholds", None)
    # Backward compatible single-days input (older UI/scripts)
    days_single = max(1, min(int(data.get("days") or 30), 365))
    exit_mode = str(data.get("exit_mode") or "policy").strip().lower()
    exit_modes = data.get("exit_modes", None)
    direction_override_raw = str(data.get("direction_override") or "").strip().lower()
    direction_override = direction_override_raw if direction_override_raw in ("long", "short") else None
    override_exit_policy = bool(data.get("override_exit_policy")) if isinstance(data, dict) else False
    entry_tp_pct = data.get("entry_tp_pct", None)
    entry_sl_pct = data.get("entry_sl_pct", None)
    entry_trail_pct = data.get("entry_trail_pct", None)
    signal_exit_enabled = bool(data.get("signal_exit_enabled")) if isinstance(data, dict) else False
    signal_exit_window = data.get("signal_exit_window", None)
    signal_exit_start_pct = data.get("signal_exit_start_pct", None)
    signal_exit_threshold_grid_enabled = bool(data.get("signal_exit_threshold_grid_enabled")) if isinstance(data, dict) else False
    signal_exit_thresholds_raw = data.get("signal_exit_thresholds", None)
    signal_exit_threshold_raw = data.get("signal_exit_threshold", None)
    p_enter_threshold_raw = data.get("p_enter_threshold", None)
    try:
        p_enter_threshold = float(p_enter_threshold_raw) if p_enter_threshold_raw not in (None, "") else None
    except Exception:
        p_enter_threshold = None
    atr_len = data.get("atr_len", None)
    atr_mult = data.get("atr_mult", None)
    if not isinstance(result_dirs, list) or not result_dirs:
        return jsonify({"success": False, "error": "result_dirs[] required"}), 400
    if days_grid is not None and not isinstance(days_grid, list):
        return jsonify({"success": False, "error": "days_grid must be a list (e.g. [30,60,90])"}), 400
    p_enter_list: List[float | None] = [p_enter_threshold]
    if p_enter_grid_enabled:
        if not isinstance(p_enter_thresholds_raw, list) or not p_enter_thresholds_raw:
            return jsonify({"success": False, "error": "p_enter_thresholds[] required when threshold grid is enabled"}), 400
        parsed_thresholds: List[float] = []
        for raw_thr in p_enter_thresholds_raw:
            try:
                thr = float(raw_thr)
            except Exception:
                return jsonify({"success": False, "error": f"bad p_enter_threshold value: {raw_thr!r}"}), 400
            if not (0.0 < thr < 1.0):
                return jsonify({"success": False, "error": f"p_enter_threshold must be between 0 and 1: {thr}"}), 400
            parsed_thresholds.append(round(thr, 6))
        p_enter_list = sorted(set(parsed_thresholds), reverse=True)
    signal_exit_thr_list: List[float | None] = [None]
    if signal_exit_enabled:
        if signal_exit_threshold_grid_enabled:
            if not isinstance(signal_exit_thresholds_raw, list) or not signal_exit_thresholds_raw:
                return jsonify({"success": False, "error": "signal_exit_thresholds[] required when exit-threshold grid is enabled"}), 400
            parsed_exit_thresholds: List[float] = []
            for raw_thr in signal_exit_thresholds_raw:
                try:
                    thr = float(raw_thr)
                except Exception:
                    return jsonify({"success": False, "error": f"bad signal_exit_threshold value: {raw_thr!r}"}), 400
                if not (0.0 < thr < 1.0):
                    return jsonify({"success": False, "error": f"signal_exit_threshold must be between 0 and 1: {thr}"}), 400
                parsed_exit_thresholds.append(round(thr, 6))
            signal_exit_thr_list = list(dict.fromkeys(parsed_exit_thresholds))
        elif signal_exit_threshold_raw not in (None, ""):
            try:
                single_thr = float(signal_exit_threshold_raw)
            except Exception:
                return jsonify({"success": False, "error": f"bad signal_exit_threshold value: {signal_exit_threshold_raw!r}"}), 400
            if not (0.0 < single_thr < 1.0):
                return jsonify({"success": False, "error": f"signal_exit_threshold must be between 0 and 1: {single_thr}"}), 400
            signal_exit_thr_list = [round(single_thr, 6)]
    allowed_exit = ("policy", "hold_steps", "atr_trail")
    # New UI: exit_modes[] grid
    exit_list: List[str] = []
    if isinstance(exit_modes, list) and exit_modes:
        for em in exit_modes:
            s = str(em or "").strip().lower()
            if not s:
                continue
            if s not in allowed_exit:
                return jsonify({"success": False, "error": f"bad exit_modes value: {em!r}"}), 400
            exit_list.append(s)
        exit_list = sorted(set(exit_list))
    else:
        # Backward compatible single exit_mode
        if exit_mode not in allowed_exit:
            exit_mode = "policy"
        exit_list = [exit_mode]
    try:
        atr_len = int(atr_len) if atr_len not in (None, "") else None
    except Exception:
        atr_len = None
    try:
        atr_mult = float(atr_mult) if atr_mult not in (None, "") else None
    except Exception:
        atr_mult = None
    try:
        entry_tp_pct = float(entry_tp_pct) if (override_exit_policy and entry_tp_pct not in (None, "")) else None
    except Exception:
        entry_tp_pct = None
    try:
        entry_sl_pct = float(entry_sl_pct) if (override_exit_policy and entry_sl_pct not in (None, "")) else None
    except Exception:
        entry_sl_pct = None
    try:
        entry_trail_pct = float(entry_trail_pct) if (override_exit_policy and entry_trail_pct not in (None, "")) else None
    except Exception:
        entry_trail_pct = None
    try:
        signal_exit_window = int(signal_exit_window) if (signal_exit_enabled and signal_exit_window not in (None, "")) else 20
    except Exception:
        signal_exit_window = 20
    signal_exit_window = max(1, min(int(signal_exit_window or 20), 500))
    try:
        signal_exit_start_pct = float(signal_exit_start_pct) if (signal_exit_enabled and signal_exit_start_pct not in (None, "")) else 0.65
    except Exception:
        signal_exit_start_pct = 0.65
    if signal_exit_start_pct > 1.0:
        signal_exit_start_pct = signal_exit_start_pct / 100.0
    signal_exit_start_pct = max(0.0, min(float(signal_exit_start_pct or 0.65), 1.0))

    # Normalize signs for exit-policy overrides:
    # - TP is a positive return threshold
    # - SL is a negative return threshold
    # - Trail is a positive drawdown-from-peak threshold
    if override_exit_policy:
        try:
            if entry_tp_pct is not None and float(entry_tp_pct) < 0:
                entry_tp_pct = abs(float(entry_tp_pct))
        except Exception:
            entry_tp_pct = None
        try:
            if entry_sl_pct is not None and float(entry_sl_pct) > 0:
                entry_sl_pct = -abs(float(entry_sl_pct))
        except Exception:
            entry_sl_pct = None
        try:
            if entry_trail_pct is not None and float(entry_trail_pct) < 0:
                entry_trail_pct = abs(float(entry_trail_pct))
        except Exception:
            entry_trail_pct = None

    days_list: List[int] = []
    if isinstance(days_grid, list) and days_grid:
        allowed = {30, 60, 90}
        for d in days_grid:
            try:
                di = int(d)
            except Exception:
                return jsonify({"success": False, "error": f"bad days_grid value: {d!r}"}), 400
            if di not in allowed:
                return jsonify({"success": False, "error": f"days_grid supports only 30/60/90 (got {di})"}), 400
            days_list.append(di)
        days_list = sorted(set(days_list))
    else:
        days_list = [int(days_single)]

    from tasks.xgb_oos_tasks import _oos_end_ts_ms
    batch_oos_end_ts = _oos_end_ts_ms()

    # Prefetch once per symbol before enqueueing parallel OOS tasks.
    pre = _prefetch_oos_to_db(result_dirs=[str(x or "").strip() for x in result_dirs], days_list=days_list)
    if not bool(pre.get("success")):
        return jsonify({"success": False, "error": pre.get("error") or "prefetch failed", "prefetch": pre}), 400

    tasks: List[Dict[str, str]] = []
    batch_id = new_batch_id()
    ts = datetime.utcnow().isoformat() + "Z"
    for rd in result_dirs:
        rd = str(rd).strip()
        if not rd:
            continue
        for days in days_list:
            for em in exit_list:
                for pe_thr in p_enter_list:
                    for sig_thr in signal_exit_thr_list:
                        t = celery.send_task(
                            "tasks.xgb_oos_tasks.run_xgb_oos_test",
                            kwargs={
                                "result_dir": rd,
                                "days": int(days),
                                "ts": ts,
                                "exit_mode": em,
                                "atr_len": atr_len,
                                "atr_mult": atr_mult,
                                "override_exit_policy": bool(override_exit_policy),
                                "entry_tp_pct": entry_tp_pct,
                                "entry_sl_pct": entry_sl_pct,
                                "entry_trail_pct": entry_trail_pct,
                                "p_enter_threshold": pe_thr,
                                "direction_override": direction_override,
                                "signal_exit_enabled": bool(signal_exit_enabled),
                                "signal_exit_window": int(signal_exit_window),
                                "signal_exit_start_pct": float(signal_exit_start_pct),
                                "signal_exit_threshold": sig_thr,
                                "batch_id": batch_id,
                                "oos_end_ts_ms": batch_oos_end_ts,
                            },
                            queue="oos",
                        )
                        tasks.append({
                            "result_dir": rd,
                            "days": int(days),
                            "exit_mode": em,
                            "p_enter_threshold": pe_thr,
                            "signal_exit_threshold": sig_thr,
                            "task_id": t.id,
                        })
    if not tasks:
        return jsonify({"success": False, "error": "no tasks enqueued"}), 400
    register_batch(
        batch_id,
        len(tasks),
        meta={
            "ts": ts,
            "days_grid": days_list,
            "exit_modes": exit_list,
            "p_enter_thresholds": p_enter_list,
            "signal_exit_thresholds": signal_exit_thr_list,
        },
        task_ids=[t["task_id"] for t in tasks],
    )
    return jsonify({
        "success": True,
        "batch_id": batch_id,
        "tasks": tasks,
        "total": len(tasks),
        "days_grid": days_list,
        "exit_modes": exit_list,
        "p_enter_thresholds": p_enter_list,
        "signal_exit_thresholds": signal_exit_thr_list,
        "prefetch": pre.get("prefetch", []),
    })


# ── CSV export / download ─────────────────────────────────────────────


@xgb_oos_bp.get("/xgb_oos_batch_status")
def xgb_oos_batch_status():
    batch_id = (request.args.get("batch_id") or "").strip()
    if not batch_id:
        return jsonify({"success": False, "error": "batch_id required"}), 400
    return jsonify(get_batch_status(batch_id))


@xgb_oos_bp.get("/xgb_oos_batch_results")
def xgb_oos_batch_results():
    batch_id = (request.args.get("batch_id") or "").strip()
    if not batch_id:
        return jsonify({"success": False, "error": "batch_id required"}), 400
    offset_raw = request.args.get("offset")
    limit_raw = request.args.get("limit")
    try:
        offset = int(offset_raw) if offset_raw not in (None, "") else 0
    except Exception:
        offset = 0
    try:
        limit = int(limit_raw) if limit_raw not in (None, "") else 500
    except Exception:
        limit = 500
    return jsonify(get_batch_results(batch_id=batch_id, offset=offset, limit=limit))


@xgb_oos_bp.get("/xgb_oos_recent_batches")
def xgb_oos_recent_batches():
    limit_raw = request.args.get("limit")
    try:
        limit = int(limit_raw) if limit_raw not in (None, "") else 10
    except Exception:
        limit = 10
    from utils.xgb_oos_batch_csv import get_recent_batches
    return jsonify({"success": True, "batches": get_recent_batches(limit)})


@xgb_oos_bp.post("/xgb_oos_batch_save_csv")
def xgb_oos_batch_save_csv():
    """Legacy/manual: сохранить CSV из переданных results[] (UI больше не использует)."""
    data = request.get_json(silent=True) or {}
    results = data.get("results", [])
    if not isinstance(results, list) or not results:
        return jsonify({"success": False, "error": "results[] required"}), 400
    out = write_oos_batch_csv(results)
    if not out.get("success"):
        return jsonify(out), 400
    return jsonify(out)


@xgb_oos_bp.post("/xgb_oos_batch_force_finalize")
def xgb_oos_batch_force_finalize():
    """Явно сохранить CSV из уже собранных success results (partial)."""
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "json body required"}), 400
    batch_id = str(data.get("batch_id") or "").strip()
    confirm = str(data.get("confirm") or "").strip()
    reason = str(data.get("reason") or "partial save").strip()
    if not batch_id:
        return jsonify({"success": False, "error": "batch_id required"}), 400
    if confirm != "FORCE_PARTIAL":
        return jsonify({"success": False, "error": "confirm=FORCE_PARTIAL required"}), 400
    out = force_finalize_batch_csv(batch_id=batch_id, reason=reason)
    if not out.get("success"):
        return jsonify(out), 400
    return jsonify(out)


@xgb_oos_bp.post("/xgb_oos_batch_cancel")
def xgb_oos_batch_cancel():
    """Отменить запущенный batch: отозвать задачи Celery и пометить статус как cancelled."""
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "json body required"}), 400
    batch_id = str(data.get("batch_id") or "").strip()
    if not batch_id:
        return jsonify({"success": False, "error": "batch_id required"}), 400
    from utils.xgb_oos_batch_csv import cancel_batch_csv
    out = cancel_batch_csv(batch_id=batch_id)
    if not out.get("success"):
        return jsonify(out), 400
    return jsonify(out)


@xgb_oos_bp.post("/xgb_oos_save_experiment")
def xgb_oos_save_experiment():
    """Сохраняет snapshot OOS эксперимента: active preset + выбранные результаты."""
    data = request.get_json(silent=True) or {}
    results = data.get("results", [])
    if not isinstance(results, list) or not results:
        return jsonify({"success": False, "error": "results[] required"}), 400

    valid_results = [item for item in results if isinstance(item, dict) and item.get("success") is True]
    if not valid_results:
        return jsonify({"success": False, "error": "no successful results"}), 400
    if not _ACTIVE_XGB_PRESET.exists():
        return jsonify({"success": False, "error": f"active preset not found: {_ACTIVE_XGB_PRESET.as_posix()}"}), 404

    symbol = str(data.get("symbol") or valid_results[0].get("symbol") or "XGB")
    name = _safe_slug(str(data.get("experiment_name") or "oos_experiment"))
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
    experiment_dir = _EXPERIMENTS_DIR / _symbol_dir_name(symbol) / f"{ts}_{name}"
    experiment_dir.mkdir(parents=True, exist_ok=False)

    preset_path = experiment_dir / "preset.json"
    summary_path = experiment_dir / "summary.json"
    shutil.copy2(_ACTIVE_XGB_PRESET, preset_path)

    models = [_serialize_oos_experiment_model(item) for item in valid_results]
    summary = {
        "schema_version": 1,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "symbol": symbol,
        "experiment_name": name,
        "source_preset": _ACTIVE_XGB_PRESET.as_posix(),
        "oos_csv": data.get("oos_csv"),
        "selection_metric": data.get("selection_metric") or "manual",
        "selected_count": len(models),
        "selected_models": models,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return jsonify({
        "success": True,
        "path": experiment_dir.as_posix(),
        "preset": preset_path.as_posix(),
        "summary": summary_path.as_posix(),
        "selected_count": len(models),
    })


@xgb_oos_bp.post("/xgb_oos_save_top_csv_experiment")
def xgb_oos_save_top_csv_experiment():
    """Сохраняет top-N моделей из выбранного OOS CSV как experiment snapshot."""
    data = request.get_json(silent=True) or {}
    filename = str(data.get("filename") or "").strip()
    if not filename:
        return jsonify({"success": False, "error": "filename required"}), 400
    if not _ACTIVE_XGB_PRESET.exists():
        return jsonify({"success": False, "error": f"active preset not found: {_ACTIVE_XGB_PRESET.as_posix()}"}), 404

    safe = _CSV_DIR.resolve()
    target = (safe / filename).resolve()
    if safe not in target.parents and target.parent != safe:
        return jsonify({"success": False, "error": "forbidden"}), 403
    if not target.exists() or not target.is_file():
        return jsonify({"success": False, "error": "csv not found"}), 404

    try:
        top_n = max(1, min(int(data.get("top_n") or 3), 50))
    except (TypeError, ValueError):
        top_n = 3

    rows: List[Dict[str, Any]] = []
    with target.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            roi = _float_or_none(row.get("roi_pct"))
            trades = _int_or_none(row.get("trades_count")) or 0
            if roi is None or trades <= 0:
                continue
            rows.append(row)

    if not rows:
        return jsonify({"success": False, "error": "csv has no rows with roi_pct and trades_count > 0"}), 400

    rows.sort(key=lambda item: _float_or_none(item.get("roi_pct")) or float("-inf"), reverse=True)
    selected_rows = rows[:top_n]
    symbol = str(data.get("symbol") or selected_rows[0].get("symbol") or "XGB")
    name = _safe_slug(str(data.get("experiment_name") or f"top_{top_n}_from_csv"))
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
    experiment_dir = _EXPERIMENTS_DIR / _symbol_dir_name(symbol) / f"{ts}_{name}"
    experiment_dir.mkdir(parents=True, exist_ok=False)

    preset_path = experiment_dir / "preset.json"
    summary_path = experiment_dir / "summary.json"
    shutil.copy2(_ACTIVE_XGB_PRESET, preset_path)

    models = [_serialize_oos_experiment_csv_row(row) for row in selected_rows]
    summary = {
        "schema_version": 1,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "symbol": symbol,
        "experiment_name": name,
        "source_preset": _ACTIVE_XGB_PRESET.as_posix(),
        "oos_csv": target.as_posix(),
        "selection_metric": "roi_pct",
        "selected_count": len(models),
        "selected_models": models,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return jsonify({
        "success": True,
        "path": experiment_dir.as_posix(),
        "preset": preset_path.as_posix(),
        "summary": summary_path.as_posix(),
        "selected_count": len(models),
    })


@xgb_oos_bp.get("/xgb_oos_csv_list")
def xgb_oos_csv_list():
    """Список сохранённых CSV файлов."""
    files: List[Dict[str, Any]] = []
    if _CSV_DIR.exists():
        for f in sorted(_CSV_DIR.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True):
            files.append({"name": f.name, "size": f.stat().st_size,
                          "mtime": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")})
    return jsonify({"success": True, "files": files})


@xgb_oos_bp.get("/xgb_oos_csv_download/<filename>")
def xgb_oos_csv_download(filename: str):
    """Скачивание CSV файла."""
    safe = _CSV_DIR.resolve()
    target = (safe / filename).resolve()
    if safe not in target.parents and target.parent != safe:
        abort(403)
    if not target.exists():
        abort(404)
    return send_file(str(target), mimetype="text/csv", as_attachment=True, download_name=filename)


@xgb_oos_bp.post("/xgb_oos_csv_delete")
def xgb_oos_csv_delete():
    """Удаление CSV файла."""
    data = request.get_json(silent=True) or {}
    filename = str(data.get("filename") or "").strip()
    if not filename:
        return jsonify({"success": False, "error": "filename required"}), 400
    safe = _CSV_DIR.resolve()
    target = (safe / filename).resolve()
    if safe not in target.parents and target.parent != safe:
        return jsonify({"success": False, "error": "forbidden"}), 403
    if not target.exists():
        return jsonify({"success": False, "error": "not found"}), 404
    try:
        target.unlink()
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    return jsonify({"success": True, "deleted": filename})


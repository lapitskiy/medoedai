"""Server-side XGB OOS batch CSV aggregation (Redis + Celery finalize)."""
from __future__ import annotations

import csv
import io
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.redis_utils import get_redis_client

logger = logging.getLogger(__name__)

CSV_DIR = Path("predict_test") / "xgb_oos"
BATCH_TTL_SEC = 3600 * 24 * 7

CSV_FIELDS = [
    "run_dir", "symbol", "task", "direction", "days",
    "exit_mode", "atr_len", "atr_mult",
    "signal_exit_enabled", "signal_exit_window", "signal_exit_start_pct", "signal_exit_start_step",
    "signal_exit_threshold",
    "horizon_steps", "threshold", "p_enter_thr", "max_hold_steps", "min_profit", "fee_bps",
    "n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
    "reg_lambda", "min_child_weight", "gamma", "scale_pos_weight",
    "val_acc", "f1_buy_sell", "f1_1", "prec_1", "recall_1",
    "y_non_hold", "pred_non_hold",
    "pnl_total", "roi_pct", "winrate", "profit_factor", "max_dd",
    "trades_count", "wins", "losses", "avg_trade_pnl", "avg_bars_held",
    "equity_end",
    "exit_timeout_cnt", "exit_trail_cnt", "exit_tp_cnt", "exit_sl_cnt", "exit_signal_cnt",
    "exit_weak_signal_cnt", "exit_eof_cnt", "exit_atr_cnt",
    "exit_timeout_share", "exit_trail_share", "exit_tp_share", "exit_sl_share",
    "exit_signal_share", "exit_weak_signal_share", "exit_eof_share", "exit_atr_share",
    "error",
]


def _batch_key(batch_id: str, suffix: str) -> str:
    return f"xgb_oos:batch:{batch_id}:{suffix}"


def register_batch(batch_id: str, expected: int, meta: Optional[Dict[str, Any]] = None, task_ids: Optional[List[str]] = None) -> None:
    r = get_redis_client()
    payload = {
        "batch_id": batch_id,
        "expected": int(expected),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "status": "running",
        **(meta or {}),
    }
    pipe = r.pipeline()
    pipe.set(_batch_key(batch_id, "meta"), json.dumps(payload, ensure_ascii=False))
    pipe.set(_batch_key(batch_id, "expected"), str(int(expected)))
    pipe.set(_batch_key(batch_id, "done"), "0")
    pipe.set(_batch_key(batch_id, "success"), "0")
    pipe.set(_batch_key(batch_id, "last_activity"), str(datetime.utcnow().timestamp()))
    pipe.delete(_batch_key(batch_id, "results"))
    pipe.delete(_batch_key(batch_id, "csv"))
    pipe.delete(_batch_key(batch_id, "finalizing"))
    pipe.delete(_batch_key(batch_id, "task_ids"))
    if task_ids:
        for i in range(0, len(task_ids), 500):
            pipe.rpush(_batch_key(batch_id, "task_ids"), *task_ids[i:i+500])
    for k in ("meta", "expected", "done", "success", "results", "csv", "finalizing", "last_activity", "task_ids"):
        pipe.expire(_batch_key(batch_id, k), BATCH_TTL_SEC)
    pipe.execute()


def oos_result_to_csv_row(res: Dict[str, Any]) -> Dict[str, Any]:
    cfg = res.get("cfg_snapshot") or {}
    m = res.get("oos_metrics") or {}
    bt = res.get("backtest") or {}
    signal_exit = bt.get("signal_exit") if isinstance(bt.get("signal_exit"), dict) else {}
    f1v = m.get("f1_val")
    pv = m.get("precision_val")
    rv = m.get("recall_val")
    try:
        thr_pct = float(cfg.get("threshold")) * 100 if cfg.get("threshold") is not None else None
    except (TypeError, ValueError):
        thr_pct = None
    return {
        "run_dir": res.get("run_dir", ""),
        "symbol": res.get("symbol", ""),
        "task": res.get("task", ""),
        "direction": res.get("direction", ""),
        "days": res.get("days", ""),
        "exit_mode": res.get("exit_mode", ""),
        "atr_len": res.get("atr_len", ""),
        "atr_mult": res.get("atr_mult", ""),
        "signal_exit_enabled": res.get("signal_exit_enabled", signal_exit.get("enabled")),
        "signal_exit_window": res.get("signal_exit_window", signal_exit.get("window")),
        "signal_exit_start_pct": (
            float(res.get("signal_exit_start_pct")) * 100.0
            if res.get("signal_exit_start_pct") not in (None, "")
            else (
                float(signal_exit.get("start_pct")) * 100.0
                if signal_exit.get("start_pct") not in (None, "")
                else None
            )
        ),
        "signal_exit_start_step": res.get("signal_exit_start_step", signal_exit.get("start_step")),
        "signal_exit_threshold": (
            res.get("signal_exit_threshold")
            if res.get("signal_exit_threshold") not in (None, "")
            else signal_exit.get("exit_threshold")
        ),
        "horizon_steps": cfg.get("horizon_steps"),
        "threshold": thr_pct,
        "p_enter_thr": cfg.get("p_enter_threshold"),
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
        "val_acc": m.get("val_acc"),
        "f1_buy_sell": m.get("f1_buy_sell_val"),
        "f1_1": f1v[1] if isinstance(f1v, list) and len(f1v) > 1 else None,
        "prec_1": pv[1] if isinstance(pv, list) and len(pv) > 1 else None,
        "recall_1": rv[1] if isinstance(rv, list) and len(rv) > 1 else None,
        "y_non_hold": m.get("y_non_hold_rate_val"),
        "pred_non_hold": m.get("pred_non_hold_rate_val"),
        "pnl_total": bt.get("pnl_total"),
        "roi_pct": bt.get("roi_pct"),
        "winrate": bt.get("winrate"),
        "profit_factor": bt.get("profit_factor"),
        "max_dd": bt.get("max_dd"),
        "trades_count": bt.get("trades_count"),
        "wins": bt.get("wins"),
        "losses": bt.get("losses"),
        "avg_trade_pnl": bt.get("avg_trade_pnl"),
        "avg_bars_held": bt.get("avg_bars_held"),
        "equity_end": bt.get("equity_end"),
        "exit_timeout_cnt": (bt.get("reason_counts", {}) or {}).get("timeout"),
        "exit_trail_cnt": (bt.get("reason_counts", {}) or {}).get("trail"),
        "exit_tp_cnt": (bt.get("reason_counts", {}) or {}).get("tp"),
        "exit_sl_cnt": (bt.get("reason_counts", {}) or {}).get("sl"),
        "exit_signal_cnt": (bt.get("reason_counts", {}) or {}).get("signal"),
        "exit_weak_signal_cnt": (bt.get("reason_counts", {}) or {}).get("weak_signal_avg"),
        "exit_eof_cnt": (bt.get("reason_counts", {}) or {}).get("eof"),
        "exit_atr_cnt": (bt.get("reason_counts", {}) or {}).get("atr_trail"),
        "exit_timeout_share": (bt.get("reason_share", {}) or {}).get("timeout"),
        "exit_trail_share": (bt.get("reason_share", {}) or {}).get("trail"),
        "exit_tp_share": (bt.get("reason_share", {}) or {}).get("tp"),
        "exit_sl_share": (bt.get("reason_share", {}) or {}).get("sl"),
        "exit_signal_share": (bt.get("reason_share", {}) or {}).get("signal"),
        "exit_weak_signal_share": (bt.get("reason_share", {}) or {}).get("weak_signal_avg"),
        "exit_eof_share": (bt.get("reason_share", {}) or {}).get("eof"),
        "exit_atr_share": (bt.get("reason_share", {}) or {}).get("atr_trail"),
        "error": res.get("error", ""),
    }


def write_oos_batch_csv(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for res in results:
        if not isinstance(res, dict):
            continue
        rows.append(oos_result_to_csv_row(res))
    if not rows:
        return {"success": False, "error": "no valid results"}

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sym = rows[0].get("symbol", "XGB")
    dir_set = sorted({str(r.get("direction") or "").strip().lower() for r in rows if str(r.get("direction") or "").strip()})
    direction = dir_set[0] if len(dir_set) == 1 else ("mixed" if dir_set else "na")
    safe_direction = "".join(ch for ch in direction if ch.isalnum() or ch in ("_", "-"))[:16] or "na"
    em_set = sorted({str(r.get("exit_mode") or "").strip().lower() for r in rows if str(r.get("exit_mode") or "").strip()})
    em = em_set[0] if len(em_set) == 1 else ("mixed" if em_set else "na")
    safe_em = "".join(ch for ch in em if ch.isalnum() or ch in ("_", "-"))[:32] or "na"
    filename = f"xgb_oos_{sym}_{safe_direction}_{safe_em}_{ts_str}.csv"
    csv_path = CSV_DIR / filename

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, extrasaction="ignore", restval="")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    csv_path.write_text(buf.getvalue(), encoding="utf-8")
    return {"success": True, "filename": filename, "rows": len(rows)}


def on_batch_child_task_done(batch_id: str, result: Dict[str, Any]) -> None:
    r = get_redis_client()
    if not r.exists(_batch_key(batch_id, "expected")):
        return
    r.set(_batch_key(batch_id, "last_activity"), str(datetime.utcnow().timestamp()), ex=BATCH_TTL_SEC)

    try:
        r.rpush(_batch_key(batch_id, "results"), json.dumps(result, ensure_ascii=False))
        if result.get("success"):
            r.incr(_batch_key(batch_id, "success"))
    except Exception:
        logger.exception("xgb oos batch: failed to store result batch_id=%s", batch_id)

    done = int(r.incr(_batch_key(batch_id, "done")))
    expected = int(r.get(_batch_key(batch_id, "expected")) or "0")
    if expected <= 0:
        return

    if done < expected:
        return

    if not r.set(_batch_key(batch_id, "finalizing"), "1", nx=True, ex=600):
        return

    try:
        from tasks.xgb_oos_tasks import finalize_xgb_oos_batch_csv

        finalize_xgb_oos_batch_csv.delay(batch_id)
    except Exception:
        logger.exception("xgb oos batch: finalize enqueue failed batch_id=%s", batch_id)
        r.delete(_batch_key(batch_id, "finalizing"))


def finalize_batch_csv(batch_id: str, error_status: str = None) -> Dict[str, Any]:
    r = get_redis_client()
    key = _batch_key(batch_id, "results")
    total = int(r.llen(key) or 0)
    results: List[Dict[str, Any]] = []
    
    for i in range(0, total, 500):
        raw_items = r.lrange(key, i, i + 499)
        for raw in raw_items:
            try:
                item = json.loads(raw)
                if isinstance(item, dict):
                    results.append(item)
            except Exception:
                continue

    meta_raw = r.get(_batch_key(batch_id, "meta"))
    meta: Dict[str, Any] = {}
    if meta_raw:
        try:
            meta = json.loads(meta_raw)
        except Exception:
            meta = {}

    if not results:
        meta["status"] = error_status or "failed"
        meta["error"] = "no successful OOS results in batch"
        r.set(_batch_key(batch_id, "meta"), json.dumps(meta, ensure_ascii=False))
        r.expire(_batch_key(batch_id, "meta"), BATCH_TTL_SEC)
        return {"success": False, "batch_id": batch_id, "error": meta["error"]}

    out = write_oos_batch_csv(results)
    if not out.get("success"):
        meta["status"] = error_status or "failed"
        meta["error"] = out.get("error") or "csv write failed"
        r.set(_batch_key(batch_id, "meta"), json.dumps(meta, ensure_ascii=False))
        return {"success": False, "batch_id": batch_id, **out}

    filename = str(out["filename"])
    meta["status"] = error_status or "done"
    meta["csv_filename"] = filename
    meta["csv_rows"] = int(out.get("rows") or 0)
    meta["finished_at"] = datetime.utcnow().isoformat() + "Z"
    r.set(_batch_key(batch_id, "meta"), json.dumps(meta, ensure_ascii=False))
    r.set(_batch_key(batch_id, "csv"), filename)
    for k in ("meta", "csv"):
        r.expire(_batch_key(batch_id, k), BATCH_TTL_SEC)
    r.delete(_batch_key(batch_id, "finalizing"))
    return {"success": True, "batch_id": batch_id, **out}


def get_batch_status(batch_id: str) -> Dict[str, Any]:
    r = get_redis_client()
    if not batch_id or not r.exists(_batch_key(batch_id, "expected")):
        return {"success": False, "error": "batch not found"}

    expected = int(r.get(_batch_key(batch_id, "expected")) or "0")
    done = int(r.get(_batch_key(batch_id, "done")) or "0")
    success_count = int(r.get(_batch_key(batch_id, "success")) or "0")
    csv_filename = r.get(_batch_key(batch_id, "csv"))
    last_activity_raw = r.get(_batch_key(batch_id, "last_activity"))
    status = "running"
    meta: Dict[str, Any] = {}
    meta_raw = r.get(_batch_key(batch_id, "meta"))
    if meta_raw:
        try:
            meta = json.loads(meta_raw)
            status = str(meta.get("status") or status)
        except Exception:
            pass

    if csv_filename:
        # if meta says cancelled/partial_timeout, keep that, else done
        if status not in ("cancelled", "partial_timeout", "failed"):
            status = "done"
    elif status == "running" and done >= expected > 0 and r.get(_batch_key(batch_id, "finalizing")):
        status = "writing"
    elif status == "running":
        # check if stalled
        try:
            if last_activity_raw and (datetime.utcnow().timestamp() - float(last_activity_raw)) > 1200:
                status = "stalled"
        except Exception:
            pass

    out = {
        "success": True,
        "batch_id": batch_id,
        "expected": expected,
        "done": done,
        "success_count": success_count,
        "status": status,
        "csv_filename": csv_filename,
        "csv_rows": meta.get("csv_rows"),
        "error": meta.get("error"),
        "created_at": meta.get("created_at"),
        "finished_at": meta.get("finished_at"),
    }

    # Add timing stats
    created_at_str = meta.get("created_at")
    if created_at_str and status in ("running", "writing", "stalled"):
        try:
            import time
            created_ts = datetime.fromisoformat(created_at_str.replace('Z', '+00:00')).timestamp()
            elapsed = time.time() - created_ts
            out["elapsed_sec"] = round(elapsed)
            if done > 0 and expected > 0:
                avg_per_run = elapsed / done
                out["avg_per_run_sec"] = round(avg_per_run, 1)
                out["eta_sec"] = round((expected - done) * avg_per_run)
        except Exception:
            pass

    return out


def get_batch_results(batch_id: str, offset: int = 0, limit: int = 500) -> Dict[str, Any]:
    """
    Returns successful batch results stored in Redis.
    Results are appended by on_batch_child_task_done() into list key <batch_id>:results.
    """
    r = get_redis_client()
    if not batch_id or not r.exists(_batch_key(batch_id, "expected")):
        return {"success": False, "error": "batch not found"}

    try:
        offset_i = max(0, int(offset or 0))
    except Exception:
        offset_i = 0
    try:
        limit_i = int(limit or 0)
    except Exception:
        limit_i = 500
    limit_i = max(1, min(limit_i, 2000))

    key = _batch_key(batch_id, "results")
    total = int(r.llen(key) or 0)
    if offset_i >= total:
        return {
            "success": True,
            "batch_id": batch_id,
            "offset": offset_i,
            "limit": limit_i,
            "total": total,
            "results": [],
        }

    raw_items = r.lrange(key, offset_i, offset_i + limit_i - 1)
    results: List[Dict[str, Any]] = []
    for raw in raw_items:
        try:
            item = json.loads(raw)
            if isinstance(item, dict):
                results.append(item)
        except Exception:
            continue

    return {
        "success": True,
        "batch_id": batch_id,
        "offset": offset_i,
        "limit": limit_i,
        "total": total,
        "results": results,
    }


def force_finalize_batch_csv(batch_id: str, reason: str = "snapshot") -> Dict[str, Any]:
    """
    Force-write CSV from currently collected successful results WITHOUT closing the batch.
    """
    r = get_redis_client()
    if not batch_id or not r.exists(_batch_key(batch_id, "expected")):
        return {"success": False, "error": "batch not found"}

    expected = int(r.get(_batch_key(batch_id, "expected")) or "0")
    done = int(r.get(_batch_key(batch_id, "done")) or "0")
    success_count = int(r.get(_batch_key(batch_id, "success")) or "0")

    key = _batch_key(batch_id, "results")
    total = int(r.llen(key) or 0)
    results: List[Dict[str, Any]] = []
    
    for i in range(0, total, 500):
        raw_items = r.lrange(key, i, i + 499)
        for raw in raw_items:
            try:
                item = json.loads(raw)
                if isinstance(item, dict):
                    results.append(item)
            except Exception:
                continue

    out = write_oos_batch_csv(results)
    if out.get("success"):
        old_fn = Path(CSV_DIR) / str(out["filename"])
        new_fn = old_fn.with_name(old_fn.stem + "_snapshot" + old_fn.suffix)
        old_fn.rename(new_fn)
        out["filename"] = new_fn.name

    return {**out, "snapshot": True, "reason": reason, "expected": expected, "done": done, "success_count": success_count}


def cancel_batch_csv(batch_id: str) -> Dict[str, Any]:
    """
    Cancels a running batch: revokes all queued tasks, marks meta as cancelled, and snapshots.
    """
    r = get_redis_client()
    if not batch_id or not r.exists(_batch_key(batch_id, "expected")):
        return {"success": False, "error": "batch not found"}

    from celery.execute import send_task
    try:
        from tasks import celery
    except ImportError:
        pass

    # Mark as cancelled
    meta_raw = r.get(_batch_key(batch_id, "meta"))
    if meta_raw:
        try:
            meta = json.loads(meta_raw)
            meta["status"] = "cancelled"
            meta["finished_at"] = datetime.utcnow().isoformat() + "Z"
            r.set(_batch_key(batch_id, "meta"), json.dumps(meta, ensure_ascii=False))
        except Exception:
            pass

    # Try revoking individual tasks if we have them
    task_ids = r.lrange(_batch_key(batch_id, "task_ids"), 0, -1)
    if task_ids:
        try:
            for tid in task_ids:
                if tid:
                    celery.control.revoke(tid.decode('utf-8') if isinstance(tid, bytes) else str(tid), terminate=True)
            logger.info(f"Revoked {len(task_ids)} tasks for batch {batch_id}")
        except Exception as e:
            logger.error(f"Failed to revoke tasks for batch {batch_id}: {e}")

    # Optionally take a snapshot
    force_finalize_batch_csv(batch_id, reason="cancelled")
    
    return get_batch_status(batch_id)


def get_recent_batches(limit: int = 10) -> List[Dict[str, Any]]:
    r = get_redis_client()
    batches = []
    for k in r.scan_iter("xgb_oos:batch:*:expected", count=100):
        try:
            k_str = k.decode('utf-8') if isinstance(k, bytes) else str(k)
            batch_id = k_str.split(":")[2]
        except Exception:
            continue
        st = get_batch_status(batch_id)
        if st.get("success"):
            batches.append(st)

    def parse_dt(s):
        try:
            return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()
        except:
            return 0

    batches.sort(key=lambda x: parse_dt(x.get("created_at") or ""), reverse=True)
    return batches[:limit]


def new_batch_id() -> str:
    return uuid.uuid4().hex

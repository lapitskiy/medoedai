from flask import Blueprint, render_template, request, jsonify
from pathlib import Path
import json
import shutil
import time
from typing import Any, Dict, List

from tasks import celery
from utils.redis_utils import get_redis_client

analytics_bp = Blueprint('analytics', __name__)

def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _scan_xgb_runs() -> List[Dict[str, Any]]:
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
            metrics = _safe_read_json(run_dir / "metrics.json")
            meta = _safe_read_json(run_dir / "meta.json")
            cfg = meta.get("cfg_snapshot") if isinstance(meta.get("cfg_snapshot"), dict) else {}
            try:
                mtime = run_dir.stat().st_mtime
            except Exception:
                mtime = 0.0
            out.append({
                "symbol": (manifest.get("symbol") or sym_dir.name),
                "symbol_code": sym_dir.name,
                "run_name": manifest.get("run_name") or run_dir.name,
                "direction": manifest.get("direction") or cfg.get("direction"),
                "task": manifest.get("task") or cfg.get("task") or metrics.get("task"),
                "source": manifest.get("source") or "",
                "grid_id": manifest.get("grid_id") or "",
                "horizon_steps": cfg.get("horizon_steps"),
                "threshold": cfg.get("threshold"),
                "max_hold_steps": cfg.get("max_hold_steps"),
                "min_profit": cfg.get("min_profit"),
                "metrics": metrics,
                "meta": meta,
                "result_dir": str(run_dir),
                "mtime": mtime,
            })
    out.sort(key=lambda r: float(r.get("mtime") or 0.0), reverse=True)
    return out


def _active_xgb_tasks() -> List[Dict[str, Any]]:
    """
    Returns active XGB Celery tasks with progress meta (done/total) if available.
    """
    out: List[Dict[str, Any]] = []
    by_task_id: Dict[str, Dict[str, Any]] = {}
    try:
        rc = get_redis_client()
    except Exception:
        rc = None
    try:
        insp = celery.control.inspect(timeout=1.0)
        active = insp.active() or {}
    except Exception:
        active = {}
    for hostname, items in (active or {}).items():
        if not isinstance(items, list):
            continue
        for t in items:
            try:
                if not isinstance(t, dict):
                    continue
                name = str(t.get("name") or "")
                if not name.startswith("tasks.xgb_tasks."):
                    continue
                task_id = str(t.get("id") or "")
                kwargs = t.get("kwargs") if isinstance(t.get("kwargs"), dict) else {}
                info = celery.AsyncResult(task_id).info
                meta = info if isinstance(info, dict) else {}
                done = meta.get("done")
                total = meta.get("total")
                started_at = meta.get("started_at")
                eta_sec = None
                avg_per_run_sec = None
                # Monotonic progress from Redis (works even without celery-worker restart).
                if task_id and rc is not None:
                    pkey = f"analytics:xgb:progress:{task_id}"
                    try:
                        prev_done = int(rc.hget(pkey, "done") or 0)
                        prev_total = int(rc.hget(pkey, "total") or 0)
                        cur_done = int(done or 0)
                        cur_total = int(total or 0)
                        done = max(cur_done, prev_done)
                        total = max(cur_total, prev_total)
                        prev_started = rc.hget(pkey, "started_at")
                        if started_at is None and prev_started:
                            started_at = float(prev_started)
                        if started_at is not None:
                            try:
                                started_at = min(float(started_at), float(prev_started)) if prev_started else float(started_at)
                            except Exception:
                                pass
                        rc.hset(pkey, mapping={
                            "done": int(done or 0),
                            "total": int(total or 0),
                            "started_at": float(started_at) if started_at is not None else "",
                        })
                        rc.expire(pkey, 7 * 24 * 3600)
                    except Exception:
                        pass
                try:
                    d_i = int(done) if done is not None else 0
                    t_i = int(total) if total is not None else 0
                    sa_f = float(started_at) if started_at is not None else 0.0
                    if sa_f > 0 and d_i > 0 and t_i > d_i:
                        elapsed = max(0.0, time.time() - sa_f)
                        avg_per_run_sec = elapsed / max(1, d_i)
                        eta_sec = int(max(0.0, (t_i - d_i) * avg_per_run_sec))
                except Exception:
                    eta_sec = None
                    avg_per_run_sec = None

                row = {
                    "task_id": task_id,
                    "name": name,
                    "hostname": hostname,
                    "symbol": meta.get("symbol") or kwargs.get("symbol"),
                    "task": meta.get("task") or kwargs.get("task"),
                    "direction": kwargs.get("direction"),
                    "done": done,
                    "total": total,
                    "started_at": started_at,
                    "eta_sec": eta_sec,
                    "avg_per_run_sec": avg_per_run_sec,
                    "workers_seen": 1,
                }
                if not task_id:
                    out.append(row)
                    continue
                prev = by_task_id.get(task_id)
                if prev is None:
                    by_task_id[task_id] = row
                else:
                    # Same task_id can be visible in several worker slots (re-delivery/stale active entries).
                    # Keep a single row and preserve max progress so UI does not jump backwards.
                    prev["workers_seen"] = int(prev.get("workers_seen", 1)) + 1
                    try:
                        prev["done"] = max(int(prev.get("done") or 0), int(done or 0))
                    except Exception:
                        pass
                    try:
                        prev["total"] = max(int(prev.get("total") or 0), int(total or 0))
                    except Exception:
                        pass
                    # Recompute ETA after merged max(done/total).
                    try:
                        d_i = int(prev.get("done") or 0)
                        t_i = int(prev.get("total") or 0)
                        sa_f = float(prev.get("started_at") or 0.0)
                        if sa_f > 0 and d_i > 0 and t_i > d_i:
                            elapsed = max(0.0, time.time() - sa_f)
                            avg_per_run_sec = elapsed / max(1, d_i)
                            prev["avg_per_run_sec"] = avg_per_run_sec
                            prev["eta_sec"] = int(max(0.0, (t_i - d_i) * avg_per_run_sec))
                    except Exception:
                        pass
            except Exception:
                continue
    out.extend(by_task_id.values())
    return out


@analytics_bp.route('/analitika')
def analytics_page():
    """Страница аналитики результатов обучения"""
    return render_template('analitika/index.html')


@analytics_bp.route('/analitika/xgb')
def analytics_xgb_page():
    """Статистика по XGB моделям (result/xgb/*/runs/*)."""
    runs = _scan_xgb_runs()
    active_xgb = _active_xgb_tasks()
    return render_template('analitika/xgb.html', runs=runs, active_xgb=active_xgb)


@analytics_bp.get('/analitika/xgb/manifest')
def analytics_xgb_view_manifest():
    """
    Render a single XGB run manifest.json by path (read-only).
    Security: only allows reading result/xgb/<SYM>/runs/<RUN>/manifest.json.
    """
    raw_path = (request.args.get("path") or "").strip()
    if not raw_path:
        return jsonify({"success": False, "error": "path required"}), 400

    base = (Path("result") / "xgb").resolve()
    target = Path(raw_path).resolve()
    if base not in target.parents:
        return jsonify({"success": False, "error": "path outside result/xgb"}), 400
    if target.name != "manifest.json":
        return jsonify({"success": False, "error": "only manifest.json is allowed"}), 400
    if target.parent.name != "runs" and target.parent.parent.name != "runs":
        # expected .../runs/<run>/manifest.json
        return jsonify({"success": False, "error": "not a run manifest path"}), 400
    data = _safe_read_json(target)
    if not data:
        return jsonify({"success": False, "error": "manifest not found or unreadable"}), 404
    return render_template("analitika/xgb_manifest.html", path=str(target), manifest=data)


@analytics_bp.post('/analitika/xgb/delete_runs')
def analytics_xgb_delete_runs():
    """
    Deletes selected XGB run directories.
    Security: only allows deleting paths under result/xgb/<SYM>/runs/<RUN>.
    """
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
            # Must be inside base
            if base not in target.parents:
                raise ValueError("path outside result/xgb")
            # Must match .../runs/<run_id>
            if target.name == "runs" or target.parent.name != "runs":
                raise ValueError("not a run directory (expected .../runs/<run>)")
            if not target.exists() or not target.is_dir():
                raise ValueError("path does not exist or not a directory")
            # Extra guard: must contain manifest.json
            if not (target / "manifest.json").exists():
                raise ValueError("manifest.json not found in run dir")
            shutil.rmtree(target)
            deleted.append(str(target))
        except Exception as e:
            errors.append({"path": str(p), "error": str(e)})

    return jsonify({"success": True, "deleted": deleted, "errors": errors, "deleted_count": len(deleted)})

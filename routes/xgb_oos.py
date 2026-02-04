from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from celery.result import AsyncResult  # type: ignore
from flask import Blueprint, jsonify, render_template, request  # type: ignore

from tasks import celery  # type: ignore


xgb_oos_bp = Blueprint("xgb_oos", __name__)


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
    """
    Сканирует result/xgb/*/runs/* и возвращает список запусков для выбора в UI.
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
            cfg = meta.get("cfg_snapshot") if isinstance(meta.get("cfg_snapshot"), dict) else {}
            try:
                mtime = float(run_dir.stat().st_mtime)
            except Exception:
                mtime = 0.0

            # Model path is stored in manifest for XGB
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
                }
            )
    out.sort(key=lambda r: float(r.get("mtime") or 0.0), reverse=True)
    return out


@xgb_oos_bp.get("/oos_xgb")
def xgb_oos_page():
    runs = _scan_xgb_runs()
    return render_template("oos/xgb_oos.html", xgb_runs=runs)


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
        days = max(1, min(days, 365))
        if not result_dir:
            return jsonify({"success": False, "error": "result_dir required"}), 400
        # Send by name to avoid import cycles
        task = celery.send_task(
            "tasks.xgb_oos_tasks.run_xgb_oos_test",
            kwargs={"result_dir": result_dir, "days": days, "ts": datetime.utcnow().isoformat() + "Z"},
            queue="celery",
        )
        return jsonify({"success": True, "task_id": task.id})
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
        if ar.state in ("SUCCESS", "FAILURE"):
            try:
                res = ar.result
                if isinstance(res, dict):
                    resp["result"] = res
            except Exception:
                pass
        return jsonify(resp)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


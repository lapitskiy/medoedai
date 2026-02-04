from flask import Blueprint, render_template, request, jsonify
from pathlib import Path
import json
import shutil
from typing import Any, Dict, List

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


@analytics_bp.route('/analitika')
def analytics_page():
    """Страница аналитики результатов обучения"""
    return render_template('analitika/index.html')


@analytics_bp.route('/analitika/xgb')
def analytics_xgb_page():
    """Статистика по XGB моделям (result/xgb/*/runs/*)."""
    runs = _scan_xgb_runs()
    return render_template('analitika/xgb.html', runs=runs)


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

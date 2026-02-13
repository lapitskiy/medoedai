from __future__ import annotations

import csv
import io
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from celery.result import AsyncResult  # type: ignore
from flask import Blueprint, jsonify, render_template, request, send_file, abort  # type: ignore

from tasks import celery  # type: ignore

xgb_oos_bp = Blueprint("xgb_oos", __name__)

_CSV_DIR = Path("predict_test") / "xgb_oos"


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
            queue="oos",
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


# ── Batch OOS ──────────────────────────────────────────────────────────

@xgb_oos_bp.post("/xgb_oos_batch_async")
def xgb_oos_batch_async():
    """Запускает OOS для нескольких run'ов параллельно через Celery."""
    data = request.get_json(silent=True) or {}
    result_dirs = data.get("result_dirs", [])
    days = max(1, min(int(data.get("days") or 30), 365))
    if not isinstance(result_dirs, list) or not result_dirs:
        return jsonify({"success": False, "error": "result_dirs[] required"}), 400

    tasks: List[Dict[str, str]] = []
    ts = datetime.utcnow().isoformat() + "Z"
    for rd in result_dirs:
        rd = str(rd).strip()
        if not rd:
            continue
        t = celery.send_task(
            "tasks.xgb_oos_tasks.run_xgb_oos_test",
            kwargs={"result_dir": rd, "days": days, "ts": ts},
            queue="oos",
        )
        tasks.append({"result_dir": rd, "task_id": t.id})
    return jsonify({"success": True, "tasks": tasks, "total": len(tasks)})


# ── CSV export / download ─────────────────────────────────────────────

_CSV_FIELDS = [
    "run_dir", "symbol", "task", "direction", "days",
    "horizon_steps", "threshold", "max_hold_steps", "min_profit", "fee_bps",
    "n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
    "reg_lambda", "min_child_weight", "gamma", "scale_pos_weight",
    "val_acc", "f1_buy_sell", "f1_1", "prec_1", "recall_1",
    "y_non_hold", "pred_non_hold",
    "pnl_total", "roi_pct", "winrate", "profit_factor", "max_dd",
    "trades_count", "wins", "losses", "avg_trade_pnl", "avg_bars_held",
    "equity_end",
]


@xgb_oos_bp.post("/xgb_oos_batch_save_csv")
def xgb_oos_batch_save_csv():
    """Принимает результаты batch OOS, сохраняет CSV на сервер."""
    data = request.get_json(silent=True) or {}
    results = data.get("results", [])
    if not isinstance(results, list) or not results:
        return jsonify({"success": False, "error": "results[] required"}), 400

    rows: List[Dict[str, Any]] = []
    for res in results:
        if not isinstance(res, dict) or not res.get("success"):
            continue
        cfg = res.get("cfg_snapshot") or {}
        m = res.get("oos_metrics") or {}
        bt = res.get("backtest") or {}
        f1v = m.get("f1_val")
        pv = m.get("precision_val")
        rv = m.get("recall_val")
        rows.append({
            "run_dir": res.get("run_dir", ""),
            "symbol": res.get("symbol", ""),
            "task": res.get("task", ""),
            "direction": res.get("direction", ""),
            "days": res.get("days", ""),
            "horizon_steps": cfg.get("horizon_steps"),
            "threshold": cfg.get("threshold"),
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
        })

    if not rows:
        return jsonify({"success": False, "error": "no valid results"}), 400

    _CSV_DIR.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sym = rows[0].get("symbol", "XGB")
    filename = f"xgb_oos_{sym}_{ts_str}.csv"
    csv_path = _CSV_DIR / filename

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=_CSV_FIELDS, extrasaction="ignore", restval="")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    csv_path.write_text(buf.getvalue(), encoding="utf-8")

    return jsonify({"success": True, "filename": filename, "rows": len(rows)})


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


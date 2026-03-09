from __future__ import annotations

import csv
import io
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
        exit_mode = str(data.get("exit_mode") or "policy").strip().lower()
        atr_len = data.get("atr_len", None)
        atr_mult = data.get("atr_mult", None)
        direction_override_raw = str(data.get("direction_override") or "").strip().lower()
        direction_override = direction_override_raw if direction_override_raw in ("long", "short") else None
        override_exit_policy = bool(data.get("override_exit_policy")) if isinstance(data, dict) else False
        entry_tp_pct = data.get("entry_tp_pct", None)
        entry_sl_pct = data.get("entry_sl_pct", None)
        entry_trail_pct = data.get("entry_trail_pct", None)
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
            },
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
            scored.append({"path": str(run_dir), "score": float(score), "info": info})
        except Exception as e:
            errors.append({"path": str(p), "error": str(e)})

    if not scored:
        return jsonify({"success": False, "error": "No valid run dirs to prune", "errors": errors}), 400

    scored.sort(key=lambda x: float(x.get("score") or -1e9), reverse=True)
    n = len(scored)
    keep_n = max(1, int(math.ceil(n * (keep_pct / 100.0))))
    keep = scored[:keep_n]
    drop = scored[keep_n:]

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

    tasks: List[Dict[str, str]] = []
    ts = datetime.utcnow().isoformat() + "Z"
    for rd in result_dirs:
        rd = str(rd).strip()
        if not rd:
            continue
        for days in days_list:
            for em in exit_list:
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
                        "p_enter_threshold": p_enter_threshold,
                        "direction_override": direction_override,
                    },
                    queue="oos",
                )
                tasks.append({"result_dir": rd, "days": int(days), "exit_mode": em, "task_id": t.id})
    return jsonify({"success": True, "tasks": tasks, "total": len(tasks), "days_grid": days_list, "exit_modes": exit_list})


# ── CSV export / download ─────────────────────────────────────────────

_CSV_FIELDS = [
    "run_dir", "symbol", "task", "direction", "days",
    "exit_mode", "atr_len", "atr_mult",
    "horizon_steps", "threshold", "p_enter_thr", "max_hold_steps", "min_profit", "fee_bps",
    "n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
    "reg_lambda", "min_child_weight", "gamma", "scale_pos_weight",
    "val_acc", "f1_buy_sell", "f1_1", "prec_1", "recall_1",
    "y_non_hold", "pred_non_hold",
    "pnl_total", "roi_pct", "winrate", "profit_factor", "max_dd",
    "trades_count", "wins", "losses", "avg_trade_pnl", "avg_bars_held",
    "equity_end",
    # backtest exit breakdown (best-effort)
    "exit_timeout_cnt", "exit_trail_cnt", "exit_tp_cnt", "exit_sl_cnt", "exit_signal_cnt", "exit_eof_cnt", "exit_atr_cnt",
    "exit_timeout_share", "exit_trail_share", "exit_tp_share", "exit_sl_share", "exit_signal_share", "exit_eof_share", "exit_atr_share",
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
        try:
            thr_pct = float(cfg.get("threshold")) * 100 if cfg.get("threshold") is not None else None
        except (TypeError, ValueError):
            thr_pct = None
        rows.append({
            "run_dir": res.get("run_dir", ""),
            "symbol": res.get("symbol", ""),
            "task": res.get("task", ""),
            "direction": res.get("direction", ""),
            "days": res.get("days", ""),
            "exit_mode": res.get("exit_mode", ""),
            "atr_len": res.get("atr_len", ""),
            "atr_mult": res.get("atr_mult", ""),
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
            # exit reasons
            "exit_timeout_cnt": (bt.get("reason_counts", {}) or {}).get("timeout"),
            "exit_trail_cnt": (bt.get("reason_counts", {}) or {}).get("trail"),
            "exit_tp_cnt": (bt.get("reason_counts", {}) or {}).get("tp"),
            "exit_sl_cnt": (bt.get("reason_counts", {}) or {}).get("sl"),
            "exit_signal_cnt": (bt.get("reason_counts", {}) or {}).get("signal"),
            "exit_eof_cnt": (bt.get("reason_counts", {}) or {}).get("eof"),
            "exit_atr_cnt": (bt.get("reason_counts", {}) or {}).get("atr_trail"),
            "exit_timeout_share": (bt.get("reason_share", {}) or {}).get("timeout"),
            "exit_trail_share": (bt.get("reason_share", {}) or {}).get("trail"),
            "exit_tp_share": (bt.get("reason_share", {}) or {}).get("tp"),
            "exit_sl_share": (bt.get("reason_share", {}) or {}).get("sl"),
            "exit_signal_share": (bt.get("reason_share", {}) or {}).get("signal"),
            "exit_eof_share": (bt.get("reason_share", {}) or {}).get("eof"),
            "exit_atr_share": (bt.get("reason_share", {}) or {}).get("atr_trail"),
        })

    if not rows:
        return jsonify({"success": False, "error": "no valid results"}), 400

    _CSV_DIR.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sym = rows[0].get("symbol", "XGB")
    # include exit_mode in filename for clarity (policy/hold_steps/atr_trail)
    em_set = sorted({str(r.get("exit_mode") or "").strip().lower() for r in rows if str(r.get("exit_mode") or "").strip()})
    em = em_set[0] if len(em_set) == 1 else ("mixed" if em_set else "na")
    safe_em = "".join(ch for ch in em if ch.isalnum() or ch in ("_", "-"))[:32] or "na"
    filename = f"xgb_oos_{sym}_{safe_em}_{ts_str}.csv"
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


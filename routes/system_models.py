from flask import Blueprint, jsonify, request, render_template, current_app  # type: ignore
from pathlib import Path
import json
import os
import pickle
import time
from utils.path import resolve_run_dir  # type: ignore


system_models_bp = Blueprint("system_models", __name__)


@system_models_bp.get("/system/models")
def system_models_page():
    return render_template("system/models_tools.html")


def _sanitize_trade(t):
    if isinstance(t, dict):
        return {
            k: v
            for k, v in t.items()
            if isinstance(k, str) and isinstance(v, (int, float, str, bool, type(None)))
        }
    return t


def _summarize_value(v, max_list_items: int = 30, max_str: int = 2000):
    try:
        if v is None or isinstance(v, (bool, int, float)):
            return v
        if isinstance(v, str):
            if len(v) > max_str:
                return v[: max_str] + f"...(truncated,{len(v)})"
            return v
        if isinstance(v, (list, tuple)):
            n = len(v)
            head = []
            for item in list(v)[: max_list_items]:
                head.append(_summarize_value(item, max_list_items=10, max_str=300))
            return {"__type__": "list", "len": n, "head": head}
        if isinstance(v, dict):
            out = {"__type__": "dict", "keys": sorted([str(k) for k in v.keys()])[:200]}
            # keep a small preview of scalar-ish values
            preview = {}
            for k in list(v.keys())[:50]:
                try:
                    preview[str(k)] = _summarize_value(v.get(k), max_list_items=10, max_str=300)
                except Exception:
                    continue
            out["preview"] = preview
            out["len"] = len(v)
            return out
        # fallback to string
        s = str(v)
        if len(s) > max_str:
            return s[: max_str] + f"...(truncated,{len(s)})"
        return s
    except Exception:
        return str(v)


def _flatten(obj, prefix: str = "", out: list | None = None):
    if out is None:
        out = []
    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(v, dict):
                    _flatten(v, key, out)
                else:
                    out.append({"key": key, "value": _summarize_value(v)})
        else:
            out.append({"key": prefix or "value", "value": _summarize_value(obj)})
    except Exception:
        out.append({"key": prefix or "value", "value": str(obj)})
    return out


@system_models_bp.get("/api/system/run_details")
def api_system_run_details():
    try:
        model_type = (request.args.get("model_type") or "dqn").strip().lower()
        symbol = (request.args.get("symbol") or "").strip()
        run_id = (request.args.get("run_id") or "").strip()
        if not symbol or not run_id:
            return jsonify({"success": False, "error": "symbol and run_id required"}), 400

        run_dir = resolve_run_dir(model_type, run_id, symbol_hint=symbol, create=False)
        if run_dir is None or not run_dir.exists():
            return jsonify({"success": False, "error": "run directory not found"}), 404

        manifest_path = run_dir / "manifest.json"
        train_path = run_dir / "train_result.pkl"

        manifest = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}

        train = {}
        if train_path.exists():
            try:
                with open(train_path, "rb") as f:
                    train = pickle.load(f)
                if not isinstance(train, dict):
                    train = {"__non_dict__": _summarize_value(train)}
            except Exception as e:
                train = {"__error__": str(e)}

        # never dump huge trades inline here
        if isinstance(train, dict) and "all_trades" in train:
            try:
                if isinstance(train.get("all_trades"), list) and len(train.get("all_trades")) > 0:
                    train["all_trades"] = {"__type__": "list", "len": len(train.get("all_trades")), "head": []}
            except Exception:
                pass

        return jsonify(
            {
                "success": True,
                "run_dir": run_dir.as_posix(),
                "manifest_items": _flatten(manifest, prefix="manifest"),
                "train_items": _flatten(train, prefix="train_result"),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@system_models_bp.post("/api/system/migrate_all_trades")
def api_system_migrate_all_trades():
    """
    Миграция старых train_result.pkl:
    - выносим results['all_trades'] -> all_trades.json рядом с train_result.pkl
    - в train_result.pkl оставляем all_trades=[], all_trades_path, all_trades_count
    """
    t0 = time.time()
    payload = request.get_json(silent=True) or {}
    dry_run = bool(payload.get("dry_run", False))
    limit = payload.get("limit", None)
    try:
        limit = int(limit) if limit is not None else None
    except Exception:
        limit = None

    base = Path("result")
    if not base.exists():
        return jsonify({"success": False, "error": "result/ not found"}), 404

    scanned = 0
    migrated = 0
    skipped = 0
    errors = 0
    details = []

    for pkl_path in base.rglob("train_result.pkl"):
        if limit is not None and migrated >= limit:
            break
        scanned += 1
        run_dir = pkl_path.parent
        trades_json = run_dir / "all_trades.json"

        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, dict):
                skipped += 1
                continue

            trades = data.get("all_trades")
            if not isinstance(trades, list) or len(trades) == 0:
                # already migrated or просто не было сделок
                skipped += 1
                continue

            if trades_json.exists():
                # json уже есть — просто проставим метаданные в pkl (лёгкая миграция)
                if not dry_run:
                    data["all_trades"] = []
                    data["all_trades_path"] = trades_json.as_posix()
                    data["all_trades_count"] = int(data.get("all_trades_count") or len(trades))
                    tmp = pkl_path.with_suffix(".pkl.tmp")
                    with open(tmp, "wb") as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(tmp, pkl_path)
                migrated += 1
                continue

            safe_trades = [_sanitize_trade(t) for t in trades]
            if not dry_run:
                trades_json.write_text(json.dumps(safe_trades, ensure_ascii=False), encoding="utf-8")
                data["all_trades"] = []
                data["all_trades_path"] = trades_json.as_posix()
                data["all_trades_count"] = len(trades)
                tmp = pkl_path.with_suffix(".pkl.tmp")
                with open(tmp, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, pkl_path)

            migrated += 1
            if len(details) < 20:
                details.append({"run_dir": run_dir.as_posix(), "trades": len(trades)})
        except Exception as e:
            errors += 1
            if len(details) < 20:
                details.append({"file": pkl_path.as_posix(), "error": str(e)})
            try:
                current_app.logger.warning(f"[migrate_all_trades] failed for {pkl_path}: {e}")
            except Exception:
                pass

    return jsonify(
        {
            "success": True,
            "dry_run": dry_run,
            "scanned": scanned,
            "migrated": migrated,
            "skipped": skipped,
            "errors": errors,
            "details": details,
            "elapsed_sec": round(time.time() - t0, 3),
        }
    )


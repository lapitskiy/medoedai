from flask import Blueprint, render_template, jsonify, request  # type: ignore
from pathlib import Path
import json
import os
import shutil
import math
from datetime import datetime
from typing import Any, Dict, List
from utils.trade_utils import get_model_predictions
from utils.redis_utils import get_redis_client
from utils.trading_sessions import load_session


xgb_bp = Blueprint("xgb", __name__)

_XGB_GRID_PRESET_PATH = Path("predict_test") / "xgb_hypo" / "xgb_grid_full_preset.json"
_XGB_GRID_PRESET_ALLOWED_KEYS = {
    "gfSymbol", "gfTask", "gfDirection", "gfLimitCandles", "gfEarlyStopping", "gfKeepTopN",
    "gfRankByProxy", "gfDeleteRest", "gfUseSrFeatures",
    "gfHsFrom", "gfHsTo", "gfHsStep", "gfThrFrom", "gfThrTo", "gfThrStep",
    "gfMhFrom", "gfMhTo", "gfMhStep", "gfMpFrom", "gfMpTo", "gfMpStep",
    "gfFeeFrom", "gfFeeTo", "gfFeeStep",
    "gfPeFrom", "gfPeTo", "gfPeStep",
    "gfEtpFrom", "gfEtpTo", "gfEtpStep",
    "gfEslFrom", "gfEslTo", "gfEslStep",
    "gfEtrFrom", "gfEtrTo", "gfEtrStep",
    "gfMdFrom", "gfMdTo", "gfMdStep", "gfLrFrom", "gfLrTo", "gfLrStep",
    "gfNeFrom", "gfNeTo", "gfNeStep", "gfSsFrom", "gfSsTo", "gfSsStep",
    "gfCbFrom", "gfCbTo", "gfCbStep", "gfRlFrom", "gfRlTo", "gfRlStep",
    "gfMcwFrom", "gfMcwTo", "gfMcwStep", "gfGmFrom", "gfGmTo", "gfGmStep",
    "gfSpwFrom", "gfSpwTo", "gfSpwStep",
    "gfUse1mMicrovol", "gfUse1mMomentum", "gfUse1mCandleStructure", "gfUse1mVolume",
    "gfUse1dRegime",
}


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _normalize_xgb_symbol(value: str) -> str:
    symbol = str(value or "").strip().upper().replace("/", "")
    if symbol.endswith("USDT"):
        symbol = symbol[:-4]
    return symbol


def _pick_best_oos_entry(history: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    best_item: Dict[str, Any] | None = None
    best_score: tuple[float, float, float] | None = None
    for item in history:
        if not isinstance(item, dict):
            continue
        backtest = item.get("backtest") if isinstance(item.get("backtest"), dict) else {}
        try:
            pnl_total = float(backtest.get("pnl_total"))
        except Exception:
            continue
        try:
            profit_factor = float(backtest.get("profit_factor") or 0.0)
        except Exception:
            profit_factor = 0.0
        try:
            max_dd = float(backtest.get("max_dd") or 0.0)
        except Exception:
            max_dd = 0.0
        score = (pnl_total, profit_factor, -max_dd)
        if best_score is None or score > best_score:
            best_score = score
            best_item = item
    return best_item


def _is_xgb_model_path(value: Any) -> bool:
    try:
        path = str(value or "").replace("\\", "/")
    except Exception:
        return False
    return "/models/xgb/" in path


def _sanitize_for_json(value: Any) -> Any:
    try:
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, (list, tuple)):
            return [_sanitize_for_json(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _sanitize_for_json(v) for k, v in value.items()}
        return value
    except Exception:
        return None


def _xgb_model_uuid_from_manifest(manifest: Dict[str, Any], version_name: str) -> str:
    value = (
        manifest.get("run_name")
        or manifest.get("run_id")
        or manifest.get("id")
        or version_name
    )
    return str(value or version_name)


def _version_sort_key(version_name: str) -> int:
    return int(version_name[1:]) if version_name.startswith("v") and version_name[1:].isdigit() else 0


def _current_xgb_version(ensemble_dir: Path) -> str | None:
    current_link = ensemble_dir / "current"
    if current_link.is_symlink():
        try:
            return current_link.resolve().name
        except Exception:
            return None
    if current_link.is_file():
        try:
            return current_link.read_text(encoding="utf-8").strip()
        except Exception:
            return None
    return None


def _version_dirs(ensemble_dir: Path) -> List[Path]:
    return [
        item for item in ensemble_dir.iterdir()
        if item.is_dir() and item.name.startswith("v")
    ]


def _set_current_xgb_version(ensemble_dir: Path, version_name: str | None) -> None:
    current_link = ensemble_dir / "current"
    if current_link.exists() or current_link.is_symlink():
        if current_link.is_dir() and not current_link.is_symlink():
            shutil.rmtree(current_link)
        else:
            current_link.unlink()

    if version_name:
        os.symlink(version_name, current_link)


def _serialize_xgb_prod_version(symbol_code: str, ensemble_name: str, version_dir: Path, current_version: str | None) -> Dict[str, Any]:
    manifest = _safe_read_json(version_dir / "manifest.json")
    metrics = _safe_read_json(version_dir / "metrics.json")
    meta = _safe_read_json(version_dir / "meta.json")
    oos = _safe_read_json(version_dir / "oos_xgb_results.json")
    cfg = meta.get("cfg_snapshot") if isinstance(meta.get("cfg_snapshot"), dict) else {}
    oos_history = oos.get("history") if isinstance(oos.get("history"), list) else []
    best_oos = _pick_best_oos_entry(oos_history)
    best_oos_backtest = best_oos.get("backtest") if isinstance(best_oos, dict) and isinstance(best_oos.get("backtest"), dict) else {}
    model_uuid = _xgb_model_uuid_from_manifest(manifest, version_dir.name)

    return {
        "symbol": symbol_code,
        "ensemble": ensemble_name,
        "version": version_dir.name,
        "path": version_dir.as_posix(),
        "is_current": version_dir.name == current_version,
        "model_id": manifest.get("run_id") or manifest.get("id") or version_dir.name,
        "run_id": manifest.get("run_id") or manifest.get("id") or version_dir.name,
        "model_uuid": model_uuid,
        "direction": manifest.get("direction") or cfg.get("direction"),
        "task": manifest.get("task") or cfg.get("task"),
        "trained_as": manifest.get("direction") or cfg.get("direction"),
        "created_at": manifest.get("created_at"),
        "source_run_path": manifest.get("source_run_path"),
        "train_metrics": {
            "val_acc": metrics.get("val_acc"),
            "f1_buy_sell_val": metrics.get("f1_buy_sell_val"),
            "f1_val": metrics.get("f1_val"),
            "precision_val": metrics.get("precision_val"),
            "recall_val": metrics.get("recall_val"),
            "proxy_pnl_val": metrics.get("proxy_pnl_val"),
        },
        "cfg": {
            "horizon_steps": cfg.get("horizon_steps"),
            "threshold": cfg.get("threshold"),
            "max_hold_steps": cfg.get("max_hold_steps"),
            "min_profit": cfg.get("min_profit"),
            "fee_bps": cfg.get("fee_bps"),
            "p_enter_threshold": cfg.get("p_enter_threshold"),
            "entry_tp_pct": cfg.get("entry_tp_pct"),
            "entry_sl_pct": cfg.get("entry_sl_pct"),
            "entry_trail_pct": cfg.get("entry_trail_pct"),
            "n_estimators": cfg.get("n_estimators"),
            "max_depth": cfg.get("max_depth"),
            "learning_rate": cfg.get("learning_rate"),
            "subsample": cfg.get("subsample"),
            "colsample_bytree": cfg.get("colsample_bytree"),
            "reg_lambda": cfg.get("reg_lambda"),
            "min_child_weight": cfg.get("min_child_weight"),
            "gamma": cfg.get("gamma"),
        },
        "best_oos": {
            "days": best_oos.get("days") if isinstance(best_oos, dict) else None,
            "ts": best_oos.get("ts") if isinstance(best_oos, dict) else None,
            "pnl_total": best_oos_backtest.get("pnl_total"),
            "roi_pct": best_oos_backtest.get("roi_pct"),
            "profit_factor": best_oos_backtest.get("profit_factor"),
            "max_dd": best_oos_backtest.get("max_dd"),
            "trades_count": best_oos_backtest.get("trades_count"),
            "winrate": best_oos_backtest.get("winrate"),
            "avg_trade_pnl": best_oos_backtest.get("avg_trade_pnl"),
            "avg_bars_held": best_oos_backtest.get("avg_bars_held"),
            "equity_end": best_oos_backtest.get("equity_end"),
        },
    }


def _candidate_symbol_values(value: str) -> List[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    code = _normalize_xgb_symbol(raw)
    values = [raw, raw.upper(), code, f"{code}USDT", f"{code}/USDT"]
    unique: List[str] = []
    for item in values:
        candidate = str(item or "").strip()
        if candidate and candidate not in unique:
            unique.append(candidate)
    return unique


def _prediction_to_dict(prediction: Any) -> Dict[str, Any]:
    try:
        q_values = json.loads(prediction.q_values) if prediction.q_values else []
    except Exception:
        q_values = []
    try:
        market_conditions = (
            json.loads(prediction.market_conditions) if prediction.market_conditions else {}
        )
    except Exception:
        market_conditions = {}
    try:
        model_path = str(getattr(prediction, "model_path", "") or "")
        if _is_xgb_model_path(model_path):
            meta = _safe_read_json(Path(model_path).with_name("meta.json"))
            manifest = _safe_read_json(Path(model_path).with_name("manifest.json"))
            cfg = meta.get("cfg_snapshot") if isinstance(meta.get("cfg_snapshot"), dict) else {}
            symbol_value = str(getattr(prediction, "symbol", "") or "").strip().upper()
            if isinstance(market_conditions, dict):
                if cfg.get("p_enter_threshold") is not None and "xgb_p_enter_threshold" not in market_conditions:
                    market_conditions["xgb_p_enter_threshold"] = cfg.get("p_enter_threshold")
                try:
                    rc = get_redis_client()
                    runtime_thr = rc.get(f"trading:xgb_entry_threshold:{symbol_value}") if rc and symbol_value else None
                    if runtime_thr not in (None, ""):
                        runtime_thr_s = (
                            runtime_thr.decode("utf-8")
                            if isinstance(runtime_thr, (bytes, bytearray))
                            else str(runtime_thr)
                        )
                        market_conditions["xgb_runtime_threshold"] = float(runtime_thr_s)
                        market_conditions["xgb_threshold_override_active"] = True
                except Exception:
                    pass
                if cfg.get("task") and "xgb_task" not in market_conditions:
                    market_conditions["xgb_task"] = cfg.get("task")
                if cfg.get("direction") and "xgb_direction" not in market_conditions:
                    market_conditions["xgb_direction"] = cfg.get("direction")
                if manifest.get("run_name") and "xgb_run_name" not in market_conditions:
                    market_conditions["xgb_run_name"] = manifest.get("run_name")
    except Exception:
        pass
    return {
        "id": getattr(prediction, "id", None),
        "session_id": (
            market_conditions.get("session_id")
            if isinstance(market_conditions, dict)
            else None
        ),
        "timestamp": (
            prediction.timestamp.isoformat()
            if getattr(prediction, "timestamp", None)
            else None
        ),
        "symbol": getattr(prediction, "symbol", None),
        "action": getattr(prediction, "action", None),
        "q_values": q_values,
        "current_price": getattr(prediction, "current_price", None),
        "position_status": getattr(prediction, "position_status", None),
        "confidence": getattr(prediction, "confidence", None),
        "model_path": getattr(prediction, "model_path", None),
        "market_conditions": (
            market_conditions if isinstance(market_conditions, dict) else {}
        ),
        "created_at": (
            prediction.created_at.isoformat()
            if getattr(prediction, "created_at", None)
            else None
        ),
    }


def _read_xgb_grid_preset() -> dict:
    if not _XGB_GRID_PRESET_PATH.exists():
        return {}
    try:
        data = json.loads(_XGB_GRID_PRESET_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


@xgb_bp.get("/api/xgb/grid_full_preset")
def api_xgb_grid_full_preset_get():
    try:
        data = _read_xgb_grid_preset()
        values = data.get("values") if isinstance(data.get("values"), dict) else {}
        return jsonify({
            "success": True,
            "values": values,
            "updated_at": data.get("updated_at"),
            "source": "server_json",
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@xgb_bp.post("/api/xgb/grid_full_preset")
def api_xgb_grid_full_preset_save():
    try:
        payload = request.get_json(silent=True) or {}
        raw_values = payload.get("values") if isinstance(payload.get("values"), dict) else payload
        if not isinstance(raw_values, dict):
            return jsonify({"success": False, "error": "values must be an object"}), 400

        values = {}
        for k, v in raw_values.items():
            key = str(k)
            if key in _XGB_GRID_PRESET_ALLOWED_KEYS:
                values[key] = "" if v is None else str(v)

        _XGB_GRID_PRESET_PATH.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "values": values,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        _XGB_GRID_PRESET_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        return jsonify({"success": True, "saved_keys": len(values), "path": _XGB_GRID_PRESET_PATH.as_posix()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@xgb_bp.get("/xgb_models")
def xgb_models_page():
    """Страница обучения XGB моделей."""
    return render_template("xgb_models.html", include_sidebar=True)


@xgb_bp.get("/xgb_models_content")
def xgb_models_content_page():
    """Контент обучения XGB без legacy sidebar."""
    return render_template("xgb_models.html", include_sidebar=False)


@xgb_bp.get("/api/xgb/symbols")
def api_xgb_symbols():
    try:
        base = Path("result") / "xgb"
        if not base.exists():
            return jsonify({"success": True, "symbols": []})
        symbols = sorted(
            d.name for d in base.iterdir()
            if d.is_dir() and (d / "runs").is_dir()
        )
        return jsonify({"success": True, "symbols": symbols})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@xgb_bp.get("/api/xgb/ensembles")
def api_xgb_ensembles():
    try:
        symbol_raw = (request.args.get("symbol") or "").strip()
        if not symbol_raw:
            return jsonify({"success": False, "error": "symbol required"}), 400

        symbol_code = _normalize_xgb_symbol(symbol_raw)
        if not symbol_code:
            return jsonify({"success": False, "error": "bad symbol"}), 400

        models_root = Path("models") / "xgb" / symbol_code.lower()
        if not models_root.exists():
            return jsonify({"success": True, "symbol": symbol_code, "ensembles": {}})

        ensembles: Dict[str, Any] = {}
        for ensemble_dir in sorted(models_root.iterdir(), key=lambda item: item.name):
            if not ensemble_dir.is_dir() or not ensemble_dir.name.startswith("ensemble-"):
                continue

            versions: List[Dict[str, Any]] = []
            current_version = _current_xgb_version(ensemble_dir)

            for version_dir in sorted(ensemble_dir.iterdir(), key=lambda item: item.name):
                if not version_dir.is_dir() or not version_dir.name.startswith("v"):
                    continue

                versions.append(_serialize_xgb_prod_version(symbol_code, ensemble_dir.name, version_dir, current_version))

            versions.sort(
                key=lambda item: _version_sort_key(str(item.get("version") or "v0")),
                reverse=True,
            )

            current_model = next((item for item in versions if item.get("is_current")), None)
            if current_model is None and versions:
                current_model = versions[0]

            ensembles[ensemble_dir.name] = {
                "current_version": current_version,
                "current_model": current_model,
                "versions": versions,
            }

        return jsonify({"success": True, "symbol": symbol_code, "ensembles": ensembles})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@xgb_bp.get("/api/xgb/prod/models")
def api_xgb_prod_models():
    try:
        base = Path("models") / "xgb"
        if not base.exists():
            return jsonify({"success": True, "models": []})

        models: List[Dict[str, Any]] = []
        for symbol_dir in sorted(base.iterdir(), key=lambda item: item.name):
            if not symbol_dir.is_dir():
                continue
            symbol_code = _normalize_xgb_symbol(symbol_dir.name)
            for ensemble_dir in sorted(symbol_dir.iterdir(), key=lambda item: item.name):
                if not ensemble_dir.is_dir() or not ensemble_dir.name.startswith("ensemble-"):
                    continue
                current_version = _current_xgb_version(ensemble_dir)
                for version_dir in sorted(ensemble_dir.iterdir(), key=lambda item: item.name):
                    if not version_dir.is_dir() or not version_dir.name.startswith("v"):
                        continue
                    models.append(_serialize_xgb_prod_version(symbol_code, ensemble_dir.name, version_dir, current_version))

        models.sort(
            key=lambda item: (
                str(item.get("symbol") or ""),
                str(item.get("ensemble") or ""),
                -_version_sort_key(str(item.get("version") or "v0")),
            )
        )
        return jsonify({"success": True, "models": models})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@xgb_bp.delete("/api/xgb/prod/models")
def api_xgb_prod_model_delete():
    try:
        data = request.get_json(silent=True) or {}
        symbol_code = _normalize_xgb_symbol(str(data.get("symbol") or ""))
        ensemble = str(data.get("ensemble") or "").strip()
        version = str(data.get("version") or "").strip()

        if not symbol_code or not ensemble or not version:
            return jsonify({"success": False, "error": "symbol, ensemble and version required"}), 400
        if not ensemble.startswith("ensemble-"):
            return jsonify({"success": False, "error": "bad ensemble"}), 400
        if not version.startswith("v") or not version[1:].isdigit():
            return jsonify({"success": False, "error": "bad version"}), 400

        ensemble_dir = Path("models") / "xgb" / symbol_code.lower() / ensemble
        version_dir = ensemble_dir / version
        if not version_dir.is_dir():
            return jsonify({"success": False, "error": "model version not found"}), 404

        current_version = _current_xgb_version(ensemble_dir)
        shutil.rmtree(version_dir)

        next_current = current_version
        if current_version == version:
            remaining = sorted(_version_dirs(ensemble_dir), key=lambda item: _version_sort_key(item.name), reverse=True)
            next_current = remaining[0].name if remaining else None
            _set_current_xgb_version(ensemble_dir, next_current)

        if not _version_dirs(ensemble_dir):
            next_current = None
            _set_current_xgb_version(ensemble_dir, None)

        return jsonify({
            "success": True,
            "deleted": {
                "symbol": symbol_code,
                "ensemble": ensemble,
                "version": version,
            },
            "current_version": next_current,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@xgb_bp.get("/api/xgb/predictions")
def api_xgb_predictions():
    try:
        symbol_raw = (request.args.get("symbol") or "").strip()
        session_id_raw = (request.args.get("session_id") or "").strip()
        model_path_raw = (request.args.get("model_path") or "").strip()
        limit = min(max(int(request.args.get("limit", 100)), 1), 500)
        session_doc = None
        session_model_paths: set[str] = set()
        if session_id_raw:
            try:
                rc = get_redis_client()
                session_doc = load_session(rc, session_id_raw)
                if isinstance(session_doc, dict):
                    for item in (session_doc.get("model_paths") or []):
                        if item:
                            session_model_paths.add(str(item))
                    if session_doc.get("model_path"):
                        session_model_paths.add(str(session_doc.get("model_path")))
            except Exception:
                session_doc = None

        predictions_data: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        for candidate_symbol in _candidate_symbol_values(symbol_raw):
            predictions = get_model_predictions(symbol=candidate_symbol, limit=limit)
            for prediction in predictions or []:
                model_path = getattr(prediction, "model_path", None)
                if not _is_xgb_model_path(model_path):
                    continue
                row = _prediction_to_dict(prediction)
                row_model_path = str(row.get("model_path") or "").strip()
                row_session_id = str(row.get("session_id") or "").strip()
                if model_path_raw and row_model_path != model_path_raw:
                    continue
                if session_id_raw:
                    matches_session = row_session_id == session_id_raw
                    if (not matches_session) and row_model_path and row_model_path in session_model_paths:
                        matches_session = True
                        row["session_id"] = session_id_raw
                    if not matches_session:
                        continue
                row_id = str(row.get("id") or f"{row.get('timestamp')}|{row.get('model_path')}")
                if row_id in seen_ids:
                    continue
                seen_ids.add(row_id)
                predictions_data.append(row)

        predictions_data.sort(
            key=lambda item: str(item.get("created_at") or item.get("timestamp") or ""),
            reverse=True,
        )
        predictions_data = predictions_data[:limit]

        if not predictions_data:
            try:
                rc = get_redis_client()
                keys = sorted(rc.keys("trading:latest_result_*") or [], reverse=True)
                target_symbols = set(_candidate_symbol_values(symbol_raw))
                for key in keys:
                    try:
                        raw = rc.get(key)
                        if not raw:
                            continue
                        snapshot = json.loads(raw)
                        snapshot_symbols = snapshot.get("symbols") or []
                        if target_symbols and not any(s in target_symbols for s in snapshot_symbols):
                            continue
                        preds = snapshot.get("predictions") or []
                        for item in preds:
                            if not isinstance(item, dict):
                                continue
                            if not _is_xgb_model_path(item.get("model_path")):
                                continue
                            if model_path_raw and str(item.get("model_path") or "").strip() != model_path_raw:
                                continue
                            predictions_data.append(
                                {
                                    "id": None,
                                    "session_id": snapshot.get("session_id"),
                                    "timestamp": snapshot.get("timestamp"),
                                    "symbol": (snapshot_symbols[0] if snapshot_symbols else symbol_raw),
                                    "action": item.get("action"),
                                    "q_values": item.get("q_values") or [],
                                    "current_price": None,
                                    "position_status": None,
                                    "confidence": item.get("confidence"),
                                    "model_path": item.get("model_path"),
                                    "market_conditions": {
                                        "market_regime": snapshot.get("market_regime"),
                                    },
                                    "created_at": snapshot.get("timestamp"),
                                }
                            )
                        if predictions_data:
                            break
                    except Exception:
                        continue
            except Exception:
                pass

        safe_payload = _sanitize_for_json(
            {
                "success": True,
                "symbol": _normalize_xgb_symbol(symbol_raw) if symbol_raw else None,
                "session_id": session_id_raw or None,
                "model_path": model_path_raw or None,
                "predictions": predictions_data[:limit],
                "total_predictions": len(predictions_data[:limit]),
            }
        )
        return jsonify(safe_payload)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@xgb_bp.get("/api/xgb/runs")
def api_xgb_runs():
    try:
        symbol = (request.args.get("symbol") or "").strip().upper()
        if not symbol:
            return jsonify({"success": False, "error": "symbol required"}), 400
        runs_dir = Path("result") / "xgb" / symbol / "runs"
        if not runs_dir.exists():
            return jsonify({"success": True, "runs": []})
        runs = []
        for rd in sorted(runs_dir.iterdir()):
            if not rd.is_dir():
                continue
            manifest = {}
            mf = rd / "manifest.json"
            if mf.exists():
                try:
                    manifest = json.loads(mf.read_text(encoding="utf-8"))
                except Exception:
                    manifest = {}
            metrics = {}
            mef = rd / "metrics.json"
            if mef.exists():
                try:
                    metrics = json.loads(mef.read_text(encoding="utf-8"))
                except Exception:
                    metrics = {}
            has_model = (rd / "model.json").exists()
            if not has_model:
                continue
            runs.append({
                "run_id": rd.name,
                "direction": manifest.get("direction"),
                "task": manifest.get("task"),
                "val_acc": metrics.get("val_acc"),
                "path": rd.as_posix(),
            })
        return jsonify({"success": True, "runs": runs})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@xgb_bp.post("/create_xgb_model_version")
def create_xgb_model_version():
    """Копирует XGB run из result/xgb/ в models/xgb/<symbol>/ensemble-a/vN."""
    try:
        data = request.get_json(silent=True) or {}
        symbol = (data.get("symbol") or "").strip().upper()
        run_id = (data.get("run_id") or "").strip()
        ensemble = (data.get("ensemble") or "ensemble-a").strip() or "ensemble-a"

        if not symbol or not run_id:
            return jsonify({"success": False, "error": "symbol и run_id обязательны"}), 400

        run_dir = Path("result") / "xgb" / symbol / "runs" / run_id
        if not run_dir.is_dir():
            return jsonify({"success": False, "error": f"Run {run_id} не найден"}), 404

        models_root = Path("models") / "xgb" / symbol.lower()
        ensemble_dir = models_root / ensemble
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        existing = []
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

        copied_files = []
        for f in run_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, version_dir / f.name)
                copied_files.append(f.name)

        # manifest.yaml
        run_manifest = {}
        mf = run_dir / "manifest.json"
        if mf.exists():
            try:
                run_manifest = json.loads(mf.read_text(encoding="utf-8"))
            except Exception:
                pass
        yaml_lines = [
            f'id: "{run_id}"',
            f'symbol: "{symbol}"',
            f'ensemble: "{ensemble}"',
            f'version: "{version_name}"',
            f'created_at: "{datetime.utcnow().isoformat()}"',
            f'run_id: "{run_id}"',
            f'model_type: "xgb"',
        ]
        for key in ("direction", "task"):
            val = run_manifest.get(key)
            if val:
                yaml_lines.append(f'{key}: "{val}"')
        yaml_lines.append(f'source_run_path: "{run_dir.as_posix()}"')
        yaml_lines.append("files:")
        for fn in copied_files:
            yaml_lines.append(f'  - "{fn}"')
        (version_dir / "manifest.yaml").write_text(
            "\n".join(yaml_lines) + "\n", encoding="utf-8"
        )

        # symlink current -> vN
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

        return jsonify({
            "success": True,
            "model_id": run_id,
            "version": version_name,
            "files": copied_files,
            "path": version_dir.as_posix(),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


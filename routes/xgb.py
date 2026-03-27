from flask import Blueprint, render_template, jsonify, request  # type: ignore
from pathlib import Path
import json
import os
import shutil
from datetime import datetime


xgb_bp = Blueprint("xgb", __name__)

_XGB_GRID_PRESET_PATH = Path("predict_test") / "xgb_hypo" / "xgb_grid_full_preset.json"
_XGB_GRID_PRESET_ALLOWED_KEYS = {
    "gfSymbol", "gfTask", "gfDirection", "gfLimitCandles", "gfEarlyStopping", "gfKeepTopN",
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
    return render_template("xgb_models.html")


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


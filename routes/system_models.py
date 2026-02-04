from flask import Blueprint, jsonify, request, render_template, current_app  # type: ignore
from pathlib import Path
import json
import os
import pickle
import time
import csv
import io
from utils.path import resolve_run_dir  # type: ignore
from utils.db_utils import db_get_ohlcv_only  # type: ignore
from utils.adaptive_normalization import adaptive_normalizer  # type: ignore


system_models_bp = Blueprint("system_models", __name__)


@system_models_bp.get("/system/models")
def system_models_page():
    return render_template("system/models_tools.html")


@system_models_bp.get("/system/adaptive")
def system_adaptive_page():
    return render_template("system/adaptive.html")


@system_models_bp.get("/system/hypotheses")
def system_hypotheses_page():
    return render_template("system/hypotheses.html")


def _safe_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _pick(d: dict, path: str):
    """Pick nested key via dot-path."""
    try:
        cur = d
        for k in path.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k)
        return cur
    except Exception:
        return None


def _iter_symbol_runs(symbol: str, limit_runs: int = 200):
    base = Path("result") / str(symbol).upper() / "runs"
    if not base.exists():
        return []
    runs = []
    for run_dir in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        pkl_path = run_dir / "train_result.pkl"
        if not manifest_path.exists() or not pkl_path.exists():
            continue
        runs.append({"run_dir": run_dir, "manifest_path": manifest_path, "pkl_path": pkl_path})
        if limit_runs and len(runs) >= limit_runs:
            break
    return runs


@system_models_bp.get("/api/system/hypotheses_export")
def api_system_hypotheses_export():
    """
    Экспорт агрегированных данных по runs для генерации гипотез.
    Возвращает JSON: {csv, prompt}.
    Query:
      - symbol (required)
      - limit_runs (default 200)
    """
    try:
        symbol = (request.args.get("symbol") or "").strip().upper()
        if not symbol:
            return jsonify({"success": False, "error": "symbol is required"}), 400
        try:
            limit_runs = int(request.args.get("limit_runs") or "200")
        except Exception:
            limit_runs = 200

        rows = []
        items = _iter_symbol_runs(symbol, limit_runs=limit_runs)
        for it in items:
            run_dir: Path = it["run_dir"]
            manifest = {}
            try:
                manifest = json.loads(it["manifest_path"].read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
            data = {}
            try:
                with open(it["pkl_path"], "rb") as f:
                    data = pickle.load(f)
                if not isinstance(data, dict):
                    data = {}
            except Exception:
                data = {}

            cfg = data.get("cfg_snapshot") if isinstance(data.get("cfg_snapshot"), dict) else {}
            meta = data.get("train_metadata") if isinstance(data.get("train_metadata"), dict) else {}
            stats = data.get("final_stats") if isinstance(data.get("final_stats"), dict) else {}
            winrates = data.get("winrates") if isinstance(data.get("winrates"), dict) else {}
            wr_all = _pick(winrates, "train_all")
            wr_expl = _pick(winrates, "train_exploit")

            # adaptive params stored in manifest by our recent change (compact)
            adapt = manifest.get("adaptive_params") if isinstance(manifest.get("adaptive_params"), dict) else {}

            row = {
                "symbol": symbol,
                "run_id": manifest.get("run_id") or run_dir.name,
                "created_at": manifest.get("created_at") or data.get("training_date"),
                "direction": manifest.get("direction") or manifest.get("trained_as") or data.get("trained_as"),
                # key metrics
                "winrate_train_all": _safe_float(wr_all),
                "winrate_train_exploit": _safe_float(wr_expl),
                "trades_count": int(stats.get("trades_count") or 0),
                "avg_roi": _safe_float(stats.get("avg_roi")),
                "avg_profit": _safe_float(stats.get("avg_profit")),
                "avg_loss": _safe_float(stats.get("avg_loss")),
                "pl_ratio": _safe_float(stats.get("pl_ratio")),
                # cfg (subset)
                "batch_size": cfg.get("batch_size"),
                "memory_size": cfg.get("memory_size"),
                "hidden_sizes": str(cfg.get("hidden_sizes")),
                "train_repeats": cfg.get("train_repeats"),
                "use_amp": cfg.get("use_amp"),
                "use_torch_compile": cfg.get("use_torch_compile"),
                "learning_rate": cfg.get("lr") or cfg.get("learning_rate"),
                "eps_decay_steps": cfg.get("eps_decay_steps"),
                "dropout_rate": cfg.get("dropout_rate"),
                # env/risk params (from adaptive)
                "sl_pct": adapt.get("stop_loss_pct"),
                "tp_pct": adapt.get("take_profit_pct"),
                "min_hold": adapt.get("min_hold_steps"),
                "volume_threshold": adapt.get("volume_threshold"),
                "risk_calc_source": adapt.get("risk_calc_source"),
                "atr_rel_med": adapt.get("atr_rel_med"),
                "returns_std": adapt.get("returns_std"),
                # platform
                "gpu_name": meta.get("gpu_name"),
                "pytorch_version": meta.get("pytorch_version"),
            }
            rows.append(row)

        # CSV
        buf = io.StringIO()
        fieldnames = sorted(set(k for r in rows for k in r.keys()))
        w = csv.DictWriter(buf, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        csv_text = buf.getvalue()

        # TXT prompt (short, structured)
        def _top(items, key, n=10):
            try:
                arr = [x for x in items if isinstance(x.get(key), (int, float)) and x.get(key) is not None]
                arr.sort(key=lambda z: float(z.get(key) or 0.0), reverse=True)
                return arr[:n]
            except Exception:
                return []

        top_wr = _top(rows, "winrate_train_all", 10)
        top_roi = _top(rows, "avg_roi", 10)

        lines = []
        lines.append(f"Symbol: {symbol}")
        lines.append(f"Runs scanned: {len(rows)}")
        lines.append("")
        lines.append("Goal: propose hypotheses about which settings correlate with higher winrate and avg_roi.")
        lines.append("Use only the provided columns; suggest next experiments (what to vary) and what to monitor.")
        lines.append("")
        lines.append("Top-10 by winrate_train_all:")
        for r in top_wr:
            lines.append(
                f"- run_id={r.get('run_id')} winrate={r.get('winrate_train_all')} avg_roi={r.get('avg_roi')} "
                f"batch={r.get('batch_size')} mem={r.get('memory_size')} hs={r.get('hidden_sizes')} repeats={r.get('train_repeats')} "
                f"amp={r.get('use_amp')} compile={r.get('use_torch_compile')} sl={r.get('sl_pct')} tp={r.get('tp_pct')} src={r.get('risk_calc_source')}"
            )
        lines.append("")
        lines.append("Top-10 by avg_roi:")
        for r in top_roi:
            lines.append(
                f"- run_id={r.get('run_id')} avg_roi={r.get('avg_roi')} winrate={r.get('winrate_train_all')} "
                f"batch={r.get('batch_size')} mem={r.get('memory_size')} hs={r.get('hidden_sizes')} repeats={r.get('train_repeats')} "
                f"amp={r.get('use_amp')} compile={r.get('use_torch_compile')} sl={r.get('sl_pct')} tp={r.get('tp_pct')} src={r.get('risk_calc_source')}"
            )
        lines.append("")
        lines.append("Now analyze the CSV and produce:")
        lines.append("1) 5-10 hypotheses")
        lines.append("2) suggested next runs (3-5 configs) to validate")
        lines.append("3) risks/overfitting caveats")
        prompt_text = "\n".join(lines)

        return jsonify({"success": True, "symbol": symbol, "runs_count": len(rows), "csv": csv_text, "prompt": prompt_text})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@system_models_bp.get("/api/system/adaptive_params")
def api_system_adaptive_params():
    """
    Возвращает (и при необходимости пересчитывает) full-auto adaptive params по символам.
    Query:
      - symbols: CSV (если пусто — берём первые N из defaults)
      - timeframe: (default 5m)
      - limit_candles: (default 100000)
      - recalc: 0/1 (default 0) — принудительный пересчёт (обходит кеш)
    """
    try:
        timeframe = (request.args.get("timeframe") or "5m").strip()
        try:
            limit_candles = int(request.args.get("limit_candles") or "100000")
        except Exception:
            limit_candles = 100000
        recalc = str(request.args.get("recalc") or "0").strip().lower() in ("1", "true", "yes", "on")
        raw = (request.args.get("symbols") or "").strip()
        if raw:
            symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
        else:
            # небольшой дефолтный список
            symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "TONUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT"]

        items = []
        for sym in symbols:
            try:
                df = db_get_ohlcv_only(sym, timeframe, limit_candles=limit_candles)
                if df is None or df.empty:
                    items.append({"symbol": sym, "success": False, "error": "no_ohlcv"})
                    continue
                train_split_point = None
                try:
                    train_split_point = int(len(df) * 0.8)
                except Exception:
                    train_split_point = None

                if recalc:
                    # обходим кеш самым простым способом: временно TTL=0 для этого вызова
                    try:
                        prev = getattr(adaptive_normalizer, "_cache_ttl_sec", 3600)
                        setattr(adaptive_normalizer, "_cache_ttl_sec", 0)
                        params = adaptive_normalizer.get_trading_params(sym, df, train_split_point=train_split_point)
                        setattr(adaptive_normalizer, "_cache_ttl_sec", prev)
                    except Exception:
                        params = adaptive_normalizer.get_trading_params(sym, df, train_split_point=train_split_point)
                else:
                    params = adaptive_normalizer.get_trading_params(sym, df, train_split_point=train_split_point)
                if isinstance(params, dict):
                    params.pop("regime_precomputed", None)
                items.append({"symbol": sym, "success": True, "params": params})
            except Exception as e:
                items.append({"symbol": sym, "success": False, "error": str(e)})

        return jsonify({"success": True, "timeframe": timeframe, "limit_candles": limit_candles, "items": items})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


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


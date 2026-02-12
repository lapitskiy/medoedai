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


def _symbol_to_dir_name(symbol: str) -> str:
    """
    DQN stores in result/dqn/<BASE>/... where BASE = symbol without USDT/USD/USDC/...
    Example: TONUSDT -> TON.
    """
    s = (symbol or "").strip().upper().replace("/", "")
    for suf in ("USDT", "USD", "USDC", "BUSD", "USDP"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return s or "UNKNOWN"


def _iter_symbol_runs(symbol: str, model_type: str = "all", limit_runs: int = 200):
    """
    Returns combined list of run dirs (new layout):
      - DQN: result/dqn/<SYMBOL_DIR>/runs/<run_id>/
      - SAC: result/sac/<symbol_lower>/runs/<run_id>/
    """
    mt = (model_type or "all").strip().lower()
    sym_in = (symbol or "").strip()
    if not sym_in:
        return []

    roots: list[tuple[str, Path]] = []
    if mt in ("all", "dqn"):
        sym_dir = _symbol_to_dir_name(sym_in)
        roots.append(("dqn", Path("result") / "dqn" / sym_dir / "runs"))
    if mt in ("all", "sac"):
        sym_dir = sym_in.strip().lower().replace("/", "")
        roots.append(("sac", Path("result") / "sac" / sym_dir / "runs"))

    items = []
    for kind, base in roots:
        if not base.exists():
            continue
        for run_dir in base.iterdir():
            if not run_dir.is_dir():
                continue
            manifest_path = run_dir / "manifest.json"
            pkl_path = run_dir / "train_result.pkl"
            if not manifest_path.exists() or not pkl_path.exists():
                continue
            try:
                mtime = float(run_dir.stat().st_mtime)
            except Exception:
                mtime = 0.0
            items.append(
                {
                    "model_type": kind,
                    "run_dir": run_dir,
                    "manifest_path": manifest_path,
                    "pkl_path": pkl_path,
                    "mtime": mtime,
                }
            )

    items.sort(key=lambda x: float(x.get("mtime") or 0.0), reverse=True)
    if limit_runs and limit_runs > 0:
        items = items[: int(limit_runs)]
    return items


@system_models_bp.get("/api/system/hypotheses_symbols")
def api_system_hypotheses_symbols():
    """
    Lists available symbols for hypotheses page (from result/dqn, result/sac, result/xgb).
    Query:
      - model_type: all|dqn|sac|xgb (default all)
    """
    try:
        mt = (request.args.get("model_type") or "all").strip().lower()
        out = set()
        if mt in ("all", "dqn"):
            base = Path("result") / "dqn"
            if base.exists():
                for p in base.iterdir():
                    if p.is_dir() and (p / "runs").exists():
                        out.add(p.name.upper())
        if mt in ("all", "sac"):
            base = Path("result") / "sac"
            if base.exists():
                for p in base.iterdir():
                    if p.is_dir() and (p / "runs").exists():
                        out.add(p.name.upper())
        if mt == "xgb":
            base = Path("result") / "xgb"
            if base.exists():
                for p in base.iterdir():
                    if p.is_dir() and (p / "runs").exists():
                        out.add(p.name.upper())
        syms = sorted(out)
        return jsonify({"success": True, "model_type": mt, "symbols": syms})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@system_models_bp.get("/api/system/hypotheses_export")
def api_system_hypotheses_export():
    """
    Экспорт агрегированных данных по runs для генерации гипотез.
    Возвращает JSON: {csv, prompt}.
    Query:
      - symbol (required)
      - model_type: all|dqn|sac (default all)
      - limit_runs (default 200)
    """
    try:
        symbol = (request.args.get("symbol") or "").strip().upper()
        if not symbol:
            return jsonify({"success": False, "error": "symbol is required"}), 400
        model_type = (request.args.get("model_type") or "all").strip().lower()
        try:
            limit_runs = int(request.args.get("limit_runs") or "200")
        except Exception:
            limit_runs = 200

        rows = []
        items = _iter_symbol_runs(symbol, model_type=model_type, limit_runs=limit_runs)
        for it in items:
            run_dir: Path = it["run_dir"]
            mt = str(it.get("model_type") or "unknown")
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

            gym = data.get("gym_snapshot") if isinstance(data.get("gym_snapshot"), dict) else {}
            risk = gym.get("risk_management") if isinstance(gym.get("risk_management"), dict) else {}
            adapt = manifest.get("adaptive_params") if isinstance(manifest.get("adaptive_params"), dict) else {}

            # Risk params: prefer gym_snapshot (actual per-run values), else adaptive_params as fallback.
            risk_source = "none"
            sl_v = None
            tp_v = None
            min_hold_v = None
            vol_thr_v = None
            trail_v = None
            if risk and any(k in risk for k in ("STOP_LOSS_PCT", "TAKE_PROFIT_PCT", "min_hold_steps", "volume_threshold")):
                risk_source = "gym_snapshot"
                sl_v = risk.get("STOP_LOSS_PCT")
                tp_v = risk.get("TAKE_PROFIT_PCT")
                min_hold_v = risk.get("min_hold_steps")
                vol_thr_v = risk.get("volume_threshold")
                trail_v = risk.get("atr_trail_mult")
            elif adapt and any(k in adapt for k in ("stop_loss_pct", "take_profit_pct", "min_hold_steps", "volume_threshold")):
                risk_source = "adaptive_params"
                sl_v = adapt.get("stop_loss_pct")
                tp_v = adapt.get("take_profit_pct")
                min_hold_v = adapt.get("min_hold_steps")
                vol_thr_v = adapt.get("volume_threshold")
            trades_roi = data.get("trades_roi") if isinstance(data.get("trades_roi"), list) else []
            roi_sum = None
            try:
                vals = [float(x) for x in trades_roi if x is not None]
                if vals:
                    roi_sum = float(sum(vals))
            except Exception:
                roi_sum = None

            model_file = None
            try:
                mp = run_dir / "model.pth"
                model_file = mp.as_posix() if mp.exists() else None
            except Exception:
                model_file = None

            row = {
                "model_type": mt,
                "run_id": manifest.get("run_id") or run_dir.name,
                "direction": manifest.get("direction") or manifest.get("trained_as"),
                # risk knobs
                "sl_pct": _safe_float(sl_v),
                "tp_pct": _safe_float(tp_v),
                "min_hold_steps": (int(min_hold_v) if isinstance(min_hold_v, (int, float)) else None),
                "volume_threshold": _safe_float(vol_thr_v),
                "atr_trail_mult": _safe_float(trail_v),
                # DQN/SAC hyperparams (from cfg_snapshot in train_result.pkl)
                "cfg_lr": _safe_float(cfg.get("lr")),
                "cfg_batch_size": (int(cfg.get("batch_size")) if isinstance(cfg.get("batch_size"), (int, float)) else None),
                "cfg_memory_size": (int(cfg.get("memory_size")) if isinstance(cfg.get("memory_size"), (int, float)) else None),
                "cfg_hidden_sizes": (
                    ",".join(str(int(x)) for x in cfg.get("hidden_sizes"))
                    if isinstance(cfg.get("hidden_sizes"), (list, tuple)) and cfg.get("hidden_sizes")
                    else None
                ),
                "cfg_train_repeats": (int(cfg.get("train_repeats")) if isinstance(cfg.get("train_repeats"), (int, float)) else None),
                "cfg_use_amp": (bool(cfg.get("use_amp")) if cfg.get("use_amp") is not None else None),
                "cfg_use_gpu_storage": (bool(cfg.get("use_gpu_storage")) if cfg.get("use_gpu_storage") is not None else None),
                "cfg_use_torch_compile": (bool(cfg.get("use_torch_compile")) if cfg.get("use_torch_compile") is not None else None),
                "cfg_eps_decay_steps": (int(cfg.get("eps_decay_steps")) if isinstance(cfg.get("eps_decay_steps"), (int, float)) else None),
                "cfg_dropout_rate": _safe_float(cfg.get("dropout_rate")),
                # outcome metrics
                "roi_sum": _safe_float(roi_sum),
                "avg_roi": _safe_float(stats.get("avg_roi")),
                "pl_ratio": _safe_float(stats.get("pl_ratio")),
                "winrate_train_all": _safe_float(wr_all),
                "trades_count": int(stats.get("trades_count") or 0),
                "avg_profit": _safe_float(stats.get("avg_profit")),
                "avg_loss": _safe_float(stats.get("avg_loss")),
            }

            # sell_types_total → store as percents (agent/stop_loss/take_profit/timeout/trailing)
            try:
                st = data.get("sell_types_total") if isinstance(data.get("sell_types_total"), dict) else {}
                total = float(sum(float(v) for v in st.values() if isinstance(v, (int, float))))
                if total > 0:
                    row["sell_agent_pct"] = float(st.get("agent", 0) or 0) / total
                    row["sell_stop_loss_pct"] = float(st.get("stop_loss", 0) or 0) / total
                    row["sell_take_profit_pct"] = float(st.get("take_profit", 0) or 0) / total
                    row["sell_trailing_pct"] = float(st.get("trailing", 0) or 0) / total
                    row["sell_timeout_pct"] = float(st.get("timeout", 0) or 0) / total
                else:
                    row["sell_agent_pct"] = None
                    row["sell_stop_loss_pct"] = None
                    row["sell_take_profit_pct"] = None
                    row["sell_trailing_pct"] = None
                    row["sell_timeout_pct"] = None
            except Exception:
                row["sell_agent_pct"] = None
                row["sell_stop_loss_pct"] = None
                row["sell_take_profit_pct"] = None
                row["sell_timeout_pct"] = None
            rows.append(row)

        # CSV
        buf = io.StringIO()
        # CSV columns: keep stable order (by importance). Any unknown fields go to the end.
        preferred = [
            "model_type",
            "run_id",
            "direction",
            # risk knobs
            "sl_pct",
            "tp_pct",
            "min_hold_steps",
            "volume_threshold",
            "atr_trail_mult",
            # cfg hyperparams
            "cfg_lr",
            "cfg_batch_size",
            "cfg_memory_size",
            "cfg_hidden_sizes",
            "cfg_train_repeats",
            "cfg_use_amp",
            "cfg_use_gpu_storage",
            "cfg_use_torch_compile",
            "cfg_eps_decay_steps",
            "cfg_dropout_rate",
            # outcome
            "roi_sum",
            "avg_roi",
            "pl_ratio",
            "winrate_train_all",
            "trades_count",
            "avg_profit",
            "avg_loss",
            # sell composition
            "sell_agent_pct",
            "sell_stop_loss_pct",
            "sell_take_profit_pct",
            "sell_trailing_pct",
            "sell_timeout_pct",
        ]
        all_fields = set(k for r in rows for k in r.keys())
        tail = sorted([k for k in all_fields if k not in preferred])
        fieldnames = [k for k in preferred if k in all_fields] + tail
        w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore", restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
        csv_text = buf.getvalue()

        # TXT prompt (compact: no duplicated top lists; CSV is the source of truth)
        lines = []
        lines.append(f"Symbol: {symbol}")
        lines.append(f"Runs scanned: {len(rows)}")
        lines.append("")
        lines.append("Task: analyze the CSV (provided separately) and propose hypotheses which settings correlate with better outcome.")
        lines.append("Primary targets: roi_sum, avg_roi, pl_ratio, winrate_train_all, trades_count.")
        lines.append("Key knobs: sl_pct, tp_pct, min_hold_steps, volume_threshold.")
        lines.append("")
        lines.append("Output format:")
        lines.append("1) 5-10 hypotheses (reference specific CSV columns and patterns)")
        lines.append("2) 3-5 next experiments (exact parameter changes to validate)")
        lines.append("3) caveats (overfitting, low trades_count, conflicting objectives)")
        prompt_text = "\n".join(lines)

        return jsonify(
            {
                "success": True,
                "symbol": symbol,
                "model_type": model_type,
                "runs_count": len(rows),
                "csv": csv_text,
                "prompt": prompt_text,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@system_models_bp.get("/api/system/hypotheses_export_xgb")
def api_system_hypotheses_export_xgb():
    """
    Экспорт XGB runs для генерации гипотез.
    Отдельный endpoint, чтобы не смешивать с DQN/SAC.
    """
    try:
        symbol = (request.args.get("symbol") or "").strip().upper()
        if not symbol:
            return jsonify({"success": False, "error": "symbol is required"}), 400
        try:
            limit_runs = int(request.args.get("limit_runs") or "500")
        except Exception:
            limit_runs = 500

        sym_dir = _symbol_to_dir_name(symbol)
        base = Path("result") / "xgb" / sym_dir / "runs"
        if not base.exists():
            return jsonify({"success": False, "error": f"No XGB runs for {symbol}"}), 404

        items = []
        for rd in base.iterdir():
            if not rd.is_dir():
                continue
            manifest_p = rd / "manifest.json"
            metrics_p = rd / "metrics.json"
            meta_p = rd / "meta.json"
            if not manifest_p.exists() or not metrics_p.exists():
                continue
            try:
                mtime = float(rd.stat().st_mtime)
            except Exception:
                mtime = 0.0
            items.append({"run_dir": rd, "manifest_p": manifest_p, "metrics_p": metrics_p, "meta_p": meta_p, "mtime": mtime})

        items.sort(key=lambda x: x["mtime"], reverse=True)
        items = items[:limit_runs]

        rows = []
        for it in items:
            manifest = {}
            metrics = {}
            meta = {}
            try:
                manifest = json.loads(it["manifest_p"].read_text(encoding="utf-8"))
            except Exception:
                pass
            try:
                metrics = json.loads(it["metrics_p"].read_text(encoding="utf-8"))
            except Exception:
                pass
            try:
                if it["meta_p"].exists():
                    meta = json.loads(it["meta_p"].read_text(encoding="utf-8"))
            except Exception:
                pass

            cfg = meta.get("cfg_snapshot") if isinstance(meta.get("cfg_snapshot"), dict) else {}

            # f1_val is a list: [hold, buy, sell] or [hold, enter]
            f1v = metrics.get("f1_val")
            f1_1 = None
            prec_1 = None
            recall_1 = None
            if isinstance(f1v, list) and len(f1v) > 1:
                f1_1 = _safe_float(f1v[1])
            pv = metrics.get("precision_val")
            if isinstance(pv, list) and len(pv) > 1:
                prec_1 = _safe_float(pv[1])
            rv = metrics.get("recall_val")
            if isinstance(rv, list) and len(rv) > 1:
                recall_1 = _safe_float(rv[1])

            proxy = metrics.get("proxy_pnl_val") if isinstance(metrics.get("proxy_pnl_val"), dict) else {}
            proxy_sum = _safe_float(proxy.get("pnl_sum"))
            proxy_trades = None
            try:
                proxy_trades = int(proxy.get("trades")) if proxy.get("trades") is not None else None
            except Exception:
                proxy_trades = None
            proxy_mean_trade = _safe_float(proxy.get("pnl_mean_per_trade"))

            row = {
                "run_id": manifest.get("run_name") or it["run_dir"].name,
                # labeling params
                "horizon_steps": cfg.get("horizon_steps"),
                "threshold": _safe_float(cfg.get("threshold")),
                "max_hold_steps": cfg.get("max_hold_steps"),
                "min_profit": _safe_float(cfg.get("min_profit")),
                "label_delta": _safe_float(cfg.get("label_delta")),
                # model hyperparams
                "max_depth": cfg.get("max_depth"),
                "learning_rate": _safe_float(cfg.get("learning_rate")),
                "n_estimators": cfg.get("n_estimators"),
                "subsample": _safe_float(cfg.get("subsample")),
                "colsample_bytree": _safe_float(cfg.get("colsample_bytree")),
                "reg_lambda": _safe_float(cfg.get("reg_lambda")),
                "min_child_weight": _safe_float(cfg.get("min_child_weight")),
                "gamma": _safe_float(cfg.get("gamma")),
                "scale_pos_weight": _safe_float(cfg.get("scale_pos_weight")),
                # metrics
                "val_acc": _safe_float(metrics.get("val_acc")),
                "f1_buy_sell": _safe_float(metrics.get("f1_buy_sell_val")),
                "f1_1": f1_1,
                "prec_1": prec_1,
                "recall_1": recall_1,
                "y_non_hold": _safe_float(metrics.get("y_non_hold_rate_val")),
                "pred_non_hold": _safe_float(metrics.get("pred_non_hold_rate_val")),
                "proxy_pnl_sum": proxy_sum,
                "proxy_pnl_mean_trade": proxy_mean_trade,
                "proxy_pnl_trades": proxy_trades,
                "train_rows": metrics.get("train_rows"),
                "val_rows": metrics.get("val_rows"),
            }
            rows.append(row)

        # CSV
        buf = io.StringIO()
        preferred = [
            "run_id",
            "horizon_steps", "threshold", "max_hold_steps", "min_profit", "label_delta",
            "max_depth", "learning_rate", "n_estimators", "subsample", "colsample_bytree",
            "reg_lambda", "min_child_weight", "gamma", "scale_pos_weight",
            "val_acc", "f1_buy_sell", "f1_1", "prec_1", "recall_1",
            "y_non_hold", "pred_non_hold", "train_rows", "val_rows",
            "proxy_pnl_sum", "proxy_pnl_mean_trade", "proxy_pnl_trades",
        ]
        all_fields = set(k for r in rows for k in r.keys())
        tail = sorted([k for k in all_fields if k not in preferred])
        fieldnames = [k for k in preferred if k in all_fields] + tail
        w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore", restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
        csv_text = buf.getvalue()

        # TXT prompt
        lines = [
            f"Symbol: {symbol}",
            f"Model: XGBoost",
            f"Runs scanned: {len(rows)}",
            "",
            "Task: analyze the CSV and propose hypotheses which XGB settings correlate with better prediction quality.",
            "Primary targets: f1_1 (F1 class=1), prec_1, recall_1, val_acc.",
            "Key labeling knobs: max_hold_steps, min_profit, threshold, fee_bps.",
            "Key model knobs: max_depth, learning_rate, n_estimators, subsample, colsample_bytree, reg_lambda, min_child_weight, gamma, scale_pos_weight.",
            "",
            "Output format:",
            "1) 5-10 hypotheses (reference specific CSV columns and patterns)",
            "2) 3-5 next experiments (exact parameter changes to validate)",
            "3) caveats (overfitting, class imbalance, y_non_hold vs pred_non_hold mismatch)",
        ]
        prompt_text = "\n".join(lines)

        return jsonify({
            "success": True, "symbol": symbol, "model_type": "xgb",
            "runs_count": len(rows), "csv": csv_text, "prompt": prompt_text,
        })
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


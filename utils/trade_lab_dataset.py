import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _symbol_to_dir_name(symbol: str) -> str:
    """
    DQN сохраняет в result/dqn/<BASE>/... где BASE = symbol без суффиксов USDT/USD/USDC/...
    Например TONUSDT -> TON.
    """
    s = (symbol or "").strip().upper().replace("/", "")
    for suf in ("USDT", "USD", "USDC", "BUSD", "USDP"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return s or "UNKNOWN"


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _safe_unpickle(path: Path, max_mb: int = 64) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        try:
            sz = path.stat().st_size
            if sz > (max_mb * 1024 * 1024):
                return {"_error": f"train_result.pkl too large: {sz} bytes"}
        except Exception:
            pass
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        return {"_error": str(e)}


def _pick(d: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in keys:
        if k in d:
            out[k] = d.get(k)
    return out


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _jsonable(v: Any) -> Any:
    """
    Make payload JSON-serializable (handles numpy scalars like np.bool_).
    """
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    try:
        if isinstance(v, Path):
            return v.as_posix()
    except Exception:
        pass
    if isinstance(v, dict):
        out: Dict[str, Any] = {}
        for k, vv in v.items():
            out[str(k)] = _jsonable(vv)
        return out
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    # numpy scalars
    try:
        import numpy as _np  # type: ignore

        if isinstance(v, (_np.bool_,)):
            return bool(v)
        if isinstance(v, (_np.integer,)):
            return int(v)
        if isinstance(v, (_np.floating,)):
            return float(v)
    except Exception:
        pass
    # common scalar containers
    try:
        if hasattr(v, "item") and callable(getattr(v, "item")):
            return _jsonable(v.item())
    except Exception:
        pass
    try:
        if hasattr(v, "tolist") and callable(getattr(v, "tolist")):
            return _jsonable(v.tolist())
    except Exception:
        pass
    # fallback
    return str(v)


@dataclass
class TradeLabDatasetConfig:
    model_type: str = "dqn"  # dqn|sac
    base_dir: str = "result"
    out_dir: str = "result/trade_lab_tmp"
    max_pkl_mb: int = 64


def build_symbol_runs_dataset(symbol: str, cfg: Optional[TradeLabDatasetConfig] = None) -> Dict[str, Any]:
    cfg = cfg or TradeLabDatasetConfig()
    model_type = (cfg.model_type or "dqn").strip().lower()
    base = Path(cfg.base_dir)
    out_sym = (symbol or "").strip()
    if not out_sym:
        raise ValueError("symbol is required")

    if model_type == "sac":
        sym_dir = out_sym.lower()
        runs_root = base / "sac" / sym_dir / "runs"
        symbol_norm = out_sym.upper()
    else:
        sym_dir = _symbol_to_dir_name(out_sym)
        runs_root = base / "dqn" / sym_dir / "runs"
        symbol_norm = sym_dir

    runs = []
    if runs_root.exists():
        for rd in runs_root.iterdir():
            if not rd.is_dir():
                continue
            mf = _safe_read_json(rd / "manifest.json")
            tr = _safe_unpickle(rd / "train_result.pkl", max_mb=int(cfg.max_pkl_mb))

            # features (whitelist)
            cfg_snap = tr.get("cfg_snapshot") if isinstance(tr.get("cfg_snapshot"), dict) else {}
            gym_snap = tr.get("gym_snapshot") if isinstance(tr.get("gym_snapshot"), dict) else {}
            adapt = tr.get("adaptive_normalization") if isinstance(tr.get("adaptive_normalization"), dict) else {}
            risk_snap = gym_snap.get("risk_management") if isinstance(gym_snap.get("risk_management"), dict) else {}

            features = {
                "direction": (mf.get("direction") or mf.get("trained_as")),
                "seed": (mf.get("seed") if mf else None),
                "cfg": _pick(
                    cfg_snap,
                    (
                        "lr",
                        "batch_size",
                        "memory_size",
                        "train_repeats",
                        "use_noisy_networks",
                        "eps_start",
                        "eps_final",
                        "eps_decay_steps",
                        "target_update_freq",
                        "soft_update_every",
                        "buffer_save_frequency",
                        "winrate_eps_threshold",
                    ),
                ),
                "gym": _pick(
                    gym_snap,
                    (
                        "lookback_window",
                        "indicators_config",
                        "episode_length",
                        "step_minutes",
                    ),
                ),
                # Risk params that materially change trading behavior (requested by user)
                "risk": _pick(
                    risk_snap,
                    (
                        "STOP_LOSS_PCT",
                        "TAKE_PROFIT_PCT",
                        "min_hold_steps",
                        "volume_threshold",
                    ),
                ),
                "adaptive": _pick(
                    adapt,
                    (
                        "trading_params",
                        "market_conditions",
                        "adapted_params",
                    ),
                ),
            }

            # metrics (whitelist)
            winrates = tr.get("winrates") if isinstance(tr.get("winrates"), dict) else {}
            final_stats = tr.get("final_stats") if isinstance(tr.get("final_stats"), dict) else {}
            trades_roi = tr.get("trades_roi") if isinstance(tr.get("trades_roi"), list) else []
            pnl_roi_sum = None
            pnl_roi_avg = None
            try:
                if trades_roi:
                    vals = [float(x) for x in trades_roi if x is not None]
                    if vals:
                        pnl_roi_sum = float(sum(vals))
                        pnl_roi_avg = float(sum(vals) / float(len(vals)))
            except Exception:
                pnl_roi_sum = None
                pnl_roi_avg = None
            metrics = {
                "best_winrate": tr.get("best_winrate"),
                "winrates": _pick(winrates, ("train_all", "train_exploit", "eps_threshold")),
                # final_stats is used as "headline" metrics for trade quality
                "final_stats": _pick(
                    final_stats,
                    (
                        "winrate",
                        "avg_roi",
                        "pl_ratio",
                        "trades_count",
                        "bad_trades_count",
                        "avg_profit",
                        "avg_loss",
                    ),
                ),
                # pnl proxy computed from compact ROI series (kept small in train_result.pkl)
                "pnl": {
                    "roi_sum": pnl_roi_sum,
                    "roi_avg": pnl_roi_avg,
                    "roi_count": (len(trades_roi) if isinstance(trades_roi, list) else 0),
                },
                "all_trades_count": tr.get("all_trades_count"),
                "episodes_planned": tr.get("episodes"),
                "episodes_completed": tr.get("actual_episodes"),
                "total_training_time": tr.get("total_training_time"),
            }

            runs.append(
                {
                    "run_id": rd.name,
                    "run_dir": rd.as_posix(),
                    "manifest": _pick(mf, ("run_id", "parent_run_id", "root_id", "symbol", "seed", "direction", "trained_as")),
                    "features": features,
                    "metrics": metrics,
                    "_errors": _pick(tr, ("_error",)),
                }
            )

    dataset = {
        "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_type": model_type,
        "symbol_input": out_sym,
        "symbol_dir": symbol_norm,
        "runs_root": runs_root.as_posix(),
        "runs_count": len(runs),
        "runs": runs,
    }
    return _jsonable(dataset)


def write_symbol_runs_dataset_to_tmp(symbol: str, cfg: Optional[TradeLabDatasetConfig] = None) -> str:
    cfg = cfg or TradeLabDatasetConfig()
    dataset = build_symbol_runs_dataset(symbol, cfg=cfg)
    sym_dir = dataset.get("symbol_dir") or _symbol_to_dir_name(symbol)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.out_dir) / str(sym_dir)
    out_path = out_dir / f"trade_lab_dataset_{str(cfg.model_type).lower()}_{sym_dir}_{ts}.json"
    _atomic_write_json(out_path, dataset)
    return out_path.as_posix()


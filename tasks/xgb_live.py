from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.xgb.config import XgbConfig
from agents.xgb.features import _build_base_features, uses_1m_features, uses_1d_regime
from agents.xgb.predictor import XgbPredictor
from tasks.xgb_tasks import _build_dfs_from_5m, _filter_1m_to_5m_window, _filter_1d_to_5m_window
from utils.db_utils import db_get_or_fetch_ohlcv


_INDICATOR_FEATURE_NAMES = [
    "RSI_14",
    "RSI_7",
    "EMA_20",
    "EMA_50",
    "EMA_100",
    "EMA_200",
    "SMA_14",
    "EMA_20_above_50",
    "EMA_20_cross_up_50",
    "EMA_20_cross_down_50",
    "EMA_100_above_200",
    "EMA_100_cross_up_200",
    "EMA_100_cross_down_200",
    "ATR_14",
    "OBV",
    "RET_1",
    "RET_3",
    "RET_12",
    "RET_60",
    "ZSCORE_50_20",
]


def _load_xgb_runtime_meta(model_path: str) -> tuple[XgbConfig, str, str]:
    cfg = XgbConfig()
    task_name = str(getattr(cfg, "task", "directional") or "directional").strip().lower()
    direction_name = str(getattr(cfg, "direction", "long") or "long").strip().lower()

    model_dir = Path(model_path).resolve().parent
    meta_path = model_dir / "meta.json"
    manifest_path = model_dir / "manifest.json"

    try:
        if meta_path.exists():
            meta_doc = json.loads(meta_path.read_text(encoding="utf-8"))
            cfg_snap = meta_doc.get("cfg_snapshot") if isinstance(meta_doc, dict) else {}
            if isinstance(cfg_snap, dict):
                for key, value in cfg_snap.items():
                    if hasattr(cfg, key):
                        try:
                            setattr(cfg, key, value)
                        except Exception:
                            pass
    except Exception:
        pass

    try:
        if manifest_path.exists():
            manifest_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(manifest_doc, dict):
                task_name = str(
                    getattr(cfg, "task", None) or manifest_doc.get("task") or task_name
                ).strip().lower()
                direction_name = str(
                    getattr(cfg, "direction", None)
                    or manifest_doc.get("direction")
                    or direction_name
                ).strip().lower()
    except Exception:
        pass

    return cfg, task_name, direction_name


def _xgb_feature_names(cfg: XgbConfig, feature_dim: int, *, include_trade_features: bool) -> list[str]:
    names = [
        "5m_open", "5m_high", "5m_low", "5m_close", "5m_volume",
        "15m_open", "15m_high", "15m_low", "15m_close", "15m_volume",
        "1h_open", "1h_high", "1h_low", "1h_close", "1h_volume",
        *_INDICATOR_FEATURE_NAMES,
    ]
    for attr in ("_feature_names_sr", "_feature_names_1m", "_feature_names_1d"):
        values = getattr(cfg, attr, None)
        if isinstance(values, list):
            names.extend(str(v) for v in values)
    if include_trade_features:
        names.extend(["trade_age_norm", "trade_pnl_fee", "trade_mfe", "trade_mae"])
    if len(names) < feature_dim:
        names.extend(f"feature_{i}" for i in range(len(names), feature_dim))
    return names[:feature_dim]


def _calc_xgb_shap(predictor: XgbPredictor, x_input: np.ndarray, feature_names: list[str], class_idx: int | None = None) -> dict[str, Any]:
    booster = predictor.model.get_booster()
    contribs = booster.predict(predictor._xgb.DMatrix(x_input), pred_contribs=True)
    arr = np.asarray(contribs, dtype=np.float64)
    if arr.ndim == 3:
        idx = int(class_idx or 0)
        idx = max(0, min(idx, arr.shape[1] - 1))
        values_with_bias = arr[0, idx, :]
    else:
        values_with_bias = arr[0, :]
    shap_values = values_with_bias[:-1]
    bias = float(values_with_bias[-1]) if values_with_bias.size else None
    names = feature_names[: len(shap_values)]
    full = [
        {"feature_name": str(name), "shap_value": float(value)}
        for name, value in zip(names, shap_values)
    ]
    full.sort(key=lambda item: abs(float(item["shap_value"])), reverse=True)
    return {"top_features": full[:5], "full_vector": full, "base_value": bias}


def predict_xgb_live(
    *,
    symbol: str,
    model_paths: list[str],
    df_5m: pd.DataFrame,
    threshold_override: float | None = None,
    threshold_overrides_by_model: dict[str, float] | None = None,
) -> dict[str, Any]:
    preds_xgb: list[dict[str, Any]] = []
    labels_xgb: list[str] = []
    q_values_first: list[float] | None = None

    df_5m_base = df_5m[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    dfs_xgb_base = _build_dfs_from_5m(df_5m_base)
    df_1m_cache: pd.DataFrame | None = None
    df_1d_cache: pd.DataFrame | None = None

    for model_path in model_paths:
        cfg, task_name, direction_name = _load_xgb_runtime_meta(str(model_path))

        dfs_xgb = dfs_xgb_base
        if uses_1m_features(cfg) or uses_1d_regime(cfg):
            if uses_1m_features(cfg) and df_1m_cache is None:
                limit_1m = int(max(1000, len(df_5m_base) * 5 + 1000))
                df_1m_cache = db_get_or_fetch_ohlcv(
                    symbol_name=symbol,
                    timeframe="1m",
                    limit_candles=limit_1m,
                    exchange_id="bybit",
                )
                if df_1m_cache is None or df_1m_cache.empty:
                    raise RuntimeError(f"1m candles are required for XGB model {model_path}")
                df_1m_cache = _filter_1m_to_5m_window(df_1m_cache, df_5m_base)
            if uses_1d_regime(cfg) and df_1d_cache is None:
                limit_1d = int(max(120, len(df_5m_base) // 288 + 120))
                df_1d_cache = db_get_or_fetch_ohlcv(
                    symbol_name=symbol,
                    timeframe="1d",
                    limit_candles=limit_1d,
                    exchange_id="bybit",
                )
                if df_1d_cache is None or df_1d_cache.empty:
                    raise RuntimeError(f"1d candles are required for XGB model {model_path}")
                df_1d_cache = _filter_1d_to_5m_window(df_1d_cache, df_5m_base)
            dfs_xgb = _build_dfs_from_5m(
                df_5m_base,
                df_1min=(df_1m_cache if uses_1m_features(cfg) else None),
                df_1d=(df_1d_cache if uses_1d_regime(cfg) else None),
            )

        X_base, closes = _build_base_features(dfs_xgb, cfg)
        if len(closes) > 300:
            X_base = X_base[200:]
        if X_base is None or len(X_base) == 0:
            raise RuntimeError(f"xgb features are empty for {model_path}")

        predictor = XgbPredictor(model_path=str(model_path))
        is_binary_task = task_name.startswith("entry") or task_name.startswith("exit")

        model_threshold_override = None
        if isinstance(threshold_overrides_by_model, dict):
            model_threshold_override = threshold_overrides_by_model.get(str(model_path))
            if model_threshold_override is None:
                model_threshold_override = threshold_overrides_by_model.get(str(model_path).replace("\\", "/"))
        runtime_threshold = (
            float(model_threshold_override)
            if isinstance(model_threshold_override, (int, float))
            else float(threshold_override)
            if isinstance(threshold_override, (int, float))
            else float(getattr(cfg, "p_enter_threshold", 0.5) or 0.5)
        )

        if is_binary_task:
            x_last = X_base[-1:].astype(np.float32)
            trade_feats = np.zeros((1, 4), dtype=np.float32)
            x_input = np.concatenate([x_last, trade_feats], axis=1)
            proba = predictor.predict_proba(x_input)[0]
            enter_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
            is_enter = enter_prob >= runtime_threshold

            if task_name == "entry_long":
                action = "buy" if is_enter else "hold"
                q_values = [float(1.0 - enter_prob), float(enter_prob), 0.0]
            elif task_name == "entry_short":
                action = "sell" if is_enter else "hold"
                q_values = [float(1.0 - enter_prob), 0.0, float(enter_prob)]
            elif task_name == "exit_long":
                action = "sell" if is_enter else "hold"
                q_values = [float(1.0 - enter_prob), 0.0, float(enter_prob)]
            elif task_name == "exit_short":
                action = "buy" if is_enter else "hold"
                q_values = [float(1.0 - enter_prob), float(enter_prob), 0.0]
            else:
                action = "hold"
                q_values = [1.0, 0.0, 0.0]
        else:
            x_input = X_base[-1:].astype(np.float32)
            pred = predictor.predict_action(x_input)
            proba = predictor.predict_proba(x_input)[0]
            action = predictor.decode_action(int(pred[0]))
            q_values = [float(x) for x in np.asarray(proba, dtype=np.float32).tolist()]
            if len(q_values) == 2:
                q_values = [q_values[0], q_values[1], 0.0]

        feature_names = _xgb_feature_names(cfg, int(x_input.shape[1]), include_trade_features=is_binary_task)
        class_idx = 1 if is_binary_task else ({"hold": 0, "buy": 1, "sell": 2}.get(str(action), 0))
        shap_doc = _calc_xgb_shap(predictor, x_input, feature_names, class_idx=class_idx)
        confidence = float(max(q_values)) if q_values else 0.0
        if q_values_first is None:
            q_values_first = list(q_values)

        preds_xgb.append(
            {
                "model_path": str(model_path),
                "action": str(action),
                "confidence": confidence,
                "q_values": list(q_values),
                "task": task_name,
                "direction": direction_name,
                "xgb_runtime_threshold": runtime_threshold,
                "xgb_qgate_disabled": True,
                "symbol": symbol,
                "xgb_shap": shap_doc,
            }
        )
        labels_xgb.append(str(action))

    votes_xgb = {"buy": 0, "sell": 0, "hold": 0}
    for label in labels_xgb:
        votes_xgb[label] = votes_xgb.get(label, 0) + 1

    return {
        "success": True,
        "decision": labels_xgb[0] if len(labels_xgb) == 1 else "hold",
        "predictions": preds_xgb,
        "q_values": q_values_first or [],
        "votes": votes_xgb,
        "threshold_used": 1,
        "xgb_qgate_disabled": True,
    }

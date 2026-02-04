from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Tuple, List

import numpy as np

from envs.dqn_model.gym.indicators_optimized import preprocess_dataframes
from .config import XgbConfig


def _upsample_15m_to_5m(df_5m: np.ndarray, df_15m: np.ndarray) -> np.ndarray:
    # Match envs/dqn_model/gym/crypto_trading_env_optimized.py logic (index-based repeat)
    out = np.zeros((len(df_5m), 5), dtype=np.float32)
    if len(df_15m) == 0:
        return out
    for i in range(len(df_5m)):
        j = i // 3
        out[i] = df_15m[j] if j < len(df_15m) else df_15m[-1]
    return out


def _upsample_1h_to_5m(df_5m: np.ndarray, df_1h: np.ndarray) -> np.ndarray:
    out = np.zeros((len(df_5m), 5), dtype=np.float32)
    if len(df_1h) == 0:
        return out
    for i in range(len(df_5m)):
        j = i // 12
        out[i] = df_1h[j] if j < len(df_1h) else df_1h[-1]
    return out


def _build_base_features(dfs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds base market feature matrix X_base aligned to 5m rows, plus close prices.
    """
    df_5m_raw = dfs["df_5min"].values if hasattr(dfs["df_5min"], "values") else dfs["df_5min"]
    df_15m_raw = dfs["df_15min"].values if hasattr(dfs["df_15min"], "values") else dfs["df_15min"]
    df_1h_raw = dfs["df_1h"].values if hasattr(dfs["df_1h"], "values") else dfs["df_1h"]

    # Removes timestamp column (if present) and computes indicators on 5m.
    df_5m, df_15m, df_1h, indicators, _ind_map = preprocess_dataframes(
        df_5m_raw, df_15m_raw, df_1h_raw, indicators_config={
            "rsi": {"length": 14},
            "ema": {"lengths": [100, 200]},
            "ema_cross": {"pairs": [(100, 200)], "include_cross_signal": True},
            "sma": {"length": 14},
        }
    )

    df_15m_5m = _upsample_15m_to_5m(df_5m, df_15m).astype(np.float32)
    df_1h_5m = _upsample_1h_to_5m(df_5m, df_1h).astype(np.float32)
    X_base = np.concatenate([df_5m.astype(np.float32), df_15m_5m, df_1h_5m, indicators.astype(np.float32)], axis=1)
    closes = df_5m[:, 3].astype(np.float64)
    return X_base, closes


def _fee_to_fraction(fee_bps: float) -> float:
    # round-trip bps -> fraction
    return float(fee_bps) / 10000.0


def _best_future_pnl_long(closes: np.ndarray, entry_idx: int, max_hold: int, fee_frac: float) -> float:
    entry = float(closes[entry_idx])
    end = min(len(closes) - 1, entry_idx + max_hold)
    if end <= entry_idx or entry <= 0:
        return 0.0
    best_exit = float(np.max(closes[entry_idx + 1 : end + 1]))
    return (best_exit - entry) / entry - fee_frac


def _best_future_pnl_short(closes: np.ndarray, entry_idx: int, max_hold: int, fee_frac: float) -> float:
    entry = float(closes[entry_idx])
    end = min(len(closes) - 1, entry_idx + max_hold)
    if end <= entry_idx or entry <= 0:
        return 0.0
    best_exit = float(np.min(closes[entry_idx + 1 : end + 1]))
    # short pnl: profit when price goes down
    return (entry - best_exit) / entry - fee_frac


def _build_trade_features_long(closes: np.ndarray, entry_idx: int, t: int) -> Tuple[float, float, float, float]:
    entry = float(closes[entry_idx])
    cur = float(closes[t])
    if entry <= 0:
        return 0.0, 0.0, 0.0, 0.0
    pnl = (cur - entry) / entry
    path = closes[entry_idx : t + 1]
    mfe = (float(np.max(path)) - entry) / entry
    mae = (float(np.min(path)) - entry) / entry
    return pnl, mfe, mae, float(t - entry_idx)


def _build_trade_features_short(closes: np.ndarray, entry_idx: int, t: int) -> Tuple[float, float, float, float]:
    entry = float(closes[entry_idx])
    cur = float(closes[t])
    if entry <= 0:
        return 0.0, 0.0, 0.0, 0.0
    pnl = (entry - cur) / entry
    path = closes[entry_idx : t + 1]
    mfe = (entry - float(np.min(path))) / entry  # best move down
    mae = (entry - float(np.max(path))) / entry  # adverse move up (negative pnl direction)
    mae = -mae
    return pnl, mfe, mae, float(t - entry_idx)


def build_xgb_dataset(dfs: Dict[str, Any], cfg: XgbConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Builds dataset depending on cfg.task.
    - directional: y in {0,1,2} = {hold,buy,sell}
    - entry_* / exit_*: y in {0,1} = {hold, enter/exit}
    """
    X_base, closes = _build_base_features(dfs)
    task = (cfg.task or "directional").strip().lower()

    # Directional (legacy)
    if task == "directional":
        H = int(cfg.horizon_steps)
        thr = float(cfg.threshold)
        y = np.zeros((len(closes),), dtype=np.int64)  # default HOLD
        if len(closes) > H + 2:
            future = np.roll(closes, -H)
            ret = (future - closes) / np.maximum(closes, 1e-12)
            up = ret > thr
            dn = ret < -thr
            direction = (cfg.direction or "long").strip().lower()
            if direction == "short":
                y[dn] = 2  # sell
                y[up] = 1  # buy
            else:
                y[up] = 1
                y[dn] = 2
            y[-H:] = 0
        meta = {
            "feature_dim": int(X_base.shape[1]),
            "rows": int(X_base.shape[0]),
            "label_mapping": {"0": "hold", "1": "buy", "2": "sell"},
            "cfg_snapshot": asdict(cfg),
        }
        return X_base, y, meta

    # Position-aware entry/exit (binary)
    fee_frac = _fee_to_fraction(cfg.fee_bps)
    max_hold = int(cfg.max_hold_steps)
    stride = max(1, int(cfg.entry_stride))
    max_trades = max(1, int(cfg.max_trades))
    min_profit = float(cfg.min_profit)
    delta = float(cfg.label_delta)

    # Build trade samples
    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []

    is_long = task.endswith("long")
    is_entry = task.startswith("entry")
    is_exit = task.startswith("exit")

    # Entry dataset: one sample per time t (like directional), but binary label from best future pnl within window
    if is_entry:
        y = np.zeros((len(closes),), dtype=np.int64)
        for t in range(0, len(closes) - max_hold - 1):
            best = _best_future_pnl_long(closes, t, max_hold, fee_frac) if is_long else _best_future_pnl_short(closes, t, max_hold, fee_frac)
            y[t] = 1 if best >= min_profit else 0
        y[-max_hold:] = 0
        # trade-features are zeros for entry (no position yet)
        trade_feats = np.zeros((len(closes), 4), dtype=np.float32)
        X = np.concatenate([X_base.astype(np.float32), trade_feats], axis=1)
        meta = {
            "feature_dim": int(X.shape[1]),
            "rows": int(X.shape[0]),
            "label_mapping": {"0": "hold", "1": "enter"},
            "cfg_snapshot": asdict(cfg),
        }
        return X, y, meta

    # Exit dataset: simulate many trades opened at sampled entry points and label "exit now" near best future
    if is_exit:
        entries = list(range(0, max(0, len(closes) - max_hold - 1), stride))[:max_trades]
        for e in entries:
            end = min(len(closes) - 1, e + max_hold)
            if end <= e + 1:
                continue
            # precompute best future pnl from each t to end (relative to entry)
            entry = float(closes[e])
            if entry <= 0:
                continue
            window = closes[e : end + 1]
            if is_long:
                best_future_prices = np.maximum.accumulate(window[::-1])[::-1]
                best_future_pnls = (best_future_prices - entry) / entry - fee_frac
            else:
                best_future_prices = np.minimum.accumulate(window[::-1])[::-1]
                best_future_pnls = (entry - best_future_prices) / entry - fee_frac

            for k, t in enumerate(range(e, end + 1)):
                pnl, mfe, mae, age = (_build_trade_features_long(closes, e, t) if is_long else _build_trade_features_short(closes, e, t))
                pnl_fee = pnl - fee_frac
                best_remain = float(best_future_pnls[k])
                # exit if we're near the best remaining and in profit
                y_t = 1 if (pnl_fee >= min_profit and pnl_fee >= (best_remain - delta)) else 0
                age_norm = float(age) / float(max_hold) if max_hold > 0 else 0.0
                feats = np.array([age_norm, float(pnl_fee), float(mfe), float(mae)], dtype=np.float32)
                X_rows.append(np.concatenate([X_base[t].astype(np.float32), feats], axis=0))
                y_rows.append(int(y_t))

        if not X_rows:
            X = np.zeros((0, X_base.shape[1] + 4), dtype=np.float32)
            y = np.zeros((0,), dtype=np.int64)
        else:
            X = np.stack(X_rows, axis=0).astype(np.float32)
            y = np.array(y_rows, dtype=np.int64)
        meta = {
            "feature_dim": int(X.shape[1]),
            "rows": int(X.shape[0]),
            "label_mapping": {"0": "hold", "1": "exit"},
            "cfg_snapshot": asdict(cfg),
        }
        return X, y, meta

    # Fallback to directional
    return X_base, np.zeros((len(closes),), dtype=np.int64), {"cfg_snapshot": asdict(cfg)}


from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Tuple, List, Optional

import numpy as np

from envs.dqn_model.gym.indicators_optimized import preprocess_dataframes
from .config import XgbConfig


_TF_5M_MS = 5 * 60 * 1000
_TF_1D_MS = 24 * 60 * 60 * 1000


def uses_1m_features(cfg: XgbConfig) -> bool:
    return bool(
        getattr(cfg, "use_1m_microvol", False)
        or getattr(cfg, "use_1m_momentum", False)
        or getattr(cfg, "use_1m_candle_structure", False)
        or getattr(cfg, "use_1m_volume", False)
    )


def uses_1d_regime(cfg: XgbConfig) -> bool:
    return bool(getattr(cfg, "use_1d_regime", False))


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


def _as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def uses_sr_features(cfg: XgbConfig) -> bool:
    return bool(getattr(cfg, "use_sr_features", False))


def _atr_abs_ohlc(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> np.ndarray:
    if len(close) == 0:
        return np.zeros((0,), dtype=np.float64)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return _ema(tr, int(length))


def _build_sr_features(df_5m: np.ndarray, cfg: XgbConfig) -> Tuple[np.ndarray, List[str]]:
    names = [
        "sr_dist_support_atr",
        "sr_dist_resistance_atr",
        "sr_support_touch_count",
        "sr_resistance_touch_count",
        "sr_near_support_flag",
        "sr_near_resistance_flag",
        "sr_breakout_up_flag",
        "sr_breakdown_down_flag",
        "local_swing_high_flag",
        "local_swing_low_flag",
        "dist_to_last_swing_high_atr",
        "dist_to_last_swing_low_atr",
        "sr_rejection_up_flag",
        "sr_rejection_down_flag",
        "sr_bounce_momentum_3",
        "sr_bounce_momentum_6",
        "range_compression_atr",
        "consolidation_flag",
    ]
    if not uses_sr_features(cfg):
        return np.zeros((len(df_5m), 0), dtype=np.float32), []

    lookback = max(2, int(getattr(cfg, "sr_lookback_steps", 288)))
    min_window = max(2, int(getattr(cfg, "sr_min_window_steps", 48)))
    tol_atr = max(0.0, float(getattr(cfg, "sr_touch_tolerance_atr", 0.25)))
    swing_window = max(2, int(getattr(cfg, "sr_swing_window_steps", 12)))
    consolidation_window = max(2, int(getattr(cfg, "sr_consolidation_window_steps", 24)))
    consolidation_atr = max(0.0, float(getattr(cfg, "sr_consolidation_atr_threshold", 4.0)))

    open_ = df_5m[:, 0].astype(np.float64, copy=False)
    high = df_5m[:, 1].astype(np.float64, copy=False)
    low = df_5m[:, 2].astype(np.float64, copy=False)
    close = df_5m[:, 3].astype(np.float64, copy=False)
    atr = np.maximum(_atr_abs_ohlc(high, low, close, 14), np.maximum(close, 1e-12) * 1e-6)
    rows = np.zeros((len(close), len(names)), dtype=np.float32)
    last_swing_high = np.nan
    last_swing_low = np.nan

    for i in range(len(close)):
        start = max(0, i - lookback)
        if i - start < min_window:
            continue
        w_high = high[start:i]
        w_low = low[start:i]
        c = float(close[i])
        a = float(atr[i])
        tol = max(a * tol_atr, c * 1e-6)

        support_candidates = w_low[w_low <= c]
        resistance_candidates = w_high[w_high >= c]
        support = float(np.max(support_candidates)) if len(support_candidates) else float(np.min(w_low))
        resistance = float(np.min(resistance_candidates)) if len(resistance_candidates) else float(np.max(w_high))

        rows[i, 0] = float((c - support) / a)
        rows[i, 1] = float((resistance - c) / a)
        rows[i, 2] = float(np.sum(np.abs(w_low - support) <= tol))
        rows[i, 3] = float(np.sum(np.abs(w_high - resistance) <= tol))
        rows[i, 4] = 1.0 if (float(low[i]) <= support + tol and c >= support - tol) else 0.0
        rows[i, 5] = 1.0 if (float(high[i]) >= resistance - tol and c <= resistance + tol) else 0.0
        prev_close = float(close[i - 1]) if i > 0 else c
        rows[i, 6] = 1.0 if (c > resistance + tol and prev_close <= resistance + tol) else 0.0
        rows[i, 7] = 1.0 if (c < support - tol and prev_close >= support - tol) else 0.0

        if i >= swing_window:
            prev_highs = high[i - swing_window:i]
            prev_lows = low[i - swing_window:i]
            is_swing_high = float(high[i]) >= float(np.max(prev_highs))
            is_swing_low = float(low[i]) <= float(np.min(prev_lows))
            if is_swing_high:
                last_swing_high = float(high[i])
            if is_swing_low:
                last_swing_low = float(low[i])
            rows[i, 8] = 1.0 if is_swing_high else 0.0
            rows[i, 9] = 1.0 if is_swing_low else 0.0
        if not np.isnan(last_swing_high):
            rows[i, 10] = float((last_swing_high - c) / a)
        if not np.isnan(last_swing_low):
            rows[i, 11] = float((c - last_swing_low) / a)

        body_high = max(float(open_[i]), c)
        body_low = min(float(open_[i]), c)
        upper_wick = float(high[i]) - body_high
        lower_wick = body_low - float(low[i])
        rows[i, 12] = 1.0 if (float(high[i]) >= resistance - tol and c < resistance and upper_wick >= a * tol_atr) else 0.0
        rows[i, 13] = 1.0 if (float(low[i]) <= support + tol and c > support and lower_wick >= a * tol_atr) else 0.0
        if i >= 3:
            rows[i, 14] = float((c - float(close[i - 3])) / a)
        if i >= 6:
            rows[i, 15] = float((c - float(close[i - 6])) / a)
        if i >= consolidation_window:
            r0 = max(0, i - consolidation_window + 1)
            range_atr = float((np.max(high[r0:i + 1]) - np.min(low[r0:i + 1])) / a)
            rows[i, 16] = range_atr
            rows[i, 17] = 1.0 if range_atr <= consolidation_atr else 0.0

    return rows, names


def _build_1m_features(dfs: Dict[str, Any], cfg: XgbConfig) -> Tuple[np.ndarray, List[str]]:
    if not uses_1m_features(cfg):
        return np.zeros((len(dfs["df_5min"]), 0), dtype=np.float32), []
    df5 = dfs.get("df_5min")
    df1 = dfs.get("df_1min")
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if df1 is None or not hasattr(df1, "columns") or not required.issubset(set(df1.columns)):
        raise ValueError("1m features enabled but df_1min with OHLCV columns is missing")
    if df5 is None or not hasattr(df5, "columns") or "timestamp" not in df5.columns:
        raise ValueError("1m features enabled but df_5min timestamp column is missing")

    df1_sorted = df1.sort_values("timestamp")
    ts1 = _as_float_array(df1_sorted["timestamp"].to_numpy())
    o1 = _as_float_array(df1_sorted["open"].to_numpy())
    h1 = _as_float_array(df1_sorted["high"].to_numpy())
    l1 = _as_float_array(df1_sorted["low"].to_numpy())
    c1 = _as_float_array(df1_sorted["close"].to_numpy())
    v1 = _as_float_array(df1_sorted["volume"].to_numpy())
    ts5 = _as_float_array(df5["timestamp"].to_numpy())

    rows: List[List[float]] = []
    names: List[str] = []
    if getattr(cfg, "use_1m_microvol", False):
        names += ["1m_ret_std_5", "1m_ret_max_abs_5", "1m_max_move_abs_5"]
    if getattr(cfg, "use_1m_momentum", False):
        names += ["1m_cum_ret_3", "1m_cum_ret_5", "1m_ret_accel"]
    if getattr(cfg, "use_1m_candle_structure", False):
        names += ["1m_above_open_share", "1m_close_pos_5m", "1m_green_share"]
    if getattr(cfg, "use_1m_volume", False):
        names += ["1m_volume_spike", "1m_last_volume_share", "1m_up_volume_share"]

    for t5 in ts5:
        left = int(np.searchsorted(ts1, float(t5), side="left"))
        right = int(np.searchsorted(ts1, float(t5) + _TF_5M_MS, side="left"))
        if right <= left:
            raise ValueError(f"missing 1m candles for 5m timestamp={int(t5)}")
        oo, hh, ll, cc, vv = o1[left:right], h1[left:right], l1[left:right], c1[left:right], v1[left:right]
        ret = np.diff(cc) / np.maximum(cc[:-1], 1e-12) if len(cc) >= 2 else np.zeros((0,), dtype=np.float64)
        row: List[float] = []
        if getattr(cfg, "use_1m_microvol", False):
            move = np.maximum(np.abs(hh / np.maximum(oo, 1e-12) - 1.0), np.abs(ll / np.maximum(oo, 1e-12) - 1.0))
            row += [float(np.std(ret)) if len(ret) else 0.0, float(np.max(np.abs(ret))) if len(ret) else 0.0, float(np.max(move))]
        if getattr(cfg, "use_1m_momentum", False):
            row += [
                float(cc[-1] / max(cc[max(0, len(cc) - 3)], 1e-12) - 1.0),
                float(cc[-1] / max(cc[0], 1e-12) - 1.0),
                float(ret[-1] - ret[-2]) if len(ret) >= 2 else 0.0,
            ]
        if getattr(cfg, "use_1m_candle_structure", False):
            bar_low, bar_high = float(np.min(ll)), float(np.max(hh))
            row += [
                float(np.mean(cc > oo[0])),
                float((cc[-1] - bar_low) / max(bar_high - bar_low, 1e-12)),
                float(np.mean(cc >= oo)),
            ]
        if getattr(cfg, "use_1m_volume", False):
            total_vol = float(np.sum(vv))
            avg_prev = float(np.mean(vv[:-1])) if len(vv) > 1 else float(np.mean(vv))
            up_vol = float(np.sum(vv[cc >= oo]))
            row += [
                float(vv[-1] / max(avg_prev, 1e-12)),
                float(vv[-1] / max(total_vol, 1e-12)),
                float(up_vol / max(total_vol, 1e-12)),
            ]
        rows.append(row)

    return np.asarray(rows, dtype=np.float32), names


def _ema(values: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    if len(values) == 0:
        return out
    alpha = 2.0 / float(int(length) + 1)
    out[0] = float(values[0])
    for i in range(1, len(values)):
        out[i] = alpha * float(values[i]) + (1.0 - alpha) * out[i - 1]
    return out


def _atr_pct_daily(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> np.ndarray:
    if len(close) == 0:
        return np.zeros((0,), dtype=np.float64)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = _ema(tr, int(length))
    return atr / np.maximum(close, 1e-12)


def _build_1d_regime_features(dfs: Dict[str, Any], cfg: XgbConfig) -> Tuple[np.ndarray, List[str]]:
    names = [
        "daily_return_1",
        "daily_return_3",
        "daily_ema20_slope",
        "daily_close_vs_ema20",
        "daily_trend_down_flag",
        "daily_atr_pct",
        "daily_range_pct",
    ]
    if not uses_1d_regime(cfg):
        return np.zeros((len(dfs["df_5min"]), 0), dtype=np.float32), []
    df5 = dfs.get("df_5min")
    df1d = dfs.get("df_1d")
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if df1d is None or not hasattr(df1d, "columns") or not required.issubset(set(df1d.columns)):
        raise ValueError("1d regime enabled but df_1d with OHLCV columns is missing")
    if df5 is None or not hasattr(df5, "columns") or "timestamp" not in df5.columns:
        raise ValueError("1d regime enabled but df_5min timestamp column is missing")

    d = df1d.sort_values("timestamp")
    ts_d = _as_float_array(d["timestamp"].to_numpy())
    high = _as_float_array(d["high"].to_numpy())
    low = _as_float_array(d["low"].to_numpy())
    close = _as_float_array(d["close"].to_numpy())
    ts5 = _as_float_array(df5["timestamp"].to_numpy())

    ret1 = np.zeros_like(close, dtype=np.float64)
    ret3 = np.zeros_like(close, dtype=np.float64)
    if len(close) > 1:
        ret1[1:] = close[1:] / np.maximum(close[:-1], 1e-12) - 1.0
    if len(close) > 3:
        ret3[3:] = close[3:] / np.maximum(close[:-3], 1e-12) - 1.0
    ema20 = _ema(close, 20)
    ema20_prev = np.roll(ema20, 1)
    ema20_prev[0] = ema20[0]
    ema20_slope = (ema20 - ema20_prev) / np.maximum(ema20_prev, 1e-12)
    close_vs_ema20 = close / np.maximum(ema20, 1e-12) - 1.0
    trend_down = ((close < ema20) & (ema20_slope < 0.0)).astype(np.float64)
    atr_pct = _atr_pct_daily(high, low, close, 14)
    range_pct = (high - low) / np.maximum(close, 1e-12)
    daily_matrix = np.stack([ret1, ret3, ema20_slope, close_vs_ema20, trend_down, atr_pct, range_pct], axis=1)

    closed_ts = ts_d + _TF_1D_MS
    rows: List[np.ndarray] = []
    for t5 in ts5:
        j = int(np.searchsorted(closed_ts, float(t5), side="right") - 1)
        if j < 0:
            raise ValueError(f"missing closed 1d candle before 5m timestamp={int(t5)}")
        rows.append(daily_matrix[min(j, len(daily_matrix) - 1)])
    return np.asarray(rows, dtype=np.float32), names


def _atr_col_index_in_indicators() -> int:
    """
    Index of ATR_14 column inside indicators_array produced by indicators_config below.
    Keep in sync with the indicators_config ordering (Python dict preserves insertion order).
    """
    # rsi(14), rsi_7(7) -> 2
    # ema lengths [20,50,100,200] -> +4 => 6
    # ema_cross pairs 2 * (above,cross_up,cross_down) -> +6 => 12
    # sma(14) -> +1 => 13
    # atr(14) -> +1 => ATR index = 13 (0-based)
    return 13


def _build_base_features(dfs: Dict[str, Any], cfg: XgbConfig) -> Tuple[np.ndarray, np.ndarray]:
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
            "rsi_7": {"length": 7},
            "ema": {"lengths": [20, 50, 100, 200]},
            "ema_cross": {"pairs": [(20, 50), (100, 200)], "include_cross_signal": True},
            "sma": {"length": 14},
            "atr": {"length": 14},
            "obv": {},
            "returns": {"periods": [1, 3, 12, 60]},
            "zscore": {"ema_length": 50, "window": 20},
        }
    )

    # Disable ATR without breaking feature shape for old models.
    try:
        if not bool(getattr(cfg, "use_atr_feature", True)):
            j = _atr_col_index_in_indicators()
            if indicators is not None and hasattr(indicators, "shape") and indicators.ndim == 2 and indicators.shape[1] > j:
                indicators[:, j] = 0.0
    except Exception:
        pass

    df_15m_5m = _upsample_15m_to_5m(df_5m, df_15m).astype(np.float32)
    df_1h_5m = _upsample_1h_to_5m(df_5m, df_1h).astype(np.float32)
    X_sr, feature_names_sr = _build_sr_features(df_5m, cfg)
    X_1m, feature_names_1m = _build_1m_features(dfs, cfg)
    X_1d, feature_names_1d = _build_1d_regime_features(dfs, cfg)
    parts = [df_5m.astype(np.float32), df_15m_5m, df_1h_5m, indicators.astype(np.float32)]
    if X_sr.shape[1] > 0:
        parts.append(X_sr)
    if X_1m.shape[1] > 0:
        parts.append(X_1m)
    if X_1d.shape[1] > 0:
        parts.append(X_1d)
    X_base = np.concatenate(parts, axis=1)
    try:
        setattr(cfg, "_feature_names_sr", feature_names_sr)
        setattr(cfg, "_feature_names_1m", feature_names_1m)
        setattr(cfg, "_feature_names_1d", feature_names_1d)
    except Exception:
        pass
    closes = df_5m[:, 3].astype(np.float64)
    return X_base, closes


def _fee_to_fraction(fee_bps: float) -> float:
    # round-trip bps -> fraction
    return float(fee_bps) / 10000.0


def _policy_entry_return(
    closes: np.ndarray,
    highs: np.ndarray | None,
    entry_idx: int,
    max_hold: int,
    fee_frac: float,
    is_long: bool,
    tp_pct: float | None,
    sl_pct: float | None,
    trail_pct: float | None,
) -> float:
    """
    Entry labeling: simulate the SAME exit-policy as OOS backtest:
    - optional TP/SL/trailing (trailing only for long, same as tasks/xgb_oos_tasks.py)
    - forced exit at max_hold timeout
    Returns net return fraction (after fee).
    """
    n = int(len(closes))
    if entry_idx < 0 or entry_idx >= n:
        return 0.0
    entry = float(closes[int(entry_idx)])
    if entry <= 0:
        return 0.0
    end = min(n - 1, int(entry_idx) + int(max_hold))
    if end <= entry_idx:
        return 0.0

    peak = entry
    for i in range(int(entry_idx) + 1, int(end) + 1):
        px = float(closes[int(i)])
        if px <= 0:
            continue

        # peak update (for trailing)
        if is_long:
            ph = px
            if highs is not None and int(i) < len(highs):
                try:
                    ph = float(highs[int(i)])
                except Exception:
                    ph = px
            peak = max(float(peak), float(ph))

        raw_ret = (px - entry) / max(entry, 1e-12) if is_long else (entry - px) / max(entry, 1e-12)

        # trailing from peak (only long, consistent with OOS)
        if trail_pct is not None and is_long:
            dd = (peak - px) / max(peak, 1e-12)
            if dd >= float(trail_pct):
                return float(raw_ret) - float(fee_frac)

        if tp_pct is not None and raw_ret >= float(tp_pct):
            return float(raw_ret) - float(fee_frac)
        if sl_pct is not None and raw_ret <= float(sl_pct):
            return float(raw_ret) - float(fee_frac)

    # timeout exit at end
    px = float(closes[int(end)])
    raw_ret = (px - entry) / max(entry, 1e-12) if is_long else (entry - px) / max(entry, 1e-12)
    return float(raw_ret) - float(fee_frac)


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


def build_xgb_dataset(
    dfs: Dict[str, Any], cfg: XgbConfig
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    Builds dataset depending on cfg.task.
    - directional: y in {0,1,2} = {hold,buy,sell}
    - entry_* / exit_*: y in {0,1} = {hold, enter/exit}
    """
    X_base, closes = _build_base_features(dfs, cfg)

    # Trim warmup rows where EMA-200 etc. are NaN→0 garbage
    WARMUP = 200
    if len(closes) > WARMUP + 100:
        X_base = X_base[WARMUP:]
        closes = closes[WARMUP:]

    task = (cfg.task or "directional").strip().lower()
    # For entry-like backtests/labels we need highs/lows for trailing/diagnostics.
    highs = None
    lows = None
    try:
        if isinstance(X_base, np.ndarray) and X_base.ndim == 2 and X_base.shape[1] >= 5:
            highs = X_base[:, 1].astype(np.float64, copy=False)
            lows = X_base[:, 2].astype(np.float64, copy=False)
    except Exception:
        highs = None
        lows = None

    aux: Dict[str, Any] = {"closes_mode": "timeseries", "closes": closes, "highs": highs, "lows": lows}

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
        return X_base, y, meta, aux

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
        # Exit-policy knobs (match OOS policy-mode semantics)
        tp_pct = getattr(cfg, "entry_tp_pct", None)
        sl_pct = getattr(cfg, "entry_sl_pct", None)
        trail_pct = getattr(cfg, "entry_trail_pct", None)
        for t in range(0, len(closes) - max_hold):
            ret = _policy_entry_return(
                closes=closes,
                highs=highs,
                entry_idx=t,
                max_hold=max_hold,
                fee_frac=fee_frac,
                is_long=is_long,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                trail_pct=trail_pct,
            )
            y[t] = 1 if float(ret) >= float(min_profit) else 0
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
        aux_entry = dict(aux)
        aux_entry.update(
            {
                "fee_frac": fee_frac,
                "max_hold_steps": max_hold,
                "position_side": ("long" if is_long else "short"),
                "entry_exit_policy": {
                    "tp_pct": (float(tp_pct) if tp_pct is not None else None),
                    "sl_pct": (float(sl_pct) if sl_pct is not None else None),
                    "trail_pct": (float(trail_pct) if trail_pct is not None else None),
                },
            }
        )
        return X, y, meta, aux_entry

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
        aux_exit = {"closes_mode": "exit_rows", "fee_frac": fee_frac, "max_hold_steps": max_hold, "position_side": ("long" if is_long else "short")}
        return X, y, meta, aux_exit

    # Fallback to directional
    return X_base, np.zeros((len(closes),), dtype=np.int64), {"cfg_snapshot": asdict(cfg)}, aux


import numpy as np
import torch
from typing import Dict, Deque, Iterable, Optional
from collections import deque
from enum import IntEnum


class MarketState(IntEnum):
    NORMAL = 0
    HIGH_VOL = 1
    PANIC = 2
    DRAWDOWN = 3


def compute_market_state(
    current_step: int,
    df_5min: np.ndarray,
    roi_buf: Optional[Iterable[float]] = None,
    vol_buf: Optional[Iterable[float]] = None,
    trend_regime: int = 0,
    atr_rel: float | None = None,
    high_vol_atr: float = 0.006,
    high_vol_ret: float = 0.008,
    panic_atr: float = 0.010,
    panic_drop: float = -0.020,
) -> MarketState:
    """Pure market-regime detector.

    - No logging, no side-effects, no access to env/self.
    - Uses only inputs available inside env.
    """
    # Safety: not enough data
    try:
        step = int(current_step)
    except Exception:
        step = 0
    if df_5min is None or not hasattr(df_5min, "shape") or df_5min.shape[0] < 3:
        return MarketState.NORMAL

    # --- price returns volatility ---
    end = max(0, min(step, int(df_5min.shape[0])))
    if end <= 2:
        end = int(df_5min.shape[0])
    start = max(0, end - 40)  # short window for regime
    closes = df_5min[start:end, 3].astype(np.float32, copy=False)
    if closes.shape[0] >= 3:
        rets = np.diff(closes) / (closes[:-1] + 1e-8)
        vol_ret = float(np.std(rets[-20:])) if rets.shape[0] >= 2 else 0.0
        last_ret = float(rets[-1]) if rets.shape[0] >= 1 else 0.0
    else:
        vol_ret = 0.0
        last_ret = 0.0

    # --- ATR proxy (relative) ---
    if atr_rel is None:
        try:
            idx = max(1, min(end - 1, int(df_5min.shape[0] - 1)))
            atr_rel = float(calc_relative_vol_numpy(df_5min, idx, lookback=12))
        except Exception:
            atr_rel = 0.0
    else:
        try:
            atr_rel = float(atr_rel)
        except Exception:
            atr_rel = 0.0

    # --- drawdown proxy from recent ROI buffer (unrealized ROI series) ---
    dd_flag = False
    try:
        rb = list(roi_buf) if roi_buf is not None else []
        if rb:
            tail = rb[-10:]
            # "drawdown" proxy: sustained negative ROI or deep local minimum
            if (float(np.mean(tail)) <= -0.03) or (float(np.min(tail)) <= -0.06):
                dd_flag = True
    except Exception:
        dd_flag = False

    # Thresholds are intentionally conservative and stable.
    # NOTE: values are overridable via function args to avoid hardcoding in env.
    HIGH_VOL_ATR = float(high_vol_atr)
    HIGH_VOL_RET = float(high_vol_ret)
    PANIC_ATR = float(panic_atr)
    PANIC_DROP = float(panic_drop)

    try:
        tr = int(trend_regime)
    except Exception:
        tr = 0

    if dd_flag and tr <= 0:
        return MarketState.DRAWDOWN
    if (atr_rel >= PANIC_ATR) and (last_ret <= PANIC_DROP):
        return MarketState.PANIC
    if (atr_rel >= HIGH_VOL_ATR) or (vol_ret >= HIGH_VOL_RET):
        return MarketState.HIGH_VOL
    return MarketState.NORMAL

def calc_relative_vol_numpy(df_5min: np.ndarray, idx: int, lookback: int = 12) -> float:
    """
    Устойчивый proxy‑ATR на базе 5‑мин. свечей.
    Возвращает относительную волатильность (0.0…0.1 ≈ 0‑10 %).

    Parameters
    ----------
    df_5min : np.ndarray
        OHLCV‑массив с колонками [open, high, low, close, volume]
    idx : int
        Текущий индекс (exclusive) — будем смотреть назад
    lookback : int, default 12
        Сколько 5‑мин. свечей брать (12 ⇒ ~1 час)

    Returns
    -------
    float
        (mean TrueRange) / lastClose
    """
    start = max(0, idx - lookback)
    end = min(idx, len(df_5min))
    
    if end - start < 2:           # защитная заглушка для старта
        return 0.0015

    recent = df_5min[start:end]
    
    # True Range каждой свечи
    high = recent[:, 1]  # high column
    low = recent[:, 2]   # low column
    close = recent[:, 3] # close column
    
    # Предыдущие close цены
    prev_close = np.concatenate([[close[0]], close[:-1]])
    
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev_close),
                               np.abs(low - prev_close)))
    
    return float(np.mean(tr) / close[-1])

def update_vol_stats(vol_buf: Deque, vol_stats: Dict):
    """Обновляет статистики объема"""
    if len(vol_buf) > 0:
        vol_array = np.array(list(vol_buf))
        vol_stats['mean'] = np.mean(vol_array)
        vol_stats['std'] = np.std(vol_array) + 1e-8

def update_roi_stats(roi_buf: Deque, roi_stats: Dict):
    """Обновляет статистики ROI"""
    if len(roi_buf) > 0:
        roi_array = np.array(list(roi_buf))
        roi_stats['mean'] = np.mean(roi_array)
        roi_stats['std'] = np.std(roi_array) + 1e-8

def commission_penalty(fee: float, init_balance: float, kappa: float = 1000) -> float:
    """
    Штраф за комиссию.
    fee          – абсолютная комиссия за сделку
    init_balance – стартовый баланс эпизода (масштаб)
    kappa        – коэффициент веса (1 000‑3 000 по опыту)
    Возвращает отрицательное значение (penalty).
    """
    return - fee / init_balance * kappa

def check_nan(tag: str, *tensors: torch.Tensor, report_every: int = None) -> bool:
    """
    Возвращает True если всё ок. Если NaN/Inf обнаружен:
    - увеличивает счётчики
    - ничего не печатает (кроме редкого отчёта, если задан report_every)
    """
    global _NAN_TOTAL, _NAN_COUNTS
    
    if '_NAN_TOTAL' not in globals():
        globals()['_NAN_TOTAL'] = 0
    if '_NAN_COUNTS' not in globals():
        globals()['_NAN_COUNTS'] = {}
    
    ok = True
    for t in tensors:
        if not torch.isfinite(t).all():
            ok = False
            _NAN_COUNTS[tag] = _NAN_COUNTS.get(tag, 0) + 1
            _NAN_TOTAL += 1
            break

    # редкий, «rate-limited» отчёт
    if (not ok) and report_every and (_NAN_TOTAL % report_every == 0):
        print(f"[NaN-guard] total={_NAN_TOTAL} | {tag}={_NAN_COUNTS.get(tag, 0)}")
    return ok

def get_nan_stats(reset: bool = False) -> dict:
    """Вернёт агрегированную статистику по NaN/Inf. reset=True — обнулит счётчики."""
    global _NAN_TOTAL, _NAN_COUNTS
    
    if '_NAN_TOTAL' not in globals():
        globals()['_NAN_TOTAL'] = 0
    if '_NAN_COUNTS' not in globals():
        globals()['_NAN_COUNTS'] = {}
    
    stats = {"total": _NAN_TOTAL} | dict(_NAN_COUNTS)
    if reset:
        _NAN_COUNTS.clear()
        globals()["_NAN_TOTAL"] = 0
    return stats

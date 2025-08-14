import numpy as np
import torch
from typing import Dict, Deque
from collections import deque

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

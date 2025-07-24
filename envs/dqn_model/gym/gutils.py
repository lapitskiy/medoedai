import numpy as np
from collections import deque
import os
import random
import wandb
import pandas as pd

def update_vol_stats(vol_now: float, buf: deque, bootstrap_med=0.0015, bootstrap_iqr=0.0008):
    """
    В buf (deque) храним последние VOL_WINDOW значений.
    Возвращаем (median, IQR).
    """
    buf.append(vol_now)

    if len(buf) < 30:
        return bootstrap_med, bootstrap_iqr

    arr = np.fromiter(buf, float)
    q25, q75 = np.percentile(arr, [25, 75])
    return np.median(arr), q75 - q25


def update_roi_stats(roi_now: float, buf: deque,
                     bootstrap_q75: float = 0.003):
    """
    Обновляем буфер ROI и возвращаем 75‑й процентиль (q75).
    Пока в буфере < 30 значений — возвращаем заглушку.
    """
    if roi_now > 0:
        buf.append(roi_now)

    if len(buf) < 30:
        return bootstrap_q75          # 0.3 % по умолчанию

    arr = np.fromiter(buf, float)
    return np.quantile(arr, 0.75)

def calc_relative_vol(df_5min: pd.DataFrame, idx: int, lookback: int = 12) -> float:
    """
    Устойчивый proxy‑ATR на базе 5‑мин. свечей.
    Возвращает относительную волатильность (0.0…0.1 ≈ 0‑10 %).

    Parameters
    ----------
    df_5min : DataFrame
        OHLCV‑таблица с колонками 'high', 'low', 'close'
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
    recent = df_5min.iloc[start:idx].copy()

    if len(recent) < 2:           # защитная заглушка для старта
        return 0.0015

    # True Range каждой свечи
    prev_close = recent['close'].shift(1)
    tr = np.maximum(recent['high'] - recent['low'],
                    np.maximum(abs(recent['high'] - prev_close),
                               abs(recent['low']  - prev_close)))
    return float(tr.mean() / recent['close'].iloc[-1])

def commission_penalty(fee: float,
                       init_balance: float,
                       kappa: float = 2_000.0) -> float:
    """
    Штраф за комиссию.
    fee          – абсолютная комиссия за сделку
    init_balance – стартовый баланс эпизода (масштаб)
    kappa        – коэффициент веса (1 000‑3 000 по опыту)
    Возвращает отрицательное значение (penalty).
    """
    return - fee / init_balance * kappa

def setup_wandb(cfg, project: str = "medoedai‑medoedai"):
    """
    Инициализирует W&B ровно один раз, даже если модуль импортируют несколько раз
    (например, из‑за перезагрузчика Flask).

    Parameters
    ----------
    cfg : vDqnConfig
        Конфиг с гиперпараметрами и полем `run_name`.
    project : str, optional
        Название проекта в W&B.

    Returns
    -------
    wandb.sdk.wandb_run.Run | None
        Объект run, если инициализация произошла, либо None, если
        функция была вызвана во «внешнем» процессе Flask‑reloader
        (или уже была выполнена ранее).
    """
    # 1) Если уже инициализировали — просто вернём текущий run
    if wandb.run is not None:
        return wandb.run

    # 2) Во Flask‑reloader есть «родительский» процесс‑наблюдатель.
    #    Его можно распознать по отсутствию переменной WERKZEUG_RUN_MAIN.
    if os.getenv("FLASK_ENV") == "development" and os.getenv("WERKZEUG_RUN_MAIN") != "true":
        # Внешний процесс — пропускаем, чтобы не плодить лишние run‑ы
        return None

    # 3) Генерируем запоминающееся имя
    suffixes = [
        "alpha", "bravo", "vpnblock", "echo",
        "stalin", "matrix", "hton", "kaput",
        "vodka", "balalaika", "medved", "sssr",
    ]
    suffix = random.choice(suffixes)
    run_name = f"{cfg.run_name}-{suffix}-{random.randint(1, 10)}"

    # 4) Инициализируем W&B
    run = wandb.init(
        project=project,
        name=run_name,
        config={
            "learning_rate":      cfg.lr,
            "batch_size":         cfg.batch_size,
            "ε_max":              cfg.eps_start,
            "ε_min":              cfg.eps_final,
            "ε_decay":            cfg.eps_decay_steps,
            "memory_size":        cfg.memory_size,
            "gamma":              cfg.gamma,
        },
    )

    return run
import logging
from envs.dqn_model.gym.gconfig import GymConfig
import numpy as np
from collections import deque
import os
import random
from utils.f_logs import get_train_logger
import wandb
import pandas as pd
import torch
import socket

cfg = GymConfig()

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
                       init_balance: float) -> float:
    kappa = cfg.comission_kappa
    """
    Штраф за комиссию.
    fee          – абсолютная комиссия за сделку
    init_balance – стартовый баланс эпизода (масштаб)
    kappa        – коэффициент веса (1 000‑3 000 по опыту)
    Возвращает отрицательное значение (penalty).
    """
    return - fee / init_balance * kappa

def check_nan(tag: str, *tensors: torch.Tensor) -> bool:
    """
    Если в каком‑либо тензоре NaN/Inf – пишет предупреждение и
    возвращает False.
    """
    for t in tensors:
        if not torch.isfinite(t).all():
            print(f"[NaN‑guard] {tag}: detected NaN/Inf")
            return False
    return True

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
    
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    
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
    hostname = socket.gethostname() 

    run_name = f"{suffix}-{random.randint(1, 10)}-({cfg.run_name}-{hostname})"

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
        settings=wandb.Settings(
            init_timeout=180,          # > 90 с, чтобы не упасть
        )
    )
    
        # Лог‑файл с уникальным именем на основе run.id (лучше, чем только name)
    log_path = f"./logs/{run.name}-{run.id}.log"
    base_logger = get_train_logger(log_path, fmt_extra="[run:%(run)s id:%(run_id)s] ")

    # Оборачиваем в LoggerAdapter, чтобы добавлять контекст
    logger = logging.LoggerAdapter(base_logger, {"run": run.name, "run_id": run.id})

    logger.info("W&B run started | url=%s | project=%s | entity=%s",
                run.url, run.project, run.entity)

    return run, logger
"""
Blueprint: страница обучения DQN для акций (РФ рынок / T-Invest API).
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request, render_template
from celery.result import AsyncResult
from utils.redis_utils import get_redis_client
import logging

stock_models_bp = Blueprint("stock_models", __name__)
logger = logging.getLogger(__name__)
redis_client = get_redis_client()


@stock_models_bp.get("/stock_models")
def stock_models_page():
    return render_template("stock_models.html")


@stock_models_bp.post("/api/stock/train")
def api_stock_train():
    """Запустить обучение DQN для акции."""
    data = request.get_json(silent=True) or {}
    ticker = (data.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"success": False, "error": "ticker обязателен"}), 400

    episodes = int(data.get("episodes", 5))
    episode_length = int(data.get("episode_length", 2000))
    seed = data.get("seed")
    figi = data.get("figi")

    # Проверка дупликата
    running_key = f"celery:train:stock:task:{ticker}"
    if redis_client.get(running_key):
        return jsonify({"success": False, "error": f"{ticker} уже обучается"}), 409

    from tasks.stock_tasks import train_stock_dqn

    task = train_stock_dqn.apply_async(
        kwargs={
            "ticker": ticker,
            "episodes": episodes,
            "episode_length": episode_length,
            "seed": int(seed) if seed else None,
            "figi": figi or None,
        },
        queue="train",
    )

    # Помечаем running
    redis_client.setex(running_key, 48 * 3600, task.id)

    return jsonify({"success": True, "task_id": task.id, "ticker": ticker})


@stock_models_bp.get("/api/stock/task_status/<task_id>")
def api_stock_task_status(task_id: str):
    """Статус celery-задачи обучения акции."""
    res = AsyncResult(task_id)
    info = res.info if isinstance(res.info, dict) else {}
    return jsonify({
        "task_id": task_id,
        "state": res.state,
        "progress": info.get("progress", 0),
        "result": info if res.state in ("SUCCESS", "FAILURE") else None,
    })


@stock_models_bp.post("/api/stock/clear_lock")
def api_stock_clear_lock():
    data = request.get_json(silent=True) or {}
    ticker = (data.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"success": False, "error": "ticker обязателен"}), 400
    redis_client.delete(f"celery:train:stock:task:{ticker}")
    return jsonify({"success": True, "ticker": ticker})


@stock_models_bp.get("/api/stock/tickers")
def api_stock_tickers():
    """Список доступных тикеров (захардкожен MVP, потом можно сделать поиск)."""
    tickers = [
        {"ticker": "SBER", "name": "Сбербанк"},
        {"ticker": "GAZP", "name": "Газпром"},
        {"ticker": "LKOH", "name": "Лукойл"},
        {"ticker": "GMKN", "name": "Норникель"},
        {"ticker": "ROSN", "name": "Роснефть"},
        {"ticker": "YNDX", "name": "Яндекс"},
        {"ticker": "MTSS", "name": "МТС"},
        {"ticker": "MGNT", "name": "Магнит"},
        {"ticker": "NVTK", "name": "Новатэк"},
        {"ticker": "POLY", "name": "Полиметалл"},
        {"ticker": "VTBR", "name": "ВТБ"},
        {"ticker": "ALRS", "name": "Алроса"},
        {"ticker": "MOEX", "name": "МосБиржа"},
        {"ticker": "TATN", "name": "Татнефть"},
        {"ticker": "PLZL", "name": "Полюс Золото"},
    ]
    return jsonify({"success": True, "tickers": tickers})

import os
import json
from datetime import datetime
from celery import states
from tasks import celery
from routes.oos import oos_test_model as _sync_oos_test_model
from flask import Flask


def _make_fake_request_ctx(payload: dict):
    app = Flask(__name__)
    # minimal config to allow using request context inside called function
    app.config['TESTING'] = True
    return app.test_request_context(path='/oos_test_model', method='POST', json=payload)


@celery.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 0}, queue='oos')
def run_oos_test(self, payload: dict):
    """
    Обёртка Celery над синхронной функцией oos_test_model, чтобы выполнять OOS в фоне.
    Аргумент payload повторяет тело запроса UI (filename/code/days/gate_* ...).
    Возвращает JSON как у старого эндпоинта.
    """
    # Проставим прогресс
    try:
        self.update_state(state="IN_PROGRESS", meta={"queued_at": datetime.utcnow().isoformat() + 'Z'})
    except Exception:
        pass

    # Запускаем синхронную реализацию во временном Flask request context
    with _make_fake_request_ctx(payload):
        resp = _sync_oos_test_model()
    # resp — это (Response) или (Response, code)
    try:
        if isinstance(resp, tuple):
            data = resp[0].get_json(silent=True) or {}
        else:
            data = resp.get_json(silent=True) or {}
    except Exception:
        # Фоллбек: попытка сериализации
        try:
            data = json.loads(str(resp))
        except Exception:
            data = {"success": False, "error": "oos task returned non-json response"}

    # Возвращаем результат — Celery сам выставит SUCCESS
    if isinstance(data, dict):
        return data
    return {"success": False, "error": "unknown error"}



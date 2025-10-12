import warnings
import time
import pandas as pd
import json
import requests
from redis import Redis
import numpy as np
import uuid
from celery import Celery
from kombu import Queue
from celery.schedules import crontab
from utils.config_loader import get_config_value

# Подавляем DeprecationWarning от distutils, которые могут исходить из setuptools
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*distutils.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*distutils.*")

# Инициализируем Celery
celery = Celery(
    "tasks",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0"
)

# Настройки устойчивости к потере соединения с брокером
celery.conf.worker_cancel_long_running_tasks_on_connection_loss = True
celery.conf.broker_connection_retry_on_startup = True

# Настройки для автоматической очистки результатов задач
celery.conf.result_expires = 3600 * 24 * 7 # 7 дней TTL для результатов задач
celery.conf.result_backend_transport_options = {
    'master_name': 'mymaster',
    'visibility_timeout': 3600 * 24 * 7, # 7 дней
}

# Определяем очереди и маршрутизацию задач
celery.conf.task_queues = (
    Queue('celery'),
    Queue('train'),
    Queue('trade'),
    Queue('oos'),
)
celery.conf.task_default_queue = 'celery'
celery.conf.task_routes = {
    'tasks.celery_tasks.train_dqn': {'queue': 'train'},
    'tasks.celery_tasks.train_dqn_symbol': {'queue': 'train'},
    'tasks.celery_tasks.train_dqn_multi_crypto': {'queue': 'train'},
    'tasks.celery_tasks.train_cnn_model': {'queue': 'train'},
    'tasks.sac_tasks.train_sac_symbol': {'queue': 'train'},
    'tasks.celery_task_trade.execute_trade': {'queue': 'trade'},
    'tasks.celery_task_trade.start_trading_task': {'queue': 'trade'},
    'tasks.celery_task_trade.refresh_trading_status': {'queue': 'celery'},
    'tasks.oos_tasks.run_oos_test': {'queue': 'oos'},
}

# Включаем периодический запуск торговли
if str(get_config_value('ENABLE_TRADING_BEAT', '0')).lower() in ('1', 'true', 'yes', 'on'):
    celery.conf.beat_schedule = {
        'start-trading-every-5-minutes': {
            'task': 'tasks.celery_task_trade.start_trading_task',
            'schedule': crontab(minute='*/5'),
            'args': ([], None)
        },
        'refresh-trading-status-every-minute': {
            'task': 'tasks.celery_task_trade.refresh_trading_status',
            'schedule': crontab(minute='*'),
            'args': (),
        },
    }
    celery.conf.timezone = 'UTC'
    print("✅ Периодическая торговля включена (каждые 5 минут)")
else:
    print("⚠️ Периодическая торговля отключена (ENABLE_TRADING_BEAT=0)")

# Импортируем модули с задачами, чтобы Celery их обнаружил
import tasks.celery_tasks
import tasks.celery_task_trade
import tasks.oos_tasks
import tasks.sac_tasks

from __future__ import annotations

import os
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from orm.models import Base

# Один engine на процесс (Flask/Celery). Важно: создаём лениво (после fork), иначе можно унаследовать соединения.
_ENGINE: Optional[Engine] = None
_SessionLocal = None

def get_db_url():
    """Получает URL базы данных из переменных окружения или использует настройки из Docker"""
    # Получаем параметры из переменных окружения (приоритет)
    db_host = os.getenv('DB_HOST', 'postgres')  # Изменено с localhost на postgres для Docker
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'medoed_db')
    db_user = os.getenv('DB_USER', 'medoed_user')
    db_password = os.getenv('DB_PASSWORD', 'medoed')
    
    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

def get_engine():
    """Создает и возвращает SQLAlchemy engine (singleton per-process) с пулом соединений."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    db_url = os.getenv("DATABASE_URL") or get_db_url()

    # Параметры пула можно подкрутить через env, чтобы не упереться в max_connections Postgres
    pool_size = int(os.getenv("DB_POOL_SIZE", "2"))
    max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "3"))
    pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "1800"))

    _ENGINE = create_engine(
        db_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        pool_recycle=pool_recycle,
        pool_pre_ping=True,
        echo=False,
    )
    return _ENGINE

def get_db_session():
    """Создает и возвращает сессию базы данных"""
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return _SessionLocal()

def create_tables():
    """Создает все таблицы из моделей SQLAlchemy"""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("✅ Все таблицы созданы успешно")

def drop_tables():
    """Удаляет все таблицы (осторожно!)"""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    print("⚠️ Все таблицы удалены")

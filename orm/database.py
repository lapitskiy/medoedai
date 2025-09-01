from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from orm.models import Base
import os

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
    """Создает и возвращает SQLAlchemy engine"""
    db_url = get_db_url()
    return create_engine(db_url)

def get_db_session():
    """Создает и возвращает сессию базы данных"""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()

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

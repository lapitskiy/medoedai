# Управление миграциями базы данных

## Обзор

Проект использует **SQLAlchemy ORM** (не Django) для работы с базой данных. Для управления схемой базы данных используется **Alembic** - стандартный инструмент миграций для SQLAlchemy.

## Основные команды

### 1. Инициализация (уже выполнена)
```bash
# Создание структуры миграций
alembic init alembic
```

### 2. Создание миграции
```bash
# Автоматическое создание миграции на основе изменений в моделях
alembic revision --autogenerate -m "Описание изменений"

# Примеры:
alembic revision --autogenerate -m "Добавлена таблица Trade"
alembic revision --autogenerate -m "Добавлено поле confidence в Trade"
alembic revision --autogenerate -m "Изменен тип поля price на DECIMAL"
```

### 3. Применение миграций
```bash
# Применить все миграции
alembic upgrade head

# Применить конкретную миграцию
alembic upgrade 0001

# Откатить на одну миграцию назад
alembic downgrade -1

# Откатить все миграции
alembic downgrade base
```

### 4. Просмотр статуса
```bash
# Показать текущую версию
alembic current

# Показать историю миграций
alembic history

# Показать что будет применено
alembic show 0001
```

## Структура файлов

```
alembic/
├── env.py              # Конфигурация окружения
├── script.py.mako      # Шаблон для генерации миграций
└── versions/           # Папка с файлами миграций
    ├── 0001_initial.py
    ├── 0002_add_trade_table.py
    └── ...
```

## Примеры использования

### Добавление новой таблицы

1. **Добавьте модель в `orm/models.py`:**
```python
class NewTable(Base):
    __tablename__ = 'new_table'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
```

2. **Создайте миграцию:**
```bash
alembic revision --autogenerate -m "Добавлена таблица NewTable"
```

3. **Примените миграцию:**
```bash
alembic upgrade head
```

### Изменение существующей таблицы

1. **Измените модель в `orm/models.py`:**
```python
class Trade(Base):
    # ... существующие поля ...
    new_field = Column(String, nullable=True)  # Добавлено новое поле
```

2. **Создайте и примените миграцию:**
```bash
alembic revision --autogenerate -m "Добавлено поле new_field в Trade"
alembic upgrade head
```

## Важные моменты

### 1. Автоматическое создание таблиц (текущий подход)
В `main.py` используется:
```python
from orm.database import create_tables
create_tables()  # Создает все таблицы при запуске
```

**Это подходит только для разработки!** В продакшене используйте миграции.

### 2. Переход на миграции
Для перехода с `create_tables()` на миграции:

1. **Создайте начальную миграцию:**
```bash
alembic revision --autogenerate -m "Initial migration"
```

2. **Примените миграцию:**
```bash
alembic upgrade head
```

3. **Удалите `create_tables()` из `main.py`**

### 3. Конфигурация базы данных
URL базы данных настраивается в `alembic.ini`:
```ini
sqlalchemy.url = postgresql://medoed_user:medoed@localhost:5432/medoed_db
```

Для разных окружений используйте переменные окружения:
```python
# В env.py
import os
config.set_main_option("sqlalchemy.url", os.getenv("DATABASE_URL"))
```

## Troubleshooting

### Ошибка "Target database is not up to date"
```bash
# Проверьте текущую версию
alembic current

# Примените все миграции
alembic upgrade head
```

### Ошибка "Can't locate revision identified by"
```bash
# Проверьте историю миграций
alembic history

# При необходимости откатитесь к базовой версии
alembic downgrade base
```

### Конфликт миграций
Если несколько разработчиков создали миграции одновременно:
1. Согласуйте порядок миграций
2. Объедините изменения в одной миграции
3. Удалите конфликтующие файлы миграций

## Рекомендации

1. **Всегда создавайте миграции** при изменении моделей
2. **Тестируйте миграции** на копии базы данных
3. **Делайте бэкапы** перед применением миграций
4. **Используйте описательные сообщения** в миграциях
5. **Проверяйте SQL** перед применением миграций

## Сравнение с Django

| Django ORM | SQLAlchemy + Alembic |
|------------|---------------------|
| `makemigrations` | `alembic revision --autogenerate` |
| `migrate` | `alembic upgrade head` |
| `showmigrations` | `alembic history` |
| `sqlmigrate` | `alembic show <revision>` |

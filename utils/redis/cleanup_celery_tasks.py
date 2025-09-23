#!/usr/bin/env python3
"""
Скрипт для очистки старых задач Celery из Redis
"""

import redis
import json
import sys
from datetime import datetime, timedelta

def cleanup_celery_tasks():
    """Очищает старые задачи Celery из Redis"""
    try:
        # Подключаемся к Redis
        redis_hosts = ['redis', 'localhost', '127.0.0.1']
        r = None
        
        for host in redis_hosts:
            try:
                r = redis.Redis(host=host, port=6379, db=0, socket_connect_timeout=5)
                r.ping()
                print(f"✅ Подключились к Redis на {host}")
                break
            except Exception:
                continue
        
        if r is None:
            print("❌ Не удалось подключиться к Redis")
            return False
        
        # Получаем все ключи задач
        task_keys = r.keys('celery-task-meta-*')
        print(f"🔍 Найдено {len(task_keys)} задач в Redis")
        
        if not task_keys:
            print("✅ Нет задач для очистки")
            return True
        
        # Анализируем задачи
        now = datetime.now()
        old_tasks = []
        recent_tasks = []
        
        for key in task_keys:
            try:
                task_data = r.get(key)
                if not task_data:
                    continue
                
                # Парсим данные задачи
                try:
                    task_info = json.loads(task_data)
                except:
                    # Если не JSON, считаем задачу старой
                    old_tasks.append(key)
                    continue
                
                # Проверяем время создания
                date_done = task_info.get('date_done')
                if date_done:
                    try:
                        # Парсим время завершения
                        task_time = datetime.fromisoformat(date_done.replace('Z', '+00:00'))
                        # Задачи старше 7 дней считаем старыми
                        if now - task_time > timedelta(days=7):
                            old_tasks.append(key)
                        else:
                            recent_tasks.append(key)
                    except:
                        old_tasks.append(key)
                else:
                    # Задачи без времени завершения считаем старыми
                    old_tasks.append(key)
                    
            except Exception as e:
                print(f"⚠️ Ошибка обработки задачи {key}: {e}")
                old_tasks.append(key)
        
        print(f"📊 Статистика:")
        print(f"   • Старые задачи (>7 дней): {len(old_tasks)}")
        print(f"   • Недавние задачи (<7 дней): {len(recent_tasks)}")
        
        if not old_tasks:
            print("✅ Нет старых задач для удаления")
            return True
        
        # Удаляем старые задачи
        deleted_count = 0
        for key in old_tasks:
            try:
                r.delete(key)
                deleted_count += 1
            except Exception as e:
                print(f"⚠️ Ошибка удаления {key}: {e}")
        
        print(f"🗑️ Удалено {deleted_count} старых задач")
        
        # Показываем оставшиеся задачи
        remaining = r.keys('celery-task-meta-*')
        print(f"📈 Осталось {len(remaining)} задач в Redis")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при очистке задач Celery: {e}")
        return False

if __name__ == "__main__":
    success = cleanup_celery_tasks()
    sys.exit(0 if success else 1)

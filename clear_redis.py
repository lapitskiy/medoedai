#!/usr/bin/env python3
"""
Скрипт для очистки Redis
"""

import redis
import sys

def clear_redis():
    """Очищает Redis"""
    try:
        # Пробуем разные хосты
        redis_hosts = ['redis', 'localhost', '127.0.0.1']
        r = None
        
        for host in redis_hosts:
            try:
                r = redis.Redis(host=host, port=6379, db=0, socket_connect_timeout=5)
                r.ping()  # Проверяем соединение
                print(f"✅ Подключились к Redis на {host}")
                break
            except Exception:
                continue
        
        if r is None:
            print("❌ Не удалось подключиться к Redis")
            return False
        
        # Очищаем все данные
        r.flushall()
        print("✅ Redis очищен успешно")
        
        # Проверяем, что очистка прошла успешно
        if r.dbsize() == 0:
            print("✅ Redis пуст, готов к работе")
        else:
            print(f"⚠️ В Redis осталось {r.dbsize()} ключей")
            
        return True
            
    except Exception as e:
        print(f"❌ Ошибка при очистке Redis: {e}")
        return False

if __name__ == "__main__":
    success = clear_redis()
    sys.exit(0 if success else 1)

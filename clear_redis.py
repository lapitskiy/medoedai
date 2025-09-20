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
        
        # Сохраняем настройки торговли перед очисткой
        trading_keys_to_preserve = [
            'trading:model_paths',
            'trading:last_model_paths', 
            'trading:consensus',
            'trading:last_consensus',
            'trading:symbols',
            'trading:model_path',
            'trading:account_id',
            'trading:current_status',
            'trading:current_status_ts'
        ]
        
        preserved_data = {}
        for key in trading_keys_to_preserve:
            try:
                value = r.get(key)
                if value is not None:
                    preserved_data[key] = value
            except Exception:
                pass
        
        # Очищаем все данные
        r.flushall()
        print("✅ Redis очищен успешно")
        
        # Восстанавливаем настройки торговли
        restored_count = 0
        for key, value in preserved_data.items():
            try:
                r.set(key, value)
                restored_count += 1
            except Exception:
                pass
        
        if restored_count > 0:
            print(f"✅ Восстановлено {restored_count} настроек торговли")
        
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

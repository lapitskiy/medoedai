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
        
        # Сохраняем настройки торговли и задачи celery-trade перед очисткой
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
        
        # Сохраняем задачи celery-trade (очередь trade и связанные данные)
        celery_trade_keys_to_preserve = []
        try:
            # Получаем все ключи Redis
            all_keys = r.keys('*')
            for key in all_keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else str(key)
                
                # Сохраняем очередь trade и связанные с ней данные
                if (key_str == 'trade' or  # Очередь celery-trade
                    key_str.startswith('_kombu.binding.trade') or  # Биндинги очереди trade
                    key_str.startswith('unacked_trade') or  # Неподтвержденные задачи trade
                    key_str.startswith('_celery-trade') or  # Внутренние данные celery-trade
                    key_str.startswith('trade:')):  # Пользовательские ключи trade
                    celery_trade_keys_to_preserve.append(key_str)
                    
        except Exception as e:
            print(f"⚠️ Ошибка при получении ключей celery-trade: {e}")
        
        # НЕ сохраняем celery-task-meta-* - они будут автоматически очищены
        print("🧹 Не сохраняем celery-task-meta-* ключи - они будут очищены")
        
        # Автоматически добавляем все trading:* ключи для полного сохранения статуса
        try:
            trading_dynamic_keys = [
                k.decode('utf-8') for k in r.keys('trading:*')
                if k.decode('utf-8') not in trading_keys_to_preserve
                and not k.decode('utf-8').startswith('trading:agent_lock')  # Не сохраняем локи агента
            ]
            trading_keys_to_preserve.extend(trading_dynamic_keys)
            if trading_dynamic_keys:
                print(f"✅ Добавлено {len(trading_dynamic_keys)} динамических trading:* ключей для сохранения")
                print("📋 Список: " + ", ".join(trading_dynamic_keys))  # Отладка: показываем, какие ключи найдены
        except Exception as e:
            print(f"⚠️ Ошибка при получении trading:* ключей: {e}")
        
        # Объединяем все ключи для сохранения, исключая любые agent_lock
        safe_trading_keys = [k for k in trading_keys_to_preserve if not str(k).startswith('trading:agent_lock')]
        all_keys_to_preserve = safe_trading_keys + celery_trade_keys_to_preserve
        
        print(f"🔑 Ключи для сохранения ({len(all_keys_to_preserve)}): " + ", ".join(all_keys_to_preserve))  # Отладка: все ключи

        preserved_data = {}
        for key in all_keys_to_preserve:
            try:
                key_type = r.type(key).decode('utf-8')
                print(f"💾 Сохраняем {key} (тип: {key_type})")  # Отладка: ключ и тип
                
                if key_type == 'list':
                    value = r.lrange(key, 0, -1)
                elif key_type == 'set':
                    value = r.smembers(key)
                elif key_type == 'hash':
                    value = r.hgetall(key)
                elif key_type == 'string':
                    value = r.get(key)
                else:
                    value = r.get(key)  # Fallback
                
                if value is not None and value != b'none':  # Игнорируем 'none'
                    preserved_data[key] = (key_type, value)  # Сохраняем с типом
            except Exception as e:
                print(f"⚠️ Ошибка при сохранении ключа {key}: {e}")
                pass
        
        # Очищаем все данные
        r.flushall()
        print("✅ Redis очищен успешно")
        
        # Восстанавливаем настройки торговли и задачи celery-trade
        restored_count = 0
        for key, (orig_type, value) in preserved_data.items():
            try:
                value_type = type(value).__name__
                print(f"🔄 Восстанавливаем {key} (ориг. тип: {orig_type}, знач. тип: {value_type})")  # Отладка
                
                if orig_type == 'list':
                    for item in value:
                        r.lpush(key, item)
                elif orig_type == 'set':
                    for item in value:
                        r.sadd(key, item)
                elif orig_type == 'hash':
                    r.hmset(key, value)
                else:
                    r.set(key, value)
                restored_count += 1
            except Exception as e:
                print(f"⚠️ Ошибка при восстановлении ключа {key}: {e}")
                pass
        
        if restored_count > 0:
            print(f"✅ Восстановлено {restored_count} ключей (торговля + celery-trade)")
            print("💡 Очередь 'trade' и задачи celery-trade сохранены")
        
        # Отладка: Проверяем ключ trading:current_status после восстановления
        try:
            status_value = r.get('trading:current_status')
            if status_value:
                print(f"🔑 trading:current_status после восстановления: {status_value.decode('utf-8')}")
            else:
                print("⚠️ trading:current_status не найден после восстановления")
        except Exception as e:
            print(f"⚠️ Ошибка проверки trading:current_status: {e}")
            
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

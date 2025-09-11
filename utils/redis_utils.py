from redis import Redis as _Redis
import redis

_redis_client = None

def get_redis_client():
    """Глобальный Redis-клиент (decode_responses=True)."""
    global _redis_client
    if _redis_client is None:
        _redis_client = _Redis(host='redis', port=6379, db=0, decode_responses=True)
    return _redis_client

def clear_redis_on_startup():
    """Очищает Redis при запуске приложения. Возвращает подключение или None."""
    try:
        redis_hosts = ['localhost', 'redis', '127.0.0.1']
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
            print("⚠️ Не удалось подключиться к Redis")
            return None
        r.flushall()
        print("✅ Redis очищен при запуске")
        if r.dbsize() == 0:
            print("✅ Redis пуст, готов к работе")
        else:
            print(f"⚠️ В Redis осталось {r.dbsize()} ключей")
        return r
    except Exception as e:
        print(f"⚠️ Не удалось очистить Redis: {e}")
        print("Продолжаем работу без очистки Redis")
        return None



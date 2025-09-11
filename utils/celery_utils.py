import redis
import docker

def ensure_symbol_worker(app, redis_client, queue_name: str) -> dict:
    """Гарантирует наличие отдельного Celery-воркера для указанной очереди.

    - Проверяет Redis-лок `celery:worker:<queue>`
    - Проверяет процесс внутри контейнера `celery-worker`
    - При отсутствии — запускает новый процесс воркера для очереди
    """
    try:
        lock_key = f"celery:worker:{queue_name}"
        try:
            if redis_client.get(lock_key):
                return {"started": False, "reason": "already_marked_running"}
        except Exception:
            pass

        client = docker.from_env()
        container = client.containers.get('celery-worker')
        app.logger.info(f"[ensure_worker] container 'celery-worker' status={container.status}")

        check_cmd = f"sh -lc 'pgrep -af \"celery.*-Q {queue_name}\" >/dev/null 2>&1'"
        exec_res = container.exec_run(check_cmd, tty=True)
        already_running = (exec_res.exit_code == 0)
        app.logger.info(f"[ensure_worker] check queue={queue_name} exit={exec_res.exit_code}")

        if already_running:
            try:
                redis_client.setex(lock_key, 24 * 3600, 'running')
            except Exception:
                pass
            return {"started": False, "reason": "process_exists"}

        start_cmd = (
            "sh -lc '"
            "mkdir -p /workspace/logs && "
            f"(OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4 nohup celery -A tasks.celery_tasks worker -Q {queue_name} -P solo -c 1 --loglevel=info "
            f"> /workspace/logs/{queue_name}.log 2>&1 & echo $! > /workspace/logs/{queue_name}.pid) "
            "'"
        )
        res = container.exec_run(start_cmd, tty=True)
        app.logger.info(f"[ensure_worker] start queue={queue_name} exit={res.exit_code}")

        exec_res2 = container.exec_run(check_cmd, tty=True)
        app.logger.info(f"[ensure_worker] recheck queue={queue_name} exit={exec_res2.exit_code}")

        try:
            redis_client.setex(lock_key, 24 * 3600, 'running')
        except Exception:
            pass

        return {"started": True}
    except Exception as e:
        app.logger.error(f"[ensure_worker] error: {e}")
        return {"started": False, "error": str(e)}



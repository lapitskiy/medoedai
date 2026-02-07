#!/bin/sh
# wait-for-redis.sh

REDIS_HOST=${REDIS_HOST:-redis}
REDIS_PORT=${REDIS_PORT:-6379}
TIMEOUT=${TIMEOUT:-30}

python - <<'PY'
import os, socket, sys, time
host = os.environ.get("REDIS_HOST", "redis")
port = int(os.environ.get("REDIS_PORT", "6379"))
timeout = int(os.environ.get("TIMEOUT", "30"))
print(f"Waiting for Redis at {host}:{port} to be available...", file=sys.stderr)
for i in range(1, timeout + 1):
    try:
        with socket.create_connection((host, port), timeout=1):
            print(f"Redis is available after {i} seconds.", file=sys.stderr)
            sys.exit(0)
    except OSError:
        time.sleep(1)
print(f"Redis did not become available within {timeout} seconds. Exiting.", file=sys.stderr)
sys.exit(1)
PY

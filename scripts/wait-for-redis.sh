#!/bin/sh
# wait-for-redis.sh

REDIS_HOST=${REDIS_HOST:-redis}
REDIS_PORT=${REDIS_PORT:-6379}
TIMEOUT=${TIMEOUT:-30}

>&2 echo "Waiting for Redis at $REDIS_HOST:$REDIS_PORT to be available..."

for i in $(seq $TIMEOUT);
do
  nc -z "$REDIS_HOST" "$REDIS_PORT" > /dev/null 2>&1
  result=$?
  if [ $result -eq 0 ]; then
    >&2 echo "Redis is available after $i seconds."
    exit 0
  fi
  sleep 1
done

>&2 echo "Redis did not become available within $TIMEOUT seconds. Exiting."
exit 1

#!/usr/bin/env python
"""Purge Celery train queue and all dedup marker keys on startup."""
import redis

r = redis.Redis('redis', 6379)
keys = r.keys('celery:train:*')
if keys:
    r.delete(*keys)
r.delete('train')
print(f'🧹 Purged train queue + {len(keys)} marker keys')

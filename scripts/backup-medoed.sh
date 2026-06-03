#!/usr/bin/env bash
set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-/home/lapitsky/backup/daily}"
RCLONE_REMOTE="${RCLONE_REMOTE:-selectel}"
S3_BUCKET="${S3_BUCKET:-medoedai-backups}"
S3_PREFIX="${S3_PREFIX:-daily}"
PG_CONTAINER="${PG_CONTAINER:-postgres}"
PG_USER="${PG_USER:-medoed_user}"
PG_DB="${PG_DB:-medoed_db}"
LOCAL_KEEP_DAYS="${LOCAL_KEEP_DAYS:-7}"
REMOTE_KEEP_DAYS="${REMOTE_KEEP_DAYS:-30}"

DATE_TAG="$(date +%F)"
DUMP_PATH="${BACKUP_DIR}/medoed_db_${DATE_TAG}.dump"
REMOTE_PATH="${RCLONE_REMOTE}:${S3_BUCKET}/${S3_PREFIX}/"

mkdir -p "${BACKUP_DIR}"

docker exec "${PG_CONTAINER}" pg_dump -U "${PG_USER}" -Fc "${PG_DB}" -f /tmp/medoed.dump
docker cp "${PG_CONTAINER}:/tmp/medoed.dump" "${DUMP_PATH}"
docker exec "${PG_CONTAINER}" rm -f /tmp/medoed.dump

rclone copy "${BACKUP_DIR}/" "${REMOTE_PATH}" --include "medoed_db_*.dump" -v

find "${BACKUP_DIR}" -name 'medoed_db_*.dump' -type f -mtime +"${LOCAL_KEEP_DAYS}" -delete
rclone delete "${REMOTE_PATH}" --include "medoed_db_*.dump" --min-age "${REMOTE_KEEP_DAYS}d"

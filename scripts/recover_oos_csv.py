#!/usr/bin/env python3
"""Recover OOS batch CSV from Celery task results (by task_id list)."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from celery.result import AsyncResult  # noqa: E402

from tasks import celery  # noqa: E402


def _load_ids(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return sorted(set(re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", text)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover OOS CSV from Celery results")
    parser.add_argument("task_ids_file", type=Path, help="file with task UUIDs (one per line or log dump)")
    args = parser.parse_args()

    ids = _load_ids(args.task_ids_file)
    if not ids:
        print("no task ids found", file=sys.stderr)
        return 1

    results: list[dict] = []
    missing = 0
    failed = 0
    for i, task_id in enumerate(ids, 1):
        ar = AsyncResult(task_id, app=celery)
        if ar.state != "SUCCESS":
            if ar.state in ("PENDING", "STARTED", "RETRY"):
                missing += 1
            else:
                failed += 1
            continue
        res = ar.result
        if isinstance(res, dict) and res.get("success"):
            results.append(res)
        else:
            failed += 1
        if i % 200 == 0:
            print(f"polled {i}/{len(ids)} ok={len(results)} missing={missing} failed={failed}")

    if not results:
        print("no successful results in redis", file=sys.stderr)
        return 1

    from utils.xgb_oos_batch_csv import write_oos_batch_csv  # noqa: E402

    payload = write_oos_batch_csv(results)
    print(payload)
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())

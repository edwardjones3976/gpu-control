"""Overnight GPU queue entrypoint (1–7 AM PT) — source stub.

Live task name: GPU-Overnight-Queue. Register only after Cursor PASS via Desk Code.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")


def in_window(now: datetime | None = None) -> bool:
    now = now or datetime.now(PT)
    return 1 <= now.hour < 7


def main() -> int:
    if not in_window():
        print(json.dumps({"ran": False, "reason": "outside 01:00-07:00 America/Los_Angeles"}))
        return 0
    sys.path.append(r"C:\Users\edwar\Tools\gpu")
    # Import runner from installed runtime copy after merge.
    try:
        from jobs.runner import list_pending, load_job, run_job_stub  # type: ignore
    except ImportError:
        print(json.dumps({"ran": False, "reason": "jobs.runner not installed under Tools/gpu"}))
        return 1
    results = []
    for path in list_pending():
        job = load_job(path)
        results.append(run_job_stub(job, provider=str(job.get("provider", "scheduler"))))
    print(json.dumps({"ran": True, "results": results}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

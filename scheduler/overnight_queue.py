"""Overnight GPU queue (ACCEPT item 4): drains Tools/gpu/jobs/queue between 01:00 and 07:00 America/Los_Angeles.

Run by the scheduled task GPU-Overnight-Queue (registered only after Cursor PASS, see install_task.py).
Outside the window it exits at once. Inside it calls the installed jobs/runner.py drain() with a 07:00
deadline: jobs near the deadline get the stop file, save progress and stay queued; Warden sessions come first.

A scheduled task has no Agent Room or Slack access, so failures and yields go to Tools/gpu/jobs/notices.jsonl
(written by the runner) and a summary goes to Tools/gpu/scheduler/runs.jsonl. The first peer active after 07:00
posts the night's notices, once, to #director-admin-ops (owner rule: stop, save, one Slack notice).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
START, END = time(1, 0), time(7, 0)
RUNTIME = Path(r"C:\Users\edwar\Tools\gpu")
RUNNER = RUNTIME / "jobs" / "runner.py"
LOG = RUNTIME / "scheduler" / "runs.jsonl"


def in_window(now: datetime) -> bool:
    t = now.astimezone(PT).time()
    return START <= t < END


def deadline_for(now: datetime) -> datetime:
    """07:00 PT on the same Pacific date as now (DST-safe: built from the local date, not now + offset)."""
    local = now.astimezone(PT)
    return datetime.combine(local.date(), END, tzinfo=PT)


def _load_runner():
    spec = importlib.util.spec_from_file_location("gpu_runner", RUNNER)
    if spec is None or spec.loader is None or not RUNNER.is_file():
        raise ImportError("jobs runner not installed at %s" % RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _log(entry: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main(now: datetime | None = None) -> int:
    now = now or datetime.now(PT)
    entry = {"at": now.isoformat()}
    if not in_window(now):
        print(json.dumps({**entry, "ran": False, "reason": "outside 01:00-07:00 America/Los_Angeles"}))
        return 0
    try:
        runner = _load_runner()
    except ImportError as e:
        entry.update(ran=False, error=str(e))
        _log(entry)
        print(json.dumps(entry))
        return 1
    result = runner.drain(deadline_for(now).timestamp())
    entry.update(ran=True, deadline=deadline_for(now).isoformat(), **result)
    _log(entry)
    print(json.dumps(entry))
    return 0 if result.get("ok") else 1


def selftest() -> dict:
    def pt(y, mo, d, h, mi):
        return datetime(y, mo, d, h, mi, tzinfo=PT)

    assert not in_window(pt(2026, 11, 22, 0, 59)) and in_window(pt(2026, 11, 22, 1, 0))
    assert in_window(pt(2026, 11, 22, 6, 59)) and not in_window(pt(2026, 11, 22, 7, 0))
    assert not in_window(pt(2026, 11, 22, 13, 0))
    assert in_window(datetime(2026, 11, 22, 10, 30, tzinfo=ZoneInfo("UTC")))  # 02:30 PST
    assert deadline_for(pt(2026, 11, 22, 3, 0)) == pt(2026, 11, 22, 7, 0)
    # DST ends 2026-11-01 at 02:00 PDT: the window spans 7 wall-clock hours that night, deadline still 07:00 PST.
    assert deadline_for(pt(2026, 11, 1, 1, 30)).utcoffset() == timedelta(hours=-8)
    assert main(pt(2026, 11, 22, 12, 0)) == 0  # outside the window: no runner import, no side effects
    return {"offline": "pass"}


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        print(json.dumps(selftest()))
        raise SystemExit(0)
    raise SystemExit(main())

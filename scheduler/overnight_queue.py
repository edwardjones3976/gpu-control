"""Overnight GPU queue (ACCEPT item 4): drains Tools/gpu/jobs/queue between 01:00 and 07:00 America/Los_Angeles.

Run by the scheduled task GPU-Overnight-Queue (registered only after Cursor PASS, see install_task.py).
Outside the window it exits at once. Inside it calls the installed jobs/runner.py drain() with a deadline that
is the earlier of 07:00 PT and the task's 6-hour hard limit, minus a margin longer than the runner's stop grace,
so a job always gets its stop file and time to checkpoint before anything kills it.

A scheduled task has no Agent Room or Slack access, so failures and yields go to Tools/gpu/jobs/notices.jsonl
(written by the runner) and a summary goes to Tools/gpu/scheduler/runs.jsonl. The first peer active after 07:00
posts the night's notices, once, to #director-admin-ops (owner rule: stop, save, one Slack notice).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")
START, END = time(1, 0), time(7, 0)
TASK_LIMIT = timedelta(hours=6)  # ExecutionTimeLimit PT6H in GPU-Overnight-Queue.xml (elapsed time, not wall clock)
MARGIN = timedelta(minutes=5)  # > runner STOP_GRACE_S (120 s) + its 30 s wait + polling
RUNTIME = Path(r"C:\Users\edwar\Tools\gpu")
RUNNER = RUNTIME / "jobs" / "runner.py"
LOG = RUNTIME / "scheduler" / "runs.jsonl"


def in_window(now: datetime) -> bool:
    t = now.astimezone(PT).time()
    return START <= t < END


def deadline_for(now: datetime) -> datetime:
    """The earlier of 07:00 PT (from the Pacific date, DST-safe) and now + the task limit, minus MARGIN."""
    local = now.astimezone(PT)
    seven = datetime.combine(local.date(), END, tzinfo=PT).astimezone(timezone.utc)
    # Elapsed-time arithmetic must be in UTC: aware datetimes sharing one tzinfo add wall-clock hours across DST.
    return (min(seven, now.astimezone(timezone.utc) + TASK_LIMIT) - MARGIN).astimezone(PT)


def _load_runner():
    if not RUNNER.is_file():
        raise ImportError("jobs runner not installed at %s" % RUNNER)
    spec = importlib.util.spec_from_file_location("gpu_runner", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _log(entry: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main(now: datetime | None = None, load=_load_runner) -> int:
    now = now or datetime.now(PT)
    entry = {"at": now.isoformat()}
    if not in_window(now):
        print(json.dumps({**entry, "ran": False, "reason": "outside 01:00-07:00 America/Los_Angeles"}))
        return 0
    deadline = deadline_for(now)
    try:
        result = load().drain(deadline.timestamp(), now=now)
        entry.update(ran=True, deadline=deadline.isoformat(), drain=result)
        code = 0 if result.get("ok") else 1
    except Exception as e:  # noqa: BLE001 - a scheduled task must always leave a log line
        entry.update(ran=False, deadline=deadline.isoformat(), error="%s: %s" % (type(e).__name__, e))
        code = 1
    _log(entry)
    print(json.dumps(entry))
    return code


def selftest() -> dict:
    def pt(y, mo, d, h, mi):
        return datetime(y, mo, d, h, mi, tzinfo=PT)

    assert not in_window(pt(2026, 11, 22, 0, 59)) and in_window(pt(2026, 11, 22, 1, 0))
    assert in_window(pt(2026, 11, 22, 6, 59)) and not in_window(pt(2026, 11, 22, 7, 0))
    assert not in_window(pt(2026, 11, 22, 13, 0))
    assert in_window(datetime(2026, 11, 22, 10, 30, tzinfo=ZoneInfo("UTC")))  # 02:30 PST
    assert deadline_for(pt(2026, 11, 22, 1, 0)) == pt(2026, 11, 22, 6, 55)
    assert deadline_for(pt(2026, 11, 22, 3, 0)) == pt(2026, 11, 22, 6, 55)
    # DST ends 2026-11-01 at 02:00 PDT: a 01:00 PDT start is hard-killed 6 elapsed hours later, at 06:00 PST.
    fall = datetime(2026, 11, 1, 1, 0, tzinfo=PT)  # fold=0: the first 01:00, PDT
    d = deadline_for(fall)
    utc = timezone.utc
    assert d.astimezone(utc) - fall.astimezone(utc) == TASK_LIMIT - MARGIN, (d, fall)
    assert (d.hour, d.minute, d.utcoffset()) == (5, 55, timedelta(hours=-8)), d  # 05:55 PST, before the 06:00 kill

    class Stub:
        calls = []

        @staticmethod
        def drain(deadline, now=None):
            Stub.calls.append(deadline)
            return {"ok": True, "jobs_run": 0, "results": [], "left": 0}

    global LOG
    real_log, LOG = LOG, Path(__import__("tempfile").mkdtemp()) / "runs.jsonl"
    try:
        assert main(pt(2026, 11, 22, 2, 0), load=lambda: Stub) == 0 and Stub.calls  # in-window path, no TypeError
        assert main(pt(2026, 11, 22, 2, 0), load=lambda: (_ for _ in ()).throw(SyntaxError("bad"))) == 1
        lines = LOG.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2 and "SyntaxError" in lines[1], lines
    finally:
        LOG = real_log
    assert main(pt(2026, 11, 22, 12, 0)) == 0  # outside the window: no runner load, no log write
    return {"offline": "pass"}


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        print(json.dumps(selftest()))
        raise SystemExit(0)
    raise SystemExit(main())

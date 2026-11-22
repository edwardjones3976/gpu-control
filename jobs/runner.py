"""GPU job runner: runs reviewed job scripts one at a time under the local_gpu heavy lease (ACCEPT item 3).

A job is a JSON file in Tools/gpu/jobs/queue/:
  {"label": "embed-sweep-1", "script": "cuda_matmul.py", "sha256": "<64 hex>", "provider": "claude",
   "vram_gb": 2, "ram_gb": 3, "max_minutes": 30, "args": ["--size", "4096"]}

Rules enforced here (owner design interview, 2026-11-21):
- Reviewed by hash: the script is a bare name in Tools/gpu/jobs/scripts/; its sha256 is listed in approved.json
  with a reviewer from a known provider other than the author. The verified bytes are copied to a per-run file
  and that copy is what runs, so the file cannot change between the check and the start.
- One heavy job at a time: every job takes the heavy local_gpu.lease() (capped Windows job, writer lock).
  The child also runs in its own kill-on-close job, so no process of it outlives the lease.
- Warden first: a Warden session, or the end of the window, writes the job's stop file. A job that then exits
  with YIELD_EXIT (3) within STOP_GRACE_S is "yielded" and requeued; anything else is done or failed.
- No automatic retry: a job moves to running/ before it starts. Done/failed jobs go to done/ or failed/
  (unique names). A running/ entry whose runner died (crash, or the lease's max_minutes kill) is moved to
  failed/ at the next drain, with a notice. Each failure or yield adds one line to notices.jsonl.
- Queue jobs (drain, or run on a queue/ file) start only inside 01:00-07:00 America/Los_Angeles.

Subcommands: list | validate JOB | run JOB | drain --deadline ISO | setup-venv | --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from datetime import time as dtime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

RUNTIME = Path(r"C:\Users\edwar\Tools\gpu")
JOBS = RUNTIME / "jobs"
QUEUE, RUNNING, DONE, FAILED, LOGS = (JOBS / d for d in ("queue", "running", "done", "failed", "logs"))
SCRIPTS = JOBS / "scripts"
APPROVED = JOBS / "approved.json"  # installed from the reviewed repo copy, never hand-edited
NOTICES = JOBS / "notices.jsonl"  # one line per failure or yield, for the morning peer to post
RUNS = JOBS / "runs.jsonl"
VENV_PY = RUNTIME / "venv" / "Scripts" / "python.exe"
SCRATCH = RUNTIME / "scratch"
BASE_PY = Path(r"C:\Users\edwar\AppData\Local\Python\pythoncore-3.14-64\python.exe")
PT = ZoneInfo("America/Los_Angeles")
WINDOW = (dtime(1, 0), dtime(7, 0))
YIELD_EXIT = 3  # a job that saved its checkpoint after the stop file exits with this code
STOP_GRACE_S = 120  # time a job gets to save its progress after the stop file appears
LEASE_MARGIN_MIN = 5  # the lease's own max_minutes kill (which also ends this runner) is only a backstop
MAX_VRAM_GB, MAX_RAM_GB, MAX_MINUTES = 10.5, 16, 360
PROVIDERS = frozenset({"openai", "claude", "cursor", "antigravity", "grok", "copilot", "local"})
JOB_KEYS = {"label", "script", "sha256", "provider", "vram_gb", "ram_gb", "max_minutes", "args", "notes"}
_LABEL = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")
_SCRIPT = re.compile(r"^[A-Za-z0-9_\-]{1,64}\.py$")
_SHA = re.compile(r"^[0-9a-f]{64}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _append(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": _now(), **entry}) + "\n")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _provider(handle: Any) -> str:
    return str(handle or "").strip().lower().split("@")[0]


def in_window(when: datetime) -> bool:
    t = when.astimezone(PT).time()
    return WINDOW[0] <= t < WINDOW[1]


def load_approved(path: Path = APPROVED) -> dict[str, dict[str, Any]]:
    """sha256 -> entry. An entry counts only if its reviewer is a known provider and not the author's provider."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    ok = {}
    for sha, e in (data.get("approved") or {}).items():
        if not (isinstance(e, dict) and isinstance(sha, str) and _SHA.match(sha)):
            continue
        author, reviewer = _provider(e.get("author")), _provider(e.get("reviewed_by"))
        if author in PROVIDERS and reviewer in PROVIDERS and reviewer != author:
            ok[sha] = e
    return ok


def validate(job: Any, approved: dict[str, dict[str, Any]], scripts: Path = SCRIPTS) -> list[str]:
    if not isinstance(job, dict):
        return ["job must be a JSON object"]
    e = ["unknown key %r" % k for k in job if k not in JOB_KEYS]
    if not (isinstance(job.get("label"), str) and _LABEL.match(job["label"])):
        e.append("label must be 1-64 letters, digits, '.', '_' or '-'")
    script = job.get("script")
    if not (isinstance(script, str) and _SCRIPT.match(script)):
        e.append("script must be a bare file name like name.py (no folders)")
        script = None
    sha = job.get("sha256")
    if not (isinstance(sha, str) and _SHA.match(sha)):
        e.append("sha256 must be 64 lowercase hex characters")
        sha = None
    if script and sha:
        path = scripts / script
        if not path.is_file():
            e.append("script not installed: %s" % path)
        elif _sha_bytes(path.read_bytes()) != sha:
            e.append("script bytes do not match sha256 (changed after review)")
        entry = approved.get(sha)
        if not entry:
            e.append("sha256 is not in approved.json with a reviewer from another provider")
        elif entry.get("script") != script:
            e.append("approved.json lists this hash for %r, not %r" % (entry.get("script"), script))
    for key, lo, hi in (("vram_gb", 0, MAX_VRAM_GB), ("ram_gb", 0, MAX_RAM_GB), ("max_minutes", 1, MAX_MINUTES)):
        v = job.get(key, 0 if key != "max_minutes" else 30)
        if not (_num(v) and lo <= v <= hi):
            e.append("%s must be a number from %s to %s" % (key, lo, hi))
    args = job.get("args", [])
    if not (isinstance(args, list) and all(isinstance(a, str) and len(a) <= 200 for a in args) and len(args) <= 32):
        e.append("args must be a list of at most 32 strings")
    return e


def _lg():
    sys.path.append(str(RUNTIME))  # append, never insert
    import local_gpu  # type: ignore

    return local_gpu


def _unique(dest_dir: Path, name: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    p = dest_dir / name
    if not p.exists():
        return p
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return dest_dir / ("%s-%s%s" % (p.stem, stamp, p.suffix))


def _pid_alive(pid: int) -> bool:
    import ctypes
    import ctypes.wintypes as W

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
    k.OpenProcess.restype = W.HANDLE
    k.GetExitCodeProcess.argtypes = [W.HANDLE, ctypes.POINTER(W.DWORD)]
    k.CloseHandle.argtypes = [W.HANDLE]
    h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    try:
        code = W.DWORD()
        return bool(k.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
    finally:
        k.CloseHandle(h)


class _ChildJob:
    """A kill-on-close Windows job holding only the child, so its whole tree ends with the run."""

    def __init__(self, lg, pid: int):
        import ctypes
        import ctypes.wintypes as W

        self.lg, self.job = lg, lg.k32.CreateJobObjectW(None, None)
        if not self.job:
            raise OSError("CreateJobObjectW failed")
        info = lg._Ext()
        info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        lg.k32.SetInformationJobObject(self.job, 9, ctypes.byref(info), ctypes.sizeof(info))
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
        k.OpenProcess.restype = W.HANDLE
        h = k.OpenProcess(0x0101, False, pid)  # PROCESS_SET_QUOTA | PROCESS_TERMINATE
        # shortcut: the child is assigned just after it starts; a grandchild it spawns in the first milliseconds
        # (before its script runs) would escape. Python's start-up makes that window practically empty.
        self.assigned = bool(h) and bool(lg.k32.AssignProcessToJobObject(self.job, h))
        if h:
            lg.k32.CloseHandle(h)

    def close(self) -> None:
        if self.job:
            self.lg.k32.TerminateJobObject(self.job, 1)
            self.lg.k32.CloseHandle(self.job)
            self.job = None


def recover_dead_runs() -> list[dict[str, Any]]:
    """Move running/ entries whose runner process is gone to failed/, one notice each (no automatic retry)."""
    out: list[dict[str, Any]] = []
    for marker in sorted(RUNNING.glob("*.marker.json")) if RUNNING.is_dir() else []:
        try:
            m = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if _pid_alive(int(m.get("runner_pid", 0))):
            continue
        job_file = RUNNING / str(m.get("job", ""))
        if m.get("job") and job_file.is_file():
            shutil.move(str(job_file), str(_unique(FAILED, job_file.name)))
        shutil.move(str(marker), str(_unique(FAILED, marker.name)))
        entry = {"kind": "failed", "job": m.get("job"), "status": "failed", "provider": m.get("provider"),
                 "error": "runner ended while the job ran (crash, or the lease's max_minutes kill)",
                 "started": m.get("started")}
        _append(NOTICES, entry)
        _append(RUNS, entry)
        out.append(entry)
    return out


def run_job(job_path: Path, approved: dict[str, dict[str, Any]], deadline: float | None = None,
            now: datetime | None = None) -> dict[str, Any]:
    """Run one job file. Returns {status: done|failed|yielded|refused|busy, ...}."""
    job_path = job_path.resolve()
    from_queue = job_path.parent == QUEUE.resolve()
    if from_queue and not in_window(now or datetime.now(PT)):
        return {"status": "refused", "job": job_path.name,
                "errors": ["queued batch jobs start only 01:00-07:00 America/Los_Angeles"]}
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"status": "refused", "job": job_path.name, "errors": ["unreadable job: %s" % e]}
    errors = validate(job, approved)
    if errors:
        return _finish(job_path, job, {"status": "refused", "errors": errors})
    if not VENV_PY.is_file():
        return _finish(job_path, job, {"status": "refused", "errors": ["GPU environment missing: run setup-venv"]})
    lg = _lg()
    label, provider = job["label"], str(job.get("provider", "unknown"))
    if lg.warden_session():
        return {"status": "busy", "job": job_path.name, "reason": "a Warden session is running (Warden first)"}
    # The verified bytes are what runs: copy them to a per-run folder nobody else writes to.
    data = (SCRIPTS / job["script"]).read_bytes()
    if _sha_bytes(data) != job["sha256"]:
        return _finish(job_path, job, {"status": "refused", "errors": ["script changed after validation"]})
    run_dir = SCRATCH / ("run-" + label)
    run_dir.mkdir(parents=True, exist_ok=True)
    script_copy = run_dir / job["script"]
    script_copy.write_bytes(data)
    LOGS.mkdir(parents=True, exist_ok=True)
    stop_file = run_dir / ".stop"
    stop_file.unlink(missing_ok=True)
    log_path = LOGS / ("%s-%s.log" % (label, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    env = {**os.environ, "GPU_JOB_STOP_FILE": str(stop_file), "GPU_JOB_LABEL": label,
           "GPU_JOB_CHECKPOINT_DIR": str(SCRATCH / ("ckpt-" + label))}
    max_s = float(job.get("max_minutes", 30)) * 60
    # No automatic retry: the job leaves queue/ before it starts, with a marker naming this runner.
    RUNNING.mkdir(parents=True, exist_ok=True)
    if from_queue:
        moved = RUNNING / job_path.name
        shutil.move(str(job_path), str(moved))
        job_path = moved
    marker = RUNNING / (job_path.stem + ".marker.json")
    marker.write_text(json.dumps({"job": job_path.name if from_queue else None, "runner_pid": os.getpid(),
                                  "started": _now(), "provider": provider}), encoding="utf-8")
    t0 = time.monotonic()
    stop_reason, killed, code = None, None, None
    try:
        with lg.lease(label, vram_gb=float(job.get("vram_gb", 0)), ram_gb=float(job.get("ram_gb", 0)),
                      max_minutes=int(job.get("max_minutes", 30)) + LEASE_MARGIN_MIN, provider=provider,
                      cpu_workers=lg.HEAVY_WORKERS + 1, job_id=job_path.stem):  # always the heavy path
            with open(log_path, "wb") as log:
                p = subprocess.Popen([str(VENV_PY), "-I", str(script_copy), *job.get("args", [])],
                                     cwd=str(run_dir), env=env, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT)
                child = None
                try:
                    child = _ChildJob(lg, p.pid)
                    stop_at = None
                    while p.poll() is None:
                        time.sleep(2)
                        if time.monotonic() - t0 > max_s:
                            killed = "max_minutes"
                            break
                        if stop_at is None:
                            if lg.warden_session():
                                stop_reason = "Warden session started"
                            elif deadline is not None and time.time() > deadline - STOP_GRACE_S:
                                stop_reason = "window ends"
                            if stop_reason:
                                stop_file.write_text(stop_reason, encoding="utf-8")
                                stop_at = time.monotonic() + STOP_GRACE_S
                        elif time.monotonic() > stop_at:
                            killed = "ignored the stop file for %d s" % STOP_GRACE_S
                            break
                finally:
                    if child is not None:
                        child.close()  # ends the child and everything it started, whatever happened above
                    if p.poll() is None:
                        p.kill()
                    p.wait()
                code = p.returncode
    except Exception as e:  # noqa: BLE001 - Busy/Refused from the lease, or a start failure
        if type(e).__name__ == "Busy" and code is None:
            back = _requeue(job_path, from_queue)
            marker.unlink(missing_ok=True)
            return {"status": "busy", "job": back.name, "reason": str(e)}
        return _finish(job_path, job, {"status": "failed", "error": "%s: %s" % (type(e).__name__, e)}, marker)
    finally:
        stop_file.unlink(missing_ok=True)
    result = {"exit_code": code, "seconds": round(time.monotonic() - t0, 1), "log": str(log_path)}
    if killed:
        return _finish(job_path, job, {"status": "failed", "error": "ended: " + killed, **result}, marker)
    if stop_reason and code == YIELD_EXIT:
        back = _requeue(job_path, from_queue)
        marker.unlink(missing_ok=True)
        result.update(status="yielded", reason=stop_reason, job=back.name)
        _append(NOTICES, {"kind": "yielded", "provider": provider, **result})
        _append(RUNS, result)
        return result
    return _finish(job_path, job, {"status": "done" if code == 0 else "failed", **result}, marker)


def _requeue(job_path: Path, from_queue: bool) -> Path:
    if from_queue and job_path.parent == RUNNING.resolve():
        back = QUEUE / job_path.name
        shutil.move(str(job_path), str(back))
        return back
    return job_path


def _finish(job_path: Path, job: Any, result: dict[str, Any], marker: Path | None = None) -> dict[str, Any]:
    status = result["status"]
    if job_path.parent in (QUEUE.resolve(), RUNNING.resolve()):
        shutil.move(str(job_path), str(_unique(DONE if status == "done" else FAILED, job_path.name)))
    if marker is not None:
        marker.unlink(missing_ok=True)
    entry = {"job": job_path.name, "label": job.get("label") if isinstance(job, dict) else None, **result}
    _append(RUNS, entry)
    if status != "done":
        _append(NOTICES, {"kind": status, "provider": job.get("provider") if isinstance(job, dict) else None,
                          **entry})
    return entry


def pending() -> list[Path]:
    return sorted(p for p in QUEUE.glob("*.json") if p.is_file()) if QUEUE.is_dir() else []


def drain(deadline: float, now: datetime | None = None) -> dict[str, Any]:
    """Run queued jobs in name order until the queue is empty, the card is busy, a job fails outside its own
    code, or the deadline nears. Only inside the overnight window."""
    now = now or datetime.now(PT)
    if not in_window(now):
        return {"ok": False, "jobs_run": 0, "results": [], "left": len(pending()),
                "error": "drain runs only 01:00-07:00 America/Los_Angeles"}
    recovered = recover_dead_runs()
    approved, results = load_approved(), []
    for path in pending():
        if time.time() > deadline - STOP_GRACE_S:
            results.append({"job": path.name, "status": "not-started", "reason": "window ends"})
            break
        r = run_job(path, approved, deadline)
        results.append(r)
        if r["status"] in ("busy", "yielded") or (r["status"] == "failed" and r.get("exit_code") is None):
            break  # the card is taken, or something outside the job broke; leave the rest queued
    return {"ok": all(r["status"] in ("done", "not-started", "busy", "yielded") for r in results),
            "jobs_run": len(results), "recovered": recovered, "results": results, "left": len(pending())}


def setup_venv() -> dict[str, Any]:
    """Create Tools/gpu/venv on the base Python (which holds the CUDA torch build), then install pinned extras."""
    out: dict[str, Any] = {}
    if not VENV_PY.is_file():
        p = subprocess.run([str(BASE_PY), "-m", "venv", "--system-site-packages", str(RUNTIME / "venv")],
                           capture_output=True, text=True, timeout=300, check=False)
        out["venv"] = {"exit_code": p.returncode, "stderr": p.stderr[-500:]}
        if p.returncode != 0:
            return {"ok": False, **out}
    req = Path(__file__).with_name("requirements-gpu.txt")
    lines = [ln for ln in req.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    if lines:
        p = subprocess.run([str(VENV_PY), "-m", "pip", "install", "--require-hashes", "--no-deps", "-r", str(req)],
                           capture_output=True, text=True, timeout=1800, check=False)
        out["pip"] = {"exit_code": p.returncode, "tail": (p.stdout + p.stderr)[-800:]}
    p = subprocess.run([str(VENV_PY), "-I", "-c", "import torch,json;print(json.dumps([torch.__version__,"
                        "torch.cuda.is_available()]))"], capture_output=True, text=True, timeout=120, check=False)
    out["torch"] = p.stdout.strip() or p.stderr[-300:]
    out["ok"] = p.returncode == 0 and out.get("pip", {}).get("exit_code", 0) == 0
    return out


def selftest() -> dict[str, Any]:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        scripts = Path(tmp)
        (scripts / "ok.py").write_text("print('hi')\n", encoding="utf-8")
        sha = _sha_bytes((scripts / "ok.py").read_bytes())
        ap_file = scripts / "approved.json"
        ap_file.write_text(json.dumps({"approved": {
            sha: {"script": "ok.py", "author": "claude@blackbox", "reviewed_by": "cursor@blackbox"},
            "0" * 64: {"script": "self.py", "author": "claude@blackbox", "reviewed_by": "claude@blacktop"},
            "1" * 64: {"script": "case.py", "author": "claude@blackbox", "reviewed_by": " Claude@BlackTop "},
            "2" * 64: {"script": "made.py", "author": "claude@blackbox", "reviewed_by": "live-test@harness"},
            "3" * 64: {"script": "none.py", "author": "claude@blackbox", "reviewed_by": None},
        }}), encoding="utf-8")
        approved = load_approved(ap_file)
        assert list(approved) == [sha], approved  # self, case-variant self, unknown and missing reviewers drop
        good = {"label": "t1", "script": "ok.py", "sha256": sha, "vram_gb": 1, "ram_gb": 1, "max_minutes": 5}
        assert validate(good, approved, scripts) == [], validate(good, approved, scripts)
        bad = [{**good, "script": "..\\x.py"}, {**good, "script": "sub/ok.py"}, {**good, "sha256": "0" * 64},
               {**good, "vram_gb": 99}, {**good, "max_minutes": 0}, {**good, "args": "--x"}, {**good, "cmd": "x"},
               {**good, "label": "a b"}, {**good, "vram_gb": True}]
        for case in bad:
            assert validate(case, approved, scripts), case
        (scripts / "ok.py").write_text("print('changed')\n", encoding="utf-8")
        assert any("changed after review" in m for m in validate(good, approved, scripts))
        (scripts / "a.json").write_text("1", encoding="utf-8")
        assert _unique(scripts, "a.json") != scripts / "a.json" and _unique(scripts, "b.json") == scripts / "b.json"
    assert not in_window(datetime(2026, 11, 22, 14, 0, tzinfo=PT))
    assert in_window(datetime(2026, 11, 22, 2, 0, tzinfo=PT))
    report: dict[str, Any] = {"offline": "pass"}
    if sys.platform == "win32":
        assert _pid_alive(os.getpid()) and not _pid_alive(0)
        noon = datetime(2026, 11, 22, 12, 0, tzinfo=PT)
        assert drain(time.time() + 3600, now=noon)["ok"] is False  # refused outside the window, nothing touched
        report["live"] = {"approved_entries": len(load_approved()), "pending": len(pending()),
                          "venv": VENV_PY.is_file()}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list")
    for name in ("validate", "run"):
        sub.add_parser(name).add_argument("job", type=Path)
    sub.add_parser("drain").add_argument("--deadline", required=True, help="ISO time with a UTC offset")
    sub.add_parser("setup-venv")
    a = ap.parse_args(argv)
    try:
        if a.selftest:
            out: dict[str, Any] = selftest()
        elif a.cmd == "list":
            out = {"ok": True, "pending": [p.name for p in pending()], "approved": len(load_approved())}
        elif a.cmd == "validate":
            errs = validate(json.loads(a.job.read_text(encoding="utf-8")), load_approved())
            out = {"ok": not errs, "errors": errs}
        elif a.cmd == "run":
            out = run_job(a.job, load_approved())
            out["ok"] = out["status"] in ("done", "yielded", "busy")
        elif a.cmd == "drain":
            dl = datetime.fromisoformat(a.deadline)
            if dl.tzinfo is None:
                raise ValueError("--deadline needs a UTC offset, e.g. 2026-11-23T06:55:00-08:00")
            out = drain(dl.timestamp())
        elif a.cmd == "setup-venv":
            out = setup_venv()
        else:
            ap.print_help()
            return 2
    except (OSError, ValueError) as e:
        out = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
    print(json.dumps(out))
    return 0 if out.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())

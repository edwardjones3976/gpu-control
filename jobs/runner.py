"""GPU job runner: runs reviewed job scripts one at a time under the local_gpu heavy lease (ACCEPT item 3).

A job is a JSON file in Tools/gpu/jobs/queue/:
  {"label": "embed-sweep-1", "script": "cuda_matmul.py", "sha256": "<64 hex>", "provider": "claude",
   "vram_gb": 2, "ram_gb": 3, "max_minutes": 30, "args": ["--size", "4096"]}

Rules enforced here (owner design interview, 2026-11-21):
- The script must sit in Tools/gpu/jobs/scripts/ (no paths, no ..), and its sha256 must be listed in
  approved.json with a reviewer who is not its author. approved.json ships through a reviewed PR.
- One heavy job at a time, through local_gpu.lease(): VRAM/RAM admission, a capped Windows job, max_minutes.
- Warden first: while a job runs, a Warden session makes the runner write the job's stop file. The job saves
  its progress and exits; after STOP_GRACE_S the runner ends it. A yielded job stays queued and resumes later.
- A failed job is not retried by the runner; it moves to failed/ and one notice line is written for peers.
- Jobs run with the GPU environment's python (Tools/gpu/venv) and cwd Tools/gpu/scratch.

Subcommands: list | validate JOB | run JOB | drain [--deadline ISO] | setup-venv | --selftest
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
from pathlib import Path
from typing import Any

RUNTIME = Path(r"C:\Users\edwar\Tools\gpu")
JOBS = RUNTIME / "jobs"
QUEUE, DONE, FAILED, LOGS = JOBS / "queue", JOBS / "done", JOBS / "failed", JOBS / "logs"
SCRIPTS = JOBS / "scripts"
APPROVED = JOBS / "approved.json"  # installed from the reviewed repo copy, never hand-edited
NOTICES = JOBS / "notices.jsonl"  # one line per failure or yield, for the morning peer to post
RUNS = JOBS / "runs.jsonl"
VENV_PY = RUNTIME / "venv" / "Scripts" / "python.exe"
SCRATCH = RUNTIME / "scratch"
BASE_PY = Path(r"C:\Users\edwar\AppData\Local\Python\pythoncore-3.14-64\python.exe")
STOP_GRACE_S = 120  # time a job gets to save its progress after the stop file appears
MAX_VRAM_GB, MAX_RAM_GB, MAX_MINUTES = 10.5, 16, 360
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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_approved(path: Path = APPROVED) -> dict[str, dict[str, Any]]:
    """sha256 -> {script, author, reviewed_by, review}. Entries whose reviewer is missing or is the author drop."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    ok = {}
    for sha, e in (data.get("approved") or {}).items():
        if not (isinstance(e, dict) and _SHA.match(sha)):
            continue
        author, reviewer = str(e.get("author") or ""), str(e.get("reviewed_by") or "")
        if reviewer and author and reviewer.split("@")[0] != author.split("@")[0]:
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
        elif _sha(path) != sha:
            e.append("script bytes do not match sha256 (changed after review)")
        entry = approved.get(sha)
        if not entry:
            e.append("sha256 is not in approved.json with a reviewer other than its author")
        elif entry.get("script") != script:
            e.append("approved.json lists this hash for %r, not %r" % (entry.get("script"), script))
    for key, hi in (("vram_gb", MAX_VRAM_GB), ("ram_gb", MAX_RAM_GB), ("max_minutes", MAX_MINUTES)):
        v = job.get(key, 0 if key != "max_minutes" else 30)
        if not (_num(v) and 0 <= v <= hi) or (key == "max_minutes" and v < 1):
            e.append("%s must be a number from %s to %s" % (key, 1 if key == "max_minutes" else 0, hi))
    args = job.get("args", [])
    if not (isinstance(args, list) and all(isinstance(a, str) and len(a) <= 200 for a in args) and len(args) <= 32):
        e.append("args must be a list of at most 32 strings")
    return e


def _lg():
    sys.path.append(str(RUNTIME))  # append, never insert
    import local_gpu  # type: ignore

    return local_gpu


def run_job(job_path: Path, approved: dict[str, dict[str, Any]], deadline: float | None = None) -> dict[str, Any]:
    """Run one queued job. Returns {status: done|failed|yielded|refused|busy, ...}; moves the job file."""
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"status": "refused", "job": job_path.name, "errors": ["unreadable job: %s" % e]}
    errors = validate(job, approved)
    if errors:
        return _finish(job_path, job, {"status": "refused", "errors": errors})
    py = VENV_PY if VENV_PY.is_file() else None
    if py is None:
        return _finish(job_path, job, {"status": "refused", "errors": ["GPU environment missing: run setup-venv"]})
    lg = _lg()
    label, provider = job["label"], str(job.get("provider", "unknown"))
    if lg.warden_session():
        return {"status": "busy", "job": job_path.name, "reason": "a Warden session is running (Warden first)"}
    LOGS.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    stop_file = SCRATCH / (".stop-" + label)
    stop_file.unlink(missing_ok=True)
    log_path = LOGS / ("%s-%s.log" % (label, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    env = {**os.environ, "GPU_JOB_STOP_FILE": str(stop_file), "GPU_JOB_LABEL": label,
           "GPU_JOB_CHECKPOINT_DIR": str(SCRATCH / ("ckpt-" + label))}
    t0 = time.monotonic()
    stop_reason = None
    try:
        with lg.lease(label, vram_gb=float(job.get("vram_gb", 0)), ram_gb=float(job.get("ram_gb", 0)),
                      max_minutes=int(job.get("max_minutes", 30)), provider=provider, job_id=job_path.stem):
            with open(log_path, "wb") as log:
                p = subprocess.Popen([str(py), "-I", str(SCRIPTS / job["script"]), *job.get("args", [])],
                                     cwd=str(SCRATCH), env=env, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT)
                stop_at = None
                while p.poll() is None:
                    time.sleep(2)
                    if stop_at is None:
                        if lg.warden_session():
                            stop_reason = "Warden session started"
                        elif deadline is not None and time.time() > deadline - STOP_GRACE_S:
                            stop_reason = "window ends"
                        if stop_reason:
                            stop_file.write_text(stop_reason, encoding="utf-8")
                            stop_at = time.monotonic() + STOP_GRACE_S
                    elif time.monotonic() > stop_at:
                        p.terminate()
                        p.wait(30)
                code = p.returncode
    except Exception as e:  # noqa: BLE001 - Busy/Refused from the lease, or a start failure
        name = type(e).__name__
        if name == "Busy":
            return {"status": "busy", "job": job_path.name, "reason": str(e)}
        return _finish(job_path, job, {"status": "failed", "error": "%s: %s" % (name, e)})
    finally:
        stop_file.unlink(missing_ok=True)
    result = {"exit_code": code, "seconds": round(time.monotonic() - t0, 1), "log": str(log_path)}
    if stop_reason:
        result.update(status="yielded", reason=stop_reason)  # stays queued; resumes from its checkpoint
        _append(NOTICES, {"kind": "yielded", "job": job_path.name, **result})
        _append(RUNS, {"job": job_path.name, **result})
        return {"job": job_path.name, **result}
    return _finish(job_path, job, {"status": "done" if code == 0 else "failed", **result})


def _finish(job_path: Path, job: Any, result: dict[str, Any]) -> dict[str, Any]:
    status = result["status"]
    dest = DONE if status == "done" else FAILED
    dest.mkdir(parents=True, exist_ok=True)
    if job_path.parent == QUEUE:
        shutil.move(str(job_path), str(dest / job_path.name))
    entry = {"job": job_path.name, "label": job.get("label") if isinstance(job, dict) else None, **result}
    _append(RUNS, entry)
    if status != "done":
        _append(NOTICES, {"kind": status, "provider": job.get("provider") if isinstance(job, dict) else None,
                          **entry})
    return entry


def pending() -> list[Path]:
    return sorted(p for p in QUEUE.glob("*.json") if p.is_file()) if QUEUE.is_dir() else []


def drain(deadline: float | None) -> dict[str, Any]:
    """Run queued jobs in name order until the queue is empty, the card is busy, or the deadline nears."""
    approved, results = load_approved(), []
    for path in pending():
        if deadline is not None and time.time() > deadline - STOP_GRACE_S:
            results.append({"job": path.name, "status": "not-started", "reason": "window ends"})
            break
        r = run_job(path, approved, deadline)
        results.append(r)
        if r["status"] in ("busy", "yielded"):
            break  # Warden or another heavy job has the card; leave the rest queued
    return {"ok": all(r["status"] in ("done", "not-started", "busy", "yielded") for r in results),
            "ran": len(results), "results": results, "left": len(pending())}


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
        sha = _sha(scripts / "ok.py")
        ap_file = scripts / "approved.json"
        ap_file.write_text(json.dumps({"approved": {
            sha: {"script": "ok.py", "author": "claude@blackbox", "reviewed_by": "cursor@blackbox"},
            "0" * 64: {"script": "self.py", "author": "claude@blackbox", "reviewed_by": "claude@blacktop"},
        }}), encoding="utf-8")
        approved = load_approved(ap_file)
        assert sha in approved and "0" * 64 not in approved, approved  # same provider on another device = self
        good = {"label": "t1", "script": "ok.py", "sha256": sha, "vram_gb": 1, "ram_gb": 1, "max_minutes": 5}
        assert validate(good, approved, scripts) == [], validate(good, approved, scripts)
        bad = [{**good, "script": "..\\x.py"}, {**good, "script": "sub/ok.py"}, {**good, "sha256": "0" * 64},
               {**good, "vram_gb": 99}, {**good, "max_minutes": 0}, {**good, "args": "--x"}, {**good, "cmd": "x"},
               {**good, "label": "a b"}, {**good, "vram_gb": True}]
        for case in bad:
            assert validate(case, approved, scripts), case
        (scripts / "ok.py").write_text("print('changed')\n", encoding="utf-8")
        assert any("changed after review" in m for m in validate(good, approved, scripts))
    report: dict[str, Any] = {"offline": "pass"}
    if sys.platform == "win32":
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
    sub.add_parser("drain").add_argument("--deadline", help="ISO time with offset; stop starting jobs before it")
    sub.add_parser("setup-venv")
    a = ap.parse_args(argv)
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
        out = drain(datetime.fromisoformat(a.deadline).timestamp() if a.deadline else None)
    elif a.cmd == "setup-venv":
        out = setup_venv()
    else:
        ap.print_help()
        return 2
    print(json.dumps(out))
    return 0 if out.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())

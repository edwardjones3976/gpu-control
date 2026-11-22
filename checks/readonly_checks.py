"""Read-only host checks for the GPU control plane (ACCEPT item 1).

Sections (each returns a dict and never raises):
  processes  top programs by committed memory (name, pid, parent, working set, commit). No command lines.
  memory     physical memory and commit charge (WDDM counts GPU allocations as commit).
  events     recent Critical/Error events from the System and Application logs.
  crashes    crash, hang, unexpected-shutdown and GPU-reset (TDR) events, plus LiveKernelReports files.
  env        HKCU\\Environment only; values outside the allowlist are redacted.
  ollama     /api/tags from the local Ollama.
  gpu        local_gpu.py gpu_status from the runtime install.

Reads only. Standard library only. No PowerShell, no shell=True, no writes, no other registry keys.
The one subprocess is %SystemRoot%\\System32\\wevtutil.exe "qe" (query events), called by absolute path.
Usage: python -I readonly_checks.py [--section NAME ...] [--top N] [--events N] [--selftest]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

MAX_MESSAGE_CHARS = 300  # event text is trimmed so one run stays well under Desk Code's 64 KiB line limit
EVENT_NS = {"e": "http://schemas.microsoft.com/win/2004/08/events/event"}
LEVELS = {"1": "critical", "2": "error", "3": "warning", "4": "info"}
CRASH_QUERIES = {
    # Event IDs are only unique per provider, so each ID is paired with the provider that defines it.
    "Application": "*[System[(Provider[@Name='Application Error'] and EventID=1000)"
                   " or (Provider[@Name='Windows Error Reporting'] and EventID=1001)"
                   " or (Provider[@Name='Application Hang'] and EventID=1002)]]",
    "System": "*[System[(Provider[@Name='Microsoft-Windows-Kernel-Power'] and EventID=41)"
              " or (Provider[@Name='Microsoft-Windows-WER-SystemErrorReporting'] and EventID=1001)"
              " or (Provider[@Name='Display'] and EventID=4101)"
              " or (Provider[@Name='EventLog'] and EventID=6008)]]",
}
ERROR_QUERY = "*[System[(Level=1 or Level=2)]]"
# HKCU\Environment values print only for these names (allowlist: an unknown name stays redacted).
ENV_VISIBLE_NAMES = frozenset({"PATH", "TEMP", "TMP", "JAVA_HOME", "PYTHONHOME", "PYTHONPATH"})
ENV_VISIBLE_PREFIXES = ("OLLAMA_", "CUDA_", "NVIDIA_", "GHIDRA_", "GGML_")


def _err(e: BaseException) -> dict[str, Any]:
    return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}


# ---------- processes and memory (ctypes, Windows only) ----------

def _kernel32():
    import ctypes
    import ctypes.wintypes as W

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateToolhelp32Snapshot.argtypes = [W.DWORD, W.DWORD]
    k.CreateToolhelp32Snapshot.restype = W.HANDLE
    k.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
    k.OpenProcess.restype = W.HANDLE
    k.CloseHandle.argtypes = [W.HANDLE]
    k.CloseHandle.restype = W.BOOL
    return ctypes, W, k


def processes(top: int = 25) -> dict[str, Any]:
    try:
        ctypes, W, k = _kernel32()

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [("dwSize", W.DWORD), ("cntUsage", W.DWORD), ("th32ProcessID", W.DWORD),
                        ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", W.DWORD),
                        ("cntThreads", W.DWORD), ("th32ParentProcessID", W.DWORD),
                        ("pcPriClassBase", ctypes.c_long), ("dwFlags", W.DWORD),
                        ("szExeFile", ctypes.c_wchar * 260)]

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", W.DWORD), ("PageFaultCount", W.DWORD)] + [
                (n, ctypes.c_size_t) for n in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        k.Process32FirstW.argtypes = k.Process32NextW.argtypes = [W.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
        k.Process32FirstW.restype = k.Process32NextW.restype = W.BOOL
        k.K32GetProcessMemoryInfo.argtypes = [W.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), W.DWORD]
        k.K32GetProcessMemoryInfo.restype = W.BOOL

        snap = k.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
        if not snap or snap == W.HANDLE(-1).value:
            return {"ok": False, "error": "CreateToolhelp32Snapshot failed, winerror %d" % ctypes.get_last_error()}
        rows, unreadable = [], 0
        try:
            pe = PROCESSENTRY32W()
            pe.dwSize = ctypes.sizeof(pe)
            more = k.Process32FirstW(snap, ctypes.byref(pe))
            while more:
                row = {"name": pe.szExeFile, "pid": pe.th32ProcessID, "ppid": pe.th32ParentProcessID,
                       "threads": pe.cntThreads, "working_set_mb": None, "commit_mb": None}
                h = k.OpenProcess(0x1000, False, pe.th32ProcessID)  # PROCESS_QUERY_LIMITED_INFORMATION
                if h:
                    try:
                        pmc = PROCESS_MEMORY_COUNTERS()
                        pmc.cb = ctypes.sizeof(pmc)
                        if k.K32GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                            row["working_set_mb"] = round(pmc.WorkingSetSize / 2**20, 1)
                            row["commit_mb"] = round(pmc.PagefileUsage / 2**20, 1)
                    finally:
                        k.CloseHandle(h)
                if row["commit_mb"] is None:
                    unreadable += 1
                rows.append(row)
                more = k.Process32NextW(snap, ctypes.byref(pe))
        finally:
            k.CloseHandle(snap)
        rows.sort(key=lambda r: r["commit_mb"] or 0, reverse=True)
        return {"ok": True, "count": len(rows), "memory_unreadable": unreadable, "top_by_commit": rows[:top]}
    except Exception as e:  # noqa: BLE001 - report, never crash the checker
        return _err(e)


def memory() -> dict[str, Any]:
    try:
        ctypes, W, k = _kernel32()

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", W.DWORD), ("dwMemoryLoad", W.DWORD)] + [
                (n, ctypes.c_ulonglong) for n in (
                    "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                    "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]

        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(m)
        if not k.GlobalMemoryStatusEx(ctypes.byref(m)):
            return {"ok": False, "error": "GlobalMemoryStatusEx failed, winerror %d" % ctypes.get_last_error()}
        gb = 2**30
        return {"ok": True, "load_pct": m.dwMemoryLoad,
                "ram_total_gb": round(m.ullTotalPhys / gb, 2), "ram_avail_gb": round(m.ullAvailPhys / gb, 2),
                "commit_limit_gb": round(m.ullTotalPageFile / gb, 2),
                "commit_avail_gb": round(m.ullAvailPageFile / gb, 2)}
    except Exception as e:  # noqa: BLE001
        return _err(e)


# ---------- event logs (wevtutil qe, read-only) ----------

def _wevtutil() -> str:
    return os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "wevtutil.exe")


def parse_events(xml_text: str) -> list[dict[str, Any]]:
    """Parse wevtutil /f:RenderedXml output (a run of <Event> elements with no root)."""
    root = ET.fromstring("<Events>" + xml_text + "</Events>")
    out = []
    for ev in root.findall("e:Event", EVENT_NS):
        sysn = ev.find("e:System", EVENT_NS)
        prov = sysn.find("e:Provider", EVENT_NS) if sysn is not None else None
        tc = sysn.find("e:TimeCreated", EVENT_NS) if sysn is not None else None
        msg = ev.findtext("e:RenderingInfo/e:Message", default="", namespaces=EVENT_NS) or ""
        msg = " ".join(msg.split())
        level = sysn.findtext("e:Level", default="", namespaces=EVENT_NS) if sysn is not None else ""
        out.append({
            "time": tc.get("SystemTime") if tc is not None else None,
            "provider": prov.get("Name") if prov is not None else None,
            "event_id": sysn.findtext("e:EventID", default="", namespaces=EVENT_NS) if sysn is not None else "",
            "level": LEVELS.get(level, level),
            "message": msg[:MAX_MESSAGE_CHARS] + ("…" if len(msg) > MAX_MESSAGE_CHARS else ""),
        })
    return out


def query_log(log: str, query: str, count: int) -> dict[str, Any]:
    if log not in ("System", "Application"):  # fixed set; never pass caller text to wevtutil
        return {"ok": False, "error": "log not allowed: %r" % log}
    try:
        p = subprocess.run([_wevtutil(), "qe", log, "/c:%d" % int(count), "/rd:true", "/f:RenderedXml",
                            "/q:" + query], capture_output=True, timeout=60, check=False)
        if p.returncode != 0:
            detail = p.stderr.decode("mbcs", "replace")[:300]
            return {"ok": False, "error": "wevtutil exit %d: %s" % (p.returncode, detail)}
        raw = p.stdout
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("mbcs", "replace")
        return {"ok": True, "events": parse_events(text)}
    except Exception as e:  # noqa: BLE001
        return _err(e)


def events(count: int = 15) -> dict[str, Any]:
    return {log: query_log(log, ERROR_QUERY, count) for log in ("System", "Application")}


def live_kernel_reports(limit: int = 20) -> dict[str, Any]:
    base = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "LiveKernelReports")
    files, errors = [], []
    try:
        for top in os.scandir(base):
            entries = os.scandir(top.path) if top.is_dir(follow_symlinks=False) else [top]
            try:
                for f in entries:
                    if f.is_file(follow_symlinks=False):
                        st = f.stat(follow_symlinks=False)
                        files.append({"path": os.path.relpath(f.path, base), "bytes": st.st_size,
                                      "modified": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()})
            except OSError as e:
                errors.append("%s: %s" % (top.name, e.strerror))
    except OSError as e:
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, e.strerror or e)}
    files.sort(key=lambda f: f["modified"], reverse=True)
    return {"ok": True, "count": len(files), "newest": files[:limit], "unreadable_folders": errors}


def crashes(count: int = 15) -> dict[str, Any]:
    out: dict[str, Any] = {log: query_log(log, q, count) for log, q in CRASH_QUERIES.items()}
    out["live_kernel_reports"] = live_kernel_reports()
    return out


# ---------- HKCU\Environment (names always, values only when allowlisted) ----------

def env_value_visible(name: str) -> bool:
    """Return True if this HKCU\\Environment value may be printed; False redacts it to its length.

    The output goes into Desk Code receipts and Agent Room posts, so a user variable holding an API key
    must never be printed.
    """
    n = name.upper()
    if any(word in n for word in ("KEY", "TOKEN", "SECRET", "PASS", "AUTH", "CRED", "COOKIE", "SESSION")):
        return False  # belt and braces: an allowlisted prefix never wins over a secret-looking word
    return n in ENV_VISIBLE_NAMES or n.startswith(ENV_VISIBLE_PREFIXES)


def env() -> dict[str, Any]:
    try:
        import winreg

        rows = []
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ) as key:
            i = 0
            while True:
                try:
                    name, value, vtype = winreg.EnumValue(key, i)
                except OSError:
                    break
                i += 1
                text = value if isinstance(value, str) else repr(value)
                shown = env_value_visible(name)
                rows.append({"name": name, "type": vtype,
                             "value": text if shown else None, "redacted_len": None if shown else len(text)})
        rows.sort(key=lambda r: r["name"].upper())
        return {"ok": True, "count": len(rows), "values": rows}
    except Exception as e:  # noqa: BLE001
        return _err(e)


# ---------- Ollama and local_gpu (unchanged from the skeleton) ----------

def ollama_tags(endpoint: str = "http://127.0.0.1:11434", timeout: float = 5.0) -> dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(endpoint + "/api/tags")
    try:
        with opener.open(req, timeout=timeout) as r:
            return {"ok": True, "body": json.loads(r.read().decode())}
    except Exception as e:  # noqa: BLE001
        return _err(e)


def local_gpu_status() -> dict[str, Any]:
    sys.path.append(r"C:\Users\edwar\Tools\gpu")  # append, never insert: the runtime copy must not shadow stdlib
    try:
        from local_gpu import tool_gpu_status  # type: ignore

        return {"ok": True, "status": tool_gpu_status()}
    except Exception as e:  # noqa: BLE001
        return _err(e)


SECTIONS = ("processes", "memory", "events", "crashes", "env", "ollama", "gpu")


def run(sections, top: int, count: int) -> dict[str, Any]:
    fns = {"processes": lambda: processes(top), "memory": memory, "events": lambda: events(count),
           "crashes": lambda: crashes(count), "env": env, "ollama": ollama_tags, "gpu": local_gpu_status}
    out: dict[str, Any] = {"checked_at": datetime.now(timezone.utc).isoformat()}
    for s in sections:
        out[s] = fns[s]()
    return out


def selftest() -> dict[str, Any]:
    sample = (
        '<Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event"><System>'
        '<Provider Name="Display"/><EventID>4101</EventID><Level>3</Level>'
        '<TimeCreated SystemTime="2026-11-21T03:00:00.000Z"/></System>'
        '<RenderingInfo Culture="en-US"><Message>Display driver nvlddmkm stopped responding\n and has recovered.'
        "</Message></RenderingInfo></Event>"
    )
    ev = parse_events(sample + sample)
    assert len(ev) == 2 and ev[0]["event_id"] == "4101" and ev[0]["provider"] == "Display", ev
    assert ev[0]["level"] == "warning" and "\n" not in ev[0]["message"], ev
    assert parse_events("") == []
    assert query_log("Security", ERROR_QUERY, 1)["ok"] is False  # only System and Application are queryable
    for secret in ("OPENAI_API_KEY", "GITHUB_TOKEN", "SLACK_BOT_TOKEN", "MY_PASSWORD", "AWS_SECRET_ACCESS_KEY"):
        assert env_value_visible(secret) is False, secret
    assert env_value_visible("ollama_api_key") is False  # the block word wins over an allowlisted prefix
    assert env_value_visible("ollama_keep_alive") is True  # case-insensitive, and KEEP is not KEY
    assert env_value_visible("Path") and env_value_visible("OLLAMA_MODELS") and not env_value_visible("ZAI_X")
    report: dict[str, Any] = {"offline": "pass"}
    if sys.platform == "win32":
        live = run(SECTIONS, 25, 15)
        size = len(json.dumps(live))
        assert size < 60_000, "output too large for one Desk Code line: %d bytes" % size
        assert live["memory"]["ok"] and live["memory"]["ram_total_gb"] > 1, live["memory"]
        assert live["processes"]["ok"] and live["processes"]["count"] > 10, live["processes"]
        assert live["env"]["ok"], live["env"]
        report.update({"live": "pass", "bytes": size,
                       "sections_ok": {s: bool(live[s].get("ok", True)) for s in SECTIONS}})
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--section", action="append", choices=SECTIONS, help="repeatable; default: all")
    ap.add_argument("--top", type=int, default=25, help="processes to list (default 25)")
    ap.add_argument("--events", type=int, default=15, help="events per log (default 15)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        print(json.dumps(selftest()))
        return 0
    print(json.dumps(run(a.section or SECTIONS, max(1, min(a.top, 200)), max(1, min(a.events, 100)))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Check or register the GPU-Overnight-Queue scheduled task from GPU-Overnight-Queue.xml.

  check    parse the XML and verify name, 01:00 trigger, 6 h limit, least privilege and paths (no side effects)
  install  check, then write a UTF-16 copy to Tools/gpu/scratch and run schtasks /Create /XML /TN ... /F
           Only after Cursor PASS on the PR that contains this exact XML (owner rule: live register after PASS).
  status   schtasks /Query for the task (read-only)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

NAME = "GPU-Overnight-Queue"
XML = Path(__file__).with_name("GPU-Overnight-Queue.xml")
SCRATCH = Path(r"C:\Users\edwar\Tools\gpu\scratch")
NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
SCHTASKS = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "schtasks.exe")


def check(xml_text: str | None = None) -> list[str]:
    text = xml_text if xml_text is not None else XML.read_text(encoding="utf-8")
    root = ET.fromstring(text.split("?>", 1)[1] if text.lstrip().startswith("<?xml") else text)
    e = []

    def val(path: str) -> str | None:
        node = root.find(path, NS)
        return node.text.strip() if node is not None and node.text else None

    if val("t:RegistrationInfo/t:URI") != "\\" + NAME:
        e.append("URI must be \\%s" % NAME)
    sb = val("t:Triggers/t:CalendarTrigger/t:StartBoundary") or ""
    if not sb.endswith("T01:00:00"):
        e.append("trigger must start at 01:00 local")
    if val("t:Triggers/t:CalendarTrigger/t:ScheduleByDay/t:DaysInterval") != "1":
        e.append("trigger must be daily")
    if val("t:Settings/t:ExecutionTimeLimit") != "PT6H":
        e.append("ExecutionTimeLimit must be PT6H (stop by 07:00)")
    if val("t:Principals/t:Principal/t:RunLevel") != "LeastPrivilege":
        e.append("RunLevel must be LeastPrivilege")
    if val("t:Settings/t:StartWhenAvailable") != "false":
        e.append("StartWhenAvailable must be false (no catch-up runs in the day)")
    if val("t:Settings/t:WakeToRun") != "false":
        e.append("WakeToRun must be false")
    execs = root.findall("t:Actions/t:Exec", NS)
    if len(execs) != 1:
        e.append("exactly one Exec action")
    else:
        cmd = val("t:Actions/t:Exec/t:Command") or ""
        args = val("t:Actions/t:Exec/t:Arguments") or ""
        if not cmd.lower().endswith("python.exe") or "\\\\" in cmd:
            e.append("Command must be one python.exe path with single backslashes")
        if args != r"-I C:\Users\edwar\Tools\gpu\scheduler\overnight_queue.py":
            e.append("Arguments must run the installed overnight_queue.py with -I")
        if "powershell" in (cmd + args).lower() or "pwsh" in (cmd + args).lower():
            e.append("no PowerShell")
    return e


def install() -> dict:
    errors = check()
    if errors:
        return {"ok": False, "errors": errors}
    SCRATCH.mkdir(parents=True, exist_ok=True)
    copy = SCRATCH / (NAME + ".utf16.xml")
    text = XML.read_text(encoding="utf-8")
    copy.write_text(text, encoding="utf-16")
    p = subprocess.run([SCHTASKS, "/Create", "/TN", NAME, "/XML", str(copy), "/F"], capture_output=True,
                       timeout=60, check=False)
    return {"ok": p.returncode == 0, "xml_sha256": hashlib.sha256(XML.read_bytes()).hexdigest(),
            "exit_code": p.returncode, "output": (p.stdout + p.stderr).decode("mbcs", "replace")[-500:]}


def status() -> dict:
    p = subprocess.run([SCHTASKS, "/Query", "/TN", NAME, "/V", "/FO", "LIST"], capture_output=True, timeout=60,
                       check=False)
    return {"ok": p.returncode == 0, "registered": p.returncode == 0,
            "output": (p.stdout + p.stderr).decode("mbcs", "replace")[-1500:]}


def selftest() -> dict:
    assert check() == [], check()
    text = XML.read_text(encoding="utf-8")
    assert check(text.replace("PT6H", "PT12H")), "accepted a 12 h limit"
    assert check(text.replace("LeastPrivilege", "HighestAvailable")), "accepted elevation"
    assert check(text.replace("<StartWhenAvailable>false", "<StartWhenAvailable>true")), "accepted catch-up"
    assert check(text.replace("T01:00:00", "T13:00:00")), "accepted a daytime trigger"
    assert check(text.replace(r"pythoncore-3.14-64\python.exe", r"pythoncore-3.14-64\\python.exe"))
    return {"offline": "pass", "xml_sha256": hashlib.sha256(XML.read_bytes()).hexdigest()}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "--selftest":
        out = selftest()
    elif cmd == "check":
        errs = check()
        out = {"ok": not errs, "errors": errs}
    elif cmd == "install":
        out = install()
    elif cmd == "status":
        out = status()
    else:
        print(__doc__)
        raise SystemExit(2)
    print(json.dumps(out))
    raise SystemExit(0 if out.get("ok", True) else 1)

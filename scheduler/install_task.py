"""Check or register the GPU-Overnight-Queue scheduled task from GPU-Overnight-Queue.xml.

  check    verify the XML (one daily 01:00 trigger, one python.exe action, 6 h limit, least privilege, no
           catch-up, no wake, IgnoreNew) and that the installed scripts match the repo copies (no side effects)
  install  check, then register from the same bytes: a UTF-16 copy in Tools/gpu/scratch and schtasks /Create /XML.
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
HERE = Path(__file__).resolve().parent
XML = HERE / "GPU-Overnight-Queue.xml"
RUNTIME = Path(r"C:\Users\edwar\Tools\gpu")
SCRATCH = RUNTIME / "scratch"
# Installed file -> reviewed repo copy. The task runs the installed files, so they must be the reviewed bytes.
INSTALLED = {RUNTIME / "scheduler" / "overnight_queue.py": HERE / "overnight_queue.py",
             RUNTIME / "jobs" / "runner.py": HERE.parent / "jobs" / "runner.py"}
NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
T = "{%s}" % NS["t"]
SCHTASKS = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "schtasks.exe")
ARGS = r"-I C:\Users\edwar\Tools\gpu\scheduler\overnight_queue.py"


def check_xml(text: str) -> list[str]:
    root = ET.fromstring(text.split("?>", 1)[1] if text.lstrip().startswith("<?xml") else text)
    e = []

    def val(path: str) -> str | None:
        node = root.find(path, NS)
        return node.text.strip() if node is not None and node.text else None

    if val("t:RegistrationInfo/t:URI") != "\\" + NAME:
        e.append("URI must be \\%s" % NAME)
    tnode = root.find("t:Triggers", NS)
    triggers = list(tnode) if tnode is not None else []
    if len(triggers) != 1 or triggers[0].tag != T + "CalendarTrigger":
        e.append("exactly one trigger, a CalendarTrigger")
    else:
        trig = triggers[0]
        if trig.find("t:Repetition", NS) is not None or trig.find("t:RandomDelay", NS) is not None:
            e.append("no Repetition or RandomDelay on the trigger")
        if not (val("t:Triggers/t:CalendarTrigger/t:StartBoundary") or "").endswith("T01:00:00"):
            e.append("trigger must start at 01:00 local")
        if val("t:Triggers/t:CalendarTrigger/t:ScheduleByDay/t:DaysInterval") != "1":
            e.append("trigger must be daily")
    if len(root.findall("t:Principals/t:Principal", NS)) != 1:
        e.append("exactly one Principal")
    for path, want in (("t:Principals/t:Principal/t:RunLevel", "LeastPrivilege"),
                       ("t:Settings/t:ExecutionTimeLimit", "PT6H"), ("t:Settings/t:StartWhenAvailable", "false"),
                       ("t:Settings/t:WakeToRun", "false"), ("t:Settings/t:MultipleInstancesPolicy", "IgnoreNew")):
        if val(path) != want:
            e.append("%s must be %s" % (path.split(":")[-1], want))
    anode = root.find("t:Actions", NS)
    actions = list(anode) if anode is not None else []
    if len(actions) != 1 or actions[0].tag != T + "Exec":
        e.append("exactly one action, an Exec (no ComHandler or others)")
    else:
        cmd = val("t:Actions/t:Exec/t:Command") or ""
        args = val("t:Actions/t:Exec/t:Arguments") or ""
        if not cmd.lower().endswith("python.exe") or "\\\\" in cmd:
            e.append("Command must be one python.exe path with single backslashes")
        if args != ARGS:
            e.append("Arguments must be exactly: " + ARGS)
        if "powershell" in (cmd + args).lower() or "pwsh" in (cmd + args).lower():
            e.append("no PowerShell")
    return e


def check_installed() -> tuple[list[str], dict[str, str]]:
    e, hashes = [], {}
    for live, repo in INSTALLED.items():
        if not live.is_file():
            e.append("not installed: %s" % live)
            continue
        hashes[str(live)] = hashlib.sha256(live.read_bytes()).hexdigest()
        if hashes[str(live)] != hashlib.sha256(repo.read_bytes()).hexdigest():
            e.append("installed %s differs from the reviewed repo copy" % live.name)
    return e, hashes


def install() -> dict:
    data = XML.read_bytes()  # one read: what is checked, copied and hashed are the same bytes
    text = data.decode("utf-8")
    errors = check_xml(text)
    inst_errors, hashes = check_installed()
    if errors or inst_errors:
        return {"ok": False, "registered": False, "errors": errors + inst_errors}
    SCRATCH.mkdir(parents=True, exist_ok=True)
    copy = SCRATCH / (NAME + ".utf16.xml")
    copy.write_text(text, encoding="utf-16")
    out = {"xml_sha256": hashlib.sha256(data).hexdigest(), "installed_sha256": hashes}
    try:
        p = subprocess.run([SCHTASKS, "/Create", "/TN", NAME, "/XML", str(copy), "/F"], capture_output=True,
                           timeout=60, check=False)
    except (subprocess.TimeoutExpired, OSError) as e:
        return {**out, "ok": False, "registered": "UNKNOWN", "error": "%s: %s" % (type(e).__name__, e),
                "next": "run `install_task.py status` before any retry"}
    return {**out, "ok": p.returncode == 0, "registered": p.returncode == 0, "exit_code": p.returncode,
            "output": (p.stdout + p.stderr).decode("mbcs", "replace")[-500:]}


def status() -> dict:
    try:
        p = subprocess.run([SCHTASKS, "/Query", "/TN", NAME, "/V", "/FO", "LIST"], capture_output=True, timeout=60,
                           check=False)
    except (subprocess.TimeoutExpired, OSError) as e:
        return {"ok": False, "registered": "UNKNOWN", "error": "%s: %s" % (type(e).__name__, e)}
    return {"ok": True, "registered": p.returncode == 0,
            "output": (p.stdout + p.stderr).decode("mbcs", "replace")[-1500:]}


def selftest() -> dict:
    text = XML.read_text(encoding="utf-8")
    assert check_xml(text) == [], check_xml(text)
    bad = {
        "12 h limit": text.replace("PT6H", "PT12H"),
        "elevation": text.replace("LeastPrivilege", "HighestAvailable"),
        "catch-up": text.replace("<StartWhenAvailable>false", "<StartWhenAvailable>true"),
        "daytime": text.replace("T01:00:00", "T13:00:00"),
        "doubled backslashes": text.replace(r"pythoncore-3.14-64\python.exe", r"pythoncore-3.14-64\\python.exe"),
        "parallel instances": text.replace("IgnoreNew", "Parallel"),
        "extra logon trigger": text.replace("</CalendarTrigger>", "</CalendarTrigger><LogonTrigger/>"),
        "repetition": text.replace("<Enabled>true</Enabled>\n      <ScheduleByDay>",
                                   "<Repetition><Interval>PT1H</Interval></Repetition>\n      <ScheduleByDay>"),
        "COM action": text.replace("</Exec>", "</Exec><ComHandler><ClassId>{0}</ClassId></ComHandler>"),
        "second principal": text.replace("</Principal>", "</Principal><Principal id='x'/>"),
    }
    for why, t in bad.items():
        assert check_xml(t), "accepted: " + why
    return {"offline": "pass", "xml_sha256": hashlib.sha256(XML.read_bytes()).hexdigest(),
            "installed_check": check_installed()[0] or "match"}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "--selftest":
        out = selftest()
    elif cmd == "check":
        errs = check_xml(XML.read_text(encoding="utf-8")) + check_installed()[0]
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

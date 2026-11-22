"""Ollama control for the BlackBox GPU (ACCEPT item 2).

Subcommands (each prints one JSON line, also on errors):
  status                               Ollama's user variables, processes, version, loaded models, GPU and RAM
  plan     --settings FILE             validate FILE and show which variables would change (no side effects)
  apply    --settings FILE             write the variables, restart Ollama, confirm them, load each model and check
                                       the limits under load; any failure or breach rolls back at once
  rollback                             restore the variables saved by the last apply and restart Ollama
  restart                              restart Ollama with the current variables (same gate as apply)
  pull     --model ID:TAG --digest SHA256 [--replace]   download one model by exact ID, then check its digest
  bench    --model ID [--levels 1,2,4] [--load-twice] [--num-ctx N]   measure throughput and load time
  --selftest

Owner rules (2026-11-21): measure before tuning; restart only when no Warden session runs and T7 is not loaded;
roll back at once if GPU memory passes 11.5 GB or the PC drops under 2 GB free, then notify; re-check T7 after an
8-bit KV cache or flash-attention change. Only the variables in ENV_MAP are ever written, in HKCU\\Environment only.
Standard library only; no PowerShell, no shell=True.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUNTIME = Path(r"C:\Users\edwar\Tools\gpu")
STATE_DIR = RUNTIME / "ollama"
PREVIOUS = STATE_DIR / "previous.json"  # variables as they were before the last apply
RECHECK = STATE_DIR / "t7_recheck_pending.json"  # present until T7 is re-checked after an FA/KV change
LOG = STATE_DIR / "control.log.jsonl"
ENDPOINT = "http://127.0.0.1:11434"
VRAM_LIMIT_GB, RAM_FLOOR_GB = 11.5, 2.0  # owner rollback rule, 2026-11-21
ENV_MAP = {  # settings key -> the one user variable it controls
    "keep_alive": "OLLAMA_KEEP_ALIVE",
    "parallel": "OLLAMA_NUM_PARALLEL",
    "flash_attention": "OLLAMA_FLASH_ATTENTION",
    "kv_cache_type": "OLLAMA_KV_CACHE_TYPE",
    "max_loaded_models": "OLLAMA_MAX_LOADED_MODELS",
    "cuda_cache_maxsize": "CUDA_CACHE_MAXSIZE",
}
T7_SENSITIVE = ("OLLAMA_FLASH_ATTENTION", "OLLAMA_KV_CACHE_TYPE")  # change T7's numbers: T7 must be re-checked
OLLAMA_PROCS = ("ollama app.exe", "ollama.exe")
LOCAL = Path(os.environ.get("LOCALAPPDATA", r"C:\Users\edwar\AppData\Local"))
APP = LOCAL / "Programs" / "Ollama" / "ollama app.exe"
SERVER_LOG = LOCAL / "Ollama" / "server.log"
BENCH_PROMPT = ("Explain in plain words how a graphics card's memory speed limits how fast a language model writes "
                "text, and what running several requests at once changes.")

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_KEEP = re.compile(r"^-?\d+(\.\d+)?(ms|s|m|h)?$")
_MODEL = re.compile(r"^[A-Za-z0-9._\-/]+(:[A-Za-z0-9._\-]+)?$")
_TAGGED = re.compile(r"^[A-Za-z0-9._\-/]+:[A-Za-z0-9._\-]+$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
TOP_KEYS = {"endpoint", "models", "restart_policy", "notes", *ENV_MAP}
MODEL_KEYS = {"id", "digest", "keep_alive", "num_ctx"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


# ---------- settings ----------

def _pair_rule(fa: Any, kv: Any) -> list[str]:
    """Ollama ignores a quantized KV cache without flash attention; values may be settings or variable strings."""
    fa_on = fa in (True, "1", "true", "True")
    if kv in ("q8_0", "q4_0") and not fa_on:
        return ["a quantized kv_cache_type needs flash_attention true (Ollama ignores it otherwise)"]
    return []


def validate(s: Any) -> list[str]:
    """Return every problem with a settings object; an empty list means it may be applied."""
    if not isinstance(s, dict):
        return ["settings must be a JSON object"]
    e = ["unknown key %r" % k for k in s if k not in TOP_KEYS]
    if s.get("endpoint") not in (ENDPOINT, "http://localhost:11434"):
        e.append("endpoint must be the local Ollama, %s" % ENDPOINT)
    models = s.get("models")
    if not isinstance(models, dict) or not models:
        e.append("models must be a non-empty object")
    else:
        for role, m in models.items():
            if not isinstance(m, dict):
                e.append("models.%s must be an object" % role)
                continue
            e += ["models.%s: unknown key %r" % (role, k) for k in m if k not in MODEL_KEYS]
            if not (isinstance(m.get("id"), str) and _MODEL.match(m["id"])):
                e.append("models.%s.id must be an exact model ID such as name:tag" % role)
            d = m.get("digest")
            if d is not None and not (isinstance(d, str) and _DIGEST.match(d)):
                e.append("models.%s.digest must be null or 64 lowercase hex characters" % role)
            if m.get("num_ctx") is not None and not (_int(m["num_ctx"]) and 512 <= m["num_ctx"] <= 131072):
                e.append("models.%s.num_ctx must be an integer from 512 to 131072" % role)
    ka = s.get("keep_alive")
    if ka is not None and not (_int(ka) or (isinstance(ka, str) and _KEEP.match(ka))):
        e.append("keep_alive must be seconds (integer) or a duration such as '30m'")
    if s.get("parallel") is not None and not (_int(s["parallel"]) and 1 <= s["parallel"] <= 8):
        e.append("parallel must be an integer from 1 to 8")
    fa = s.get("flash_attention")
    if fa is not None and not isinstance(fa, bool):
        e.append("flash_attention must be true, false or null")
    kv = s.get("kv_cache_type")
    if kv is not None and kv not in ("f16", "q8_0", "q4_0"):
        e.append("kv_cache_type must be f16, q8_0, q4_0 or null")
    e += _pair_rule(fa, kv)
    ml = s.get("max_loaded_models")
    if ml is not None and not (_int(ml) and 1 <= ml <= 3):
        e.append("max_loaded_models must be an integer from 1 to 3")
    cc = s.get("cuda_cache_maxsize")
    if cc is not None and not (_int(cc) and 0 <= cc <= 2**32):
        e.append("cuda_cache_maxsize must be bytes from 0 to 4294967296 (the CUDA maximum)")
    rp = s.get("restart_policy", {})
    if not isinstance(rp, dict) or any(k not in ("allow_restart", "notes") for k in rp) or \
            not isinstance(rp.get("allow_restart", False), bool):
        e.append("restart_policy may hold only allow_restart (true/false) and notes")
    return e


def env_values(s: dict[str, Any]) -> dict[str, str]:
    """The user variables a settings object asks for. A null setting is left alone, never cleared."""
    out = {}
    for key, var in ENV_MAP.items():
        v = s.get(key)
        if v is None:
            continue
        out[var] = ("1" if v else "0") if isinstance(v, bool) else str(v)
    return out


def load_settings(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8-sig")), hashlib.sha256(raw).hexdigest()


def _secs(v: str | None) -> float | None:
    """Seconds for an OLLAMA_KEEP_ALIVE value or a Go duration such as 1m0s; negative means forever (-1)."""
    if v is None or v == "":
        return None
    v = v.strip()
    if v.startswith("-"):
        return -1.0
    if re.fullmatch(r"\d+(\.\d+)?", v):
        return float(v)
    parts = re.findall(r"(\d+(?:\.\d+)?)(h|ms|us|µs|ns|m|s)", v)
    if not parts:
        return None
    unit = {"h": 3600, "m": 60, "s": 1, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9}
    return sum(float(n) * unit[u] for n, u in parts)


# ---------- HKCU\Environment (only the ENV_MAP names) ----------

def read_env() -> dict[str, str | None]:
    import winreg

    out: dict[str, str | None] = {}
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ) as k:
        for var in ENV_MAP.values():
            try:
                out[var] = str(winreg.QueryValueEx(k, var)[0])
            except FileNotFoundError:
                out[var] = None
    return out


def write_env(values: dict[str, str | None]) -> bool:
    """Set (or, for None, remove) user variables. Refuses any name outside ENV_MAP. Returns the broadcast result."""
    import winreg

    bad = [n for n in values if n not in ENV_MAP.values()]
    if bad:
        raise ValueError("refusing to write variables outside ENV_MAP: %s" % bad)
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE) as k:
        for name, val in values.items():
            if val is None:
                try:
                    winreg.DeleteValue(k, name)  # only to restore a variable that did not exist before apply
                except FileNotFoundError:
                    pass
            else:
                winreg.SetValueEx(k, name, 0, winreg.REG_SZ, val)
    return _broadcast_env()


def _broadcast_env() -> bool:
    """Tell running programs (Explorer included) that user variables changed: WM_SETTINGCHANGE 'Environment'."""
    import ctypes
    import ctypes.wintypes as W

    u = ctypes.WinDLL("user32", use_last_error=True)
    u.SendMessageTimeoutW.argtypes = [W.HWND, W.UINT, W.WPARAM, W.LPCWSTR, W.UINT, W.UINT,
                                      ctypes.POINTER(ctypes.c_size_t)]
    u.SendMessageTimeoutW.restype = ctypes.c_size_t
    res = ctypes.c_size_t()
    return bool(u.SendMessageTimeoutW(0xFFFF, 0x001A, 0, "Environment", 0x0002, 5000, ctypes.byref(res)))


# ---------- Ollama processes ----------

def _procs() -> list[tuple[int, str]]:
    import ctypes
    import ctypes.wintypes as W

    class PE(ctypes.Structure):
        _fields_ = [("dwSize", W.DWORD), ("cntUsage", W.DWORD), ("th32ProcessID", W.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", W.DWORD), ("cntThreads", W.DWORD),
                    ("th32ParentProcessID", W.DWORD), ("pcPriClassBase", ctypes.c_long), ("dwFlags", W.DWORD),
                    ("szExeFile", ctypes.c_wchar * 260)]

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateToolhelp32Snapshot.argtypes = [W.DWORD, W.DWORD]
    k.CreateToolhelp32Snapshot.restype = W.HANDLE
    k.Process32FirstW.argtypes = k.Process32NextW.argtypes = [W.HANDLE, ctypes.POINTER(PE)]
    k.CloseHandle.argtypes = [W.HANDLE]
    snap = k.CreateToolhelp32Snapshot(0x2, 0)
    if not snap or snap == W.HANDLE(-1).value:
        raise OSError("CreateToolhelp32Snapshot failed, winerror %d" % ctypes.get_last_error())
    found = []
    try:
        pe = PE()
        pe.dwSize = ctypes.sizeof(pe)
        more = k.Process32FirstW(snap, ctypes.byref(pe))
        while more:
            if pe.szExeFile.lower() in OLLAMA_PROCS:
                found.append((pe.th32ProcessID, pe.szExeFile))
            more = k.Process32NextW(snap, ctypes.byref(pe))
    finally:
        k.CloseHandle(snap)
    return found


def _terminate(pid: int) -> bool:
    """Request termination without waiting, so the tray and the server end together."""
    import ctypes
    import ctypes.wintypes as W

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
    k.OpenProcess.restype = W.HANDLE
    k.TerminateProcess.argtypes = [W.HANDLE, W.UINT]
    k.CloseHandle.argtypes = [W.HANDLE]
    h = k.OpenProcess(0x0001, False, pid)  # PROCESS_TERMINATE
    if not h:
        return False
    try:
        return bool(k.TerminateProcess(h, 0))
    finally:
        k.CloseHandle(h)


def _api(path: str, payload: Any = None, timeout: float = 10) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(ENDPOINT + path, data=data, headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def version() -> str | None:
    try:
        return _api("/api/version", timeout=3).get("version")
    except Exception:  # noqa: BLE001 - "not answering" is the answer
        return None


def _start(env_vals: dict[str, str | None], broadcast_ok: bool) -> str:
    """Start the Ollama tray app outside any Desk Code or lease job, so it outlives this process."""
    env = os.environ.copy()
    for k, v in env_vals.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    flags = 0x00000008 | 0x00000200 | 0x01000000  # DETACHED_PROCESS | NEW_PROCESS_GROUP | BREAKAWAY_FROM_JOB
    try:
        subprocess.Popen([str(APP)], env=env, creationflags=flags, close_fds=True, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "breakaway"
    except OSError:
        # The job forbids breakaway. Explorer starts the app outside the job, but with Explorer's own variables,
        # which match the registry only if the WM_SETTINGCHANGE broadcast reached it.
        if not broadcast_ok:
            raise RuntimeError("cannot break away from the job and the variable broadcast failed") from None
        explorer = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "explorer.exe")
        subprocess.run([explorer, str(APP)], timeout=30, check=False)
        return "explorer"


def parse_server_config(text: str) -> dict[str, str] | None:
    """The OLLAMA_* values from the last 'server config' line Ollama logs at start."""
    lines = [ln for ln in text.splitlines() if 'msg="server config"' in ln]
    if not lines:
        return None
    line = lines[-1]
    return {var: m.group(1) for var in ENV_MAP.values() if var.startswith("OLLAMA_")
            for m in [re.search(r"\b%s:(\S*)" % var, line)] if m}


def _effective_mismatch(want: dict[str, str | None], seen: dict[str, str]) -> list[str]:
    bad = []
    for var, val in want.items():
        if not var.startswith("OLLAMA_"):
            continue  # CUDA_CACHE_MAXSIZE is read by the driver; Ollama does not log it
        got = seen.get(var, "")
        if var == "OLLAMA_KEEP_ALIVE":
            same = val is None or (_secs(val) is not None and _secs(got) is not None
                                   and abs(_secs(val) - _secs(got)) < 1)
        elif var == "OLLAMA_FLASH_ATTENTION":
            same = val is None or (val in ("1", "true")) == (got == "true")
        else:
            same = val is None or str(val) == got
        if not same:
            bad.append("%s: wanted %r, Ollama started with %r" % (var, val, got))
    return bad


def _gate(lg) -> list[str]:
    why = []
    if lg.warden_session():
        why.append("a Warden session is running")
    names = lg.loaded_models()
    if any(n.startswith("<") for n in names) and _procs():
        why.append("Ollama is running but /api/ps did not answer, so a loaded T7 cannot be ruled out")
    if lg.T7_MODEL in names:
        why.append("T7 is loaded (the Warden may be mid-verdict)")
    return why


def _restart_locked(env_vals: dict[str, str | None], lg, broadcast_ok: bool) -> dict[str, Any]:
    """Stop every Ollama process at once and start the tray app with env_vals. Caller holds the lease."""
    if not APP.is_file():  # checked before anything is stopped
        return {"ok": False, "restarted": False, "error": "Ollama app not found at %s" % APP}
    why = _gate(lg)  # checked again immediately before the kill
    if why:
        return {"ok": False, "restarted": False, "refused": why}
    t_start = time.time()
    stopped = [{"pid": pid, "name": exe, "terminated": _terminate(pid)} for pid, exe in _procs()]
    deadline = time.monotonic() + 20
    while _procs() and time.monotonic() < deadline:
        time.sleep(0.25)
    left = _procs()
    if left:
        return {"ok": False, "restarted": False, "stopped": stopped, "still_running": left}
    try:
        how = _start(env_vals, broadcast_ok)
    except Exception as e:  # noqa: BLE001 - report so the caller rolls back
        return {"ok": False, "restarted": False, "stopped": stopped, "error": "start failed: %s: %s"
                % (type(e).__name__, e)}
    deadline, ver = time.monotonic() + 90, None
    while ver is None and time.monotonic() < deadline:
        time.sleep(1)
        ver = version()
    seen = None
    deadline = time.monotonic() + 20
    while seen is None and time.monotonic() < deadline:
        try:
            if SERVER_LOG.stat().st_mtime >= t_start - 1:
                seen = parse_server_config(SERVER_LOG.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
        if seen is None:
            time.sleep(0.5)
    mismatch = ["could not read Ollama's startup settings from %s" % SERVER_LOG] if seen is None \
        else _effective_mismatch(env_vals, seen)
    return {"ok": ver is not None and not mismatch, "restarted": True, "started_via": how, "stopped": stopped,
            "version": ver, "effective": seen, "mismatch": mismatch,
            "processes": [{"pid": p, "name": n} for p, n in _procs()]}


def _lg():
    sys.path.append(str(RUNTIME))  # append, never insert: the runtime copy must not shadow the standard library
    import local_gpu  # type: ignore

    return local_gpu


def _limits(lg, peak_vram: float | None = None, min_ram: float | None = None) -> dict[str, Any]:
    """Fail closed: an unreadable GPU counts as a breach."""
    g, m = lg.gpu(), lg.ram()
    vram = peak_vram if peak_vram is not None else g.get("vram_used_gb")
    ram = min(min_ram, m["ram_avail_gb"]) if min_ram is not None else m["ram_avail_gb"]
    breach = []
    if "error" in g and peak_vram is None:
        breach.append("GPU memory unreadable: %s" % g["error"])
    elif vram is not None and vram > VRAM_LIMIT_GB:
        breach.append("GPU memory %.2f GB > %.1f GB" % (vram, VRAM_LIMIT_GB))
    if ram < RAM_FLOOR_GB:
        breach.append("PC memory free %.2f GB < %.1f GB" % (ram, RAM_FLOOR_GB))
    return {"vram_used_gb": vram, "ram_avail_gb": ram, "breach": breach}


class _Sampler:
    """Samples peak VRAM and minimum free RAM every 0.5 s on a thread; counts failed GPU reads."""

    def __init__(self, lg):
        self.lg, self.vram, self.ram, self.fails, self.ok = lg, 0.0, 1e9, 0, 0
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.wait(0.5):
            g, m = self.lg.gpu(), self.lg.ram()
            if "error" in g:
                self.fails += 1
            else:
                self.ok += 1
                self.vram = max(self.vram, g["vram_used_gb"])
            self.ram = min(self.ram, m["ram_avail_gb"])

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.t.join(2)

    def breach(self) -> list[str]:
        out = []
        if self.ok == 0:
            out.append("GPU memory was never readable during the check")
        if self.vram > VRAM_LIMIT_GB:
            out.append("GPU memory peaked at %.2f GB > %.1f GB" % (self.vram, VRAM_LIMIT_GB))
        if self.ram < RAM_FLOOR_GB:
            out.append("PC memory fell to %.2f GB free < %.1f GB" % (self.ram, RAM_FLOOR_GB))
        return out


def _load_once(model: str, num_ctx: int | None) -> dict[str, Any]:
    opts = {"num_predict": 1, "temperature": 0}
    if num_ctx:
        opts["num_ctx"] = num_ctx
    try:
        r = _api("/api/generate", {"model": model, "prompt": "ok", "stream": False, "keep_alive": "30s",
                                   "options": opts}, timeout=600)
        return {"model": model, "loaded": True, "load_s": round(r.get("load_duration", 0) / 1e9, 2)}
    except urllib.error.HTTPError:  # embedding-only models do not generate
        _api("/api/embed", {"model": model, "input": "ok", "keep_alive": "30s"}, timeout=600)
        return {"model": model, "loaded": True, "embed": True}


def _load_check(lg, settings: dict[str, Any]) -> dict[str, Any]:
    """Load each configured model under the new settings while sampling memory (KV for every parallel slot is
    allocated at load time, so one short request per model shows the peak)."""
    loads = []
    with _Sampler(lg) as s:
        for m in settings.get("models", {}).values():
            if lg.warden_session():
                return {"ok": False, "loads": loads, "breach": ["a Warden session started during the check"]}
            loads.append(_load_once(m["id"], m.get("num_ctx")))
        time.sleep(1.5)  # one more sample after the last load
    breach = s.breach()
    return {"ok": not breach, "loads": loads, "peak_vram_gb": round(s.vram, 2), "min_ram_gb": round(s.ram, 2),
            "breach": breach}


def _log(entry: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": _now(), **entry}) + "\n")


def _read_previous() -> dict[str, Any] | None:
    try:
        return json.loads(PREVIOUS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _mark_recheck(changed: list[str], why: str) -> None:
    if any(v in T7_SENSITIVE for v in changed):
        RECHECK.write_text(json.dumps({"since": _now(), "why": why, "changed": changed}), encoding="utf-8")


# ---------- subcommands ----------

def cmd_status() -> dict[str, Any]:
    lg = _lg()
    return {"ok": True, "env": read_env(), "processes": [{"pid": p, "name": n} for p, n in _procs()],
            "version": version(), "loaded": lg.loaded_models(), "warden_session": lg.warden_session(),
            "gpu": lg.gpu(), "ram": lg.ram(), "previous": _read_previous(),
            "t7_recheck_pending": RECHECK.is_file()}


def cmd_plan(path: Path) -> dict[str, Any]:
    s, sha = load_settings(path)
    errors = validate(s)
    want, have = env_values(s), read_env()
    merged = {**have, **want}  # the pair rule must hold for the variables Ollama will actually start with
    errors += [e + " (after merging with the current variables)" for e in
               _pair_rule(merged.get("OLLAMA_FLASH_ATTENTION"), merged.get("OLLAMA_KV_CACHE_TYPE"))
               if e not in errors]
    if errors:
        return {"ok": False, "settings_sha256": sha, "errors": errors}
    changes = {k: {"from": have.get(k), "to": v} for k, v in want.items() if have.get(k) != v}
    return {"ok": True, "settings_sha256": sha, "changes": changes,
            "allow_restart": s.get("restart_policy", {}).get("allow_restart", False),
            "t7_recheck_required": any(k in changes for k in T7_SENSITIVE)}


def cmd_apply(path: Path, provider: str) -> dict[str, Any]:
    plan = cmd_plan(path)
    if not plan["ok"]:
        return plan
    if not plan["allow_restart"]:
        return {**plan, "ok": False, "error": "restart_policy.allow_restart is false in this settings file"}
    if not plan["changes"]:
        return {**plan, "applied": False, "note": "no variable changes; Ollama not restarted"}
    prev = _read_previous()
    if prev and not prev.get("confirmed"):
        return {**plan, "ok": False, "error": "the last apply was never confirmed; run rollback first"}
    if not APP.is_file():
        return {**plan, "ok": False, "error": "Ollama app not found at %s; nothing changed" % APP}
    settings, _ = load_settings(path)
    lg = _lg()
    try:
        lease = lg.lease("ollama-control apply", provider=provider)
        lease.__enter__()
    except Exception as e:  # noqa: BLE001 - Busy/Refused: nothing has been written
        return {**plan, "ok": False, "applied": False, "refused": ["%s: %s" % (type(e).__name__, e)]}
    out: dict[str, Any] = {**plan}
    try:
        why = _gate(lg)
        if why:  # before touching the registry
            return {**out, "ok": False, "applied": False, "refused": why}
        have = read_env()
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        PREVIOUS.write_text(json.dumps({"saved_at": _now(), "values": have, "by_settings": plan["settings_sha256"],
                                        "confirmed": False}), encoding="utf-8")
        want = {k: v["to"] for k, v in plan["changes"].items()}
        try:
            broadcast = write_env(want)
            out.update(applied=True, broadcast=broadcast)
            r = _restart_locked({**have, **want}, lg, broadcast)
            out["restart"] = r
            if not r.get("ok"):
                raise RuntimeError("restart failed: %s" % (r.get("error") or r.get("refused") or r.get("mismatch")))
            check = _load_check(lg, settings)
            out["load_check"] = check
            if not check["ok"]:
                raise RuntimeError("limit breach under load: " + "; ".join(check["breach"]))
            PREVIOUS.write_text(json.dumps({**_read_previous(), "confirmed": True, "confirmed_at": _now()}),
                                encoding="utf-8")
            _mark_recheck(list(want), "apply " + plan["settings_sha256"])
            out["ok"] = True
        except Exception as e:  # noqa: BLE001 - any failure after the write rolls back at once
            out.update(ok=False, error="%s: %s" % (type(e).__name__, e),
                       rollback=_rollback_locked(lg, provider, reason=str(e)))
    finally:
        lease.__exit__(None, None, None)
        _log({"action": "apply", "provider": provider, "result": out})
    return out


def _rollback_locked(lg, provider: str, reason: str) -> dict[str, Any]:
    prev = _read_previous()
    if not prev:
        return {"ok": False, "error": "no saved previous variables at %s" % PREVIOUS}
    if lg.warden_session():  # never kill Ollama mid-verdict; the breach is reported for the next try
        return {"ok": False, "reason": reason, "refused": ["a Warden session is running; retry rollback after it"],
                "notify": "post this to Agent Room / #director-admin-ops now"}
    if lg.T7_MODEL in lg.loaded_models():
        lg._unload_all()  # no Warden session holds it, so it may be unloaded for the rollback
    values = {k: v for k, v in prev["values"].items() if k in ENV_MAP.values()}
    current = read_env()
    broadcast = write_env(values)
    r = _restart_locked(values, lg, broadcast)
    if r.get("ok"):
        done = STATE_DIR / ("previous.rolledback-%s.json" % datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        PREVIOUS.replace(done)
    _mark_recheck([k for k in values if values[k] != current.get(k)], "rollback: " + reason)
    return {"ok": bool(r.get("ok")), "reason": reason, "restored": values, "broadcast": broadcast, "restart": r,
            "t7_recheck_required": any(values.get(k) != current.get(k) for k in T7_SENSITIVE),
            "notify": "post this result to Agent Room / #director-admin-ops (owner rule: roll back, then notify)"}


def cmd_rollback(provider: str, reason: str = "requested") -> dict[str, Any]:
    lg = _lg()
    with lg.lease("ollama-control rollback", provider=provider):
        out = _rollback_locked(lg, provider, reason)
    _log({"action": "rollback", "provider": provider, "result": out})
    return out


def cmd_restart(provider: str) -> dict[str, Any]:
    lg = _lg()
    with lg.lease("ollama-control restart", provider=provider):
        out = _restart_locked(read_env(), lg, True)
    _log({"action": "restart", "provider": provider, "result": out})
    return out


def cmd_pull(model: str, digest: str, replace: bool) -> dict[str, Any]:
    if not _TAGGED.match(model) or not _DIGEST.match(digest):
        return {"ok": False, "error": "model must be an exact ID with a tag (name:tag) and digest 64 hex characters"}
    tags = {m.get("name"): m.get("digest") for m in _api("/api/tags").get("models") or []}
    if tags.get(model) == digest:
        return {"ok": True, "model": model, "digest": digest, "note": "already installed with this digest"}
    if model in tags and not replace:
        return {"ok": False, "model": model, "installed_digest": tags[model], "expected": digest,
                "error": "a different build of this tag is installed; pulling would replace it (use --replace)"}
    req = urllib.request.Request(ENDPOINT + "/api/pull", data=json.dumps({"model": model, "stream": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    last = None
    with _OPENER.open(req, timeout=600) as r:  # each progress line resets the idle timeout
        for line in r:
            if line.strip():
                last = json.loads(line)
                if last.get("error"):
                    break
    tags = {m.get("name"): m.get("digest") for m in _api("/api/tags").get("models") or []}
    got = tags.get(model)
    out = {"ok": got == digest, "model": model, "pull_status": (last or {}).get("status"),
           "pull_error": (last or {}).get("error"), "digest": got, "expected": digest}
    if got != digest:
        out["error"] = "digest mismatch: do not use this model until a reviewed change pins the digest it has"
    _log({"action": "pull", "result": out})
    return out


def _cache_bytes() -> int | None:
    base = Path(os.environ.get("APPDATA", "")) / "NVIDIA" / "ComputeCache"
    if not base.is_dir():
        return None
    total = 0
    for f in base.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def _generate(model: str, num_ctx: int, num_predict: int, keep_alive: Any = "5m") -> dict[str, Any]:
    t = time.monotonic()
    r = _api("/api/generate", {"model": model, "prompt": BENCH_PROMPT, "stream": False, "keep_alive": keep_alive,
                               "options": {"num_predict": num_predict, "num_ctx": num_ctx, "temperature": 0,
                                           "seed": 1}}, timeout=600)
    return {"wall_s": round(time.monotonic() - t, 3), "eval_count": r.get("eval_count", 0),
            "eval_s": round(r.get("eval_duration", 0) / 1e9, 3), "load_s": round(r.get("load_duration", 0) / 1e9, 3),
            "prompt_tokens": r.get("prompt_eval_count", 0)}


def _unload(model: str, lg) -> None:
    _api("/api/generate", {"model": model, "keep_alive": 0}, timeout=120)
    deadline = time.monotonic() + 30
    while model in lg.loaded_models() and time.monotonic() < deadline:
        time.sleep(0.5)


def cmd_bench(model: str, levels: list[int], load_twice: bool, num_ctx: int, num_predict: int,
              provider: str) -> dict[str, Any]:
    if not _MODEL.match(model):
        return {"ok": False, "error": "model must be an exact ID"}
    lg = _lg()
    # shortcut: a light lease keeps heavy jobs off the card but not light embed calls, which can evict the model
    # (OLLAMA_MAX_LOADED_MODELS=1); a load_s spike mid-level means rerun. Upgrade when the bridge joins the lock.
    with lg.lease("ollama-control bench " + model, provider=provider):
        why = []
        if lg.warden_session():
            why.append("a Warden session is running")
        if model != lg.T7_MODEL and lg.T7_MODEL in lg.loaded_models():
            why.append("T7 is loaded and benchmarking another model would evict it")
        if why:
            return {"ok": False, "refused": why}
        out: dict[str, Any] = {"ok": True, "model": model, "num_ctx": num_ctx, "num_predict": num_predict,
                               "env": read_env(), "levels": []}
        with _Sampler(lg) as s:
            if load_twice:
                loads = []
                for _ in range(2):
                    if lg.warden_session():
                        out.update(ok=False, refused=["a Warden session started"])
                        break
                    _unload(model, lg)
                    before = _cache_bytes()
                    r = _generate(model, num_ctx, 1)
                    loads.append({"load_s": r["load_s"], "wall_s": r["wall_s"], "cache_bytes_before": before,
                                  "cache_bytes_after": _cache_bytes()})
                out["load_twice"] = loads
            if out["ok"]:
                _generate(model, num_ctx, 1)  # warm: the levels measure generation, not loading
            for n in levels if out["ok"] else []:
                if lg.warden_session():
                    out.update(ok=False, refused=["a Warden session started"])
                    break
                results: list[dict[str, Any]] = [{} for _ in range(n)]

                def one(i):
                    try:
                        results[i] = _generate(model, num_ctx, num_predict)
                    except Exception as e:  # noqa: BLE001 - recorded, not lost
                        results[i] = {"error": "%s: %s" % (type(e).__name__, e)}

                t = time.monotonic()
                threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
                for x in threads:
                    x.start()
                for x in threads:
                    x.join(900)
                wall = time.monotonic() - t
                errors = [r["error"] for r in results if "error" in r] + \
                    ["request still running after 900 s" for x in threads if x.is_alive()]
                tokens = sum(r.get("eval_count", 0) for r in results)
                level = {"parallel": n, "wall_s": round(wall, 2), "tokens": tokens,
                         "total_tokens_per_s": round(tokens / wall, 1) if wall and not errors else None,
                         "per_request_tokens_per_s": [round(r["eval_count"] / r["eval_s"], 1) if r.get("eval_s")
                                                      else None for r in results],
                         "max_load_s": max((r.get("load_s", 0) for r in results), default=0), "errors": errors}
                out["levels"].append(level)
                if errors:
                    out["ok"] = False
                    break
                if s.breach():
                    break
        out["peak"] = {"vram_used_gb": round(s.vram, 2), "ram_min_avail_gb": round(s.ram, 2),
                       "gpu_read_failures": s.fails}
        out["breach"] = s.breach()
        if out["breach"]:
            out["ok"] = False
            try:
                _unload(model, lg)
            except Exception:  # noqa: BLE001
                pass
            prev = _read_previous()
            if prev:  # the active variables came from an apply: owner rule, roll back at once
                out["rollback"] = _rollback_locked(lg, provider, reason="bench breach: " + "; ".join(out["breach"]))
            else:
                out["action"] = "no apply to roll back; these are the baseline variables. Notify peers."
    _log({"action": "bench", "provider": provider, "result": out})
    return out


# ---------- self-test ----------

def selftest() -> dict[str, Any]:
    example = json.loads((Path(__file__).parent / "settings.example.json").read_text(encoding="utf-8"))
    assert validate(example) == [], validate(example)
    base = {"endpoint": ENDPOINT, "models": {"t7": {"id": "local-warden:t7-ckpt153-q8"}}}
    assert validate(base) == []
    bad_cases = [
        {**base, "endpoint": "http://10.0.0.5:11434"},
        {**base, "kv_cache_type": "q8_0"},  # quantized KV without flash attention
        {**base, "kv_cache_type": "q8_0", "flash_attention": "yes"},
        {**base, "parallel": 0}, {**base, "parallel": True}, {**base, "max_loaded_models": 9},
        {**base, "cuda_cache_maxsize": 2**33}, {**base, "keep_alive": "forever"},
        {**base, "OLLAMA_ORIGINS": "*"}, {**base, "restart_policy": {"allow_restart": "yes"}},
        {**base, "models": {"t7": {"id": "bad id; rm"}}}, {**base, "models": {"t7": {"id": "x", "digest": "abc"}}},
        {**base, "models": {}}, [],
    ]
    for case in bad_cases:
        assert validate(case), case
    assert validate({**base, "kv_cache_type": "q8_0", "flash_attention": True}) == []
    assert _pair_rule("0", "q8_0") and not _pair_rule("1", "q8_0") and not _pair_rule(None, None)
    want = env_values({**base, "keep_alive": "30m", "parallel": 2, "flash_attention": True, "kv_cache_type": None})
    assert want == {"OLLAMA_KEEP_ALIVE": "30m", "OLLAMA_NUM_PARALLEL": "2", "OLLAMA_FLASH_ATTENTION": "1"}, want
    assert _secs("1m0s") == 60 and _secs("60s") == 60 and _secs("30m") == 1800 and _secs("300") == 300
    assert _secs("1h0m0s") == 3600 and _secs("-1ns") == -1 and _secs("") is None
    line = ('time=x level=INFO msg="server config" env="map[OLLAMA_FLASH_ATTENTION:false OLLAMA_KEEP_ALIVE:1m0s '
            'OLLAMA_KV_CACHE_TYPE: OLLAMA_MAX_LOADED_MODELS:1 OLLAMA_NUM_PARALLEL:1 OLLAMA_ORIGINS:[a b]]"')
    seen = parse_server_config("other\n" + line)
    assert seen["OLLAMA_NUM_PARALLEL"] == "1" and seen["OLLAMA_KV_CACHE_TYPE"] == "", seen
    assert _effective_mismatch({"OLLAMA_KEEP_ALIVE": "60s", "OLLAMA_NUM_PARALLEL": "1",
                                "OLLAMA_FLASH_ATTENTION": "0", "CUDA_CACHE_MAXSIZE": "9"}, seen) == []
    assert _effective_mismatch({"OLLAMA_NUM_PARALLEL": "2"}, seen) and parse_server_config("nothing") is None

    class FakeLg:
        T7_MODEL = "t7"

        def __init__(self, warden=False, models=(), gpu_err=False):
            self.w, self.m, self.e = warden, list(models), gpu_err

        def warden_session(self):
            return self.w

        def loaded_models(self):
            return self.m

        def gpu(self):
            return {"error": "NVML down"} if self.e else {"vram_used_gb": 3.0}

        def ram(self):
            return {"ram_avail_gb": 20.0}

    assert _limits(FakeLg(gpu_err=True))["breach"]  # fail closed
    assert _limits(FakeLg())["breach"] == []
    assert _gate(FakeLg(warden=True)) and _gate(FakeLg(models=["t7"])) and _gate(FakeLg()) == []
    try:
        write_env({"PATH": "x"})
        raise AssertionError("write_env accepted a name outside ENV_MAP")
    except ValueError:
        pass
    report: dict[str, Any] = {"offline": "pass"}
    if sys.platform == "win32":
        st = cmd_status()
        assert st["ok"] and set(st["env"]) == set(ENV_MAP.values()), st
        plan = cmd_plan(Path(__file__).parent / "settings.example.json")
        assert plan["ok"] and plan["allow_restart"] is False, plan
        live_seen = parse_server_config(SERVER_LOG.read_text(encoding="utf-8", errors="replace")) \
            if SERVER_LOG.is_file() else None
        assert live_seen is not None and _effective_mismatch(st["env"], live_seen) == [], (live_seen, st["env"])
        report.update({"live": "pass", "version": st["version"], "ollama_processes": len(st["processes"]),
                       "example_plan_changes": sorted(plan["changes"]), "running_matches_registry": True})
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("status")
    for name in ("plan", "apply"):
        p = sub.add_parser(name)
        p.add_argument("--settings", type=Path, default=STATE_DIR / "settings.json")
    sub.add_parser("rollback")
    sub.add_parser("restart", help="restart Ollama with the current user variables (same gate as apply)")
    p = sub.add_parser("pull")
    p.add_argument("--model", required=True)
    p.add_argument("--digest", required=True)
    p.add_argument("--replace", action="store_true", help="allow replacing an installed build of the same tag")
    p = sub.add_parser("bench")
    p.add_argument("--model", required=True)
    p.add_argument("--levels", default="1,2,4")
    p.add_argument("--load-twice", action="store_true")
    p.add_argument("--num-ctx", type=int, default=8192, help="match the Warden's 8192, or Ollama reloads T7")
    p.add_argument("--num-predict", type=int, default=256)
    for p in sub.choices.values():
        p.add_argument("--provider", default="unknown")
    a = ap.parse_args(argv)
    try:
        if a.selftest:
            out = selftest()
        elif a.cmd == "status":
            out = cmd_status()
        elif a.cmd == "plan":
            out = cmd_plan(a.settings)
        elif a.cmd == "apply":
            out = cmd_apply(a.settings, a.provider)
        elif a.cmd == "rollback":
            out = cmd_rollback(a.provider)
        elif a.cmd == "restart":
            out = cmd_restart(a.provider)
        elif a.cmd == "pull":
            out = cmd_pull(a.model, a.digest, a.replace)
        elif a.cmd == "bench":
            levels = [int(x) for x in a.levels.split(",") if x.strip()]
            if not levels or any(not 1 <= n <= 8 for n in levels):
                raise ValueError("--levels must be integers from 1 to 8")
            out = cmd_bench(a.model, levels, a.load_twice, max(512, min(a.num_ctx, 131072)),
                            max(1, min(a.num_predict, 2048)), a.provider)
        else:
            ap.print_help()
            return 2
    except AssertionError:
        raise
    except Exception as e:  # noqa: BLE001 - one JSON line, always
        out = {"ok": False, "cmd": a.cmd, "error": "%s: %s" % (type(e).__name__, e)}
        _log({"action": a.cmd or "selftest", "result": out})
    print(json.dumps(out))
    return 0 if out.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())

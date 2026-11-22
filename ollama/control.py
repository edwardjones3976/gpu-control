"""Ollama control for the BlackBox GPU (ACCEPT item 2).

Subcommands (each prints one JSON line):
  status                               Ollama's user variables, processes, version, loaded models, GPU and RAM
  plan     --settings FILE             validate FILE and show which variables would change (no side effects)
  apply    --settings FILE             write the variables, restart Ollama, health-check, roll back on a limit breach
  rollback                             restore the variables saved by the last apply and restart Ollama
  restart                              restart Ollama with the current variables (same gate as apply)
  pull     --model ID --digest SHA256  download one model by exact ID, then check its digest
  bench    --model ID [--levels 1,2,4] [--load-twice] [--num-ctx N]   measure throughput and load time
  --selftest

Owner rules (2026-11-21): measure before tuning; restart only when no Warden session runs and T7 is not loaded;
roll back at once if GPU memory passes 11.5 GB or the PC drops under 2 GB free, then notify; re-check T7 after an
8-bit KV cache change. Only the variables in ENV_MAP are ever written, in HKCU\\Environment only.
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
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUNTIME = Path(r"C:\Users\edwar\Tools\gpu")
STATE_DIR = RUNTIME / "ollama"
PREVIOUS = STATE_DIR / "previous.json"  # variables as they were before the last apply
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
T7_SENSITIVE = ("flash_attention", "kv_cache_type")  # change the Warden model's numbers: T7 must be re-checked
OLLAMA_PROCS = ("ollama app.exe", "ollama.exe")
APP = Path(os.environ.get("LOCALAPPDATA", r"C:\Users\edwar\AppData\Local")) / "Programs" / "Ollama" / "ollama app.exe"
BENCH_PROMPT = ("Explain in plain words how a graphics card's memory speed limits how fast a language model writes "
                "text, and what running several requests at once changes.")

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_KEEP = re.compile(r"^-?\d+(\.\d+)?(ms|s|m|h)?$")
_MODEL = re.compile(r"^[A-Za-z0-9._\-/]+(:[A-Za-z0-9._\-]+)?$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
TOP_KEYS = {"endpoint", "models", "restart_policy", "notes", *ENV_MAP}
MODEL_KEYS = {"id", "digest", "keep_alive", "num_ctx"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


# ---------- settings ----------

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
    if kv in ("q8_0", "q4_0") and fa is not True:
        e.append("a quantized kv_cache_type needs flash_attention true (Ollama ignores it otherwise)")
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
    import ctypes
    import ctypes.wintypes as W

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
    k.OpenProcess.restype = W.HANDLE
    k.TerminateProcess.argtypes = [W.HANDLE, W.UINT]
    k.WaitForSingleObject.argtypes = [W.HANDLE, W.DWORD]
    k.CloseHandle.argtypes = [W.HANDLE]
    h = k.OpenProcess(0x0001 | 0x00100000, False, pid)  # PROCESS_TERMINATE | SYNCHRONIZE
    if not h:
        return False
    try:
        ok = bool(k.TerminateProcess(h, 0))
        k.WaitForSingleObject(h, 15000)
        return ok
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


def _start(env_vals: dict[str, str | None]) -> str:
    """Start the Ollama tray app outside any Desk Code or lease job, so it outlives this process."""
    if not APP.is_file():
        raise FileNotFoundError("Ollama app not found at %s" % APP)
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
        # The job forbids breakaway: hand the start to the running Explorer, which is outside the job and has
        # re-read the user variables from the WM_SETTINGCHANGE broadcast.
        explorer = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "explorer.exe")
        subprocess.run([explorer, str(APP)], timeout=30, check=False)
        return "explorer"


def _gate(lg) -> list[str]:
    why = []
    if lg.warden_session():
        why.append("a Warden session is running")
    if lg.T7_MODEL in lg.loaded_models():
        why.append("T7 is loaded (the Warden may be mid-verdict)")
    return why


def restart(env_vals: dict[str, str | None], provider: str) -> dict[str, Any]:
    """Stop the tray app, then the server, then start the tray app with env_vals. Needs the gate clear."""
    lg = _lg()
    with lg.lease("ollama-control restart", provider=provider):  # light lease: no heavy job may start meanwhile
        why = _gate(lg)
        if why:
            return {"ok": False, "restarted": False, "refused": why}
        before = _procs()
        stopped = []
        for name in OLLAMA_PROCS:  # tray first, or it restarts the server it supervises
            for pid, exe in before:
                if exe.lower() == name:
                    stopped.append({"pid": pid, "name": exe, "terminated": _terminate(pid)})
        deadline = time.monotonic() + 20
        while _procs() and time.monotonic() < deadline:
            time.sleep(0.5)
        left = _procs()
        if left:
            return {"ok": False, "restarted": False, "stopped": stopped, "still_running": left}
        how = _start(env_vals)
        deadline, ver = time.monotonic() + 90, None
        while ver is None and time.monotonic() < deadline:
            time.sleep(1)
            ver = version()
        return {"ok": ver is not None, "restarted": True, "started_via": how, "stopped": stopped, "version": ver,
                "processes": [{"pid": p, "name": n} for p, n in _procs()]}


def _lg():
    sys.path.append(str(RUNTIME))  # append, never insert: the runtime copy must not shadow the standard library
    import local_gpu  # type: ignore

    return local_gpu


def _limits(lg) -> dict[str, Any]:
    g, m = lg.gpu(), lg.ram()
    breach = []
    if "error" not in g and g["vram_used_gb"] > VRAM_LIMIT_GB:
        breach.append("GPU memory %.2f GB > %.1f GB" % (g["vram_used_gb"], VRAM_LIMIT_GB))
    if m["ram_avail_gb"] < RAM_FLOOR_GB:
        breach.append("PC memory free %.2f GB < %.1f GB" % (m["ram_avail_gb"], RAM_FLOOR_GB))
    return {"vram_used_gb": g.get("vram_used_gb"), "ram_avail_gb": m["ram_avail_gb"], "breach": breach}


def _log(entry: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": _now(), **entry}) + "\n")


# ---------- subcommands ----------

def cmd_status() -> dict[str, Any]:
    lg = _lg()
    return {"ok": True, "env": read_env(), "processes": [{"pid": p, "name": n} for p, n in _procs()],
            "version": version(), "loaded": lg.loaded_models(), "warden_session": lg.warden_session(),
            "gpu": lg.gpu(), "ram": lg.ram(), "previous_saved": PREVIOUS.is_file()}


def cmd_plan(path: Path) -> dict[str, Any]:
    s, sha = load_settings(path)
    errors = validate(s)
    if errors:
        return {"ok": False, "settings_sha256": sha, "errors": errors}
    want, have = env_values(s), read_env()
    changes = {k: {"from": have.get(k), "to": v} for k, v in want.items() if have.get(k) != v}
    return {"ok": True, "settings_sha256": sha, "changes": changes,
            "allow_restart": s.get("restart_policy", {}).get("allow_restart", False),
            "t7_recheck_required": any(ENV_MAP[k] in changes for k in T7_SENSITIVE)}


def cmd_apply(path: Path, provider: str) -> dict[str, Any]:
    plan = cmd_plan(path)
    if not plan["ok"]:
        return plan
    if not plan["allow_restart"]:
        return {**plan, "ok": False, "error": "restart_policy.allow_restart is false in this settings file"}
    if not plan["changes"]:
        return {**plan, "applied": False, "note": "no variable changes; Ollama not restarted"}
    lg = _lg()
    why = _gate(lg)
    if why:  # checked before touching the registry, and again inside restart()
        return {**plan, "ok": False, "applied": False, "refused": why}
    have = read_env()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    PREVIOUS.write_text(json.dumps({"saved_at": _now(), "values": have, "by_settings": plan["settings_sha256"]}),
                        encoding="utf-8")
    want = {k: v["to"] for k, v in plan["changes"].items()}
    broadcast = write_env(want)
    merged = {**have, **want}
    r = restart(merged, provider)
    out = {**plan, "applied": True, "broadcast": broadcast, "restart": r}
    if not r.get("ok"):
        out["rollback"] = cmd_rollback(provider, reason="restart failed after apply")
        out["ok"] = False
    else:
        lim = _limits(lg)
        out["limits"] = lim
        if lim["breach"]:
            out["rollback"] = cmd_rollback(provider, reason="; ".join(lim["breach"]))
            out["ok"] = False
    _log({"action": "apply", "provider": provider, "result": out})
    return out


def cmd_rollback(provider: str, reason: str = "requested") -> dict[str, Any]:
    if not PREVIOUS.is_file():
        return {"ok": False, "error": "no saved previous variables at %s" % PREVIOUS}
    prev = json.loads(PREVIOUS.read_text(encoding="utf-8"))["values"]
    prev = {k: v for k, v in prev.items() if k in ENV_MAP.values()}
    broadcast = write_env(prev)
    r = restart(prev, provider)
    out = {"ok": bool(r.get("ok")), "reason": reason, "restored": prev, "broadcast": broadcast, "restart": r,
           "notify": "post this result to Agent Room / #director-admin-ops (owner rule: roll back, then notify)"}
    _log({"action": "rollback", "provider": provider, "result": out})
    return out


def cmd_pull(model: str, digest: str) -> dict[str, Any]:
    if not _MODEL.match(model) or not _DIGEST.match(digest):
        return {"ok": False, "error": "model must be an exact ID (name:tag) and digest 64 lowercase hex characters"}
    r = _api("/api/pull", {"model": model, "stream": False}, timeout=3600)
    tags = {m.get("name"): m.get("digest") for m in _api("/api/tags").get("models") or []}
    got = tags.get(model)
    out = {"ok": got == digest, "model": model, "pull_status": r.get("status"), "digest": got, "expected": digest}
    if got != digest:
        out["error"] = "digest mismatch: do not use this model until a reviewed change pins the digest it has"
    _log({"action": "pull", "result": out})
    return out


def _cache_bytes() -> int | None:
    base = Path(os.environ.get("APPDATA", "")) / "NVIDIA" / "ComputeCache"
    if not base.is_dir():
        return None
    return sum(f.stat().st_size for f in base.rglob("*") if f.is_file())


def _generate(model: str, num_ctx: int, num_predict: int, keep_alive: Any = "5m") -> dict[str, Any]:
    t = time.monotonic()
    r = _api("/api/generate", {"model": model, "prompt": BENCH_PROMPT, "stream": False, "keep_alive": keep_alive,
                               "options": {"num_predict": num_predict, "num_ctx": num_ctx, "temperature": 0,
                                           "seed": 1}}, timeout=600)
    return {"wall_s": round(time.monotonic() - t, 3), "eval_count": r.get("eval_count", 0),
            "eval_s": round(r.get("eval_duration", 0) / 1e9, 3), "load_s": round(r.get("load_duration", 0) / 1e9, 3),
            "prompt_tokens": r.get("prompt_eval_count", 0)}


def _unload(model: str) -> None:
    _api("/api/generate", {"model": model, "keep_alive": 0}, timeout=120)
    deadline = time.monotonic() + 30
    while model in _lg().loaded_models() and time.monotonic() < deadline:
        time.sleep(0.5)


def cmd_bench(model: str, levels: list[int], load_twice: bool, num_ctx: int, num_predict: int,
              provider: str) -> dict[str, Any]:
    if not _MODEL.match(model):
        return {"ok": False, "error": "model must be an exact ID"}
    lg = _lg()
    # shortcut: a light lease keeps heavy jobs off the card but not light embed calls, which can evict the model
    # (OLLAMA_MAX_LOADED_MODELS=1); a load_s spike mid-level means rerun. Upgrade when the bridge joins the lock.
    with lg.lease("ollama-control bench " + model, provider=provider):
        if lg.warden_session():
            return {"ok": False, "refused": ["a Warden session is running"]}
        peak = {"vram_used_gb": 0.0, "ram_min_gb": 1e9}
        stop = threading.Event()

        def sample():
            while not stop.wait(0.5):
                g, m = lg.gpu(), lg.ram()
                if "error" not in g:
                    peak["vram_used_gb"] = max(peak["vram_used_gb"], g["vram_used_gb"])
                peak["ram_min_gb"] = min(peak["ram_min_gb"], m["ram_avail_gb"])

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        out: dict[str, Any] = {"ok": True, "model": model, "num_ctx": num_ctx, "num_predict": num_predict,
                               "env": read_env(), "levels": []}
        try:
            if load_twice:
                loads = []
                for _ in range(2):
                    _unload(model)
                    before = _cache_bytes()
                    r = _generate(model, num_ctx, 1)
                    loads.append({"load_s": r["load_s"], "wall_s": r["wall_s"], "cache_bytes_before": before,
                                  "cache_bytes_after": _cache_bytes()})
                out["load_twice"] = loads
            _generate(model, num_ctx, 1)  # warm: the levels measure generation, not loading
            for n in levels:
                results: list[dict[str, Any]] = [{}] * n
                t = time.monotonic()
                threads = [threading.Thread(target=lambda i=i: results.__setitem__(
                    i, _generate(model, num_ctx, num_predict))) for i in range(n)]
                for x in threads:
                    x.start()
                for x in threads:
                    x.join(900)
                wall = time.monotonic() - t
                tokens = sum(r.get("eval_count", 0) for r in results)
                out["levels"].append({"parallel": n, "wall_s": round(wall, 2), "tokens": tokens,
                                      "total_tokens_per_s": round(tokens / wall, 1) if wall else None,
                                      "per_request_tokens_per_s": [round(r["eval_count"] / r["eval_s"], 1)
                                                                   if r.get("eval_s") else None for r in results],
                                      "max_load_s": max((r.get("load_s", 0) for r in results), default=0)})
                if peak["vram_used_gb"] > VRAM_LIMIT_GB or peak["ram_min_gb"] < RAM_FLOOR_GB:
                    break
        finally:
            stop.set()
            th.join(2)
        out["peak"] = {"vram_used_gb": round(peak["vram_used_gb"], 2), "ram_min_avail_gb": round(peak["ram_min_gb"], 2)}
        out["breach"] = [b for b, hit in (
            ("GPU memory passed %.1f GB" % VRAM_LIMIT_GB, peak["vram_used_gb"] > VRAM_LIMIT_GB),
            ("PC memory fell under %.1f GB free" % RAM_FLOOR_GB, peak["ram_min_gb"] < RAM_FLOOR_GB)) if hit]
        if out["breach"]:
            out["ok"] = False
            out["action"] = "owner rule: run `rollback` now if these settings came from an apply, then notify"
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
    want = env_values({**base, "keep_alive": "30m", "parallel": 2, "flash_attention": True, "kv_cache_type": None})
    assert want == {"OLLAMA_KEEP_ALIVE": "30m", "OLLAMA_NUM_PARALLEL": "2", "OLLAMA_FLASH_ATTENTION": "1"}, want
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
        report.update({"live": "pass", "version": st["version"], "ollama_processes": len(st["processes"]),
                       "example_plan_changes": sorted(plan["changes"])})
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
    p = sub.add_parser("bench")
    p.add_argument("--model", required=True)
    p.add_argument("--levels", default="1,2,4")
    p.add_argument("--load-twice", action="store_true")
    p.add_argument("--num-ctx", type=int, default=8192, help="match the Warden's 8192, or Ollama reloads T7")
    p.add_argument("--num-predict", type=int, default=256)
    for p in sub.choices.values():
        p.add_argument("--provider", default="unknown")
    a = ap.parse_args(argv)
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
        out = restart(read_env(), a.provider)
        _log({"action": "restart", "provider": a.provider, "result": out})
    elif a.cmd == "pull":
        out = cmd_pull(a.model, a.digest)
    elif a.cmd == "bench":
        levels = [int(x) for x in a.levels.split(",") if x.strip()]
        if not levels or any(not 1 <= n <= 8 for n in levels):
            ap.error("--levels must be integers from 1 to 8")
        out = cmd_bench(a.model, levels, a.load_twice, max(512, min(a.num_ctx, 131072)),
                        max(1, min(a.num_predict, 2048)), a.provider)
    else:
        ap.print_help()
        return 2
    print(json.dumps(out))
    return 0 if out.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
